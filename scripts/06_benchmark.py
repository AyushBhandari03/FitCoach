"""
Step 6: Benchmarking
Compares 5 model variants:
  1. baseline_fp32   - original SmolLM2-360M-Instruct, NOT fine-tuned (HF, fp32, CPU)
  2. finetuned_fp32   - LoRA-fine-tuned & merged model, unquantized (HF, fp32, CPU)
  3. gguf_f16          - fine-tuned model converted to GGUF, fp16
  4. gguf_q8_0         - fine-tuned model, 8-bit quantized GGUF
  5. gguf_q4_k_m       - fine-tuned model, 4-bit quantized GGUF

For each: perplexity (on val set), generation speed (tokens/sec), and peak RAM (RSS) usage.
Each variant is benchmarked in its own subprocess so memory measurements don't leak
between models. Results are printed as a table and plotted to results/*.png.

Run with no arguments to benchmark everything:
    python scripts/06_benchmark.py
"""
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
MERGED_DIR = ROOT / "merged_model" / "fitcoach_merged"
GGUF_DIR = ROOT / "gguf_models"
VAL_PATH = ROOT / "data" / "val.jsonl"
RESULTS_DIR = ROOT / "results"

SYSTEM_MSG = "You are FitCoach, a friendly and knowledgeable personal nutrition and fitness assistant."
NUM_PPL_EXAMPLES = 30      # how many val examples to use for perplexity (keep small -> CPU speed)
GEN_NEW_TOKENS = 100       # tokens to generate for the speed benchmark
GEN_PROMPT = "What should I eat before a morning workout?"

VARIANTS = ["baseline_fp32", "finetuned_fp32", "gguf_f16", "gguf_q8_0", "gguf_q4_k_m"]


class MemSampler:
    """Samples this process's RSS memory in a background thread to approximate peak usage."""
    def __init__(self, interval=0.1):
        import psutil
        self.process = psutil.Process(os.getpid())
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop:
            try:
                rss = self.process.memory_info().rss / (1024 * 1024)
                self.peak_mb = max(self.peak_mb, rss)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop = True
        self._thread.join(timeout=1)
        return self.peak_mb


def load_val_examples(n):
    examples = []
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
            if len(examples) >= n:
                break
    return examples


# ------------------------- HF (transformers) benchmarking -------------------------
def run_hf_variant(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sampler = MemSampler()
    sampler.start()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()

    # ---- perplexity on response tokens only ----
    examples = load_val_examples(NUM_PPL_EXAMPLES)
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for ex in examples:
            prompt_msgs = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": ex["instruction"]},
            ]
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            full_msgs = prompt_msgs + [{"role": "assistant", "content": ex["response"]}]
            full_text = tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False,
                                  truncation=True, max_length=512)["input_ids"]
            if len(full_ids) <= len(prompt_ids):
                continue
            input_ids = torch.tensor([full_ids])
            labels = torch.tensor([full_ids])
            labels[0, :len(prompt_ids)] = -100
            out = model(input_ids=input_ids, labels=labels)
            n_resp_tokens = len(full_ids) - len(prompt_ids)
            total_nll += out.loss.item() * n_resp_tokens
            total_tokens += n_resp_tokens
    perplexity = math.exp(total_nll / total_tokens) if total_tokens else float("nan")

    # ---- generation speed ----
    gen_msgs = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": GEN_PROMPT},
    ]
    gen_prompt_text = tokenizer.apply_chat_template(
        gen_msgs, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(gen_prompt_text, return_tensors="pt")["input_ids"]
    start = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            input_ids, max_new_tokens=GEN_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - start
    n_new = out_ids.shape[1] - input_ids.shape[1]
    tokens_per_sec = n_new / elapsed if elapsed > 0 else float("nan")

    peak_mb = sampler.stop()
    file_size_mb = get_dir_size_mb(model_path)

    return {
        "perplexity": perplexity,
        "tokens_per_sec": tokens_per_sec,
        "peak_ram_mb": peak_mb,
        "model_size_mb": file_size_mb,
    }


# ------------------------- GGUF (llama-cpp-python) benchmarking -------------------------
def run_gguf_variant(gguf_path: str):
    from llama_cpp import Llama
    from transformers import AutoTokenizer

    sampler = MemSampler()
    sampler.start()

    llm = Llama(model_path=gguf_path, n_ctx=1024, n_threads=os.cpu_count(), verbose=False, logits_all=True)
    # Reuse the exact same chat template used during training/merging, loaded locally
    # from the merged model folder (no download needed).
    hf_tok = AutoTokenizer.from_pretrained(str(MERGED_DIR))

    def build_prompt(instruction: str) -> str:
        msgs = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": instruction},
        ]
        return hf_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # ---- perplexity on response tokens only, via echoed logprobs ----
    examples = load_val_examples(NUM_PPL_EXAMPLES)
    total_nll, total_tokens = 0.0, 0
    for ex in examples:
        prompt_text = build_prompt(ex["instruction"])
        full_text = prompt_text + ex["response"]

        prompt_tokens = llm.tokenize(prompt_text.encode("utf-8"))
        full_tokens = llm.tokenize(full_text.encode("utf-8"))
        if len(full_tokens) <= len(prompt_tokens):
            continue

        result = llm.create_completion(
            prompt=full_text, max_tokens=0, echo=True, logprobs=1
        )
        token_logprobs = result["choices"][0]["logprobs"]["token_logprobs"]
        # token_logprobs aligns with the echoed prompt's tokenization; take the tail
        # portion corresponding to the response tokens.
        n_resp = len(full_tokens) - len(prompt_tokens)
        resp_logprobs = [lp for lp in token_logprobs[-n_resp:] if lp is not None]
        if not resp_logprobs:
            continue
        total_nll += -sum(resp_logprobs)
        total_tokens += len(resp_logprobs)
    perplexity = math.exp(total_nll / total_tokens) if total_tokens else float("nan")

    # ---- generation speed ----
    gen_prompt = build_prompt(GEN_PROMPT)
    start = time.time()
    out = llm.create_completion(prompt=gen_prompt, max_tokens=GEN_NEW_TOKENS)
    elapsed = time.time() - start
    n_new = out["usage"]["completion_tokens"]
    tokens_per_sec = n_new / elapsed if elapsed > 0 else float("nan")

    peak_mb = sampler.stop()
    file_size_mb = os.path.getsize(gguf_path) / (1024 * 1024)

    return {
        "perplexity": perplexity,
        "tokens_per_sec": tokens_per_sec,
        "peak_ram_mb": peak_mb,
        "model_size_mb": file_size_mb,
    }


def get_dir_size_mb(path):
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def benchmark_single(variant: str) -> dict:
    if variant == "baseline_fp32":
        return run_hf_variant(BASE_MODEL)
    elif variant == "finetuned_fp32":
        return run_hf_variant(str(MERGED_DIR))
    elif variant == "gguf_f16":
        return run_gguf_variant(str(GGUF_DIR / "fitcoach-f16.gguf"))
    elif variant == "gguf_q8_0":
        return run_gguf_variant(str(GGUF_DIR / "fitcoach-q8_0.gguf"))
    elif variant == "gguf_q4_k_m":
        return run_gguf_variant(str(GGUF_DIR / "fitcoach-q4_k_m.gguf"))
    else:
        raise ValueError(f"Unknown variant {variant}")


def main():
    if len(sys.argv) > 1:
        # child-process mode: run one variant and print JSON to stdout
        variant = sys.argv[1]
        try:
            result = benchmark_single(variant)
        except Exception as e:
            result = {"error": str(e)}
        print("RESULT_JSON:" + json.dumps(result))
        return

    # parent mode: spawn a fresh subprocess per variant for isolated memory readings
    import subprocess

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for variant in VARIANTS:
        print(f"\n=== Benchmarking {variant} ===")
        gguf_file = GGUF_DIR / f"fitcoach-{variant.split('_', 1)[1]}.gguf" if variant.startswith("gguf") else None
        if variant.startswith("gguf") and not gguf_file.exists():
            print(f"  skipped: {gguf_file} not found (run Step 5 first)")
            continue
        if variant == "finetuned_fp32" and not MERGED_DIR.exists():
            print(f"  skipped: {MERGED_DIR} not found (run Step 4 first)")
            continue

        proc = subprocess.run(
            [sys.executable, __file__, variant],
            capture_output=True, text=True,
        )
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON:")), None)
        if line is None:
            print(f"  FAILED. stderr:\n{proc.stderr[-2000:]}")
            continue
        result = json.loads(line[len("RESULT_JSON:"):])
        if "error" in result:
            print(f"  FAILED: {result['error']}")
            continue
        all_results[variant] = result
        print(f"  perplexity={result['perplexity']:.2f}  "
              f"tokens/sec={result['tokens_per_sec']:.2f}  "
              f"peak_ram={result['peak_ram_mb']:.0f}MB  "
              f"size={result['model_size_mb']:.0f}MB")

    if not all_results:
        print("\nNo results collected — make sure earlier steps have been run.")
        return

    with open(RESULTS_DIR / "benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    make_plots(all_results)
    print(f"\nSaved results table to {RESULTS_DIR / 'benchmark_results.json'}")
    print(f"Saved plots to {RESULTS_DIR}")


def make_plots(all_results: dict):
    import matplotlib.pyplot as plt

    variants = list(all_results.keys())
    perplexities = [all_results[v]["perplexity"] for v in variants]
    speeds = [all_results[v]["tokens_per_sec"] for v in variants]
    mems = [all_results[v]["peak_ram_mb"] for v in variants]
    sizes = [all_results[v]["model_size_mb"] for v in variants]

    # Plot 1: Perplexity vs Inference Latency (tokens/sec)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(speeds, perplexities, s=100)
    for v, x, y in zip(variants, speeds, perplexities):
        ax.annotate(v, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Inference speed (tokens/sec) — higher is better")
    ax.set_ylabel("Perplexity — lower is better")
    ax.set_title("Perplexity vs Inference Latency Trade-off")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "perplexity_vs_latency.png", dpi=150)
    plt.close(fig)

    # Plot 2: Memory footprint (peak RAM and on-disk model size)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(variants))
    width = 0.35
    ax.bar([i - width / 2 for i in x], mems, width, label="Peak RAM (MB)")
    ax.bar([i + width / 2 for i in x], sizes, width, label="Model size on disk (MB)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(variants, rotation=20, ha="right")
    ax.set_ylabel("MB")
    ax.set_title("Memory Footprint by Variant")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "memory_footprint.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
