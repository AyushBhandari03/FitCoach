"""
Step 7: Chat with your final edge-deployed FitCoach model (Q4_K_M quantized, runs on CPU).
"""
from pathlib import Path
from llama_cpp import Llama
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
GGUF_PATH = ROOT / "gguf_models" / "fitcoach-q4_k_m.gguf"
MERGED_DIR = ROOT / "merged_model" / "fitcoach_merged"
SYSTEM_MSG = "You are FitCoach, a friendly and knowledgeable personal nutrition and fitness assistant."


def main():
    if not GGUF_PATH.exists():
        raise SystemExit(f"{GGUF_PATH} not found. Complete Step 5 (quantization) first.")

    print("Loading FitCoach (Q4_K_M, quantized, CPU)... this only takes a few seconds.")
    llm = Llama(model_path=str(GGUF_PATH), n_ctx=4096, verbose=False)
    tokenizer = AutoTokenizer.from_pretrained(str(MERGED_DIR))

    history = []
    print("\nFitCoach is ready! Ask about nutrition or fitness. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        history = history[-8:]
        messages = [{"role": "system", "content": SYSTEM_MSG}] + history
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        out = llm.create_completion(prompt=prompt, max_tokens=200, temperature=0.7, stop=["<|im_end|>"])
        reply = out["choices"][0]["text"].strip()
        print(f"FitCoach: {reply}\n")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
