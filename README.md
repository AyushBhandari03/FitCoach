# FitCoach-Edge
Distilling & Quantizing a Tiny LLM for On-Device Personalized Nutrition & Fitness Coaching

## What this project does :

| Brief requirement          | How we do it here                                                             |
|-----------------------------|--------------------------------------------------------------------------------|
| Data Curation                | Free Groq API generates fresh synthetic Q&A data, **merged with your existing personalized-nutrition dataset from Colab** |
| Distillation / Fine-Tuning   | LoRA fine-tuning of SmolLM2-360M-Instruct on that data, using PEFT, on CPU     |
| Quantization                 | Convert to GGUF, quantize to 8-bit (Q8_0) and 4-bit (Q4_K_M)                   |
| Benchmarking                 | Perplexity vs tokens/sec vs RAM footprint, plotted for fp16 vs Q8_0 vs Q4_K_M  |

We use **GGUF quantization instead of bitsandbytes** because bitsandbytes 4-bit/8-bit requires an
NVIDIA CUDA GPU,and not in CPU. GGUF is explicitly listed as an
acceptable quantization format in your brief, and it's the standard free/local way to do this on
CPU-only hardware — llama.cpp .

## Project layout
```
fitcoach-edge/
  scripts/
    01_generate_dataset.py   # calls Groq API to build the synthetic dataset
    02_prepare_dataset.py    # cleans, dedupes, splits into train/val
    03_finetune_lora.py      # LoRA fine-tuning on CPU
    04_merge_and_export.py   # merges LoRA adapter into full model weights
    06_benchmark.py          # perplexity / latency / memory benchmarking + plots
    07_chat_cli.py           # chat with your final quantized model in the terminal
  data/
    colab_dataset.jsonl      # your existing dataset from earlier Colab work (shipped, cleaned)
                              # everything else in data/ gets generated as you run the steps
  merged_model/              # fine-tuned model (HF format) lands here
  gguf_models/               # quantized .gguf files land here
  results/                   # benchmark plots (.png) land here
  requirements.txt
  .env.example
```

## Step 0 — One-time setup (about 15-20 minutes)

1. **Install Python** (if not already): https://www.python.org/downloads/ — get 3.10 or 3.11
   (during install, tick "Add python.exe to PATH").
2. **Install Git** (if not already): https://git-scm.com/download/win — needed to grab llama.cpp later.
3. Open this `fitcoach-edge` folder in VS Code.
4. Open a terminal in VS Code (`` Ctrl+` ``) and create a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
   You should now see `(.venv)` at the start of your terminal line. Do this every time you open
   a new terminal for this project.
5. Install the Python dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   This installs PyTorch (CPU version), Hugging Face Transformers, PEFT (LoRA), Datasets,
   llama-cpp-python (for running GGUF models), matplotlib, psutil, and the Groq client.
   This step takes a while (~5-10 min) and downloads a few GB — it's free, just be patient.

6. **Get a free Groq API key** (this is our free "teacher" model — Groq gives free, fast access
   to Llama-3.1-8B/70B):
   - Go to https://console.groq.com/keys and sign up (free, no credit card).
   - Create an API key.
   - Copy `.env.example` to `.env` and paste your key in:
     ```
     GROQ_API_KEY=your_key_here
     ```

You're set up. Everything from here runs locally except the small teacher-data-generation calls
to Groq's free API.

## Step 1 — Generate fresh synthetic data (to supplement your Colab dataset)
```powershell
python scripts\01_generate_dataset.py
```
This asks Groq's Llama-3.1-8B to act as a certified nutrition & fitness coach and generate ~800
diverse Q&A pairs (macros, meal planning, weight loss/gain, hydration, supplements, workout
recovery, special diets, etc). Saves to `data/raw_dataset.jsonl`. Takes a few minutes, free.

This project also ships with `data/colab_dataset.jsonl` — the personalized-nutrition dataset
you generated earlier in Colab (1,522 examples, already cleaned of the 6 rows that had a broken
schema). Step 2 merges both sources together.

## Step 2 — Merge sources, clean & split the dataset
```powershell
python scripts\02_prepare_dataset.py
```
Merges `data/raw_dataset.jsonl` (fresh Groq data) with `data/colab_dataset.jsonl` (your earlier
Colab dataset), removes duplicate questions **across both sources**, filters out junk/too-short
answers, truncates extreme outlier-length responses, and writes the final `data/train.jsonl` and
`data/val.jsonl` (90/10 split). It prints a breakdown of how many examples came from each source
so you can see the mix. Expect roughly ~2,200-2,300 examples total after deduping.

## Step 3 — LoRA fine-tune the student model (SmolLM2-360M-Instruct)
```powershell
python scripts\03_finetune_lora.py
```
- Downloads the base model from Hugging Face the first time (~700MB, free, one-time).
- Trains a LoRA adapter (only a few million trainable params) on CPU.
- With the merged dataset (~2,000-2,300 training examples) and 3 epochs this will take longer
  than a small dataset would — plan for roughly 2-4 hours on your CPU (it prints progress as it
  goes, so you can gauge speed early and adjust). To go faster, either lower `NUM_EPOCHS` (2 is
  often enough) or set `MAX_TRAIN_EXAMPLES` (e.g. 1000) at the top of `03_finetune_lora.py` to
  train on a representative subset first, then do a full run later if you have time.
- Saves adapter to `merged_model/lora_adapter/`.

## Step 4 — Merge LoRA into the base model
```powershell
python scripts\04_merge_and_export.py
```
Merges the LoRA weights into the base model and saves a full standalone fine-tuned model to
`merged_model/fitcoach_merged/` in HF format (fp16 safetensors).

## Step 5 — Convert to GGUF and quantize (no compiling needed)
```powershell
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt
python convert_hf_to_gguf.py ..\merged_model\fitcoach_merged --outfile ..\gguf_models\fitcoach-f16.gguf --outtype f16
python convert_hf_to_gguf.py ..\merged_model\fitcoach_merged --outfile ..\gguf_models\fitcoach-q8_0.gguf --outtype q8_0
cd ..
```
That gives you fp16 and 8-bit (Q8_0) GGUF models with **no C++ build tools required**.

For true 4-bit (Q4_K_M), download the prebuilt `llama-quantize` tool (no compiling):
1. Go to https://github.com/ggerganov/llama.cpp/releases
2. Download the latest `llama-*-bin-win-avx2-x64.zip` (or `-vulkan-x64.zip` if you want to also
   try Radeon GPU acceleration later), unzip it anywhere.
3. Run:
   ```powershell
   .\path\to\unzipped\llama-quantize.exe gguf_models\fitcoach-f16.gguf gguf_models\fitcoach-q4_k_m.gguf Q4_K_M
   ```

You now have 3 model variants to compare: `fitcoach-f16.gguf`, `fitcoach-q8_0.gguf`,
`fitcoach-q4_k_m.gguf`.

## Step 6 — Benchmark: Perplexity vs Latency vs Memory
```powershell
python scripts\06_benchmark.py
```
Loads each variant (plus the original un-fine-tuned base model as a baseline), and for each:
- computes **perplexity** on the held-out validation set
- measures **tokens/sec** generation speed
- measures **peak RAM usage** and **file size on disk**

Prints a results table and saves two plots to `results/`:
- `perplexity_vs_latency.png`
- `memory_footprint.png`

## Step 7 — Chat with your final model
```powershell
python scripts\07_chat_cli.py
```
Simple terminal chat using the Q4_K_M quantized model — this is your final "edge-deployed"
nutrition & fitness assistant.

## Notes on your hardware
- All training/inference here is CPU-only fp32/fp16, no CUDA required.
- 16GB RAM is plenty for a 360M model; peak usage during training should stay well under 8GB.
- If Step 3 feels slow, reduce `MAX_TRAIN_EXAMPLES` or `NUM_EPOCHS` at the top of
  `03_finetune_lora.py` — the pipeline still works end-to-end, just with a slightly less polished
  model.
