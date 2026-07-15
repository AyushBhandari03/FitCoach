"""
Step 4: Merge the LoRA adapter into the base model and export a standalone
fine-tuned model (HF format), ready for GGUF conversion.
"""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "merged_model" / "lora_adapter"
MERGED_OUT = ROOT / "merged_model" / "fitcoach_merged"


def main():
    if not ADAPTER_DIR.exists():
        raise SystemExit(f"{ADAPTER_DIR} not found. Run 03_finetune_lora.py first.")

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))

    print("Loading LoRA adapter and merging...")
    peft_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    merged_model = peft_model.merge_and_unload()

    MERGED_OUT.mkdir(parents=True, exist_ok=True)
    # save in fp16 to keep the on-disk footprint (and later GGUF f16 conversion) small
    merged_model.half()
    merged_model.save_pretrained(str(MERGED_OUT), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_OUT))

    print(f"\nMerged fine-tuned model saved to {MERGED_OUT}")
    print("Next: convert this folder to GGUF (see README Step 5).")


if __name__ == "__main__":
    main()
