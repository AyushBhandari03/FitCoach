"""
Step 3: Distillation/Fine-Tuning
LoRA fine-tunes a small student model (SmolLM2-360M-Instruct) on the synthetic
nutrition & fitness dataset, on CPU. Uses PEFT (LoRA), not QLoRA/bitsandbytes,
since there's no CUDA GPU available on this machine.
"""
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

# ---------------------- Config (tune these if training feels too slow) ----------------------
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
MAX_TRAIN_EXAMPLES = None   # e.g. set to 300 to train on a subset and go faster
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_LENGTH = 512
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8        # effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS = 16
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "train.jsonl"
VAL_PATH = ROOT / "data" / "val.jsonl"
ADAPTER_OUT = ROOT / "merged_model" / "lora_adapter"

SYSTEM_MSG = "You are FitCoach, a friendly and knowledgeable personal nutrition and fitness assistant."


def format_example(tokenizer, instruction: str, response: str):
    """Tokenize a (prompt, response) pair via the model's chat template, masking
    the loss on the prompt tokens so the model only learns to produce the answer."""
    prompt_messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": instruction},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_messages = prompt_messages + [{"role": "assistant", "content": response}]
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        full_text, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH
    )["input_ids"]

    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    for i in range(prompt_len):
        labels[i] = -100

    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def main():
    print(f"Loading tokenizer & base model: {BASE_MODEL} (downloads once, then cached)")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if not TRAIN_PATH.exists():
        raise SystemExit(f"{TRAIN_PATH} not found. Run 02_prepare_dataset.py first.")

    raw_train = load_dataset("json", data_files=str(TRAIN_PATH))["train"]
    raw_val = load_dataset("json", data_files=str(VAL_PATH))["train"]

    if MAX_TRAIN_EXAMPLES:
        raw_train = raw_train.select(range(min(MAX_TRAIN_EXAMPLES, len(raw_train))))

    def _map_fn(ex):
        return format_example(tokenizer, ex["instruction"], ex["response"])

    train_ds = raw_train.map(_map_fn, remove_columns=raw_train.column_names)
    val_ds = raw_val.map(_map_fn, remove_columns=raw_val.column_names)

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, label_pad_token_id=-100
    )

    training_args = TrainingArguments(
        output_dir=str(ROOT / "merged_model" / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print(f"Starting LoRA fine-tuning on CPU: {len(train_ds)} train / {len(val_ds)} val examples")
    trainer.train()

    ADAPTER_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_OUT))
    tokenizer.save_pretrained(str(ADAPTER_OUT))
    print(f"\nLoRA adapter saved to {ADAPTER_OUT}")


if __name__ == "__main__":
    main()
