"""
Step 1: Data Curation
Generates a synthetic, high-quality nutrition & fitness Q&A dataset using
Groq's free Llama-3.1-8B API as the "teacher" model.

Output: data/raw_dataset.jsonl  (one {"instruction": ..., "response": ...} per line)
"""
import os
import json
import time
import random
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---------------------- Config ----------------------
TEACHER_MODEL = "llama-3.1-8b-instant"   # fast + free on Groq
NUM_BATCHES = 80                          # ~10 Q&A pairs per batch -> ~800 examples
PAIRS_PER_BATCH = 10
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw_dataset.jsonl"
TEMPERATURE = 0.9  # higher temp -> more diverse questions across batches

TOPICS = [
    "macronutrients and calorie counting",
    "meal planning for weight loss",
    "meal planning for muscle gain",
    "hydration and electrolytes",
    "pre- and post-workout nutrition",
    "protein sources for vegetarians and vegans",
    "intermittent fasting basics",
    "common dietary supplements (creatine, whey, multivitamins)",
    "sleep and recovery for fitness",
    "beginner strength training routines",
    "cardio vs strength training trade-offs",
    "healthy eating on a budget",
    "managing cravings and mindful eating",
    "sports nutrition for endurance athletes",
    "special diets (keto, Mediterranean, gluten-free)",
    "injury prevention and warm-up routines",
    "reading nutrition labels",
    "hydration during hot weather workouts",
    "nutrition for older adults staying active",
    "balancing fitness goals with a busy work schedule",
]

SYSTEM_PROMPT = """You are a certified nutrition and fitness coach creating training data for a
smaller AI assistant. Generate realistic, diverse questions a curious, health-conscious adult
might ask, paired with clear, accurate, encouraging, safe answers (2-5 sentences each).
Always include a brief safety caveat only when medically relevant (e.g. suggest consulting a
doctor for existing conditions), but do not overdo disclaimers.
Return ONLY valid JSON: a list of objects with keys "instruction" and "response". No prose,
no markdown fences, no extra commentary."""


def build_user_prompt(topic: str, n: int) -> str:
    return (
        f"Generate {n} diverse question-answer pairs about: {topic}.\n"
        f"Vary the phrasing, the persona asking (beginner, athlete, busy parent, older adult, "
        f"student, etc), and the specificity of the questions.\n"
        f'Respond with ONLY a JSON list like: '
        f'[{{"instruction": "...", "response": "..."}}, ...]'
    )


def extract_json_list(text: str):
    """Best-effort extraction of a JSON list even if the model adds stray text/fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("No JSON list found in model output")
    return json.loads(text[start : end + 1])


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise SystemExit(
            "Missing GROQ_API_KEY. Copy .env.example to .env and paste your free key from "
            "https://console.groq.com/keys"
        )

    client = Groq(api_key=api_key)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    written = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        for i in range(NUM_BATCHES):
            topic = random.choice(TOPICS)
            try:
                completion = client.chat.completions.create(
                    model=TEACHER_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(topic, PAIRS_PER_BATCH)},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=2048,
                )
                raw = completion.choices[0].message.content
                pairs = extract_json_list(raw)
            except Exception as e:
                print(f"[batch {i+1}/{NUM_BATCHES}] skipped due to error: {e}")
                continue

            new_in_batch = 0
            for pair in pairs:
                instr = str(pair.get("instruction", "")).strip()
                resp = str(pair.get("response", "")).strip()
                if len(instr) < 5 or len(resp) < 5:
                    continue
                h = hashlib.md5(instr.lower().encode()).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                f_out.write(json.dumps({"instruction": instr, "response": resp}) + "\n")
                written += 1
                new_in_batch += 1

            print(f"[batch {i+1}/{NUM_BATCHES}] topic='{topic}' +{new_in_batch} "
                  f"(total so far: {written})")

            time.sleep(0.5)  # be a little gentle on the free rate limit

    print(f"\nDone. Wrote {written} unique Q&A pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
