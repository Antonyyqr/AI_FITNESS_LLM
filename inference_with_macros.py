#!/usr/bin/env python3
# inference_with_macros.py

import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from macros import compute_macros  # your nutrient calculator

def ask(prompt, valid=None):
    while True:
        ans = input(prompt).strip()
        if not valid or ans in valid:
            return ans
        print(f"→ Please choose one of: {valid}")

def get_workout(prompt: str, tokenizer, model) -> str:
    inputs  = tokenizer(prompt, return_tensors="pt", padding=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,               # greedy decoding
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    full  = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = full[len(prompt):].strip()
    m = re.search(r"(Today's workout.*?\.)", reply)
    if m:
        return m.group(1)
    return reply.split(".")[0].strip() + "."

def main():
    print("Interactive Workout & Meal Macro Generator\n")

    # 1) Weight input
    while True:
        w = input("Enter weight (e.g. 75kg or 165lb): ").strip()
        m = re.match(r"^(\d+(?:\.\d+)?)(kg|lb)$", w, re.I)
        if m:
            val  = float(m.group(1))
            unit = m.group(2).lower()
            break
        print("→ Invalid; enter like '75kg' or '165lb'")
    weight_lb      = val * 2.20462 if unit == "kg" else val
    weight_display = f"{val}{unit}"

    # 2) Other details
    age        = ask("Age (e.g. 28): ")
    gender     = ask("Gender [male/female]: ", valid=["male","female"])
    purpose    = ask("Purpose [build muscle/fat loss/maintenance]: ",
                     valid=["build muscle","fat loss","maintenance"])
    body_part  = ask("Body part focus [upper body/lower body/full body]: ",
                     valid=["upper body","lower body","full body"])
    gym_access = ask("Gym access [yes/no]: ", valid=["yes","no"])
    intensity  = ask("Intensity [low/moderate/high]: ",
                     valid=["low","moderate","high"])
    carb_phase = ask("Carb phase [low/moderate/high]: ",
                     valid=["low","moderate","high"])

    # 3) Build prompt for workout-only model
    prompt = (
        f"User: Age={age}, weight={weight_display}, gender={gender}, "
        f"purpose={purpose}, body_part={body_part}, gym_access={gym_access}, "
        f"intensity={intensity}.\n"
        "Assistant: Today's workout is"
    )

    # 4) Load your fine‐tuned model locally
    tokenizer = AutoTokenizer.from_pretrained("./ft_model_simple")
    model     = AutoModelForCausalLM.from_pretrained("./ft_model_simple")

    print("\nGenerating workout plan…\n")
    workout = get_workout(prompt, tokenizer, model)

    # 5) Compute meal macros
    carb_g, prot_g, fat_g = compute_macros(weight_lb, carb_phase)

    # 6) Display everything
    print("=== Your Prompt ===")
    print(prompt)
    print("\n=== Workout Recommendation ===")
    print(workout)
    print(
        f"\n🥗 Meal Macros ({carb_phase}-carb day at {weight_display}):\n"
        f"  • Carbs  : {carb_g} g\n"
        f"  • Protein: {prot_g} g\n"
        f"  • Fat    : {fat_g} g"
    )

if __name__ == "__main__":
    main()

