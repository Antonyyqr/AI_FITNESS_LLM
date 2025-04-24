#!/usr/bin/env python3
# gen_dataset.py

import json
import random

# --- 1) Define pools for random sampling ---
ages            = list(range(18, 61))           # 18–60 years
weights_kg      = list(range(50, 101))          # 50–100 kg
genders         = ["male", "female"]
purposes        = ["build muscle", "fat loss", "maintenance"]
body_parts      = ["upper body", "lower body", "full body"]
gym_access_opts = [True, False]
intensities     = ["low", "moderate", "high"]

# --- 2) Define workout templates by (body_part, intensity, gym_access) ---
WORKOUT_TEMPLATES = {
    ("upper body", "high", True): [
        "Bench Press 4×8, Pull-ups 3×6, Overhead Press 3×8",
        "Incline Dumbbell Press 3×10, Lat Pulldown 3×8, Dips 3×12"
    ],
    ("upper body", "moderate", True): [
        "Dumbbell Shoulder Press 3×12, Cable Rows 3×12, Push-ups 3×15",
        "Chest Fly 3×12, Seated Row 3×12, Triceps Pushdown 3×12"
    ],
    ("upper body", "low", True): [
        "Banded Pull-aparts 3×15, Light Dumbbell Curls 2×15, Resistance-band Push-downs 2×15",
        "Machine Chest Press 2×12, Machine Row 2×12, Lateral Raises 2×15"
    ],
    ("upper body", "high", False): [
        "Push-ups 4×15, Pike Push-ups 3×10, Inverted Rows 3×10",
        "Diamond Push-ups 3×12, Archer Push-ups 3×8, Table Rows 3×12"
    ],
    ("upper body", "moderate", False): [
        "Standard Push-ups 3×12, Bodyweight Rows 3×12, Plank to Push-up 2×10",
        "Chair Dips 3×12, Towel Rows 3×12, Decline Push-ups 2×10"
    ],
    ("upper body", "low", False): [
        "Wall Push-ups 3×15, Door-frame Rows 2×15, Isometric Bicep Holds 2×20s",
        "Incline Push-ups 2×12, Seated Wall-fly Holds 2×20s, Band Pull-aparts 2×15"
    ],
    ("lower body", "high", True): [
        "Back Squat 4×8, Deadlift 3×5, Lunges 3×10 per leg",
        "Leg Press 4×10, Romanian Deadlift 3×8, Bulgarian Split Squat 3×10"
    ],
    ("lower body", "moderate", True): [
        "Goblet Squat 3×12, Hamstring Curl 3×12, Calf Raise 3×15",
        "Step-ups 3×12, Good Mornings 3×10, Glute Bridge 3×15"
    ],
    ("lower body", "low", True): [
        "Bodyweight Squats 3×15, Lying Hip Adduction 2×15, Standing Calf Raises 2×20",
        "Box Squat to Chair 2×12, Glute Bridge Hold 2×30s, Bodyweight Lunges 2×12"
    ],
    ("lower body", "high", False): [
        "Jump Squats 4×12, Single-leg Deadlift 3×8 per leg, Pistol Squats 3×5 per leg",
        "Broad Jumps 4×8, Skater Squats 3×10, Walking Lunges 3×12 per leg"
    ],
    ("lower body", "moderate", False): [
        "Squat to Chair 3×15, Reverse Lunges 3×12 per leg, Calf Raises on Stairs 3×20",
        "Step-ups on Chair 3×12, Glute Bridge 3×15, Side-lying Clamshell 3×15"
    ],
    ("lower body", "low", False): [
        "Wall Sit 3×30s, Standing Leg Curls 2×15 per leg, Seated Calf Raise (bodyweight) 2×20",
        "Static Lunge Hold 2×30s per leg, Glute Bridge Hold 2×30s, Calf Stretch Holds 2×30s"
    ],
    ("full body", "high", True): [
        "Deadlift 3×5, Pull-ups 3×6, Squat 3×8, Bench Press 3×8",
        "Clean and Press 3×5, Front Squat 3×8, Chin-ups 3×8, Dips 3×10"
    ],
    ("full body", "moderate", True): [
        "Thrusters 3×10, Kettlebell Swings 3×15, Pull-ups 3×8, Step-ups 3×12",
        "Circuit: Goblet Squat, Push-press, Bent-over Row (3 rounds × 10 reps each)"
    ],
    ("full body", "low", True): [
        "Machine Circuit: Leg Press, Chest Press, Lat Pulldown (2 rounds × 12 reps)",
        "Body-pump Class Style Circuit (2 rounds of 12 reps each exercise)"
    ],
    ("full body", "high", False): [
        "Burpees 4×15, Jumping Lunges 3×12 per leg, Push-ups 3×15, Inverted Rows 3×12",
        "Mountain Climbers 3×30s, Squat Jumps 3×12, Pike Push-ups 3×10, Plank Rows 3×12"
    ],
    ("full body", "moderate", False): [
        "Circuit: Squats, Push-ups, Bodyweight Rows, Plank (3 rounds × 10 reps)",
        "Tabata: Burpees, Air Squats, Mountain Climbers, Push-ups (8 × 20s work)"
    ],
    ("full body", "low", False): [
        "Dynamic Stretching Circuit (leg swings, arm circles, lunges) 2×10 each",
        "Yoga Flow: Sun Salutations + Bodyweight Lunges + Push-ups (2 rounds)"
    ]
}

# --- 3) Generate examples ---
NUM_EXAMPLES = 300
OUTPUT_FILE  = "fitness_workout_only.jsonl"

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for _ in range(NUM_EXAMPLES):
        age         = random.choice(ages)
        weight      = random.choice(weights_kg)
        gender      = random.choice(genders)
        purpose     = random.choice(purposes)
        body_part   = random.choice(body_parts)
        gym_access  = random.choice(gym_access_opts)
        intensity   = random.choice(intensities)

        # pick templates
        key = (body_part, intensity, gym_access)
        template_list = WORKOUT_TEMPLATES.get(key)
        if not template_list:
            # fallback to full body moderate no-gym
            template_list = WORKOUT_TEMPLATES[("full body","moderate",False)]
        workout = random.choice(template_list)

        # Build prompt & completion
        prompt = (
            f"User: Age={age}, weight={weight}kg, gender={gender}, "
            f"purpose={purpose}, body_part={body_part}, gym_access={'yes' if gym_access else 'no'}, "
            f"intensity={intensity}. Recommend a workout."
        )
        completion = f"Assistant: Today's workout is {workout}."

        fout.write(json.dumps({
            "prompt": prompt,
            "completion": completion
        }, ensure_ascii=False) + "\n")

print(f"✅ Generated {NUM_EXAMPLES} workout-only examples in {OUTPUT_FILE}")
