#!/usr/bin/env python3
# streamlit_app.py

import re
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from macros import compute_macros

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./ft_model_simple")
    model     = AutoModelForCausalLM.from_pretrained("./ft_model_simple")
    return tokenizer, model

def generate_workout(age, weight_val, unit, gender,
                     purpose, body_part, gym_access, intensity,
                     tokenizer, model):
    print("Generating workout...")
    print(age, weight_val, unit, gender, purpose, body_part, gym_access, intensity)
    prompt = (
        f"User: Age={age}, weight={weight_val}{unit}, gender={gender}, "
        f"purpose={purpose}, body_part={body_part}, gym_access={gym_access}, "
        f"intensity={intensity}.\n"
        "Assistant: Today's workout is"
    )
    print(f'Prompt: "{prompt}"')
    inputs  = tokenizer(prompt, return_tensors="pt", padding=False)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    full  = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = full[len(prompt):].strip()
    m = re.search(r"(Today's workout.*?\.)", reply)
    if m:
        return m.group(1)
    return reply.split(".")[0] + "."

# ————————————— Streamlit UI —————————————
tokenizer, model = load_model()
st.title("🏋️ AI Workout & Meal Macro Generator")


with st.form("user_info"):
    age        = st.number_input("Age", min_value=10, max_value=100, value=30)
    weight_val = st.number_input("Weight", min_value=30.0, max_value=300.0,
                                 value=75.0, format="%.1f")
    unit       = st.selectbox("Unit", ["kg", "lb"])
    gender     = st.selectbox("Gender", ["male", "female"])
    purpose    = st.selectbox("Purpose", ["build muscle", "fat loss", "maintenance"])
    body_part  = st.selectbox("Body Part Focus", ["upper body", "lower body", "full body"])
    gym_access = st.selectbox("Gym Access", ["yes", "no"])
    intensity  = st.selectbox("Intensity", ["low", "moderate", "high"])
    carb_phase = st.selectbox("Carb phase", ["low", "moderate", "high"])
    submitted  = st.form_submit_button("Generate Workout & Macros")

# age        = st.number_input("Age", min_value=10, max_value=100, value=30)
# weight_val = st.number_input("Weight", min_value=30.0, max_value=300.0,
#                                 value=75.0, format="%.1f")
# unit       = st.selectbox("Unit", ["kg", "lb"])
# gender     = st.selectbox("Gender", ["male", "female"])
# purpose    = st.selectbox("Purpose", ["build muscle", "fat loss", "maintenance"])
# body_part  = st.selectbox("Body Part Focus", ["upper body", "lower body", "full body"])
# gym_access = st.selectbox("Gym Access", ["yes", "no"])
# intensity  = st.selectbox("Intensity", ["low", "moderate", "high"])
# carb_phase = st.selectbox("Carb phase", ["low", "moderate", "high"])
# submitted  = st.form_submit_button("Generate Workout & Macros")

if submitted:
    workout = generate_workout(
        age, weight_val, unit, gender,
        purpose, body_part, gym_access, intensity,
        tokenizer, model
    )
    # compute macros
    weight_lb = weight_val * 2.20462 if unit=="kg" else weight_val
    carb_g, prot_g, fat_g = compute_macros(weight_lb, carb_phase)

    st.subheader("💪 Workout Recommendation")
    st.write(workout)

    st.subheader("🥗 Meal Macros")
    st.markdown(
        f"- **Carbs:** {carb_g} g  \n"
        f"- **Protein:** {prot_g} g  \n"
        f"- **Fat:** {fat_g} g"
    )
# print all the inputs as a dict
# print(age, weight_val, unit, gender, purpose, body_part, gym_access, intensity, carb_phase)
# if st.button("Generate Workout & Macros"):
#     tokenizer, model = load_model()
#     workout = generate_workout(
#         age, weight_val, unit, gender,
#         purpose, body_part, gym_access, intensity,
#         tokenizer, model
#     )
#     print(age, weight_val, unit, gender, purpose, body_part, gym_access, intensity, carb_phase)
#     print(workout)
#     # compute macros
#     weight_lb = weight_val * 2.20462 if unit=="kg" else weight_val
#     carb_g, prot_g, fat_g = compute_macros(weight_lb, carb_phase)

#     st.subheader("💪 Workout Recommendation")
#     st.write(workout)

#     st.subheader("🥗 Meal Macros")
#     st.markdown(
#         f"- **Carbs:** {carb_g} g  \n"
#         f"- **Protein:** {prot_g} g  \n"
#         f"- **Fat:** {fat_g} g"
#     )