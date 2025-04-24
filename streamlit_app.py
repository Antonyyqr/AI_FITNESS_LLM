# #!/usr/bin/env python3
# # streamlit_app.py

# import re
# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from macros import compute_macros

# @st.cache_resource
# def load_model():
#     tokenizer = AutoTokenizer.from_pretrained("./ft_model_simple")
#     model     = AutoModelForCausalLM.from_pretrained("./ft_model_simple")
#     return tokenizer, model

# def generate_workout(age, weight_val, unit, gender,
#                      purpose, body_part, gym_access, intensity,
#                      tokenizer, model):
#     print("Generating workout...")
#     print(age, weight_val, unit, gender, purpose, body_part, gym_access, intensity)
#     prompt = (
#         f"User: Age={age}, weight={weight_val}{unit}, gender={gender}, "
#         f"purpose={purpose}, body_part={body_part}, gym_access={gym_access}, "
#         f"intensity={intensity}.\n"
#         "Assistant: Today's workout is"
#     )
#     print(f'Prompt: "{prompt}"')
#     inputs  = tokenizer(prompt, return_tensors="pt", padding=False)
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=50,
#         do_sample=False,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.eos_token_id,
#     )
#     full  = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     reply = full[len(prompt):].strip()
#     m = re.search(r"(Today's workout.*?\.)", reply)
#     if m:
#         return m.group(1)
#     return reply.split(".")[0] + "."

# # ————————————— Streamlit UI —————————————
# tokenizer, model = load_model()
# st.title("🏋️ AI Workout & Meal Macro Generator")


# with st.form("user_info"):
#     age        = st.number_input("Age", min_value=10, max_value=100, value=30)
#     weight_val = st.number_input("Weight", min_value=30.0, max_value=300.0,
#                                  value=75.0, format="%.1f")
#     unit       = st.selectbox("Unit", ["kg", "lb"])
#     gender     = st.selectbox("Gender", ["male", "female"])
#     purpose    = st.selectbox("Purpose", ["build muscle", "fat loss", "maintenance"])
#     body_part  = st.selectbox("Body Part Focus", ["upper body", "lower body", "full body"])
#     gym_access = st.selectbox("Gym Access", ["yes", "no"])
#     intensity  = st.selectbox("Intensity", ["low", "moderate", "high"])
#     carb_phase = st.selectbox("Carb phase", ["low", "moderate", "high"])
#     submitted  = st.form_submit_button("Generate Workout & Macros")

# # age        = st.number_input("Age", min_value=10, max_value=100, value=30)
# # weight_val = st.number_input("Weight", min_value=30.0, max_value=300.0,
# #                                 value=75.0, format="%.1f")
# # unit       = st.selectbox("Unit", ["kg", "lb"])
# # gender     = st.selectbox("Gender", ["male", "female"])
# # purpose    = st.selectbox("Purpose", ["build muscle", "fat loss", "maintenance"])
# # body_part  = st.selectbox("Body Part Focus", ["upper body", "lower body", "full body"])
# # gym_access = st.selectbox("Gym Access", ["yes", "no"])
# # intensity  = st.selectbox("Intensity", ["low", "moderate", "high"])
# # carb_phase = st.selectbox("Carb phase", ["low", "moderate", "high"])
# # submitted  = st.form_submit_button("Generate Workout & Macros")

# if submitted:
#     workout = generate_workout(
#         age, weight_val, unit, gender,
#         purpose, body_part, gym_access, intensity,
#         tokenizer, model
#     )
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
# # print all the inputs as a dict
# # print(age, weight_val, unit, gender, purpose, body_part, gym_access, intensity, carb_phase)
# # if st.button("Generate Workout & Macros"):
# #     tokenizer, model = load_model()
# #     workout = generate_workout(
# #         age, weight_val, unit, gender,
# #         purpose, body_part, gym_access, intensity,
# #         tokenizer, model
# #     )
# #     print(age, weight_val, unit, gender, purpose, body_part, gym_access, intensity, carb_phase)
# #     print(workout)
# #     # compute macros
# #     weight_lb = weight_val * 2.20462 if unit=="kg" else weight_val
# #     carb_g, prot_g, fat_g = compute_macros(weight_lb, carb_phase)

# #     st.subheader("💪 Workout Recommendation")
# #     st.write(workout)

# #     st.subheader("🥗 Meal Macros")
# #     st.markdown(
# #         f"- **Carbs:** {carb_g} g  \n"
# #         f"- **Protein:** {prot_g} g  \n"
# #         f"- **Fat:** {fat_g} g"
# #     

#!/usr/bin/env python3
# streamlit_app.py

#!/usr/bin/env python3
# streamlit_app.py

#!/usr/bin/env python3
# streamlit_app.py

import os
import re
import openai
import warnings
import streamlit as st
from openai import OpenAI
from macros import compute_macros

openai.api_key = st.secrets["OPENAI_API_KEY"]

# ─── suppress deprecation warnings ─────────────────────────────────────────────
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ─── Page config & banner ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fitness Coach",
    page_icon="💪",
    layout="wide"
)

st.markdown(
    """
    <div style="text-align:center; margin-bottom: 1rem;">
      <h1>🏋️ AI Fitness Coach</h1>
      <p style="font-size:18px; color:gray;">
        Get personalized workout plans & meal macros in seconds
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.image(
    "https://images.unsplash.com/photo-1605296867304-46d5465a13f1"
    "?auto=format&fit=crop&w=1200&q=80",
    use_container_width=True,
    caption="Train smarter, not harder"
)

# ─── OpenAI client setup ───────────────────────────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 Please set the OPENAI_API_KEY environment variable.")
    st.stop()
client = OpenAI(api_key=api_key)

# ─── Helper: split & reformat workout moves ────────────────────────────────────
def format_workout_moves(raw: str) -> list[str]:
    # 1) remove trailing period, split by commas or " and "
    parts = re.split(r",\s*|\s+and\s+", raw.strip().rstrip("."))
    moves = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 2) replace "X sets of Y" → "X x Y"
        p = re.sub(r"(\d+)\s*sets of\s*(\d+)", r"\1 x \2", p, flags=re.IGNORECASE)
        moves.append(p)
    return moves

# ─── Workout generation ────────────────────────────────────────────────────────
def generate_workout_via_api(age, weight_val, unit, gender,
                             purpose, body_part, gym_access, intensity):
    user_prompt = (
        f"User: Age={age}, weight={weight_val}{unit}, gender={gender}, "
        f"purpose={purpose}, body_part={body_part}, gym_access={gym_access}, "
        f"intensity={intensity}.\n"
        "Assistant: Today's workout is"
    )
    system_message = {
        "role": "system",
        "content": (
            "You are a professional private fitness coach "
            "Produce a 2–3 sentence workout plan beginning with “Today's workout is” "
            "and ending with a period. Include at least six exercises with sets×reps. "
            "If gym_access=yes use equipment; otherwise bodyweight only. "
            "Finish with a brief note on structure (rest/supersets) need to be a full sentence."
        )
    }
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[system_message, {"role":"user","content":user_prompt}],
        temperature=0.7,
        top_p=0.9,
        max_tokens=200,
        stop=["\n"]
    )
    text = resp.choices[0].message.content.strip()
    return text if text.endswith(".") else text + "."

# ─── Sidebar form ──────────────────────────────────────────────────────────────
with st.sidebar.form("user_info", clear_on_submit=False):
    st.header("Tell me about you")
    age        = st.number_input("Age", 10, 100, 30)
    weight_val = st.number_input("Weight", 30.0, 300.0, 75.0, format="%.1f")
    unit       = st.selectbox("Unit", ["kg", "lb"])
    gender     = st.selectbox("Gender", ["male", "female"])
    purpose    = st.selectbox("Purpose", ["build muscle", "fat loss", "maintenance"])
    body_part  = st.selectbox("Body Part Focus", ["upper body", "lower body", "full body"])
    gym_access = st.selectbox("Gym Access", ["yes", "no"])
    intensity  = st.selectbox("Intensity", ["low", "moderate", "high"])
    carb_phase = st.selectbox("Carb Phase", ["low", "moderate", "high"])
    submitted  = st.form_submit_button("Generate Workout & Macros")

# ─── Main content ──────────────────────────────────────────────────────────────
if submitted:
    # 1) Generate workout via API
    raw_workout = generate_workout_via_api(
        age, weight_val, unit, gender,
        purpose, body_part, gym_access, intensity
    )

    # 2) Compute macros
    weight_lb = weight_val * 2.20462 if unit=="kg" else weight_val
    carb_g, prot_g, fat_g = compute_macros(weight_lb, carb_phase)

    # 3) Display results in two columns
    col1, col2 = st.columns((3, 1), gap="large")
    with col1:
        st.subheader("💪 Workout Plan")
        moves = format_workout_moves(raw_workout)
        for m in moves:
            st.markdown(f"- {m}.")

    with col2:
        st.subheader("🥗 Meal Macros")
        st.metric("Carbs",    f"{carb_g} g")
        st.metric("Protein",  f"{prot_g} g")
        st.metric("Fat",      f"{fat_g} g")

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:gray; font-size:12px;'>"
        "Built with ❤️ by Qirui Yang"
        "</p>",
        unsafe_allow_html=True
    )
