import streamlit as st
from model_utils import (
    prefilter_check,
    load_mistral_classifier,
    unload_mistral_classifier,
    classify_prompt_with_mistral,
    load_juggernaut_pipeline,
    unload_juggernaut_pipeline,
    generate_image_with_juggernaut,
)
from PIL import Image
import io, os

st.set_page_config(page_title="Text→Image (Mistral 7B + Juggernaut XL)", layout="wide")

st.title("Text → Image (Mistral 7B classifier + Juggernaut XL generator)")
st.markdown(
    """
This app implements the workflow you requested.
Models used (configured): Mistral-7B-Instruct and RunDiffusion/Juggernaut-XL-v9.
"""
)

# Auto-read HF token from env if set (useful for Spaces repository secrets)
DEFAULT_HF_TOKEN = os.getenv("HF_TOKEN", "")

with st.sidebar:
    st.header("Generation options")
    mode = st.radio("Mode", options=["Only logical images", "Allow illogical images"])
    steps = st.slider("Steps", min_value=10, max_value=150, value=28, step=2)
    guidance_scale = st.slider("Guidance scale (CFG)", min_value=1.0, max_value=30.0, value=7.5, step=0.5)
    size = st.selectbox("Image size (max 1024)", options=[(512,512),(640,640),(768,768),(1024,1024)], index=0, format_func=lambda s: f\"{s[0]}x{s[1]}\" )
    seed = st.number_input("Seed (0 for random)", value=0, min_value=0, step=1)
    hf_token = st.text_input("Hugging Face Token (if models are gated)", type="password", value=DEFAULT_HF_TOKEN)

prompt = st.text_area("Enter prompt", height=140, value="A surreal portrait of a cyborg musician performing on a neon-lit rooftop at dusk.")

start = st.button("Generate")

# display area
image_slot = st.empty()
log_slot = st.empty()

# session state for models
if "mistral" not in st.session_state:
    st.session_state.mistral = None
if "juggernaut" not in st.session_state:
    st.session_state.juggernaut = None

def show_popup(title, text, kind="error"):
    try:
        with st.modal(title):
            st.write(text)
            st.button("OK")
    except Exception:
        if kind=="error":
            st.error(f\"{title}: {text}\")
        elif kind=="warning":
            st.warning(f\"{title}: {text}\")
        else:
            st.info(f\"{title}: {text}\")

if start:
    if not prompt.strip():
        show_popup(\"Empty prompt\", \"Please enter a prompt before generating.\", kind=\"warning\")
        st.stop()

    # Prefilter (always check sensitive keywords)
    pref_result = prefilter_check(prompt)
    if pref_result["sensitive_found"]:
        show_popup(\"Blocked — sensitive content detected\", f\"The prompt contains sensitive keyword(s): {', '.join(pref_result['sensitive_hits'])}\", kind=\"error\")
        st.stop()

    if mode == "Only logical images":
        if pref_result["illogical_found"]:
            show_popup(\"Illogical prompt detected by prefilter\", f\"Illogical keyword(s): {', '.join(pref_result['illogical_hits'])}\", kind=\"error\")
            st.stop()

        with st.spinner(\"Loading Mistral 7B classifier (quantized / memory efficient)...\"):
            st.session_state.mistral = load_mistral_classifier(hf_token=hf_token)
        with st.spinner(\"Classifying prompt (logical vs illogical)...\"):
            classification = classify_prompt_with_mistral(st.session_state.mistral, prompt)
        log_slot.info(f\"Classifier output: {classification}\")
        if classification.strip().upper() != \"LOGICAL\" and classification.strip().upper() != \"LOGICAL.\":
            show_popup(\"Prompt classified as ILL-LOGICAL\", \"The LLM classifier labeled this prompt as illogical; image generation stopped.\", kind=\"error\")
            unload_mistral_classifier(st.session_state.mistral)
            st.session_state.mistral = None
            st.stop()

        with st.spinner(\"Unloading Mistral to free memory...\"):
            unload_mistral_classifier(st.session_state.mistral)
            st.session_state.mistral = None

        with st.spinner(\"Loading Juggernaut XL image pipeline...\"):
            st.session_state.juggernaut = load_juggernaut_pipeline(hf_token=hf_token)
        with st.spinner(\"Generating image...\"):
            image = generate_image_with_juggernaut(st.session_state.juggernaut, prompt, steps=steps, guidance_scale=guidance_scale, size=size, seed=seed)
        image_slot.image(image, caption=\"Generated image\", use_column_width=True)
        buf = io.BytesIO()
        image.save(buf, format=\"PNG\")
        buf.seek(0)
        st.download_button(\"Download PNG\", data=buf, file_name=\"generated.png\", mime=\"image/png\")

        with st.spinner(\"Unloading Juggernaut to free memory...\"):
            unload_juggernaut_pipeline(st.session_state.juggernaut)
            st.session_state.juggernaut = None

    else:
        # Allow illogical images -> only sensitive prefilter (already passed)
        with st.spinner(\"Loading Juggernaut XL image pipeline...\"):
            st.session_state.juggernaut = load_juggernaut_pipeline(hf_token=hf_token)
        with st.spinner(\"Generating illogical-allowed image...\"):
            image = generate_image_with_juggernaut(st.session_state.juggernaut, prompt, steps=steps, guidance_scale=guidance_scale, size=size, seed=seed)
        image_slot.image(image, caption=\"Generated image\", use_column_width=True)
        buf = io.BytesIO()
        image.save(buf, format=\"PNG\")
        buf.seek(0)
        st.download_button(\"Download PNG\", data=buf, file_name=\"generated.png\", mime=\"image/png\")

        with st.spinner(\"Unloading Juggernaut to free memory...\"):
            unload_juggernaut_pipeline(st.session_state.juggernaut)
            st.session_state.juggernaut = None
