
# Text→Image: Mistral 7B (classifier) + Juggernaut XL (image generator) — v2

This version hard-codes the model repository IDs you provided:
- Juggernaut XL: RunDiffusion/Juggernaut-XL-v9
- Mistral 7B: mistralai/Mistral-7B-Instruct-v0.3

## What changed in v2
- `model_utils.py` uses the exact model IDs.
- `app.py` auto-reads the `HF_TOKEN` environment variable (helpful when using Spaces repo secrets).
- Dockerfile updated for CUDA 12.1 + matching torch wheel (change if your environment uses different CUDA).
- Requirements pin to torch >=2.4.0 to align with CUDA12 builds.
- Keep `prefilter.txt` editable for your sensitive/illogical keywords.

## Notes & licensing
- Juggernaut model cards can include usage restrictions. Make sure you accept any terms on Hugging Face and that hosting a public Space is allowed for that checkpoint.
- If a model is gated you must provide a token via a Space secret named `HF_TOKEN` or paste it into the app UI.


## Updated for Hugging Face: Microsoft Phi Mini 3.5 + SDXL 1.0 base

This version replaces the previous LLM and image model with:

- **Prompt classifier (LLM)**: `microsoft/Phi-3.5-mini-instruct`. Set a Space secret named `HF_TOKEN` if the model is gated. See the model card on Hugging Face for access requirements.
- **Image generator (SDXL base)**: `stabilityai/stable-diffusion-xl-base-1.0`. This model's weights are large (~7GB) and require GPU with sufficient VRAM. See the model card for license and usage restrictions.

**Important runtime notes**
- The app loads the classifier first and only loads SDXL if the prompt is classified as logical. This minimizes peak VRAM usage.
- On Hugging Face Spaces choose a GPU runtime and set `HF_TOKEN` as a secret if the model requires it.
- If you want me to remove the original UI structure and replace it entirely, tell me explicitly; for now I preserved the app structure and only swapped model IDs and loader logic.

(Models verified from Hugging Face model pages.)
