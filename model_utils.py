"""
Model utilities updated to use Microsoft Phi Mini 3.5 (for prompt classification) and Stable Diffusion XL 1.0 base (for image generation).

This file prioritizes memory efficiency:
- loads Phi Mini in 8-bit when possible (bitsandbytes)
- loads SDXL in fp16 and enables xformers where possible
- unloads models and clears CUDA between stages
"""
from typing import Dict, Any
import os, gc

# Model repo IDs (user-provided)
MISTRAL_ID = "microsoft/Phi-3.5-mini-instruct"
JUGGERNAUT_ID = "stabilityai/stable-diffusion-xl-base-1.0"

def prefilter_check(prompt: str) -> Dict[str, Any]:
    prompt_l = prompt.lower()
    sensitive = []
    illogical = []
    here = os.path.dirname(__file__)
    kw_file = os.path.join(here, "prefilter.txt")
    sensitive_kw = []
    illogical_kw = []
    if os.path.exists(kw_file):
        with open(kw_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "::" in line:
                    t,k = line.split("::",1)
                    if t.strip().lower()=="sensitive":
                        sensitive_kw.append(k.strip().lower())
                    elif t.strip().lower()=="illogical":
                        illogical_kw.append(k.strip().lower())
    else:
        sensitive_kw = ["child", "porn", "sexual", "rape", "nsfw", "illegal", "bomb"]
        illogical_kw = ["square circle", "infinite triangle", "four sided triangle"]

    for s in sensitive_kw:
        if s in prompt_l:
            sensitive.append(s)
    for s in illogical_kw:
        if s in prompt_l:
            illogical.append(s)

    return {
        "sensitive_found": len(sensitive)>0,
        "sensitive_hits": sensitive,
        "illogical_found": len(illogical)>0,
        "illogical_hits": illogical,
    }

def _clear_cuda():
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

def load_mistral_classifier(hf_token: str = None):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        model_name = MISTRAL_ID
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True,
            torch_dtype=torch.float16,
            use_auth_token=hf_token,
        )
        model.tokenizer = tokenizer
        return model
    except Exception as e:
        print("[WARN] Mistral 8-bit load failed, falling back to CPU:", e)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(MISTRAL_ID, use_fast=True, use_auth_token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(MISTRAL_ID, device_map="cpu", torch_dtype="auto", use_auth_token=hf_token)
        model.tokenizer = tokenizer
        return model

def classify_prompt_with_mistral(model, prompt: str) -> str:
    tok = model.tokenizer
    instr = (
        "You are a strict classifier. Answer with a single word: LOGICAL or ILLOGICAL.\\n"
        "Decide whether the meaning of the user prompt makes sense, is internally coherent, and physically/plausibly describable.\\n"
        f"User prompt: \\\"{prompt}\\\"\\nAnswer:"
    )
    inputs = tok(instr, return_tensors="pt")
    import torch
    inputs = {k: v.to(next(model.parameters()).device) for k,v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=8, do_sample=False, temperature=0.0)
    ans = tok.decode(out[0], skip_special_tokens=True)
    ans = ans.strip().split()[-1] if ans.strip() else ""
    return ans

def unload_mistral_classifier(model):
    try:
        del model
    except Exception:
        pass
    _clear_cuda()

def load_juggernaut_pipeline(hf_token: str = None, torch_dtype: str = "auto"):
    try:
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
        import torch
        model_id = JUGGERNAUT_ID
        dtype = getattr(torch, "float16") if torch_dtype=="auto" and torch.cuda.is_available() else torch.float32
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, use_auth_token=hf_token)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        return pipe
    except Exception as e:
        print("[ERROR] Failed to load Juggernaut pipeline:", e)
        raise

def generate_image_with_juggernaut(pipe, prompt: str, steps: int=28, guidance_scale: float=7.5, size=(512,512), seed: int=0):
    import torch
    from PIL import Image
    if seed and seed>0:
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    else:
        generator = None
    width, height = size
    out = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance_scale, height=height, width=width, generator=generator)
    images = out.get("images") if isinstance(out, dict) else getattr(out, "images", None)
    if images and len(images)>0:
        img = images[0]
        if not isinstance(img, Image.Image):
            from PIL import Image as PILImage
            img = PILImage.fromarray(img)
        return img
    else:
        raise RuntimeError("Pipeline returned no images")

def unload_juggernaut_pipeline(pipe):
    try:
        pipe.cpu()
    except Exception:
        pass
    try:
        del pipe
    except Exception:
        pass
    _clear_cuda()
