# app.py (root)
import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import gradio as gr
from typing import Optional
from PIL import Image

from src.evaluate import (
    embed_image, embed_text, alignment_score,
    load_reference_embeddings, mean_reference_aesthetic_score
)

HQ_REFS = [
    "data/example_images/photo-1661852818096-e74ff4f806bb.jpeg",
    "data/example_images/photo-1605749439419-80c81f67eefc.jpeg",
    "data/example_images/Mountain-Sunset.jpg"
]
HQ_EMBS = load_reference_embeddings(HQ_REFS) if HQ_REFS else []
DEFAULT_PROMPT = "A high-quality, realistic photograph of the described subject"

def evaluate(img: Image.Image, prompt: Optional[str]):
    if img is None:
        return "No image", "N/A"
    img = img.convert("RGB")
    prompt = (prompt or DEFAULT_PROMPT).strip()
    ie = embed_image(img)
    te = embed_text([prompt])
    align = float(alignment_score(ie, te))
    aest_out = "N/A"
    if HQ_EMBS:
        aest = mean_reference_aesthetic_score(ie, HQ_EMBS)
        if aest is not None:
            aest_out = round(float(aest), 4)
    return round(align, 4), aest_out

def build_app():
    with gr.Blocks(title="Generative Image Evaluator (CLIP)") as demo:
        gr.Markdown("# Generative Image Evaluator (CLIP)")
        gr.Markdown("Upload an image and a prompt. Outputs CLIP alignment and an optional aesthetic proxy.")
        with gr.Row():
            img_in = gr.Image(type="pil", label="Upload image", height=350)
            prompt_in = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT)
        btn = gr.Button("Evaluate")
        out_align = gr.Label(label="Alignment score")
        out_aesth = gr.Label(label="Aesthetic score (proxy)")
        btn.click(evaluate, [img_in, prompt_in], [out_align, out_aesth])
    return demo

demo = build_app()