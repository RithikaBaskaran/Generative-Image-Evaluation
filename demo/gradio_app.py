# demo/gradio_app.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
from typing import Optional
from PIL import Image

from src.evaluate import (
    embed_image, embed_text, alignment_score,
    load_reference_embeddings, mean_reference_aesthetic_score
)

# Optional refs for aesthetic proxy
HQ_REFS = [
    "data/example_images/beach.jpeg",
    "data/example_images/flower.jpeg",
    "data/example_images/mountain-sunset.jpg"
]
HQ_EMBS = load_reference_embeddings(HQ_REFS)

DEFAULT_PROMPT = "A high-quality, realistic photograph of the described subject"

def evaluate(img: Image.Image, prompt: Optional[str]):
    try:
        if img is None:
            return "No image", "N/A"

        img = img.convert("RGB")
        prompt = (prompt or DEFAULT_PROMPT).strip()

        img_emb = embed_image(img)
        txt_emb = embed_text([prompt])
        align = float(alignment_score(img_emb, txt_emb))

        aest_out = "N/A"
        if HQ_EMBS:
            aest = mean_reference_aesthetic_score(img_emb, HQ_EMBS)
            if aest is not None:
                aest_out = round(float(aest), 4)

        return round(align, 4), aest_out

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"error: {e}", "N/A"

def build_app():
    with gr.Blocks(title="Generative Image Evaluator (CLIP)") as demo:
        gr.Markdown("# Generative Image Evaluator (CLIP)")
        gr.Markdown("Upload an image and enter a descriptive prompt. The app returns a CLIP-based alignment score and an optional aesthetic proxy.")
        with gr.Row():
            img_in = gr.Image(type="pil", label="Upload image", height=350)
            prompt_in = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT)
        btn = gr.Button("Evaluate")
        out_align = gr.Label(label="Alignment score")
        out_aesth = gr.Label(label="Aesthetic score (proxy)")
        btn.click(fn=evaluate, inputs=[img_in, prompt_in], outputs=[out_align, out_aesth])
    return demo

demo = build_app()

if __name__ == "__main__":
    demo.launch(share=True)