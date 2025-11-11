# src/evaluate.py
import os
from typing import List, Union, Optional, Dict, Any
from io import BytesIO

import torch
import numpy as np
from PIL import Image
import requests
from sklearn.metrics.pairwise import cosine_similarity

from transformers import CLIPProcessor, CLIPModel

# --------- Global cache (so model loads once per process) ----------
_MODEL = None
_PROCESSOR = None
_DEVICE = "cpu"  # "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

def _ensure_rgb(pil_img):
    # handle grayscale / RGBA etc.
    return pil_img.convert("RGB")

def load_model_and_processor(model_name: str = "openai/clip-vit-base-patch32"):
    """
    Lazily loads CLIP model + processor and caches them.
    """
    global _MODEL, _PROCESSOR
    if _MODEL is None or _PROCESSOR is None:
        _MODEL = CLIPModel.from_pretrained(model_name).to(_DEVICE)
        _PROCESSOR = CLIPProcessor.from_pretrained(model_name)
        _MODEL.eval()
    return _MODEL, _PROCESSOR


# ---------------------- Image utilities ----------------------------
def load_image_from_path(path: str) -> Image.Image:
    return _ensure_rgb(Image.open(path))

def load_image_from_url(url: str, timeout: int = 20) -> Image.Image:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return _ensure_rgb(Image.open(BytesIO(resp.content)))


def _to_pil(img: Union[str, Image.Image]) -> Image.Image:
    if isinstance(img, str):
        if img.startswith("http://") or img.startswith("https://"):
            return load_image_from_url(img)
        return load_image_from_path(img)
    return img


# ---------------------- Embeddings --------------------------------
def embed_image(img: Union[str, Image.Image]) -> np.ndarray:
    """
    Returns L2-normalized CLIP image embedding (1, D) as numpy array.
    """
    model, processor = load_model_and_processor()
    pil = _to_pil(img)
    inputs = processor(images=pil, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu().numpy()


def embed_text(texts: List[str]) -> np.ndarray:
    """
    Returns L2-normalized CLIP text embeddings (N, D) as numpy array.
    """
    model, processor = load_model_and_processor()
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(_DEVICE)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu().numpy()


# ---------------------- Scoring -----------------------------------
def alignment_score(
    image_embedding: np.ndarray,
    text_embedding: np.ndarray
) -> float:
    """
    Cosine similarity in [-1, 1]. Higher = more aligned.
    """
    return float(cosine_similarity(image_embedding, text_embedding)[0, 0])


def batch_alignment_scores(
    image_paths_or_urls: List[str],
    prompts: List[str],
    pairwise: bool = True
) -> List[float]:
    """
    If pairwise=True: compares i-th image to i-th prompt.
    Else: compares each image to mean(text_embeds) (shared reference).
    """
    img_embs = [embed_image(p) for p in image_paths_or_urls]
    txt_embs = embed_text(prompts)

    scores: List[float] = []
    if pairwise:
        for i, emb in enumerate(img_embs):
            j = min(i, len(prompts) - 1)
            scores.append(alignment_score(emb, txt_embs[j:j+1]))
    else:
        mean_txt = txt_embs.mean(axis=0, keepdims=True)
        for emb in img_embs:
            scores.append(alignment_score(emb, mean_txt))
    return scores


def mean_reference_aesthetic_score(
    image_embedding: np.ndarray,
    hq_embeddings: List[np.ndarray]
) -> Optional[float]:
    """
    A very lightweight aesthetic proxy: similarity to a set of high-quality reference images.
    Returns None if no references provided.
    """
    if not hq_embeddings:
        return None
    sims = [float(cosine_similarity(image_embedding, he)[0, 0]) for he in hq_embeddings]
    return float(np.mean(sims))


def load_reference_embeddings(hq_image_paths_or_urls: List[str]) -> List[np.ndarray]:
    return [embed_image(p) for p in hq_image_paths_or_urls]


# ---------------------- Convenience API ----------------------------
def evaluate_images(
    images: List[str],
    prompts: List[str],
    pairwise: bool = True,
    hq_refs: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    End-to-end helper used by demos or scripts.
    Returns list of dicts: { 'image': str, 'alignment': float, 'aesthetic': Optional[float] }
    """
    results = []
    # pre-embed references if requested
    hq_embs = load_reference_embeddings(hq_refs) if hq_refs else []

    # pre-embed text
    text_embs = embed_text(prompts)
    mean_txt = text_embs.mean(axis=0, keepdims=True)

    for idx, img in enumerate(images):
        img_emb = embed_image(img)
        if pairwise:
            j = min(idx, len(prompts) - 1)
            align = alignment_score(img_emb, text_embs[j:j+1])
        else:
            align = alignment_score(img_emb, mean_txt)
        aest = mean_reference_aesthetic_score(img_emb, hq_embs) if hq_embs else None

        results.append({
            "image": img,
            "alignment_score": float(align),
            "aesthetic_score": (float(aest) if aest is not None else None)
        })
    return results


if __name__ == "__main__":
    # Minimal CLI for quick testing:
    import argparse, json
    parser = argparse.ArgumentParser(description="Evaluate images with CLIP.")
    parser.add_argument("--images", nargs="+", required=True, help="Paths or URLs")
    parser.add_argument("--prompts", nargs="+", required=True, help="Text prompts")
    parser.add_argument("--pairwise", action="store_true", help="Pair ith image with ith prompt")
    parser.add_argument("--hq_refs", nargs="*", default=None, help="High-quality reference images (optional)")
    args = parser.parse_args()

    out = evaluate_images(
        images=args.images,
        prompts=args.prompts,
        pairwise=args.pairwise,
        hq_refs=args.hq_refs
    )
    print(json.dumps(out, indent=2))
