# 🧠 Generative Image Evaluation using CLIP

This project uses **OpenAI’s CLIP model** to evaluate **how well an image matches a text prompt** (semantic alignment) and estimate its **aesthetic quality** by comparing it to reference high-quality images.

It provides both:
- A **Jupyter notebook** for step-by-step experimentation
- A **Gradio app** for an interactive demo (locally or via Hugging Face Spaces)

---

## 📂 Project Structure

 ├── app.py                     # Main entrypoint for the Gradio app (used in Hugging Face Space)

 ├── requirements.txt            # Dependencies for local setup or deployment

 ├── src/
 └── evaluate.py             # Core helper functions (embeddings, scoring, loading references)

 ├── demo/
 └── gradio_app.py           # Earlier local Gradio prototype (for reference)

 ├── data/
 └── example_images/         # Reference high-quality images for aesthetic scoring

├── notebooks/
└── GenerativeImageEvaluation_CLIP.ipynb  # Detailed step-by-step notebook version

└── README.md                   # You’re here

---

## ⚙️ How It Works

### 1. Alignment Score
Uses **CLIP cosine similarity** to measure how closely the uploaded image matches the text prompt.

- **Range:** 0.0 → ~0.45 (rarely above 0.5)
- **Interpretation:**
  | Score Range | Meaning |
  |--------------|----------|
  | 0.00 – 0.15 | Weak alignment (image unrelated to prompt) |
  | 0.15 – 0.30 | Moderate alignment |
  | 0.30 – 0.45 | Strong alignment |
  | 0.45+ | Very strong match (almost literal description) |

*(Note: CLIP embeddings are normalized, so cosine values rarely approach 1.0 even for perfect matches.)*

### 2. Aesthetic Score
Compares the uploaded image to **reference high-quality photos** stored in `data/example_images/` and computes the average cosine similarity.

- **Higher scores → more visually pleasing, balanced, and professional-looking.**
- **Typical ranges:**
  | Score Range | Meaning |
  |--------------|----------|
  | 0.00 – 0.20 | Low aesthetic similarity (noisy or cluttered) |
  | 0.20 – 0.40 | Moderate appeal |
  | 0.40 – 0.60 | High aesthetic similarity |
  | 0.60+ | Very high — visually striking / professional look |

---

## 🚀 Running the Demo Locally

### Prerequisites
- Python 3.10+
- PyTorch
- Internet connection (for downloading CLIP weights)

### Steps
1. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2.	Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.	Run the demo:
    ```bash
    python app.py
    ```
  To run locally : If you prefer to use the earlier prototype inside the /demo folder, run ```bash python -m demo.gradio_app instead.```
  
4.	Open the local URL displayed (usually http://127.0.0.1:7860).

Upload an image and a short text prompt — you’ll see Alignment Score and Aesthetic Score displayed in real time.

🌐 Public Demo (Hugging Face Space)

If you’d like to try this app directly in your browser, without setting up anything locally, visit the public Hugging Face Space (link will be added soon).

This hosted demo runs the same Gradio app on Hugging Face’s infrastructure, allowing anyone to test the evaluator interactively.

## 🧩 Technologies Used
	•	Python
	•	PyTorch
	•	Hugging Face Transformers
	•	Gradio
	•	OpenAI CLIP model

## 💡 Notes
	•	Alignment scores depend on the semantic similarity between the prompt and image content.
	•	Aesthetic scores rely on chosen reference images — replacing them with your own dataset changes the scoring context.
	•	If you face low alignment scores, try rephrasing prompts to be more literal (e.g., “a red sports car on a road” instead of “fast luxury vehicle”).

## ✨ Credits

Developed by Rithika Baskaran as part of a creative exploration in evaluating generative AI outputs with CLIP-based models.
