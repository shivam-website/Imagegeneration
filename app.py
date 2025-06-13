from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch
import io
from PIL import Image
import base64

# Create Flask app 👇
app = Flask(__name__)

# Load model once
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None
).to(device)
pipe.enable_attention_slicing()

@app.route("/")
def home():
    return "🚀 Stable Diffusion API is running!"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "a fantasy landscape")
    num_images = int(data.get("num_images", 1))

    try:
        result = pipe(prompt, num_images_per_prompt=num_images)
        images = result.images

        # Convert images to base64 strings
        encoded_images = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            img_bytes = buf.getvalue()
            encoded = base64.b64encode(img_bytes).decode('utf-8')
            encoded_images.append(encoded)

        return jsonify({"images": encoded_images})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Needed for Gunicorn on Render
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)  # Use any port for local
