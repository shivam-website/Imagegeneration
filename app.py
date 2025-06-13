from flask import Flask, request, jsonify, render_template
from diffusers import StableDiffusionPipeline
import torch
import io
from PIL import Image
import base64
import os

app = Flask(__name__)

# Load model once
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None
).to(device)
pipe.enable_attention_slicing()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form.get("prompt", "a fantasy landscape")
        negative_prompt = request.form.get("negative_prompt", "")
        num_images = int(request.form.get("num_images", 1))

        try:
            result = pipe(prompt, num_images_per_prompt=num_images)
            images = result.images

            # Convert images to base64 strings for embedding in HTML
            encoded_images = []
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                img_bytes = buf.getvalue()
                encoded = base64.b64encode(img_bytes).decode('utf-8')
                encoded_images.append(encoded)

            return render_template("index.html", images=encoded_images)

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html", images=[])

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "a fantasy landscape")
    num_images = int(data.get("num_images", 1))

    try:
        result = pipe(prompt, num_images_per_prompt=num_images)
        images = result.images

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
