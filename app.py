from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

def generate_images(prompt, num_images=4):
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Load the model (will download ~2GB on first run)
        print("Loading model...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,  # Uses less memory
            safety_checker=None  # Disable safety checker for more variety
        )
        pipe = pipe.to(device)
        
        # Optimize for low-memory systems
        pipe.enable_attention_slicing()
        
        print("Generating images...")
        # Generate images
        images = pipe(
            prompt,
            num_images_per_prompt=num_images,
            height=512,
            width=512
        ).images

        # Display images
        fig, axes = plt.subplots(1, num_images, figsize=(20, 5))
        if num_images == 1:
            axes = [axes]  # Handle single image case
            
        for i, img in enumerate(images):
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f"Image {i+1}")
        
        plt.tight_layout()
        plt.show()
        
        return images
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    prompt = "a supercar at high speed on track, high quality"
    generated_images = generate_images(prompt, num_images=1)
    
    # To save images:
    # for i, img in enumerate(generated_images):
    #     img.save(f"generated_image_{i}.png")