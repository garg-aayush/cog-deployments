import depth_pro
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Union
from pathlib import Path
import os
import torch
from cog import BasePredictor, Input, Path
import time
import subprocess

WEIGHTS_FILE = "depth_pro.pt"
WEIGHTS_URL = "https://ml-site.cdn-apple.com/models/depth-pro/" + WEIGHTS_FILE
WEIGHTS_DIR = "checkpoints"
import urllib.request


def download_weights(url, dest, name):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # Download the file using urllib
    urllib.request.urlretrieve(url, dest + "/" + name)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self, device: str = "cuda"):
        """Initialize the Predictor with model and transforms."""
        if not os.path.exists(WEIGHTS_DIR):
            download_weights(WEIGHTS_URL, WEIGHTS_DIR, WEIGHTS_FILE)
        
        self.device = torch.device(device)
        self.model, self.transform = depth_pro.create_model_and_transforms(device=self.device)
        self.model.eval()

    def predict(
        self, 
        image: Path = Input(description="Input image file"),
        auto_rotate: bool = Input(description="Auto-rotate image based on EXIF", default=True),
        remove_alpha: bool = Input(description="Remove alpha channel if present", default=True)
    ) -> Path:
        """Predict depth from a single image."""
        # Load the image from path
        input_image = Image.open(image)
        
        # Get the prediction
        depth_image, focallength = predict_depth(input_image, auto_rotate, remove_alpha, self.model, self.transform)
        
        # Save and return the result
        print(f"focal length: {focallength}")
        output_path = Path("output.png")
        depth_image.save(str(output_path))
        return output_path


def predict_depth(image: Image.Image, auto_rotate: bool, remove_alpha: bool, model, transform):
    # Convert the PIL image to a temporary file path if needed
    image_path = "temp_image.jpg"
    image.save(image_path)

    # Load and preprocess the image from the given path
    loaded_image, _, f_px = depth_pro.load_rgb(image_path, auto_rotate=auto_rotate, remove_alpha=remove_alpha)
    loaded_image = transform(loaded_image)

    # Run inference
    prediction = model.infer(loaded_image, f_px=f_px)
    depth = prediction["depth"].detach().cpu().numpy().squeeze()  # Depth in [m]

    inverse_depth = 1 / depth
    # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu
    )

    focallength = prediction["focallength_px"].cpu().numpy()

    # Normalize and colorize depth map
    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)

    # Clean up temporary image
    os.remove(image_path)

    return Image.fromarray(color_depth), focallength  # Return depth map and f_px

# can't run this in Github Actions as it requires a GPU
# def main():
#     # Load model and preprocessing transform
#     model, transform = depth_pro.create_model_and_transforms(device=torch.device("cuda"))
#     model.eval()
    
#     auto_rotate = True
#     remove_alpha = True
#     image = Image.open("example.jpg")

#     depth_image, focallength = predict_depth(image, auto_rotate, remove_alpha, model, transform)
#     depth_image.save("depth_image.png")
#     print(f"Focal length: {focallength}")


# if __name__ == "__main__":
#     main()