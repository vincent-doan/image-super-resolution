from rcan import *
import torch
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from torchvision import transforms

# -------------------- Parse command-line arguments -------------------- #
parser = argparse.ArgumentParser(description='Super-resolution using RCAN')
parser.add_argument('--lr_size', type=int, default=256, help='Low-resolution size')
parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
args = parser.parse_args()
LR_SIZE = args.lr_size
IMAGE_PATH = args.image_path

# -------------------- Load RCAN model -------------------- #
model = RCAN()
model.load_state_dict(torch.load('rcan_weights.pth', map_location=torch.device('cpu')))

# -------------------- Load image and downsize using bicubic interpolation -------------------- #

bicubic_downsize_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((LR_SIZE, LR_SIZE), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
])
input_image = bicubic_downsize_transforms(Image.open(IMAGE_PATH))

bicubic_upsize_transforms = transforms.Compose([
    transforms.Resize((LR_SIZE*4, LR_SIZE*4), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
])
bicubic_upsized_image = bicubic_upsize_transforms(input_image)

# -------------------- Plot images -------------------- #
fig, axes = plt.subplots(1, 3)
# Low-res image
axes[0].imshow(input_image.permute(1, 2, 0))
axes[0].set_title("Low-res image")

# Bicubic upsize
axes[1].imshow(bicubic_upsized_image.permute(1, 2, 0))
axes[1].set_title("Bicubic interpolated image")

# High-res output from RCAN
with torch.no_grad():
    output_image = model(input_image.unsqueeze(0))
axes[2].imshow(output_image.squeeze().permute(1, 2, 0).detach())
axes[2].set_title("RCAN super-res image")

axes[0].axis('off')
axes[1].axis('off')
axes[2].axis('off')

plt.show()
