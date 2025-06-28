import os
import torch
from PIL import Image
#from taming.models.vqgan import VQModel
from scripts.main import instantiate_from_config
from taming.modules.transformer.mingpt import sample_with_past
from torchvision import transforms
import numpy as np

# --------- CONFIG ---------
VQGAN_CONFIG = "configs/imagenet_vqgan.yaml"
VQGAN_CKPT = "checkpoints/imagenet_vqgan.ckpt"
TRANSFORMER_CONFIG = "/root/logs/2021-04-03T19-39-50_cin_transformer/config.yaml"
TRANSFORMER_CKPT = "/root/logs/2021-04-03T19-39-50_cin_transformer/model.ckpt"
IMAGE_PATH = "sample.png"
SEED_LENGTH = 16  # You can modify this value easily
MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- HELPERS ---------
def load_model(config_path, checkpoint_path):
    config = instantiate_from_config(config_path)
    model = config.model
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["state_dict"], strict=False)
    model.eval().to(DEVICE)
    return model

def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(DEVICE)

def decode_to_img(model, z_indices):
    z_indices = z_indices.view(1, 16, 16)  # 256 tokens = 16x16 grid
    x = model.decode_code(z_indices)
    x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    x = x.cpu().squeeze().permute(1, 2, 0).numpy()
    return Image.fromarray((x * 255).astype(np.uint8))

# --------- MAIN ---------
if __name__ == "__main__":
    print("Loading models...")
    vqgan = load_model(VQGAN_CONFIG, VQGAN_CKPT)
    transformer = load_model(TRANSFORMER_CONFIG, TRANSFORMER_CKPT)

    print("Encoding image...")
    img_tensor = preprocess(IMAGE_PATH)
    z, _, [z_indices] = vqgan.encode(img_tensor)
    z_indices = z_indices.view(-1)

    seed = z_indices[:SEED_LENGTH].unsqueeze(0)  # shape: (1, SEED_LENGTH)

    print(f"Sampling from seed of length {SEED_LENGTH}...")
    sampled = sample_with_past(transformer, seed, steps=MAX_LENGTH - SEED_LENGTH, temperature=1.0, top_k=100, top_p=0.95)
    out = torch.cat((seed, sampled), dim=1)

    print("Decoding output image...")
    out_img = decode_to_img(vqgan, out[0])
    out_img.save("output.png")
    print("Saved to output.png")
