import os
import torch
from PIL import Image
#from taming.models.vqgan import VQModel
from main import instantiate_from_config
from taming.modules.transformer.mingpt import sample_with_masking, sample_with_past
from torchvision import transforms
from omegaconf import OmegaConf
import numpy as np

# --------- CONFIG ---------

CONFIG_PATH = "/root/logs/2021-04-03T19-39-50_cin_transformer/configs/2021-04-03T19-39-50-project.yaml"
CHECKPOINT_PATH = "/root/logs/2021-04-03T19-39-50_cin_transformer/checkpoints/last.ckpt"
SEED_IMAGE_PATH = "sample.png"
SEED_TOKEN_COUNT = 128# You can modify this value easily
MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- HELPERS ---------
def load_model(config_path, ckpt_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict["state_dict"], strict=False)
    model.eval()
    return model

def get_seed_tokens(vqgan, image_path, seed_length):
    global encodedTokens
    image = Image.open(image_path).convert("RGB").resize((256, 256))
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        _, _, quant_output = vqgan.encode(image_tensor)
        indices = quant_output[2]
        encodedTokens = indices
    return indices[:seed_length]  # Return first `seed_length` tokens

def decode_tokens(vqgan, tokens):
    print(f"tokens shape before reshape: {tokens.shape}")  # Debug
    tokens = tokens.view(1, 16, 16)  # [B, H, W]
    print(f"tokens shape after reshape: {tokens.shape}")

    with torch.no_grad():
        z = vqgan.quantize.get_codebook_entry(tokens, shape=None)  # → []
        print(f"z shape before permute: {z.shape}")
        z = z.permute(0, 3, 1, 2)  # → [1, 256, 16, 16]
        print(f"z shape after permute: {z.shape}")

        decoded = vqgan.decode(z) # -> [1,3,256,256]

    decoded = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
    decoded = ((decoded + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(decoded)

def main():
    print("Loading models...")
    model = load_model(CONFIG_PATH, CHECKPOINT_PATH)
    vqgan = model.first_stage_model
    transformer = model.transformer

    print("Encoding seed image...")
    
    seed_tokens = get_seed_tokens(vqgan, SEED_IMAGE_PATH, SEED_TOKEN_COUNT)
    print("Generating sequence...")
    #sampled = sample_with_past(seed_tokens.unsqueeze(0),transformer, steps=256 - SEED_TOKEN_COUNT, temperature=1, top_k=100, top_p=0.95)
    sampled = sample_with_masking(seed_tokens.unsqueeze(0),encodedTokens.unsqueeze(0),transformer, steps=256 - SEED_TOKEN_COUNT, temperature=1, top_k = 1, top_p = 1.0)

    full_tokens = torch.cat([seed_tokens, sampled.squeeze(0)], dim=0)
    print(f"full_tokens shape: {full_tokens.shape}")
    print("Decoding generated image...")
    output_image = decode_tokens(vqgan, full_tokens)
    output_image.save("/root/logs/output_seeded_masked.png")
   

if __name__ == "__main__":
    main()