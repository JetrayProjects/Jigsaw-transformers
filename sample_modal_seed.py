import modal

app = modal.App("sample-from-seed")

volume = modal.Volume.from_name("taming-vol", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch", "pytorch-lightning==1.0.8", "omegaconf", "einops",
        "albumentations==0.4.3", "imageio==2.9.0", "imageio-ffmpeg==0.4.2",
        "test-tube==0.7.5", "streamlit==0.73.1", "torchvision", "more-itertools",
        "opencv-python==4.5.5.64", "transformers>=4.25.0", "numpy<2.0"
    )
)

@app.function(
    image=image,
    volumes={"/root/logs": volume},
    timeout=900,
    cpu=4,
    gpu="any"
)
def run_sample_from_seed():
    import os
    import subprocess

    os.chdir("/root")
    subprocess.run(["git", "clone", "https://github.com/JetrayProjects/Jigsaw-transformers.git"], check=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/Jigsaw-transformers"

    subprocess.run(
        ["python", "scripts/sample_from_seed.py"],
        check=True,
        cwd="/root/Jigsaw-transformers",
        env=env
    )

if __name__ == "__main__":
    run_sample_from_seed.remote()
