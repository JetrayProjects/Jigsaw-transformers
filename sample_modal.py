import modal


app = modal.App("taming-transformers-sampler")

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
def run_sampling():
    import os
    import subprocess

    os.chdir("/root")
    subprocess.run(["git", "clone", "https://github.com/JetrayProjects/Jigsaw-transformers.git"], check=True)

    subprocess.run(
        [
            "python", "scripts/sample_fast.py",
            "-r", "/root/logs/2021-04-03T19-39-50_cin_transformer/",
            "-n", "1", "-k", "100", "-t", "1.0", "-p", "0.92", "--batch_size", "10" ,"--classes", "1,2,3"
        ],
        check=True,
        cwd="/root/Jigsaw-transformers"
    )

if __name__ == "__main__":
    run_sampling.remote()