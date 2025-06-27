from setuptools import setup, find_packages

setup(
    name='taming-transformers',
    version='0.1.0',
    description='Taming Transformers for High-Resolution Image Synthesis',
    packages=find_packages(exclude=["scripts", "logs", "tests"]),
    python_requires='>=3.6, <3.9',
    install_requires=[
        'torch>=1.7.0',
        'numpy',
        'tqdm',
        'Pillow',
        'scipy',
        'omegaconf',
        'pytorch-lightning==1.0.8',
        'albumentations==0.4.3',
        'einops==0.3.0',
        'imageio==2.9.0',
        'imageio-ffmpeg==0.4.2',
        'test-tube==0.7.5',
        'streamlit==0.73.1',
        'more-itertools',
        'opencv-python==4.5.5.64',
],
)
