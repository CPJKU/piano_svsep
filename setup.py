from setuptools import setup, find_packages

try:
    import torch
except ImportError:
    raise ImportError("Pytorch is not installed. Please install it using 'pip install torch'")


setup(
    name="piano_svsep",
    version="0.0.1dev",
    packages=find_packages(),
    setup_requires=["torch"],
    install_requires=[
        "torch_geometric",
        "partitura==1.5.0",
        "torchmetrics==1.4.0",
        "scipy",
        "pytorch_lightning==2.3.3",
        "verovio==4.2.1",
        "torch-scatter==2.1.2",
        "joblib==1.0.1",
        "gitpython==3.1.14",
        "tqdm==4.61.0",
    ],
)
