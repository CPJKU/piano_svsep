from setuptools import setup, find_packages

# try:
#     import torch
# except ImportError:
#     raise ImportError("Pytorch is not installed. Please install it using the proper configurations for your system https://pytorch.org/get-started/locally/")


setup(
    name="piano_svsep",
    version="0.0.1dev",
    packages=find_packages(),
    setup_requires=["torch==2.7.0"],
    install_requires=[
        "torch==2.8.0",
        "torch_geometric",
        "partitura==1.8.0",
        "torchmetrics",
        "scipy",
        "scikit-learn==1.6.0",
        "pytorch_lightning",
        "verovio",
        "torch-scatter",
        "joblib",
        "gitpython",
        "tqdm",
    ],
)
