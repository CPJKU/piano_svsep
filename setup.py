from setuptools import setup, find_packages

setup(
    name="piano_svsep",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "partitura",
        "torchmetrics",
        "scipy",
        "pytorch_lightning",
        "verovio",
        "torch-scatter",
    ],
)