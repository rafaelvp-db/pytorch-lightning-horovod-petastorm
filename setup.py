import setuptools

setuptools.setup(
  name="carsifier",
  version="0.0.1",
  author="rafael pierre",
  author_email="rafael.pierre@gmail.com",
  description="A simple example package using PyTorch, PyTorch Lightning, Petatorm and Horovod",
  url="https://github.com/rafaelvp-db/pytorch-lightning-horovod-petastorm",
  packages=setuptools.find_packages(exclude=["tests", "tests.*", "examples"]),
  install_requires=[
    "torch",
    "pytorch_lightning==1.6.5",
    "torchvision",
    "torchmetrics",
    "pyspark==3.2.1",
    "numpy",
    "petastorm",
    "opencv-python",
    "scipy"
  ],
  python_requires='>=3.8',
)