from setuptools import find_packages, setup

with open("./README.md", "r") as f:
    long_description = f.read()

setup(
    name="mlprank",
    version="0.0.1",
    description=(
        "A neural network pruning package based on based on the"
        " PageRank centrality measure"
    ),
    package_dir={"": "mlprank"},
    packages=find_packages(where="mlprank"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.aws.dev/adavidho/mlp-rank-pruning",
    author="David Hoffmann",
    author_email="adavidho@amazon.com",
    license="Apache-2.0",
    classifiers=[
        "License :: OSI Approved :: Apache-2.0",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy ~= 1.26.4",
        "torch ~= 2.2.1",
        "torchvision ~= 0.17.1",
    ],
    python_requires=">=3.10",
)
