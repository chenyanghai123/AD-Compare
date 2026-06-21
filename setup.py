from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ad_compare",
    version="0.1.0",
    author="Yanghai Chen",
    description="AD-Compare: Industrial Anomaly Detection via Comparison-Enhanced Multimodal LLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chenyanghai123/AD-Compare",
    packages=find_packages(exclude=["scripts", "configs", "deepspeed", "assets"]),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
