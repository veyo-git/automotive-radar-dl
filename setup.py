from setuptools import setup, find_packages

setup(
    name="automotive-radar-dl",
    version="0.1.0",
    description="Automotive FMCW Radar Signal Simulation & Deep Learning-Based Interference Mitigation",
    author="RadarDL",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "h5py>=3.8.0",
        "tqdm>=4.66.0",
        "gradio>=4.0.0",
    ],
)
