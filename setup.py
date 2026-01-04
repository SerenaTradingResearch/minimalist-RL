from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="minimalist-RL",
    version="0.0.8",
    author="Ricky Ding",
    author_email="e0134117@u.nus.edu",
    description="Minimalist & Decoupled Reinforcement Learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeuroAI-Research/minimalist-RL",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.8",
    license="MIT",
    keywords=[
        "reinforcement-learning",
        "rl",
        "ppo",
        "sac",
        "pytorch",
        "minimalist",
    ],
    classifiers=[
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
