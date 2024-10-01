from setuptools import setup, find_packages
from setuptools.command.install import install

# Read the requirements.txt file
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="segment",
    version="0.1",
    packages=find_packages(),
    description="Scripts for manipulating imaegs",
    url="https://github.com/MnLgt/SEGMENT",
    author="Jordan Davis",
    author_email="jordandavis16@gmail.com",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
