import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nnetwork-eyedevelop",
    version="1.0.0",
    author="EyeDevelop",
    author_email="eyedevelop@github.com",
    description="A small framework for neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EyeDevelop/NEATNetwork",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
