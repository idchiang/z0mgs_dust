import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="z0mgs_dust-idchiang",
    version="0.1.0",
    author="I-Da Chiang",
    author_email="idchinag@ucsd.edu",
    description="Dust SED fitting using MBB",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/idchiang/z0mgs_dust",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
