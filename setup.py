import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TrackEverything",
    version="1.6.3",
    author="Ami-A",
    author_email="schrodingerbot@gmail.com",
    description="A package that combines detetction, classification and tracking in videos, using AI models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ami-a/TrackEverything",
    packages=["TrackEverything"],
    #packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.18.5",
        "tensorflow==2.2.0",
        "scipy>=1.4.1",
        "Pillow>=7.2.0",
        "opencv-contrib-python>=4.4.0.42"
        ]
)