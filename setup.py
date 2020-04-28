import setuptools

setuptools.setup(
    name="kfac-pytorch",
    version="0.1.0",
    author="Greg Pauloski",
    author_email="jgpauloski@uchicago.edu",
    description="Distributed K-FAC Preconditioner for PyTorch + Horovod",
    long_description=open('README.md').read(),
    url="https://github.com/gpauloski/kfac_pytorch",
    packages=["kfac"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "torch >= 1.0",
        "horovod",
    ],
)
