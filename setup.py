import os

from setuptools import find_packages, setup

from pandasy_version import __version__

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()


def get_version() -> str:
    tag = os.environ.get("RELEASE_TAG", "")
    if "dev" in tag.split(".")[-1]:
        return tag
    if tag != "":
        assert tag == __version__, "release tag and version mismatch"
    return __version__


setup(
    name="slide",
    version=get_version(),
    packages=find_packages(),
    description="A collection of utility functions to unify"
    " pandas-like dataframes behaviors",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    author="Han Wang",
    author_email="goodwanghan@gmail.com",
    keywords="pandas sql",
    url="http://github.com/fugue-project/pandasy",
    install_requires=["triad", "pandas>=1.1.0", "numpy>=1.19"],
    extras_require={
        "dask": ["dask[dataframe]", "cloudpickle>=1.4.0"],
        "ray": ["pandas>=1.1.2", "modin[ray]>=0.8.1.1"],
        "all": ["dask[dataframe]", "cloudpickle>=1.4.0", "modin[ray]"],
    },
    classifiers=[
        # "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires=">=3.6",
)
