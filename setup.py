import runpy
from setuptools import setup, find_packages

PACKAGE_NAME = "deep_orderbook"
version_meta = runpy.run_path("./version.py")
VERSION = version_meta["__version__"]


with open("README.md", "r") as fh:
    long_description = fh.read()


def parse_requirements(requirements_file='requirements.txt'):
    """Get the contents of a file listing the requirements"""
    lines = open(requirements_file).readlines()
    dependencies = []
    for line in lines:
        maybe_dep = line.strip()
        if maybe_dep.startswith('#'):
            # Skip pure comment lines
            continue
        if maybe_dep.startswith('git+'):
            # VCS reference for dev purposes, expect a trailing comment
            # with the normal requirement
            __, __, maybe_dep = maybe_dep.rpartition('#')
        else:
            # Ignore any trailing comment
            maybe_dep, __, __ = maybe_dep.partition('#')
        # Remove any whitespace and assume non-empty results are dependencies
        maybe_dep = maybe_dep.strip()
        if maybe_dep:
            dependencies.append(maybe_dep)
    return dependencies

if __name__ == "__main__":
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        description="Transforms orders books in temporally and spatially local-correlated images.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/gQuantCoder/deep_orderbook",
        author="gQuantCoder",
        author_email="gquantcoder@gmail.com",
        packages=find_packages(),
        install_requires=parse_requirements("requirements.txt"),
        python_requires=">=3.7.4",
        entry_points={"console_scripts": ["deepbook=deep_orderbook.__main__:main"]},
    )