import pathlib
import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent
# The text of the README file
README = (HERE / "README.md").read_text()

requirements = ["tqdm", "python_binance"]

setuptools.setup(
    name="deep_orderbook",
    version="1.0.0",
    description="Transforms orders books in temporally and spatially local-correlated images",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/gQuantCoder/deep_orderbook",
    author="gQuantCoder",
    author_email="gquantcoder@gmail.com",
    license="MIT",
    classifiers=[
        # "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={"console_scripts": ["deepbook=deep_orderbook.__main__:main"]},
)

