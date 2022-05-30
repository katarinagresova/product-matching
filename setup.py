from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="matching",
    version="0.0.1",
    description="Product matching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Katarina Gresova",
    author_email="gresova11@gmail.com",
    url="https://github.com/katarinagresova/product-matching",
    packages=find_packages("src"),
    package_dir={"": "src"},
    scripts=[],
    install_requires=[
        "requests>=2.23.0",
        "pip>=20.0.1",
        "numpy>=1.17.0",
        "pandas>=1.1.4",
        "tqdm>=4.41.1",
        "sparse_dot_topn>=0.3.1",
        "tokenizer>=0.11.6",
        "comet-ml>=3.28.2"
    ],
    tests_require=["pytest"],
    include_package_data=True,
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Development Status :: 3 - Alpha",
        # Define that your audience are developers
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        # Specify which pyhton versions that you want to support
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
