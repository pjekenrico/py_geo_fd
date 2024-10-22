from setuptools import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="py_geo_fd",
    version="1.0.0",
    description="Virtual Flow Diverter deployment tool",
    long_description=long_description,
    url="https://github.com/pjekenrico/py_geo_fd",
    author="P. Jeken-Rico",
    author_email="pablojeken@hotmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Users",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="virtual flow diverter deployment, intracranial aneurysms",
    packages=["py_geo_fd"],  # Required
    python_requires=">=3.9, <4",
    install_requires=["numpy", "scipy", "vtk", "json", "meshio"],  # Optional
)
