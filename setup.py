from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ["tensorflow-gpu==2.12.0", "tensorflow-datasets==1.0.1"]

setup(
    name="trainer",
    version="0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="My Training Package",
)
