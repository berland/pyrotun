#!/usr/bin/env python3

import setuptools
from setuptools import find_packages

REQUIREMENTS = [
    "activereader",
    "aiocron<2",  # aiocron 2.1 støtter ikke sekund-felt
    "aiofiles",
    "aiohttp",
    "aioinflux",
    "astral",
    "asyncio",
    "aiomqtt",
    "defusedxml",
    "geopy",
    "gpxpy",
    "fastapi",
 #   "habapp", # no-go for py312 elvcat
    "holidays",
    "isodate",
    "lxml",
    "metpy",
    "networkx",
    "numpy",
    "paho-mqtt",
    "pandas",
    "pylunar",
    "python-dotenv",
    "python-dotenv",
    "python-openhab",
    "pytibber",
    "pyyaml",
    "requests",
    "scikit-learn",
    "skodaconnect",
    "smappy",
    "uvicorn",
    "watchfiles",
]

SETUP_REQUIREMENTS = [
    "setuptools >=28",
    "setuptools_scm",
    "pytest-runner",
]

TEST_REQUIREMENTS = [
    "black>=20.8b0",
    "flake8",
    "isort",
    "pytest",
    "pytest-asyncio",
    "rstcheck",
]

EXTRAS_REQUIRE = {"tests": TEST_REQUIREMENTS}

setuptools.setup(
    name="pyrotun",
    description="Python code for Råtun 40",
    author="Håvard Berland",
    author_email="berland@pvv.ntnu.no",
    url="https://github.com/berland/pyrotun",
    keywords=[],
    license="Private",
    platforms="any",
    include_package_data=False,
    packages=find_packages(include=["pyrotun*"]),
    install_requires=REQUIREMENTS,
    setup_requires=SETUP_REQUIREMENTS,
    use_scm_version={"write_to": "pyrotun/version.py"},
    test_suite="tests",
    extras_require=EXTRAS_REQUIRE,
)
