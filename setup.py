import os
import codecs

from setuptools import setup

base_dir = os.path.abspath(os.path.dirname(__file__))
def read(fname):
    return codecs.open(os.path.join(base_dir, fname), encoding="utf-8").read()

setup(
    name='COMPs',
    version='V6',
    packages=['ShNAPr'],
    url='https://github.com/VadymShevtsov/COMPs',
    license='GNU LGPLv3',
    author='V. Shevtsov',
    author_email='',
    description="Composite materials with CDM for tIGAr/FEniCS.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
)
