try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

import re
from pathlib import Path


def readme(root_path):
    """Returns the text content of the README.rst of the package
    Parameters
    ----------
    root_path : pathlib.Path
        path to the root of the package
    """
    with root_path.joinpath('README.md').open(encoding='UTF-8') as f:
        return f.read()


root_path = Path(__file__).parent
README = readme(root_path)


config = {
    'name': 'qutipext',
    'version': '0.1',
    'packages': find_packages(),
    'description': 'QuTip Extension Toolbox',
    'long_description': README,
    'long_description_content_type' : 'text/x-rst',
    'author': 'Hong-Ye Hu', #'version': VERSION,
    'author_email': 'hongyehu@fas.harvard.edu',
    'install_requires': ['numpy','qutip'],
    'license': 'Modified BSD',
    'scripts': [],
    'classifiers': [
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3'
    ],
}

setup(**config)