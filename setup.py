#!/usr/bin/env python

from distutils.core import setup

setup(
    name="asnet",
    description="Keras implementation of ASNets",
    version="0.1",
    author="André F. Moukarzel",
    url="https://github.com/AndreMoukarzel/asnet-ip",
    packages=["asnet"],
    install_requires = [
        'tensorflow==2.12.0',
        'tqdm==4.66.1',
        'click==8.1.7',
        'ippddl-parser @ git+https://github.com/AndreMoukarzel/ippddl-parser'
    ]
    license="GPLv3"
)