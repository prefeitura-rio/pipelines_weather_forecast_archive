# -*- coding: utf-8 -*-
# script.py
import os

with open("requirements_venv.txt") as f:
    requirements = f.readlines()

with open("pyproject.toml", "a") as f:
    f.write("\ndependencies = [\n")
    for req in requirements:
        req = req.strip()
        # Adiciona ^ após ==
        if "==" in req:
            req = req.replace("==", ">=")
        f.write(f'    "{req}",\n')
    f.write("]\n")
