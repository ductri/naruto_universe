#!/bin/bash

python setup.py bdist_wheel

rm -r -f build/
rm -r -f naruto_skills.egg-info/

pip install --upgrade dist/*
