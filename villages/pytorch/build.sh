#!/bin/sh

python skills/setup.py bdist_wheel
cp ../../skills/dist/* asset/

sudo docker build -t ductricse/pytorch .

rm asset/* -f