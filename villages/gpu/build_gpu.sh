#!/bin/sh

echo "Copy skills"
cp -r ../../skills/ .

sudo nvidia-docker build -t ductricse/dl-gpu .