#!/bin/sh

echo "Copy skills"
cp -r ../../skills/ .

sudo docker build -t ductricse/dl-cpu .