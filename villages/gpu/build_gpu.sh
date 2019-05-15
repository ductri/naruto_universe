#!/bin/sh

cp ../../skills/dist/* asset/

sudo nvidia-docker build -t ductricse/dl-gpu .

rm asset/* -f