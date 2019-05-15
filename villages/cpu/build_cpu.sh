#!/bin/sh

cp ../../skills/dist/* asset/

sudo docker build -t ductricse/dl-cpu .

rm asset/* -f