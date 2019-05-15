#!/bin/sh

cp ../../skills/dist/* asset/

sudo docker build -t ductricse/pytorch .

rm asset/* -f