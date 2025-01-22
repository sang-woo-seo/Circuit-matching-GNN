#!/bin/sh
dataset_list=( t01 t02 ) 


for dname in ${dataset_list[*]}
do
    echo "subckt : $dname"
    python3 company_preprocess.py --dataset company --circuit $dname  --gnd_vsource_directional
done
