#!/bin/sh
dataset_list=( ctg inv main_sense precharge io nand2 nor3 nand5 nor4 delay10 atd1 atd18m col_sel1m col_sel3_8m decoder4_16 col_sel7_128 nand3 decoder2_4 col_sel9_512 deco1m deco3_8m deco7_128 deco9_512 delay4 rl_sel delay6 sen_ena delay16 xvald ctrl256kbm mc sa_pre array512x1 block512x512n decoder6_64 bank16mx1n atd18 dff1 level_shift1 level_shift2 level_shift3 dram1gbn ) 


for dname in ${dataset_list[*]}
do
    echo "subckt : $dname"
    python3 company_preprocess.py --dataset company --circuit $dname  --gnd_vsource_directional
done
