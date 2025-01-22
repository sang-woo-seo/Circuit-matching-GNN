#!/bin/sh
# dataset_list=( ctg inv main_sense precharge io nand2 nor3 nand5 nor4 delay10 atd1 atd18m col_sel1m col_sel3_8m decoder4_16 col_sel7_128 nand3 decoder2_4 col_sel9_512 deco1m deco3_8m deco7_128 deco9_512 delay4 rl_sel delay6 sen_ena delay16 xvald ctrl256kbm mc sa_pre array512x1 block512x512n decoder6_64 ) 

# dataset_list=( ctg inv main_sense precharge io nand2 nor3 nand5 nor4 delay10 atd1 atd18m col_sel1m col_sel3_8m decoder4_16 col_sel7_128 nand3 decoder2_4 col_sel9_512 deco1m deco3_8m deco7_128 deco9_512 delay4 rl_sel delay6 sen_ena delay16 xvald ctrl256kbm mc sa_pre array512x1 block512x512n decoder6_64 ) 

# dataset_list1=(ctrl256kbm io dff1 main_sense atd1 decoder4_16 decoder2_4 deco3_8m delay4 rl_sel delay6 sen_ena delay16 xvald atd18m atd18 col_sel1m deco1m decoder6_64)
# dataset_list2=(ctg inv delay10 nand2 nor3 nand5 nor4 nand3)

# dataset_list1=(decoder2_4)
# dataset_list2=(nand3)

# dataset_list1=(xvald)
# dataset_list2=(nand2)

# dataset_list1=(atd1)
# dataset_list2=(delay10)

# dataset_list1=(ctrl256kbm)
# dataset_list2=(nand3)

# dataset_list1=(precharge)
# dataset_list2=(ctg)

# dataset_list1=(main_sense)
# dataset_list2=(inv)

# dataset_list1=(io)
# dataset_list2=(inv)

# dataset_list1=(decoder4_16)
# dataset_list2=(nand5)

dataset_list1=(deco9_512)
dataset_list2=(inv nand5 nand3)
# dataset_list2=(atd1 decoder2_4 delay4 col_sel1m deco1m)


for entire_ckt in ${dataset_list1[*]}
do
    for target_ckt in ${dataset_list2[*]}
    do
        if [[ "$entire_ckt" != "$target_ckt" ]] ; then
            echo "entire_ckt : $entire_ckt   ||   target_ckt : $target_ckt"
            python3 ../src/main.py --use_predefined_split --load_parameter_file_name 967_20241213_094753 --entire_circuit $entire_ckt --target_circuit $target_ckt --radius --gnd_vsource_directional --batch_size 2048 --device 4
        fi
    done
done

# 964_20241225_083056
# 839_20250102_110949
# 967_20241213_094753
# 998_20241024_081307
# 111_20241003_224710

# --train_embedder 

# --train_verbose --test_verbose
# --use_predefined_split 
# --load_parameter_file_name 156_20240902_102809


# 111_20241003_224710 - accuracy : 84.5161  ||  precision : 0.7803  ||  recall : 0.9161 - inv 0.3
# 74_20241003_194929 - accuracy : 85.8065  ||  precision : 0.8455  ||  recall : 0.8402 - inv 0.4



# 156_20240902_102809 -0.1
# 165_20240902_104635 -0.2
# 167_20240902_105037 -0.2
# 168_20240902_105238 -0.2
# 159_20240910_215942 -0.4
# 160_20240910_220154 -0.4