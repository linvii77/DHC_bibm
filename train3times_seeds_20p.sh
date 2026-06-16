#!/bin/bash


labeled_pct=20  # default

while getopts 'm:e:c:t:l:w:p:' OPT; do
    case $OPT in
        m) method=$OPTARG;;
        e) exp=$OPTARG;;
        c) cuda=$OPTARG;;
		    t) task=$OPTARG;;
		    l) lr=$OPTARG;;
		    w) cps_w=$OPTARG;;
		    p) labeled_pct=$OPTARG;;
    esac
done
echo $method
echo $cuda

epoch=300
echo $epoch

labeled_data="labeled_${labeled_pct}p"
unlabeled_data="unlabeled_${labeled_pct}p"
folder="Task_"${task}"_${labeled_pct}p/"
cps="AB"

echo $folder

python code/train_${method}.py --task ${task} --exp ${folder}${method}${exp}/fold1 --seed 0 -g ${cuda} --base_lr ${lr} -w ${cps_w} -ep ${epoch} -sl ${labeled_data} -su ${unlabeled_data} -r
python code/test.py --task ${task} --exp ${folder}${method}${exp}/fold1 -g ${cuda} --cps ${cps}
python code/train_${method}.py --task ${task} --exp ${folder}${method}${exp}/fold2 --seed 1 -g ${cuda} --base_lr ${lr} -w ${cps_w} -ep ${epoch} -sl ${labeled_data} -su ${unlabeled_data} -r
python code/test.py --task ${task} --exp ${folder}${method}${exp}/fold2 -g ${cuda} --cps ${cps}
python code/train_${method}.py --task ${task} --exp ${folder}${method}${exp}/fold3 --seed 666 -g ${cuda} --base_lr ${lr} -w ${cps_w} -ep ${epoch} -sl ${labeled_data} -su ${unlabeled_data} -r
python code/test.py --task ${task} --exp ${folder}${method}${exp}/fold3 -g ${cuda} --cps ${cps}

python code/evaluate_Ntimes_npy.py --task ${task} --exp ${folder}${method}${exp} --folds 3 --cps ${cps}
