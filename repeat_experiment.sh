#!/bin/bash
for i in {0..5}
do
python main.py --no_graph --cifar_svhn --cifar_first --readout_num 10
done

for i in {0..5}
do
python main.py --no_graph --cifar_svhn --readout_num 10
done
