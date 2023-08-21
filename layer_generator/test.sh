#!/usr/bin/env bash
# Filename: varlog.sh
for i in {1..24}
do
	python3 attention_l2_l1_layer_generator.py --layer $i
	echo -n "L" $i " " >> performance.txt                    
	make -C ./application/ clean all run CORE=8 platform=gvsoc > temp.txt
	grep -FR 'cycles' ./temp.txt | tee -a performance.txt                 
done
