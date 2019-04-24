#!/usr/bin/env bash
set -e
set -x

for run in {0..2}
do
	for cls in {0..8}
	do
		python noft_ft_CL.py --cl $cls --device 0 --exp dat_false_par_false_hea_false30000 
	done
done
