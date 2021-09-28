#!/bin/bash

set -e 

module load python/3
source /home/gchu4/scratch/warnow/tipp_og/myenv/bin/activate
python setup.py install

run_abundance.py -G markers-v3 -f $1 -g RpsK_COG0100 -d $2 --tempdir $3 --cpu 20 -s known51_454_edited.fasta -A 500

deactivate
