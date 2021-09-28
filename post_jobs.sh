#!/bin/bash

mkdir output tmp

sbatch --partition=cs --time=128:00:00 --job-name=hier_t2f_k51_454 -o log_k51_454 run_tipp2_fast.sh known51_454_edited.fasta output/k51_454 tmp/k51_454  

#sbatch --partition=cs --time=128:00:00 --job-name=t2f_k51_ill100 -o log_k51_ill100 run_tipp2_fast.sh known51_ill100.fa output/k51_ill100 tmp/k51_ill100
#sbatch --partition=cs --time=128:00:00 --job-name=t2f_k51_ill250 -o log_k51_ill250 run_tipp2_fast.sh known51_ill250.fa output/k51_ill250 tmp/k51_ill250  

#sbatch --partition=cs --time=128:00:00 --job-name=t2f_n100_454 -o log_n100_454 run_tipp2_fast.sh novel_454_combined_edited.fa output/n100_454 tmp/n100_454  
#sbatch --partition=cs --time=128:00:00 --job-name=t2f_n100_ill150 -o log_n100_ill150 run_tipp2_fast.sh novel_ill150_combined.fa output/n100_ill150 tmp/n100_ill150
#sbatch --partition=cs --time=128:00:00 --job-name=t2f_n100_ill250 -o log_n100_ill250 run_tipp2_fast.sh novel_ill250_combined.fa output/n100_ill250 tmp/n100_ill250  

