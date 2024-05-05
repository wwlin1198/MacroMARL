#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --output=discovery/results/%j/myjob.%j.out

`module load anaconda3/2022.05`

`source activate /scratch/lin.wo/macro_marl`

echo `date +%Y%m%d-%H%M%S`

echo "[.](http://main.py/)/mac_ippo.sh"
./mac_ippo.sh

echo `date +%Y%m%d-%H%M%S`