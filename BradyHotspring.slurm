#!/bin/bash
#SBATCH --job-name="Download BradyHotspring data"
#SBATCH --time=48:00:00
#SBATCH --mem=12GB
#SBATCH --export=ALL


/u/st/by/aissah/bins/bin/aws s3 cp --no-sign-request s3://nrel-pds-porotomo/DAS/H5/DASH/20160313/ /beegfs/projects/martin/BradyHotspring/03_19_2016/ --recursive
/u/st/by/aissah/bins/bin/aws s3 cp --no-sign-request s3://nrel-pds-porotomo/DAS/H5/DASH/20160314/ /beegfs/projects/martin/BradyHotspring/03_20_2016/ --recursive
/u/st/by/aissah/bins/bin/aws s3 cp --no-sign-request s3://nrel-pds-porotomo/DAS/H5/DASH/20160313/ /beegfs/projects/martin/BradyHotspring/03_21_2016/ --recursive