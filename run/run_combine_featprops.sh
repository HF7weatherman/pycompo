#!/bin/bash

analysis_identifier=${1}
node_size=${2:-"256G"}
if [ -z "$analysis_identifier" ]; then
    echo "Usage: $0 <analysis_identifier>"
    exit 1
fi

export CONFIG_FILE=/home/m/m300738/libs/pycompo/config/settings_${analysis_identifier}.yaml
export RUNFILE=/home/m/m300738/libs/pycompo/pycompo/drivers/combine_feature_props.py
	  
# COMBINE FEATURE PROPS
jobid2=$(sbatch --parsable --constraint=${node_size} <<EOF
#!/bin/bash
#SBATCH --account=mh0731
#SBATCH --partition=compute
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name="combine_feature_props"
#SBATCH --output=LOG/combine_feature_props.%j.out
#SBATCH --error=LOG/combine_feature_props.%j.out
#SBATCH --export=ALL

source ~/.bashrc
micromamba activate TRR181L4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 "${RUNFILE}" "${CONFIG_FILE}"
EOF
)
squeue -u $USER