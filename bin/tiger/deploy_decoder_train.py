'''

python script to deploy slurm job for training the decoder

'''
import os, sys 


def deploy_decoder_train(model, nbatch): 
    ''' create slurm script for training speculator and then submit 
    '''
    cntnt = '\n'.join(["#!/bin/bash", 
        "#SBATCH -J train_decoder_%s%i" % (model, nbatch),  
        "#SBATCH --exclusive",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        '#SBATCH --gres=gpu:1', 
        "#SBATCH --partition=general",
        "#SBATCH --time=23:59:59", 
        "#SBATCH --export=ALL",
        "#SBATCH --output=ofiles/train_decoder_%s%i.o" % (model, nbatch),
        "#SBATCH --mail-type=begin", 
        "#SBATCH --mail-type=end", 
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "module load anaconda3", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/provabgs/bin/decoder.py %s %i" % (model, nbatch), 
        "", 
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_train.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _train.slurm')
    os.system('rm _train.slurm')
    return None 

model = sys.argv[1]
nbatch = int(sys.argv[2]) 
deploy_decoder_train(model, nbatch) 
