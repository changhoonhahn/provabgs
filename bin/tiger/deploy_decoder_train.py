'''

python script to deploy slurm job for training the decoder

'''
import os, sys 


def deploy_decoder_train(model, i_wave, nbatch): 
    ''' create slurm script for training speculator and then submit 
    '''
    cntnt = '\n'.join(["#!/bin/bash", 
        "#SBATCH -J train_decoder_%s%i_%i" % (model, i_wave, nbatch),  
        "#SBATCH --exclusive",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        '#SBATCH --gres=gpu:1', 
        "#SBATCH --partition=general",
        "#SBATCH --time=71:59:59", 
        "#SBATCH --export=ALL",
        "#SBATCH --output=ofiles/train_decoder_%s%i_%i.o" % (model, i_wave, nbatch),
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
        "python /home/chhahn/projects/provabgs/bin/decoder.py %s %i %i" % (model, i_wave, nbatch), 
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
i_wave = int(sys.argv[2])
nbatch = int(sys.argv[3]) 
deploy_decoder_train(model, i_wave, nbatch) 
