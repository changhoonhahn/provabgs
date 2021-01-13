'''

python script to deploy slurm jobs for constructing training set for speculator

'''
import os, sys 



def deploy_emu_train(model, i_wave, n_pcas, Nlayer, Nunits):
    ''' create slurm script for training speculator and then submit 
    '''
    cntnt = '\n'.join(["#!/bin/bash", 
        "#SBATCH -J train_emu_%s_%i_%i_%ix%i" % (model, i_wave, n_pcas, Nlayer, Nunits),  
        "#SBATCH --exclusive",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        '#SBATCH --gres=gpu:1', 
        "#SBATCH --partition=general",
        "#SBATCH --time=47:59:59", 
        "#SBATCH --export=ALL",
        "#SBATCH --output=ofiles/train_emu_%s_%i_%i_%ix%i.o" % (model, i_wave, n_pcas, Nlayer, Nunits),  
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "module load anaconda3", 
        "conda activate tf2-gpu", 
        "",
        "python /home/chhahn/projects/provabgs/bin/emulator.py %s %i %i %i %i" % (model, i_wave, n_pcas, Nlayer, Nunits),
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
n_pcas = int(sys.argv[3]) 
Nlayer = int(sys.argv[4]) 
Nunits = int(sys.argv[5]) 
deploy_emu_train(model, i_wave, n_pcas,  Nlayer, Nunits)
