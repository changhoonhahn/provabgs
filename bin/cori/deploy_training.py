'''

python script to deploy slurm jobs for constructing training set for speculator

'''
import os, sys 


def deploy_training_job(ibatch, name, ncpu=1): 
    ''' create slurm script and then submit 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH --qos=regular", 
        "#SBATCH --time=01:00:00", 
        "#SBATCH --constraint=haswell", 
        "#SBATCH -N 1", 
        "#SBATCH -J train%i" % ibatch,  
        "#SBATCH -o ofiles/train_%s%i.o" % (name, ibatch), 
        "#SBATCH -L SCRATCH,project", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "conda activate gqp", 
        "",
        "python /global/homes/c/chahah/projects/provabgs/bin/training.py %s %i %i" % (name, ibatch, ncpu), 
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


ibatch0 = int(sys.argv[1])
ibatch1 = int(sys.argv[2])
name    = sys.argv[3]
ncpu    = int(sys.argv[4]) 

for ibatch in range(ibatch0, ibatch1+1): 
    print('submitting %s batch %i' % (name, ibatch))
    deploy_training_job(ibatch, name, ncpu=ncpu)
