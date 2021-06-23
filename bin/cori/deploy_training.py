'''

python script to deploy slurm jobs for constructing training set for speculator

'''
import os, sys 


def deploy_training_job(ibatch, name, ncpu=1): 
    ''' create slurm script and then submit 
    '''
    if ibatch == 'test': 
        time="02:00:00"
    else:  
        if name =='burst': 
            time="00:20:00"
        else: 
            time="01:00:00"
    
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH --qos=regular", 
        "#SBATCH --time=%s" % time, 
        "#SBATCH --constraint=haswell", 
        "#SBATCH -N 1", 
        "#SBATCH -J train%s" % str(ibatch),  
        "#SBATCH -o ofiles/train_%s%s.o" % (name, str(ibatch)),
        "#SBATCH -L SCRATCH,project", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "module unload PrgEnv-intel",
        "module load PrgEnv-gnu",
        "",
        "conda activate gqp", 
        "",
        "python /global/homes/c/chahah/projects/provabgs/bin/training_data_%s.py %s %i" % (name, str(ibatch), ncpu), 
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

try: 
    ibatch0 = int(sys.argv[1])
    ibatch1 = int(sys.argv[2])
    name    = sys.argv[3]
    ncpu    = int(sys.argv[4]) # 20 works for cori

    for ibatch in range(ibatch0, ibatch1+1): 
        print('submitting %s batch %i' % (name, ibatch))
        deploy_training_job(ibatch, name, ncpu=ncpu)
except ValueError: 
    ibatch  = sys.argv[1]
    name    = sys.argv[2]
    ncpu    = int(sys.argv[3])
        
    print('submitting %s batch %s' % (name, ibatch))
    deploy_training_job(ibatch, name, ncpu=ncpu)
