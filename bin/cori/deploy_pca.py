'''

python script to deploy slurm jobs for constructing training set for speculator

'''
import os, sys 


def deploy_pca_job(name, ibatch0, ibatch1, n_pca, i_bin): 
    ''' create slurm script and then submit 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH --qos=debug", 
        "#SBATCH --time=00:30:00", 
        "#SBATCH --constraint=haswell", 
        "#SBATCH -N 1", 
        "#SBATCH -J pca_%i" % i_bin,  
        "#SBATCH -o ofiles/pca_%s%i.o" % (name, i_bin), 
        "#SBATCH -L SCRATCH,project", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "conda activate gqp", 
        "",
        "python /global/homes/c/chahah/projects/provabgs/bin/pca.py train %s %i %i %i %i" % (name, ibatch0, ibatch1, n_pca, i_bin), 
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_pca.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _pca.slurm')
    os.system('rm _pca.slurm')
    return None 


name    = sys.argv[1]
ibatch0 = int(sys.argv[2])
ibatch1 = int(sys.argv[3])
n_pca   = int(sys.argv[4]) 
i_bin   = int(sys.argv[5]) 

deploy_pca_job(name, ibatch0, ibatch1, n_pca, i_bin)
