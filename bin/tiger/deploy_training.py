'''

python script to deploy slurm jobs for constructing training set for speculator

'''
import os, sys 


def deploy_training_job(ibatch0, ibatch1, name, ncpu=1): 
    ''' create slurm script and then submit 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s%i_%i" % (name, ibatch0, ibatch1),
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=%i" % ncpu, 
        "#SBATCH --time=12:00:00", 
        "#SBATCH --export=ALL",
        "#SBATCH -o ofiles/%s%i_%i.o" % (name, ibatch0, ibatch1),
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "module load anaconda3", 
        "conda activate gqp", 
        "",
        "export machine='della'", 
        "for i in $(seq %i %i); do" % (ibatch0, ibatch1), 
        "   python /home/chhahn/projects/provabgs/bin/training.py %s $i %i" % (name, ncpu), 
        "done", 
        ""
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


def deploy_test_job(name, ncpu=1): 
    ''' create slurm script and then submit 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s_test" % name,
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=%i" % ncpu, 
        "#SBATCH --time=06:00:00", 
        "#SBATCH --export=ALL",
        "#SBATCH -o ofiles/%s_test.o" % name,
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "module load anaconda3", 
        "conda activate gqp", 
        "export machine='della'", 
        "",
        "python /home/chhahn/projects/provabgs/bin/training.py %s test %i" % (name, ncpu), 
        ""
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_test.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _test.slurm')
    os.system('rm _test.slurm')
    return None 



try: 
    ibatch0 = int(sys.argv[1])
    ibatch1 = int(sys.argv[2])
    name    = sys.argv[3]
    ncpu    = int(sys.argv[4]) # 20 works for cori

    print('submitting %s batch %i-%i' % (name, ibatch0, ibatch1))
    deploy_training_job(ibatch0, ibatch1, name, ncpu=ncpu)
except ValueError: 
    ibatch  = sys.argv[1]
    name    = sys.argv[2]
    ncpu    = int(sys.argv[3])
        
    print('submitting %s batch %s' % (name, ibatch))
    deploy_test_job(name, ncpu=ncpu)
