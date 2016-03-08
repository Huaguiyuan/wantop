import fileinput
from os import chdir
from subprocess import call
import yaml

files = ['hr.dat', 'rr.dat', 'run.sh.ncore', 'rndegen.dat', 'main.py', 'wannier.py', 'utility.py', 'wantop.in']
with open('wantop.in') as file:
    config = file.read()
config = yaml.load(config)
nkpts = (config['k_ndiv'])**3
job_num = config['job_num']

pkpts = nkpts / job_num
for i in range(job_num):
    call(['mkdir', str(i + 1)])
    for file in files:
        call(['cp', file, str(i + 1)])
    chdir(str(i + 1))
    for line in fileinput.input('main.py', inplace=True):
        print(line.replace('#KPTLISTMOD',
                           'kpt_list=kpt_list[' + str(int(i * pkpts)) + ':' + str(int((i + 1) * pkpts)) + ',:]'), end="")
    call(['qsub', 'run.sh.ncore'])
    chdir('..')