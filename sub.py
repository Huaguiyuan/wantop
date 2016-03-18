#!/usr/bin/env python
import fileinput
import numpy as np
from os import chdir
from subprocess import call
import numpy as np
import yaml

files = ['hr.dat', 'rr.dat', 'run.sh.ncore', 'rndegen.dat', 'wantop.in']
with open('wantop.in') as file:
    config = file.read()
config = yaml.load(config)
nkpts = np.prod(config['k_ndiv'])
job_num = config['job_num']

pkpts = nkpts / job_num
for i in range(job_num):
    call(['mkdir', str(i)])
    for file in files:
        call(['cp', file, str(i)])
    chdir(str(i))
    for line in fileinput.input('wantop.in', inplace=True):
        print(line.replace('#JOBCNT', 'job_cnt: ' + str(i)), end="")
    call(['qsub', 'run.sh.ncore'])
    chdir('..')
