import fileinput
from os import chdir
from subprocess import call

files = ['hr.dat', 'rr.dat', 'run.sh.ncore', 'rndegen.dat', 'main.py', 'wannier.py', 'utility.py']
nkpts = 8000000
split_num = 16

pkpts = nkpts / split_num
for i in range(split_num):
    call(['mkdir', str(i + 1)])
    for file in files:
        call(['cp', file, str(i + 1)])
    chdir(str(i + 1))
    for line in fileinput.input('main.py', inplace=True):
        print(line.replace('#KPTLISTMOD',
                           'kpt_list=kpt_list[' + str(int(i * pkpts)) + ':' + str(int((i + 1) * pkpts)) + ',:]'), end="")
    call(['qsub', 'run.sh.ncore'])
    chdir('..')