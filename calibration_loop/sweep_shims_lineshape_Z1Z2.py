from TReND.send2vnmr import *
import shutil
import time
import os
import numpy as np
from subprocess import call

#This is the script to collect ShimNet trainig data on Agilent spectrometers. Execution:
# 1. Open VnmrJ and type: 'listenon'
# 2. Put the lineshape sample, set standard PROTON parameters and set one scan (do not modify sw and at!)
# 3. Shim the sample and collect the data. Save the optimally shimmed datatset
# 4. Put optimum z1 and z2 shim values as optiz1 and optiz2 below
# 5. Define the calibration range as range_z1 and range_z2 (default is ok)
# 6. Start the python script: 
# python3 ./sweep_shims_lineshape_Z1Z2.py
# the spectrometer will start collecting spectra
 
Classic_path = '/home/nmr700/shimnet6/lshp' 
optiz1= 8868 #put optimum shim values here
optiz2=-297

range_z1=100 #put optimum shim ranges here
range_z2=100 #put optimum shim ranges here

z1_sweep=np.arange(optiz1-range_z1,optiz1+range_z1+1,2.0)
z2_sweep=np.arange(optiz2-range_z2,optiz2+range_z2+1,2.0)

for i in range(1,np.shape(z1_sweep)[0]+1, 1):
    for j in range(1,np.shape(z2_sweep)[0]+1, 1):
        wait_until_idle()
    
        Run_macro("sethw('z1',"+str(z1_sweep[i-1])+ ")")
        Run_macro("sethw('z2',"+str(z2_sweep[j-1])+ ")")
 
        go_if_idle()
        wait_until_idle()
        time.sleep(0.5)

        Save_experiment(Classic_path + '_z1_'+ str(int(z1_sweep[i-1])) + '_z2_'+ str(int(z2_sweep[j-1]))+ '.fid')
