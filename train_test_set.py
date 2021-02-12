# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:11:21 2020

@author: ASUS
"""

import os
import random
from shutil import copyfile
def split(source,TRAIN,TEST,SPLIT_SIZE):
    files=[]
    for filename in os.listdir(source):
        file=source+filename
        if(os.path.getsize(file)>0):
            files.append(filename)
        else:
            pass
    trainsize=int(len(files)*SPLIT_SIZE)
    testsize=int(len(files)-trainsize)
    shuffled=random.sample(files,len(files))
    training_set=shuffled[0:trainsize]
    test_set=shuffled[:testsize]
    
    for filename in training_set:
        thisfile=source+filename
        destination=TRAIN+filename
        copyfile(thisfile,destination)
    for filename in test_set:
        thisfile=source+filename
        destination=TEST+filename
        copyfile(thisfile,destination)
        