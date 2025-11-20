import os
import numpy as np

mainfolder = '/gdata/fewahab/data/WSJO-3rdPaper-dataset/data/tr' #Path to tr_x.ex files for training and tt_snr-x.ex for testing
f1 = open('/ghome/fewahab/Transformer_3rd_Paper/MHSA/MHSA-enc-bottleneck-6FB/scripts/tr_list.txt', 'a')   # file name
folderlist = os.listdir(mainfolder)

for index,folder in enumerate(folderlist):
    path2write = os.path.join(mainfolder,folder)
    f1.write(path2write+'\n')
    print(path2write)