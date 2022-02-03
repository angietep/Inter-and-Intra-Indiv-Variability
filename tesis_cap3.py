#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 12:49:29 2022

@author: angeles
"""
#%% Import everything

import pandas as pd
import scipy.io as sio
import os
import numpy as np

os.chdir('/Volumes/WD_Elements/CODE/Part2_Fingerprint/1_INDIVIDUAL_fingerprint')

parentdir='/Volumes/WD_Elements/';
parentdatadir='/Volumes/TOSHIBA/';
datadir_CL=parentdatadir + 'DATA/Reprepro/'
#%assumed structure: *datadir*/C/C01/Sess1/EPI/EPI_preprocessed/ICA-AROMA+2P/epi_prepro.nii
atlasdir = parentdatadir + 'DATA/ATLAS/'
#%atlasname='Schaefer400_7Ntw_TianSubcortex_S2.nii'; 
atlasname='Schaefer400_7Ntw_TianSubcortex_S2_2_75.nii.gz'
#%behav data
table_CL = pd.read_csv(parentdir+'CODE/Part2_Fingerprint/tabla_CL.csv')

#SESS1&2 32 subjects CHILE (C28,C41,C48 excluded for bad-looking FC??)
subjHC_CL=['C01','C02','C03','C04','C05','C07','C11','C15','C16', \
           'C18','C19','C20','C21','C24','C25','C27','C28','C30', \
           'C31','C32','C36','C41','C42','C43','C46','C47','C48' \
           'C50','C51','C52','C53','C57']
#%SESS1&2 30 subjects CHILE (1435 excluded for bad-looking FC??)
subjSZ_CL=['1401','1402','1405','1407','1413','1418','1420','1421', \
           '1423','1424','1425','1426','1428','1430','1433','1434', \
           '1435','1436','1437','1441','1442','1443','1445','1446', \
           '1449','1457','1458','1461','1465','1466']
    

FC=sio.loadmat('FCtrtDiCER1_SZCL_SCh400Nt7TianS2_vox275.mat')
FCtrt_SZ_CL=np.array(FC['FCtrt_SZ_CL'])

FC=sio.loadmat('FCtrtDiCER1_HCCL_SCh400Nt7TianS2_vox275.mat')
FCtrt_HC_CL=np.array(FC['FCtrt_HC_CL'])

#%%







