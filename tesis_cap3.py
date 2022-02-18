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
import matlab.engine
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
#%%

#TO RUN matlab .m functions
eng = matlab.engine.start_matlab()
path = '/Volumes/WD_Elements/CODE/Functions'# specify your path
eng.addpath(path, nargout= 0)

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
table_Iselfs=pd.read_csv(parentdir+'CODE/Part2_Fingerprint/1_INDIVIDUAL_fingerprint/T_Rstat.csv')


#SESS1&2 32 subjects CHILE (C28,C41,C48 exclude for bad-looking FC??)
subjHC_CL=['C01','C02','C03','C04','C05','C07','C11','C15','C16', \
           'C18','C19','C20','C21','C24','C25','C27','C28','C30', \
           'C31','C32','C36','C41','C42','C43','C46','C47','C48', \
           'C50','C51','C52','C53','C57']
#SESS1&2 30 subjects CHILE (1435 exclude for bad-looking FC??)
subjSZ_CL=['1401','1402','1405','1407','1413','1418','1420','1421', \
           '1423','1424','1425','1426','1428','1430','1433','1434', \
           '1435','1436','1437','1441','1442','1443','1445','1446', \
           '1449','1457','1458','1461','1465','1466']
 

    
Iselfs_df=pd.read_csv('Iselfs_dataframe.csv')
    
    
#%%      
HC_df= table_CL.loc[table_CL['ID'].isin(subjHC_CL)]
SZ_df=table_CL.loc[table_CL['ID'].isin(subjSZ_CL)]

HC_df["FDmax"] = HC_df[["FD1", "FD2"]].max(axis=1)
HC_df['Group'] = 'HC'

SZ_df["FDmax"] = SZ_df[["FD1", "FD2"]].max(axis=1)
SZ_df['Group'] = 'SZ'

FC=sio.loadmat('FCtrtDiCER1_SZCL_SCh400Nt7TianS2_vox275.mat')
FCtrt_SZ_CL=np.array(FC['FCtrt_SZ_CL'])

FC=sio.loadmat('FCtrtDiCER1_HCCL_SCh400Nt7TianS2_vox275.mat')
FCtrt_HC_CL=np.array(FC['FCtrt_HC_CL'])

#%% GET Iself's and write in dataframe

# define SUBNETWORKS
# Schaefer400 7Networks

ATLAS={'VIS': np.ix_(np.concatenate((np.arange(0,31,1),np.arange(200,230,1))),np.concatenate((np.arange(0,31,1),np.arange(200,230,1)))), #[1:31,201:230];  MATLAB idx
       'SOM': np.ix_(np.concatenate((np.arange(31,68,1),np.arange(230,270,1))),np.concatenate((np.arange(31,68,1),np.arange(230,270,1)))), #[32:68,231:270];
       'DORSATT': np.ix_(np.concatenate((np.arange(68,91,1),np.arange(270,293,1))),np.concatenate((np.arange(68,91,1),np.arange(270,293,1)))), #[69:91,271:293]; 
       'VENTATT': np.ix_(np.concatenate((np.arange(91,113,1),np.arange(293,318,1))),np.concatenate((np.arange(91,113,1),np.arange(293,318,1)))), #[92:113,294:318]; 
       'LIMBIC': np.ix_(np.concatenate((np.arange(113,126,1),np.arange(318,331,1))),np.concatenate((np.arange(113,126,1),np.arange(318,331,1)))), #[114:126,319:331];
       'FP': np.ix_(np.concatenate((np.arange(126,148,1),np.arange(331,361,1))),np.concatenate((np.arange(126,148,1),np.arange(331,361,1)))),# [127:148,332:361];  
       'DMN': np.ix_(np.concatenate((np.arange(148,200,1),np.arange(361,400,1))),np.concatenate((np.arange(148,200,1),np.arange(361,400,1)))), #[149:200,362:400];
       'subcort': np.ix_(np.concatenate((np.arange(400,416,1),np.arange(416,432,1))),np.concatenate((np.arange(400,416,1),np.arange(416,432,1)))), #[401:416,417:432]; 
       }

plots=0

HCfg = eng.f_PCAtoolbox(matlab.double(FCtrt_HC_CL.tolist()),plots,nargout=4)
HC_Identmat_wb = HCfg[2]
HC_Iself_wb = np.diagonal(HC_Identmat_wb, axis1=0, axis2=1)
name='Iself_WB' #whole brain
HC_df[name] = HC_Iself_wb.tolist()

SZfg = eng.f_PCAtoolbox(matlab.double(FCtrt_SZ_CL.tolist()),plots,nargout=4)
SZ_Identmat_wb = SZfg[2]
SZ_Iself_wb = np.diagonal(SZ_Identmat_wb, axis1=0, axis2=1)
name='Iself_WB' #whole brain
SZ_df[name] = SZ_Iself_wb.tolist()


HC_Identmat_ntw=np.zeros((len(subjHC_CL),len(subjHC_CL),len(ATLAS.keys())))
SZ_Identmat_ntw=np.zeros((len(subjSZ_CL),len(subjSZ_CL),len(ATLAS.keys())))

for idx, network in enumerate(ATLAS.keys()):
    print (f'Computing fingerprint {network}')
    FC_HC_ntw=FCtrt_HC_CL[ATLAS[network]]
    FC_SZ_ntw=FCtrt_SZ_CL[ATLAS[network]]
    #Plot para verificar FC subnetworks
        #plt.figure()
        #plt.imshow(FC_HC_ntw[:,:,0])
    HCfg = eng.f_PCAtoolbox(matlab.double(FC_HC_ntw.tolist()),plots,nargout=4)
    SZfg = eng.f_PCAtoolbox(matlab.double(FC_SZ_ntw.tolist()),plots,nargout=4)

    #HCfg contains:
        #HCfg[0]=HCCLdata.Idiff_orig, 
        #HCfg[1]=HCCLdata.Idiff_opt,
        #HCfg[2]=HCCLdata.Ident_mat_orig, 
        #HCfg[3]=HCCLdata.Ident_mat_opt
    
    #Plots de Identmats
    #plt.figure()
    #plt.imshow(HCfg[2])

    HC_Identmat_ntw[:,:,idx]=HCfg[2]  
    SZ_Identmat_ntw[:,:,idx]=SZfg[2]  

   
    HC_Iself = np.diagonal(HC_Identmat_ntw[:,:,idx], axis1=0, axis2=1)
    name='Iself_'+network
    HC_df[name] = HC_Iself.tolist()
    
    SZ_Iself = np.diagonal(SZ_Identmat_ntw[:,:,idx], axis1=0, axis2=1)
    name='Iself_'+network
    SZ_df[name] = SZ_Iself.tolist()
    
    del HC_Iself, SZ_Iself
#%%

Iselfs_df = pd.concat([SZ_df,HC_df])

Iselfs_df.to_csv('Iselfs_dataframe.csv', index=False)

#%%

model = ols('Iself_VIS ~ Age + Age:Age + Sex + FDmax + Group', data=Iselfs_df).fit()
print(model.summary())

#define figure size
fig = plt.figure(figsize=(12,8))
#produce regression plots
fig = sm.graphics.plot_regress_exog(model, 'FDmax', fig=fig)


model = ols('Iself_SOM ~ Age + Age:Age + Sex + FDmax + Group', data=Iselfs_df).fit()
print(model.summary())
#define figure size
fig = plt.figure(figsize=(12,8))
#produce regression plots
fig = sm.graphics.plot_regress_exog(model, 'FDmax', fig=fig)
 

model = ols('Iself_DMN ~ Age + Age:Age + Sex + FDmax + Group', data=Iselfs_df).fit()
print(model.summary())
#define figure size
fig = plt.figure(figsize=(12,8))
#produce regression plots
fig = sm.graphics.plot_regress_exog(model, 'FDmax', fig=fig)

model = ols('Iself_subcort ~ Age + Age:Age + Sex + FDmax + Group', data=Iselfs_df).fit()
print(model.summary())
#define figure size
fig = plt.figure(figsize=(12,8))
#produce regression plots
fig = sm.graphics.plot_regress_exog(model, 'FDmax', fig=fig)


model = ols('Iself_WB ~ Age + Age:Age + Sex + FDmax + Group', data=Iselfs_df).fit()
print(model.summary())
#define figure size
fig = plt.figure(figsize=(12,8))
#produce regression plots
fig = sm.graphics.plot_regress_exog(model, 'FDmax', fig=fig)



# VIOLIN PLOTS: RESIDUOS ISELF POR GRUPOS Y POR NETWORKS (EXCLUYENDO A GRUPOS DEL MODELO)

# CALCULAR IOTHERS




