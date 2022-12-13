#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 12:49:29 2022

@author: angeles
"""
#%% Import 

import pandas as pd
import scipy.io as sio
import os
import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats import multitest
import statsmodels.formula.api as smf
#from statannot import add_stat_annotation
from scipy import stats
#%% Initialize and load data frame

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
    
#%% DON'T need to RUN if already have dataframe:
HC_df= table_CL.loc[table_CL['ID'].isin(subjHC_CL)].reset_index()
SZ_df=table_CL.loc[table_CL['ID'].isin(subjSZ_CL)].reset_index()

FC=sio.loadmat('FCtrtDiCER1_SZCL_SCh400Nt7TianS2_vox275.mat')
FCtrt_SZ_CL=np.array(FC['FCtrt_SZ_CL'])

FC=sio.loadmat('FCtrtDiCER1_HCCL_SCh400Nt7TianS2_vox275.mat')
FCtrt_HC_CL=np.array(FC['FCtrt_HC_CL'])

#%%         Get Individual variability as mean(1 - corr(FC_subji_sess1,FC_subjk_sess1))

FCtrt_all=np.arctanh(np.concatenate((FCtrt_HC_CL,FCtrt_SZ_CL),axis=2))
FCs1_all=FCtrt_all[:,:,np.arange(0,FCtrt_all.shape[2],2)]
FCs2_all=FCtrt_all[:,:,np.arange(1,FCtrt_all.shape[2],2)]

maskut=np.triu_indices(len(FCtrt_all),k=1)

#HC
varS1=np.zeros(len(subjHC_CL) + len(subjSZ_CL))
varS2=np.zeros(len(subjHC_CL) + len(subjSZ_CL))

corrdistS1=np.zeros([len(subjHC_CL) + len(subjSZ_CL), len(subjHC_CL) + len(subjSZ_CL) -1])
corrdistS2=np.zeros([len(subjHC_CL) + len(subjSZ_CL), len(subjHC_CL) + len(subjSZ_CL) -1])

for i,subject in enumerate(np.arange(0,FCtrt_all.shape[2],2)):
        
        FCsubj_sess1=FCtrt_all[:,:,subject]
        FCsubj_sess2=FCtrt_all[:,:,subject+1]
       
        #idxG1 all sess1 but excluding this subject 
        idxG1=[x for x in np.arange(0,FCtrt_all.shape[2],2) if x != subject] 
        #idxG2 all sess1 but excluding this subject
        idxG2=[x for x in np.arange(1,FCtrt_all.shape[2],2) if x != subject+1]    
                  
        for k, other in enumerate(idxG1):
            corrdistS1[i,k]=1-np.corrcoef(FCsubj_sess1[maskut],FCtrt_all[:,:,other][maskut])[0,1]
        for k, other in enumerate(idxG2):
            corrdistS2[i,k]=1-np.corrcoef(FCsubj_sess2[maskut],FCtrt_all[:,:,other][maskut])[0,1]
       
        varS1[i]=np.tanh(corrdistS1[i,:].mean())
        varS2[i]=np.tanh(corrdistS2[i,:].mean())
   
HC_df.loc[:,'IndivVar_S1'] = varS1[np.arange(len(subjHC_CL))].tolist()
HC_df.loc[:,'IndivVar_S2'] = varS2[np.arange(len(subjHC_CL))].tolist()

SZ_df.loc[:,'IndivVar_S1'] = varS1[np.arange(len(subjHC_CL),len(subjHC_CL)+len(subjSZ_CL))].tolist()
SZ_df.loc[:,'IndivVar_S2'] = varS2[np.arange(len(subjHC_CL),len(subjHC_CL)+len(subjSZ_CL))].tolist()
  

#%%         Get DeviationFromHealthy as corr(FC_subji_Sess1, meanFC_group_Sess1)

maskut=np.triu_indices(len(FCtrt_HC_CL),k=1)

#HC
DevFromHealth_S1_HC=np.zeros(len(subjHC_CL))
DevFromHealth_S2_HC=np.zeros(len(subjHC_CL))

for i,subject in enumerate(np.arange(0,FCtrt_HC_CL.shape[2],2)):
        
        FC_sess1=np.arctanh(FCtrt_HC_CL[:,:,subject])
        FC_sess2=np.arctanh(FCtrt_HC_CL[:,:,subject+1])

        idxG1=[x for x in np.arange(0,FCtrt_HC_CL.shape[2],2) if x != subject] 
        idxG2=[x for x in np.arange(1,FCtrt_HC_CL.shape[2],2) if x != subject+1]    
       
        FC_HCS1 = np.arctanh(FCtrt_HC_CL[:,:, idxG1])
        FC_HCS2 = np.arctanh(FCtrt_HC_CL[:,:, idxG2])
        refFC_groupS1=np.mean(FC_HCS1,axis=2) #mean FC sess 1
        refFC_groupS2=np.mean(FC_HCS2,axis=2) #mean FC sess 2     
      
        DevFromHealth_S1_HC[i]=np.tanh(1- np.corrcoef(FC_sess1[maskut],refFC_groupS1[maskut])[0,1])
        DevFromHealth_S2_HC[i]=np.tanh(1- np.corrcoef(FC_sess2[maskut],refFC_groupS2[maskut])[0,1])
   
HC_df.loc[:,'DevFromHealth_S1'] = DevFromHealth_S1_HC.tolist() 
HC_df.loc[:,'DevFromHealth_S2'] = DevFromHealth_S2_HC.tolist()
 
#SZ   
DevFromHealth_S1_SZ=np.zeros(len(subjSZ_CL))
DevFromHealth_S2_SZ=np.zeros(len(subjSZ_CL))

idxG1=[x for x in np.arange(0,FCtrt_HC_CL.shape[2],2)]
FC_HCS1 = np.arctanh(FCtrt_HC_CL[:,:, idxG1])
refFC_groupS1=np.mean(FC_HCS1,axis=2) #mean FC sess 1

idxG2=[x for x in np.arange(1,FCtrt_HC_CL.shape[2],2)]    
FC_HCS2 = np.arctanh(FCtrt_HC_CL[:,:, idxG2])
refFC_groupS2=np.mean(FC_HCS2,axis=2) #mean FC sess 2 

for i,subject in enumerate(np.arange(0,FCtrt_SZ_CL.shape[2],2)):
   
    FC_sess1=np.arctanh(FCtrt_SZ_CL[:,:,subject])
    FC_sess2=np.arctanh(FCtrt_SZ_CL[:,:,subject+1])
       
    DevFromHealth_S1_SZ[i]=np.tanh(1- np.corrcoef(FC_sess1[maskut],refFC_groupS1[maskut])[0,1])
    DevFromHealth_S2_SZ[i]=np.tanh(1- np.corrcoef(FC_sess2[maskut],refFC_groupS2[maskut])[0,1])


SZ_df.loc[:,'DevFromHealth_S1'] = DevFromHealth_S1_SZ.tolist()
SZ_df.loc[:,'DevFromHealth_S2'] = DevFromHealth_S2_SZ.tolist()


#%%         Get Iothers as corr(FC_subji_Sess1, meanFC_group_Sess1)

maskut=np.triu_indices(len(FCtrt_HC_CL),k=1)

#HC
IothersS1_HC=np.zeros(len(subjHC_CL))
IothersS2_HC=np.zeros(len(subjHC_CL))

for i,subject in enumerate(np.arange(0,FCtrt_HC_CL.shape[2],2)):
        
        FC_sess1=np.arctanh(FCtrt_HC_CL[:,:,subject])
        FC_sess2=np.arctanh(FCtrt_HC_CL[:,:,subject+1])
       
        #idxG1 all sess1 but excluding this subject 
        #AND different sex subjects: 
        idxG1=[x for x in np.arange(0,FCtrt_HC_CL.shape[2],2) if x != subject] 
        #idxG2 all sess1 but excluding this subject
        idxG2=[x for x in np.arange(1,FCtrt_HC_CL.shape[2],2) if x != subject+1]    
       
        idx_samesex = (HC_df[HC_df["Sex"]==HC_df["Sex"][i]].index).tolist()
        idx_samesexS1 = [element * 2 for element in idx_samesex]
        idx_samesexS2 = [element * 2 + 1 for element in idx_samesex]
       
        idx_finalS1=[value for value in idxG1 if value in idx_samesexS1]
        idx_finalS2=[value for value in idxG2 if value in idx_samesexS2]
       
        FC_groupS1 = np.arctanh(FCtrt_HC_CL[:,:, idx_finalS1])  #for all subjects in group use idxG1 instead of idx_finalS1
        FC_groupS2 = np.arctanh(FCtrt_HC_CL[:,:, idx_finalS2])
      
        mFC_groupS1=np.mean(FC_groupS1,axis=2) #mean FC sess 1
        mFC_groupS2=np.mean(FC_groupS2,axis=2) #mean FC sess 2
       
       
        IothersS1_HC[i]=np.tanh(np.corrcoef(FC_sess1[maskut],mFC_groupS1[maskut])[0,1])
        IothersS2_HC[i]=np.tanh(np.corrcoef(FC_sess2[maskut],mFC_groupS2[maskut])[0,1])
   
HC_df.loc[:,'Iothers_S1'] = IothersS1_HC.tolist()
HC_df.loc[:,'Iothers_S2'] = IothersS2_HC.tolist()
 
#SZ   
IothersS1_SZ=np.zeros(len(subjSZ_CL))
IothersS2_SZ=np.zeros(len(subjSZ_CL))

for i,subject in enumerate(np.arange(0,FCtrt_SZ_CL.shape[2],2)):
   
    FC_sess1=np.arctanh(FCtrt_SZ_CL[:,:,subject])
    FC_sess2=np.arctanh(FCtrt_SZ_CL[:,:,subject+1])
   
    #idxG1 all sess1 but excluding this subject
    idxG1=[x for x in np.arange(0,FCtrt_SZ_CL.shape[2],2) if x != subject] 
    #idxG2 all sess1 but excluding this subject
    idxG2=[x for x in np.arange(1,FCtrt_SZ_CL.shape[2],2) if x != subject+1]    
   
    idx_samesex = (SZ_df[SZ_df["Sex"]==SZ_df["Sex"][i]].index).tolist()
    idx_samesexS1 = [element * 2 for element in idx_samesex]
    idx_samesexS2 = [element * 2 + 1 for element in idx_samesex]
       
    idx_finalS1=[value for value in idxG1 if value in idx_samesexS1]
    idx_finalS2=[value for value in idxG2 if value in idx_samesexS2]    
  
    FC_groupS1 = np.arctanh(FCtrt_SZ_CL[:,:, idx_finalS1])  #for all subjects in group use idxG1 instead of idx_finalS1
    FC_groupS2 = np.arctanh(FCtrt_SZ_CL[:,:, idx_finalS2])
  
    mFC_groupS1=np.mean(FC_groupS1,axis=2) #mean FC sess 1
    mFC_groupS2=np.mean(FC_groupS2,axis=2) #mean FC sess 2
      
    IothersS1_SZ[i]=np.tanh(np.corrcoef(FC_sess1[maskut],mFC_groupS1[maskut])[0,1])
    IothersS2_SZ[i]=np.tanh(np.corrcoef(FC_sess2[maskut],mFC_groupS2[maskut])[0,1])


SZ_df.loc[:,'Iothers_S1'] = IothersS1_SZ.tolist()
SZ_df.loc[:,'Iothers_S2'] = IothersS2_SZ.tolist()



#%%         Get Iothers as mean(corr(FC_subji_sess1,FC_subjk_sess1))

maskut=np.triu_indices(len(FCtrt_HC_CL),k=1)

#HC
IothersS1_HC=np.zeros(len(subjHC_CL))
IothersS2_HC=np.zeros(len(subjHC_CL))
corrS1_HC=np.zeros(len(subjHC_CL)-1)
corrS2_HC=np.zeros(len(subjHC_CL)-1)

for i,subject in enumerate(np.arange(0,FCtrt_HC_CL.shape[2],2)):
        
        FC_sess1=np.arctanh(FCtrt_HC_CL[:,:,subject])
        FC_sess2=np.arctanh(FCtrt_HC_CL[:,:,subject+1])
       
        #idxG1 all sess1 but excluding this subject 
        idxG1=[x for x in np.arange(0,FCtrt_HC_CL.shape[2],2) if x != subject] 
        #idxG2 all sess1 but excluding this subject
        idxG2=[x for x in np.arange(1,FCtrt_HC_CL.shape[2],2) if x != subject+1]    
                  
        for k, other in enumerate(idxG1):
            FC_other=np.arctanh(FCtrt_HC_CL[:,:,other])
            corrS1_HC[k]=np.corrcoef(FC_sess1[maskut],FC_other[maskut])[0,1]
        for k, other in enumerate(idxG2):
            FC_other=np.arctanh(FCtrt_HC_CL[:,:,other])
            corrS2_HC[k]=np.corrcoef(FC_sess2[maskut],FC_other[maskut])[0,1]
       
        IothersS1_HC[i]=np.tanh(corrS1_HC.mean())
        IothersS2_HC[i]=np.tanh(corrS2_HC.mean())
   
HC_df.loc[:,'Iothers_S1_WB'] = IothersS1_HC.tolist() #Whole Brain
HC_df.loc[:,'Iothers_S2_WB'] = IothersS2_HC.tolist()
 
#SZ   
IothersS1_SZ=np.zeros(len(subjSZ_CL))
IothersS2_SZ=np.zeros(len(subjSZ_CL))
corrS1_SZ=np.zeros(len(subjSZ_CL)-1)
corrS2_SZ=np.zeros(len(subjSZ_CL)-1)

for i,subject in enumerate(np.arange(0,FCtrt_SZ_CL.shape[2],2)):
        
        FC_sess1=np.arctanh(FCtrt_SZ_CL[:,:,subject])
        FC_sess2=np.arctanh(FCtrt_SZ_CL[:,:,subject+1])
       
        #idxG1 all sess1 but excluding this subject 
        idxG1=[x for x in np.arange(0,FCtrt_SZ_CL.shape[2],2) if x != subject] 
        #idxG2 all sess1 but excluding this subject
        idxG2=[x for x in np.arange(1,FCtrt_SZ_CL.shape[2],2) if x != subject+1]    
                  
        for k, other in enumerate(idxG1):
            FC_other=np.arctanh(FCtrt_SZ_CL[:,:,other])
            corrS1_SZ[k]=np.corrcoef(FC_sess1[maskut],FC_other[maskut])[0,1]
        for k, other in enumerate(idxG2):
            FC_other=np.arctanh(FCtrt_SZ_CL[:,:,other])
            corrS2_SZ[k]=np.corrcoef(FC_sess2[maskut],FC_other[maskut])[0,1]
       
        IothersS1_SZ[i]=np.tanh(corrS1_SZ.mean())
        IothersS2_SZ[i]=np.tanh(corrS2_SZ.mean())
   
SZ_df.loc[:,'Iothers_S1_WB'] = IothersS1_SZ.tolist()
SZ_df.loc[:,'Iothers_S2_WB'] = IothersS2_SZ.tolist()
 
#%%         Get Iothers subnetworks as mean(corr(FC_subji_sess1,FC_subjk_sess1))


for idx, network in enumerate(ATLAS.keys()):
    print (f'Computing Iothers {network}')
    FC_HC_ntw=FCtrt_HC_CL[ATLAS[network]]
    FC_SZ_ntw=FCtrt_SZ_CL[ATLAS[network]]
    maskut=np.triu_indices(len(FC_HC_ntw),k=1)
    #Plot para verificar FC subnetworks
    #plt.figure()
    #plt.imshow(FC_HC_ntw[:,:,0])
    
    #HC
    IothersS1_HC=np.zeros(len(subjHC_CL))
    IothersS2_HC=np.zeros(len(subjHC_CL))
    corrS1_HC=np.zeros(len(subjHC_CL)-1)
    corrS2_HC=np.zeros(len(subjHC_CL)-1)
    
    for i,subject in enumerate(np.arange(0,FC_HC_ntw.shape[2],2)):
            
            FC_sess1=np.arctanh(FC_HC_ntw[:,:,subject])
            FC_sess2=np.arctanh(FC_HC_ntw[:,:,subject+1])
           
            #idxG1 all sess1 but excluding this subject 
            idxG1=[x for x in np.arange(0,FC_HC_ntw.shape[2],2) if x != subject] 
            #idxG2 all sess1 but excluding this subject
            idxG2=[x for x in np.arange(1,FC_HC_ntw.shape[2],2) if x != subject+1]    
                      
            for k, other in enumerate(idxG1):
                FC_other=np.arctanh(FC_HC_ntw[:,:,other])
                corrS1_HC[k]=np.corrcoef(FC_sess1[maskut],FC_other[maskut])[0,1]
            for k, other in enumerate(idxG2):
                FC_other=np.arctanh(FC_HC_ntw[:,:,other])
                corrS2_HC[k]=np.corrcoef(FC_sess2[maskut],FC_other[maskut])[0,1]
           
            IothersS1_HC[i]=np.tanh(corrS1_HC.mean())
            IothersS2_HC[i]=np.tanh(corrS2_HC.mean())
       
    name_s1='Iothers_S1_'+network
    name_s2='Iothers_S2_'+network
    HC_df.loc[:,name_s1] = IothersS1_HC.tolist()
    HC_df.loc[:,name_s2] = IothersS2_HC.tolist()
    
    #SZ   
    IothersS1_SZ=np.zeros(len(subjSZ_CL))
    IothersS2_SZ=np.zeros(len(subjSZ_CL))
    corrS1_SZ=np.zeros(len(subjSZ_CL)-1)
    corrS2_SZ=np.zeros(len(subjSZ_CL)-1)
    
    for i,subject in enumerate(np.arange(0,FC_SZ_ntw.shape[2],2)):
            
            FC_sess1=np.arctanh(FC_SZ_ntw[:,:,subject])
            FC_sess2=np.arctanh(FC_SZ_ntw[:,:,subject+1])
           
            #idxG1 all sess1 but excluding this subject 
            idxG1=[x for x in np.arange(0,FC_SZ_ntw.shape[2],2) if x != subject] 
            #idxG2 all sess1 but excluding this subject
            idxG2=[x for x in np.arange(1,FC_SZ_ntw.shape[2],2) if x != subject+1]    
                      
            for k, other in enumerate(idxG1):
                FC_other=np.arctanh(FC_SZ_ntw[:,:,other])
                corrS1_SZ[k]=np.corrcoef(FC_sess1[maskut],FC_other[maskut])[0,1]
            for k, other in enumerate(idxG2):
                FC_other=np.arctanh(FC_SZ_ntw[:,:,other])
                corrS2_SZ[k]=np.corrcoef(FC_sess2[maskut],FC_other[maskut])[0,1]
           
            IothersS1_SZ[i]=np.tanh(corrS1_SZ.mean())
            IothersS2_SZ[i]=np.tanh(corrS2_SZ.mean())
    
    name_s1='Iothers_S1_'+network
    name_s2='Iothers_S2_'+network
    SZ_df.loc[:,name_s1] = IothersS1_SZ.tolist()
    SZ_df.loc[:,name_s2] = IothersS2_SZ.tolist()
     

#%%         GET Iself's

plots=0
raw_or_opt=2 # 2 = raw; 3 = optimize with PCA-toolbox


HCfg = eng.f_PCAtoolbox(matlab.double(FCtrt_HC_CL.tolist()),plots,nargout=4)
HC_Identmat_wb = HCfg[raw_or_opt]
HC_Iself_wb = np.diagonal(HC_Identmat_wb, axis1=0, axis2=1)
name='Iself_WB' #whole brain
HC_df.loc[:,name] = HC_Iself_wb.tolist()

SZfg = eng.f_PCAtoolbox(matlab.double(FCtrt_SZ_CL.tolist()),plots,nargout=4)
SZ_Identmat_wb = SZfg[raw_or_opt]
SZ_Iself_wb = np.diagonal(SZ_Identmat_wb, axis1=0, axis2=1)
name='Iself_WB' #whole brain
SZ_df.loc[:,name] = SZ_Iself_wb.tolist()


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

    HC_Identmat_ntw[:,:,idx]=HCfg[raw_or_opt]  
    SZ_Identmat_ntw[:,:,idx]=SZfg[raw_or_opt]  
 
    HC_Iself = np.diagonal(HC_Identmat_ntw[:,:,idx], axis1=0, axis2=1)
    name='Iself_'+network
    HC_df.loc[:,name] = HC_Iself.tolist()
    
    SZ_Iself = np.diagonal(SZ_Identmat_ntw[:,:,idx], axis1=0, axis2=1)
    name='Iself_'+network
    SZ_df.loc[:,name] = SZ_Iself.tolist()
    
    del HC_Iself, SZ_Iself
#

#%%         Write DataFrame

HC_df.loc[:, ('FDmax')] = HC_df.loc[:,("FD1", "FD2")].max(axis=1)
HC_df.loc[:,('Group')] = 'HC'

SZ_df.loc[:,"FDmax"] = SZ_df.loc[:,("FD1", "FD2")].max(axis=1)
SZ_df.loc[:,'Group'] = 'SZ'

Iselfs_df = pd.concat([HC_df,SZ_df])

Iselfs_df.loc[:,"Age2"] = Iselfs_df.Age*Iselfs_df.Age
agemean = Iselfs_df['Age'].mean()
Iselfs_df.loc[:,"Age_demeaned"] = Iselfs_df['Age'].sub(agemean)
age2mean = Iselfs_df['Age2'].mean()
Iselfs_df.loc[:,"Age2_demeaned"] = Iselfs_df['Age2'].sub(age2mean)

FDmaxmean = Iselfs_df['FDmax'].mean()
Iselfs_df.loc[:,"FDmax_demeaned"] = Iselfs_df['FDmax'].sub(FDmaxmean)

FD1mean = Iselfs_df['FD1'].mean()
Iselfs_df.loc[:,"FD1_demeaned"] = Iselfs_df['FD1'].sub(FD1mean)
FD2mean = Iselfs_df['FD2'].mean()
Iselfs_df.loc[:,"FD2_demeaned"] = Iselfs_df['FD2'].sub(FD2mean)


Iselfs_df.loc[:,"deltaPANSS"] = abs(Iselfs_df.TPANSS2-Iselfs_df.TPANSS)
Iselfs_df.loc[:,"deltaATP"] = abs(Iselfs_df.ATPdose-Iselfs_df.ATPdose2)
#Iselfs_df.loc[:,"deltaTP"] = abs(Iselfs_df.TP-Iselfs_df.TP2)

Iselfs_df=Iselfs_df.reset_index()


#if raw_or_opt == 2:
#    Iselfs_df.to_csv('Iselfs_dataframe.csv', index=False)
#else:
#    Iselfs_df.to_csv('IselfsOPT_dataframe.csv', index=False)

#%% (TO EXCLUDE OR NOT TO EXCLUDE subjects with fdmax >0.5)

# fig1= plt.figure(figsize=(20,20));
# ax1=fig1.add_subplot(10,1,1)
# plt.plot(Iselfs_df.index,Iselfs_df.Iself_WB)
# plt.plot(Iselfs_df.index,[x * Iselfs_df.Iself_WB.mean() for x in np.ones(len(Iselfs_df))],'--')
# xcoords = Iselfs_df.index[Iselfs_df.FDmax>0.5]
# for xc in xcoords:
#     plt.axvline(x=xc,color='r', linestyle = '--', linewidth=1)
# ax1.set_title('Iselfs ~  FDmax ')
# ax1.set_ylabel('Iself WB')
# ax2=fig1.add_subplot(10,1,10)
# plt.plot(Iselfs_df.index,Iselfs_df.FDmax)
# plt.plot(Iselfs_df.index,[x * 0.5 for x in np.ones(len(Iselfs_df))],'--r')
# ax2.set_ylabel("FDmax")

# for idx, network in enumerate(ATLAS.keys()):
#     name = 'Iself_' + network
#     ax=fig1.add_subplot(10,1,idx+2)
#     plt.plot(Iselfs_df.index,Iselfs_df[name])
#     plt.plot(Iselfs_df.index,[x * Iselfs_df[name].mean() for x in np.ones(len(Iselfs_df))],'--')
#     ax.set_ylabel(name)
#     xcoords = Iselfs_df.index[Iselfs_df.FDmax>0.5]
#     for xc in xcoords:
#         plt.axvline(x=xc,color='r', linestyle = '--', linewidth=1)
 
    
# corrFD=stats.pearsonr(Iselfs_df.FDmax,Iselfs_df["Iself_WB"])
# print(f'FDmax_corr_IselfWB: \n Pearson r: {corrFD[0]:.3f} \t p-val: {corrFD[1]:.3f}')
        
# for idx, network in enumerate(ATLAS.keys()):
#     name = 'Iself_' + network
#     corrFD=stats.pearsonr(Iselfs_df.FDmax,Iselfs_df[name])
#     print(f'FDmax_corr_{name}: \n Pearson r: {corrFD[0]:.3f} \t p-val: {corrFD[1]:.3f}')
 
    
# print(" \n Excluding fdmax > 0.5") 
# corrFD=stats.pearsonr(Iselfs_df.FDmax[Iselfs_df.FDmax<0.5],Iselfs_df.Iself_WB[Iselfs_df.FDmax<0.5])
# print(f'FDmax_corr_IselfWB: \n Pearson r: {corrFD[0]:.3f} \t p-val: {corrFD[1]:.3f}')
        
# for idx, network in enumerate(ATLAS.keys()):
#     name = 'Iself_' + network
#     corrFD=stats.pearsonr(Iselfs_df.FDmax[Iselfs_df.FDmax<0.5],Iselfs_df[name][Iselfs_df.FDmax<0.5])
#     print(f'FDmax_corr_{name}: \n Pearson r: {corrFD[0]:.3f} \t p-val: {corrFD[1]:.3f}')
    

#%% Iself GLM:  
#%%     # Model 1: Iself_subnetworks ~ age + sex + maxFD + Group

# standardizing dataframe
Iselfs_cols=Iselfs_df.filter(regex='^Iself',axis=1)
Iselfs_cols_zscore = Iselfs_cols.apply(stats.zscore)

Covars_zscore = Iselfs_df[["Age","Age2","Age_demeaned","Age2_demeaned","FDmax", "FDmax_demeaned"]].apply(stats.zscore)
df_zscore = pd.concat([Iselfs_cols_zscore,Covars_zscore, Iselfs_df[["Group","Sex"]]], axis=1)

##%% Make text file with model summaries (including Group as variable)    
uncorrected_p=np.zeros(9)
f = open("ISELFs_GLM_FDmaxdemeaned.txt", "w")

# print("The following subjects have FDmax > 0.5 \n", file = f)
# for x in Iselfs_df.ID[Iselfs_df.FDmax>0.5]:
#     print(x, file = f)

name = 'Iself_WB'
model = smf.ols((name +' ~ Age_demeaned + Sex + FDmax_demeaned + Group'), data=df_zscore).fit()
uncorrected_p[0]=model.pvalues["Group[T.SZ]"]
print(model.summary(), file = f)

# fig1= plt.figure();
# ax1=fig1.add_subplot(6,1,1)
# plt.plot(Iselfs_df.index,Iselfs_df.Iself_WB)
# ax1.set_title(name +' ~ Age + Age2 + Sex + FDmax + Group')
# ax1.set_ylabel(name)
# ax2=fig1.add_subplot(6,1,2)
# plt.plot(Iselfs_df.index,Iselfs_df.Age)
# ax2.set_ylabel("Age")
# ax2=fig1.add_subplot(6,1,3)
# plt.plot(Iselfs_df.index,Iselfs_df.Age2)
# ax2.set_ylabel("Age2")
# ax3=fig1.add_subplot(6,1,4)
# plt.plot(Iselfs_df.index,Iselfs_df.Sex,".")
# ax3.set_ylabel("Sex")
# ax4=fig1.add_subplot(6,1,5)
# plt.plot(Iselfs_df.index,Iselfs_df.FDmax)    
# ax4.set_ylabel("Fdmax") 
# ax5=fig1.add_subplot(6,1,6)
# plt.plot(Iselfs_df.index,pd.Categorical(Iselfs_df.Group,['HC','SZ']))
# ax5.set_xlabel("Subjects")
# ax5.set_ylabel("Group")

for idx, network in enumerate(ATLAS.keys()):
    name = 'Iself_' + network
    model = smf.ols( (name +' ~ Age_demeaned + Sex + FDmax_demeaned + Group'), data=df_zscore).fit()
    uncorrected_p[idx+1]=model.pvalues["Group[T.SZ]"]
    print(model.summary(), file = f)
        
f.close()

p_corr=multitest.fdrcorrection(uncorrected_p[1:], alpha=0.05, method='indep', is_sorted=False)[1]

#%%     # Plot residuals (model without Group as variable)

name = 'Iself_WB'
model = smf.ols((name +' ~ Age_demeaned + Sex + FDmax_demeaned '), data=Iselfs_df).fit()
fig = plt.figure(figsize=(20,20),dpi=100)
#plt.title("My title")
plot_opts = {
        #"cutoff_val": 0.3,
        #"cutoff_type": "abs",
        "label_fontsize": "x-large",
        "label_rotation": 0,
        #"bean_color": "#FF6F00",
        #"bean_mean_color": "#009D91",
        'bean_show_median':False,
        }
labels = ["HC","FEP"]

Resid = [model.resid[Iselfs_df.Group == id] for id in Iselfs_df.Group.unique()]
Text=stats.ttest_ind(Resid[0], Resid[1])
print(name + 'p-val: ' + str(Text[1]))

ax = fig.add_subplot(3,3,9)
sm.graphics.beanplot(Resid, ax=ax, labels=labels, plot_opts=plot_opts, jitter=True)
ax.set_xlabel("Group")
ax.set_ylabel("Residuals")
ax.set_title("Iself_WholeBrain")
#ax.text(.5,.5,"p value: {:.3f}".format(Text[1]))  
#SE ME PONE EL TEXTO EN CUALQUIER LADO. NDEAH!

for idx, network in enumerate(ATLAS.keys()):
    name = 'Iself_' + network
    model= smf.ols( (name +' ~ Age_demeaned + Sex + FDmax_demeaned '), data=Iselfs_df).fit()
       
    Resid = [model.resid[Iselfs_df.Group == id ] for id in Iselfs_df.Group.unique()]   
    Text=stats.ttest_ind(Resid[0], Resid[1])
    print(name + ' p-val: ' + str(Text[1]))
    
    ax = fig.add_subplot(3,3,idx+1)
    sm.graphics.beanplot(Resid, ax=ax, labels=labels, plot_opts=plot_opts, jitter=True)
    ax.set_xlabel("Group")
    ax.set_ylabel("Residuals")
    ax.set_title(name)

filename = '/Users/angeles/Desktop/Iself_plot1000.jpg'
plt.savefig(filename, dpi=1000)
    # plt.show()
#%%     # Model 2: # Iself_subnetworks ~ age + sex + maxFD + deltaATP + deltaPANSS       

  
SZ_only=Iselfs_df[(Iselfs_df.Group == 'SZ')] #& (Iselfs_df.FDmax<0.5)]

SZ_covar=SZ_only[["Age_demeaned","FDmax_demeaned","deltaPANSS","deltaATP"]]
SZ_covar_z = SZ_covar.dropna().apply(stats.zscore)

SZ_Iselfs=SZ_only.filter(regex='^Iself',axis=1)
SZ_Iselfs_z = SZ_Iselfs.apply(stats.zscore)

SZ_only_zscore = pd.concat([SZ_Iselfs_z,SZ_covar_z, SZ_only[["Sex"]]], axis=1)


f = open("Clinicalvars_CovDemeaned.txt", "w")


uncorrected_p_PANSS=np.zeros(9)
uncorrected_p_ATP=np.zeros(9)

name = 'Iself_WB'
model = smf.ols((name +' ~ Age_demeaned + Sex + FDmax_demeaned + deltaATP + deltaPANSS'), data=SZ_only_zscore).fit()  #
uncorrected_p_PANSS[0]=model.pvalues["deltaPANSS"]
uncorrected_p_ATP[0]=model.pvalues["deltaATP"]
print(model.summary(), file = f) 



fig1= plt.figure(figsize=(9,12));
ax1=fig1.add_subplot(7,1,1)
plt.plot(Iselfs_df.index,Iselfs_df.Iself_WB)
ax1.set_title(name +' ~ Age_demeaned + Sex + FDmax_demeaned + Group + deltaATP + deltaPANSS')
ax1.set_ylabel(name)
ax2=fig1.add_subplot(7,1,2)
plt.plot(Iselfs_df.index,Iselfs_df.Age_demeaned)
ax2.set_ylabel("Age")
ax2=fig1.add_subplot(7,1,3)
plt.plot(Iselfs_df.index,Iselfs_df.Sex,'.')
ax2.set_ylabel("Sex")
ax3=fig1.add_subplot(7,1,4)
plt.plot(Iselfs_df.index,Iselfs_df.FDmax_demeaned)
ax3.set_ylabel("FDmax_demeaned")
ax4=fig1.add_subplot(7,1,5)
plt.plot(Iselfs_df.index,pd.Categorical(Iselfs_df.Group,['HC','SZ']))    
ax4.set_ylabel("Group") 
ax5=fig1.add_subplot(7,1,6)
plt.plot(Iselfs_df.index,Iselfs_df.deltaPANSS)
ax5.set_ylabel("deltaPANSS")
ax6=fig1.add_subplot(7,1,7)
plt.plot(Iselfs_df.index,Iselfs_df.deltaATP)
ax6.set_ylabel("deltaATP")
ax6.set_xlabel("Subjects")




for idx, network in enumerate(ATLAS.keys()):
    name = 'Iself_' + network
    model = smf.ols( (name +' ~ Age_demeaned + Sex + FDmax_demeaned +  deltaATP + deltaPANSS'), data=SZ_only_zscore).fit()
    uncorrected_p_PANSS[idx+1]=model.pvalues["deltaPANSS"]
    uncorrected_p_ATP[idx+1]=model.pvalues["deltaATP"]
    print(model.summary(), file = f) 
        
f.close()

p_corrPANSS=multitest.fdrcorrection(uncorrected_p_PANSS[1:], alpha=0.05, method='indep', is_sorted=False)[1]
p_corrATP=multitest.fdrcorrection(uncorrected_p_ATP[1:], alpha=0.05, method='indep', is_sorted=False)[1]


#%% Iothers GLM:
#%%    # Mixed effects: random variable (2 Iothers per subject)

#CREAR DF CON DOS IOTHERS POR SUJETO: 
    #ID     Yvar
    #C01    IothersS1
    #C01    IothersS2
    #C02    IothersS1
    #C02    IothersS2
    # ....
uncorrected_p_GROUP=np.zeros(9)
uncorrected_p_PANSS=np.zeros(9)
uncorrected_p_ATP=np.zeros(9)

Iothers_df= pd.melt(Iselfs_df, id_vars=["ID","Age", "Age_demeaned", "Sex", "Group"], \
                             value_vars=["Iothers_S1_WB","Iothers_S2_WB"], \
                             var_name = "Sess", value_name= "Iothers_WB" )
    
FD_covar= pd.melt(Iselfs_df, id_vars=["ID"], \
                                 value_vars=["FD1_demeaned","FD2_demeaned"], \
                                 var_name = "FDsess", value_name= "FD_demeaned" )
    
PANSS_covar= pd.melt(Iselfs_df, id_vars=["ID"], \
                                 value_vars=["TPANSS","TPANSS2"], \
                                 var_name = "TPANSSsess", value_name= "TPANSS" )

ATP_covar= pd.melt(Iselfs_df, id_vars=["ID"], \
                                 value_vars=["ATPdose","ATPdose2"], \
                                 var_name = "ATPsess", value_name= "ATPdose" )
    
    
RepeatedMeasures_df = pd.concat([Iothers_df, FD_covar["FD_demeaned"], PANSS_covar["TPANSS"], ATP_covar["ATPdose"]],axis=1)

    #model: smf.mixedlm (Yvar ~ Age + ... + Group, data = Iothers_df, groups=Iothers_df["ID"])
    
f = open("Iothers_timevaryingCov.txt", "w")
print("MODEL: Iothers_WB ~ Age_demeaned + Sex + FD_demeaned + Group + (1|subject) \n", file = f)


md = smf.mixedlm("Iothers_WB ~ Age_demeaned + Sex + FD_demeaned + Group", \
                 data= RepeatedMeasures_df, groups=RepeatedMeasures_df["ID"])
mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
print(mdf.summary(),file=f)
uncorrected_p_GROUP[0]=mdf.pvalues["Group[T.SZ]"]

SZ_RepeatedMeasures_df= RepeatedMeasures_df[(RepeatedMeasures_df.Group == "SZ")]
  
    
#Model doesnt work with nan values (2 subjects have nan ATPdose2)    
SZ_RepeatedMeasures_df=SZ_RepeatedMeasures_df.dropna(axis = 0, how ='any')    
#  

print("MODEL: Iothers_WB ~ Age + Sex +FD + TPANSS+ + ATPdose + \n \
       (1|subject) dropped NaN ATPdose2 \n", file = f)


md = smf.mixedlm("Iothers_WB ~ Age_demeaned + Sex + FD_demeaned + TPANSS + ATPdose", \
                 data= SZ_RepeatedMeasures_df, groups=SZ_RepeatedMeasures_df["ID"])  
mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
print(mdf.summary(), file =f )
uncorrected_p_PANSS[0]=mdf.pvalues["TPANSS"]
uncorrected_p_ATP[0]=mdf.pvalues["ATPdose"]



print("MODEL: Iothers_WB ~ Age + Sex + FD + \n \
      Sess + (1|subject) dropped NaN ATPdose2 \n", file = f)


md = smf.mixedlm("Iothers_WB ~ Age_demeaned + Sex + FD_demeaned + Sess", \
                 data= SZ_RepeatedMeasures_df, groups=SZ_RepeatedMeasures_df["ID"])  
mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
print(mdf.summary(), file =f )

#Iothers_type en el modelo seria como agregar Sess1 vs Sess2 (tampoco tiene efecto significativo)
f.close()

#%%    # Iothers SUBNETWORKS: Mixed effects: random variable (2 Iothers per subject)

#CREAR DF CON DOS IOTHERS POR SUJETO: 
    #ID     Yvar
    #C01    IothersS1
    #C01    IothersS2
    #C02    IothersS1
    #C02    IothersS2
    # ....

Iothers_df= pd.melt(Iselfs_df, id_vars=["ID","Age", "Age_demeaned", "Sex", "Group"], \
                             value_vars=["Iothers_S1_WB","Iothers_S2_WB"], \
                             var_name = "Sess", value_name= "Iothers_WB" )
    
FD_covar= pd.melt(Iselfs_df, id_vars=["ID"], \
                                 value_vars=["FD1_demeaned","FD2_demeaned"], \
                                 var_name = "FDsess", value_name= "FD_demeaned" )
    
PANSS_covar= pd.melt(Iselfs_df, id_vars=["ID"], \
                                 value_vars=["TPANSS","TPANSS2"], \
                                 var_name = "TPANSSsess", value_name= "TPANSS" )

ATP_covar= pd.melt(Iselfs_df, id_vars=["ID"], \
                                 value_vars=["ATPdose","ATPdose2"], \
                                 var_name = "ATPsess", value_name= "ATPdose" )
     
RepeatedMeasures_df = pd.concat([Iothers_df, FD_covar["FD_demeaned"], PANSS_covar["TPANSS"], ATP_covar["ATPdose"]],axis=1)
    
for idx, network in enumerate(ATLAS.keys()):
    name_s1='Iothers_S1_'+network
    name_s2='Iothers_S2_'+network
 
    Iothers_subnet_df= pd.melt(Iselfs_df, id_vars=["ID"], \
                             value_vars=[name_s1,name_s2], \
                             var_name = "Sess", value_name= "Iothers_"+network )
    
    RepeatedMeasures_df = pd.concat([RepeatedMeasures_df,Iothers_subnet_df["Iothers_"+network]],axis=1)
        

SZ_RepeatedMeasures_df= RepeatedMeasures_df[(RepeatedMeasures_df.Group == "SZ")]  
#Model doesnt work with nan values (2 subjects have nan ATPdose2)    
SZ_RepeatedMeasures_df=SZ_RepeatedMeasures_df.dropna(axis = 0, how ='any')    
# 

    #model: smf.mixedlm (Yvar ~ Age + ... + Group, data = Iothers_df, groups=Iothers_df["ID"])

for idx, network in enumerate(ATLAS.keys()):
      
    f = open("Iothers_"+network+".txt", "w")
    print("MODEL: Iothers_"+network+" ~ Age_demeaned + Sex + FD_demeaned + Group + (1|subject) \n", file = f)

    md = smf.mixedlm("Iothers_"+network+" ~ Age_demeaned + Sex + FD_demeaned + Group", \
                 data= RepeatedMeasures_df, groups=RepeatedMeasures_df["ID"])
    mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
    print(mdf.summary(),file=f)
    uncorrected_p_GROUP[idx+1]=mdf.pvalues["Group[T.SZ]"]


    print("MODEL: Iothers_"+network+" ~ Age + + Sex + FD + TPANSS+ + ATPdose + \n \
       (1|subject) dropped NaN ATPdose2 \n", file = f)


    md = smf.mixedlm("Iothers_"+network+" ~ Age_demeaned + Sex + FD_demeaned + TPANSS + ATPdose", \
                     data= SZ_RepeatedMeasures_df, groups=SZ_RepeatedMeasures_df["ID"])  
    mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
    print(mdf.summary(), file =f )
    uncorrected_p_PANSS[idx+1]=mdf.pvalues["TPANSS"]
    uncorrected_p_ATP[idx+1]=mdf.pvalues["ATPdose"]

    
    #Iothers_type en el modelo seria como agregar Sess1 vs Sess2 (tampoco tiene efecto significativo)
    f.close()
    

p_corrGROUP=multitest.fdrcorrection(uncorrected_p_GROUP[1:], alpha=0.05, method='indep', is_sorted=False)[1]
p_corrPANSS=multitest.fdrcorrection(uncorrected_p_PANSS[1:], alpha=0.05, method='indep', is_sorted=False)[1]
p_corrATP=multitest.fdrcorrection(uncorrected_p_ATP[1:], alpha=0.05, method='indep', is_sorted=False)[1]


#%%     # Plot residuals (model without Group as variable)

name = 'Iothers_WB'
md = smf.mixedlm(name+" ~ Age_demeaned + Sex + FD_demeaned", \
               data= RepeatedMeasures_df, groups=RepeatedMeasures_df["ID"])    
mdf = md.fit()

fig = plt.figure(figsize=(20,20),dpi=100)
#plt.title("My title")
plot_opts = {
        #"cutoff_val": 0.3,
        #"cutoff_type": "abs",
        "label_fontsize": "x-large",
        "label_rotation": 0,
        #"bean_color": "#FF6F00",
        #"bean_mean_color": "#009D91",
        'bean_show_median':False,
        }
labels = ["HC","FEP"]

Resid = [mdf.resid[RepeatedMeasures_df.Group == id] for id in RepeatedMeasures_df.Group.unique()]
Text=stats.ttest_ind(Resid[0], Resid[1])
print(name + ' p-val: ' + str(Text[1]))

ax = fig.add_subplot(3,3,9)
sm.graphics.beanplot(Resid, ax=ax, labels=labels, plot_opts=plot_opts, jitter=True)
ax.set_xlabel("Group")
ax.set_ylabel("Residuals")
ax.set_title("Iothers_WholeBrain")
#ax.text(.5,.5,"p value: {:.3f}".format(Text[1]))  
#SE ME PONE EL TEXTO EN CUALQUIER LADO. NDEAH!

for idx, network in enumerate(ATLAS.keys()):
    name = 'Iothers_' + network
    md= smf.mixedlm( (name +' ~ Age_demeaned + Sex + FD_demeaned '), \
                   data= RepeatedMeasures_df, groups=RepeatedMeasures_df["ID"])    
    mdf = md.fit()
    
    Resid = [mdf.resid[RepeatedMeasures_df.Group == id ] for id in RepeatedMeasures_df.Group.unique()]   
    Text=stats.ttest_ind(Resid[0], Resid[1])
    print(name + ' p-val: ' + str(Text[1]))
    
    ax = fig.add_subplot(3,3,idx+1)
    sm.graphics.beanplot(Resid, ax=ax, labels=labels, plot_opts=plot_opts, jitter=True)
    ax.set_xlabel("Group")
    ax.set_ylabel("Residuals")
    ax.set_title(name)

    # plt.show()
filename = '/Users/angeles/Desktop/Iothers_plot1000.jpg'
plt.savefig(filename, dpi=1000)

#%% Dev from Healthy FC:
#%%    # Mixed effects: two measures per subject

#CREAR DF CON DOS IOTHERS POR SUJETO: 
    #ID     Yvar
    #C01    IothersS1
    #C01    IothersS2
    #C02    IothersS1
    #C02    IothersS2
    # ....
    
DevHealth_df= pd.melt(Iselfs_df, id_vars=["ID","Age", "Age_demeaned", "Sex", "Group"], \
                             value_vars=["DevFromHealth_S1","DevFromHealth_S2"], \
                             var_name = "Sess", value_name= "DevfromHealth_value" )
    
FD_covar= pd.melt(Iselfs_df, id_vars=["ID"], \
                                 value_vars=["FD1_demeaned","FD2_demeaned"], \
                                 var_name = "FDsess", value_name= "FD_demeaned" )
    
PANSS_covar= pd.melt(Iselfs_df, id_vars=["ID"], \
                                 value_vars=["TPANSS","TPANSS2"], \
                                 var_name = "TPANSSsess", value_name= "TPANSS" )

ATP_covar= pd.melt(Iselfs_df, id_vars=["ID"], \
                                 value_vars=["ATPdose","ATPdose2"], \
                                 var_name = "ATPsess", value_name= "ATPdose" )
    
    
RepeatedMeasures_df = pd.concat([DevHealth_df, FD_covar["FD_demeaned"], PANSS_covar["TPANSS"], ATP_covar["ATPdose"]],axis=1)

    #model: smf.mixedlm (Yvar ~ Age + ... + Group, data = Iothers_df, groups=Iothers_df["ID"])
    
f = open("DevfromHealth_FDdemeaned.txt", "w")
print("MODEL: DevfromHealth_value ~ Age_demeaned + FD_demeaned + Sex + Group + (1|subject) \n", file = f)


md = smf.mixedlm("DevfromHealth_value ~ Age_demeaned + Sex + FD_demeaned + Group", \
                 data= RepeatedMeasures_df, groups=RepeatedMeasures_df["ID"])
mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
print(mdf.summary(),file=f)


SZ_RepeatedMeasures_df= RepeatedMeasures_df[(RepeatedMeasures_df.Group == "SZ")]
  
    
#Model doesnt work with nan values (2 subjects have nan ATPdose2)    
SZ_RepeatedMeasures_df=SZ_RepeatedMeasures_df.dropna(axis = 0, how ='any')    
#  

print("MODEL: DevfromHealth_value ~ Age_demeaned + Sex + FD_demeaned + TPANSS+ + ATPdose + \n \
       (1|subject) dropped NaN ATPdose2 \n", file = f)


md = smf.mixedlm("DevfromHealth_value ~ Age_demeaned + Sex + FD_demeaned + TPANSS + ATPdose ", \
                 data= SZ_RepeatedMeasures_df, groups=SZ_RepeatedMeasures_df["ID"])  
mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
print(mdf.summary(), file =f )

print("MODEL: DevfromHealth_value ~ Age_demeaned + Sex + FD_demeaned + \n \
      Sess + (1|subject) dropped NaN ATPdose2 \n", file = f)


md = smf.mixedlm("DevfromHealth_value ~ Age_demeaned + Sex + FD_demeaned + Sess", \
                 data= SZ_RepeatedMeasures_df, groups=SZ_RepeatedMeasures_df["ID"])  
mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
print(mdf.summary(), file =f )

f.close()

#%% Plots: Deviation from healthy FC

fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,5))
fig.suptitle('Deviation from Healthy FC')
axes[0].set_title('Session 1')
axes[1].set_title('Session 2')
sns.violinplot(data= Iselfs_df,y="DevFromHealth_S1", x="Group", ax=axes[0])
sns.violinplot(data= Iselfs_df, y="DevFromHealth_S2", x="Group", ax=axes[1])


fig, (ax1,ax2) = plt.subplots(1, 2, sharex=True, figsize=(10,5))
fig.suptitle('Deviation from Healthy FC')
ax1.set_title('Session 1')
ax2.set_title('Session 2')
bins=[.2,.225,.25,.275,.3,.325,.35,.375,.4,.425,.45,.475,.50,.525,.55,.575,.60,.625,.65,.675,.70]
sns.histplot(Iselfs_df, x="DevFromHealth_S1", hue="Group", bins=bins, kde=True,ax=ax1)
sns.histplot(Iselfs_df, x="DevFromHealth_S2", hue="Group", bins=bins, kde=True, ax=ax2)

sns.displot(x=Iselfs_df.DevFromHealth_S1[Iselfs_df.Group=='HC'], kind= "kde", fill=True)

#HCsess1 - SZsess1 
stats.ttest_ind(Iselfs_df.DevFromHealth_S1[Iselfs_df.Group=='HC'],Iselfs_df.DevFromHealth_S1[Iselfs_df.Group=='SZ'])
#HCsess2 - SZsess1
stats.ttest_ind(Iselfs_df.DevFromHealth_S1[Iselfs_df.Group=='HC'],Iselfs_df.DevFromHealth_S2[Iselfs_df.Group=='SZ'])
#HCsess1 - SZsess2 
stats.ttest_ind(Iselfs_df.DevFromHealth_S1[Iselfs_df.Group=='HC'],Iselfs_df.DevFromHealth_S2[Iselfs_df.Group=='SZ'])
#HCsess2 - SZsess2 
stats.ttest_ind(Iselfs_df.DevFromHealth_S2[Iselfs_df.Group=='HC'],Iselfs_df.DevFromHealth_S2[Iselfs_df.Group=='SZ'])
#HCsess1 - HCsess2
stats.ttest_ind(Iselfs_df.DevFromHealth_S1[Iselfs_df.Group=='HC'],Iselfs_df.DevFromHealth_S2[Iselfs_df.Group=='HC'])
#SZsess1 - SZsess2
stats.ttest_ind(Iselfs_df.DevFromHealth_S1[Iselfs_df.Group=='SZ'],Iselfs_df.DevFromHealth_S2[Iselfs_df.Group=='SZ'])


stats.ttest_ind(DevFromHealth_S1_HC,DevFromHealth_S1_SZ)
#%%


fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,5))
fig.suptitle('Individual variability')
axes[0].set_title('Session 1')
axes[1].set_title('Session 2')
sns.violinplot(data= Iselfs_df,y="IndivVar_S1", x="Group", ax=axes[0])
sns.violinplot(data= Iselfs_df, y="IndivVar_S2", x="Group", ax=axes[1])

fig, (ax1,ax2) = plt.subplots(1, 2, sharex=True, figsize=(10,5))
fig.suptitle('Individual Variability')
ax1.set_title('Session 1')
ax2.set_title('Session 2')
bins=[.2,.225,.25,.275,.3,.325,.35,.375,.4,.425,.45,.475,.50,.525,.55,.575,.60,.625,.65,.675,.70]
sns.histplot(Iselfs_df, x="IndivVar_S1", hue="Group", bins=bins, kde=True,ax=ax1)
sns.histplot(Iselfs_df, x="IndivVar_S2", hue="Group", bins=bins, kde=True, ax=ax2)

#HCsess1 - SZsess1 
stats.ttest_ind(Iselfs_df.IndivVar_S1[Iselfs_df.Group=='HC'],Iselfs_df.IndivVar_S1[Iselfs_df.Group=='SZ'])
#HCsess2 - SZsess2 
stats.ttest_ind(Iselfs_df.IndivVar_S2[Iselfs_df.Group=='HC'],Iselfs_df.IndivVar_S2[Iselfs_df.Group=='SZ'])
#HCsess1 - HCsess2
stats.ttest_ind(Iselfs_df.IndivVar_S1[Iselfs_df.Group=='HC'],Iselfs_df.IndivVar_S2[Iselfs_df.Group=='HC'])
#SZsess1 - SZsess2
stats.ttest_ind(Iselfs_df.IndivVar_S1[Iselfs_df.Group=='SZ'],Iselfs_df.IndivVar_S2[Iselfs_df.Group=='SZ'])
#%% PLOTS

# CHEQUEAR CONTENIDO DE RepeatedMeasures_df

name = 'Iothers_WB'
md = smf.mixedlm(name+" ~ Age_demeaned + Sex + FD_demeaned", \
               data= RepeatedMeasures_df, groups=RepeatedMeasures_df["ID"])    
mdf = md.fit()

errSZ=stats.sem(mdf.resid[RepeatedMeasures_df.Group == 'SZ'])
errHC=stats.sem(mdf.resid[RepeatedMeasures_df.Group == 'HC'])

fig = plt.figure(figsize=(20,20),dpi=100)
#plt.title("My title")
plot_opts = {
        #"cutoff_val": 0.3,
        #"cutoff_type": "abs",
        "label_fontsize": "large",
        "label_rotation": 0,
        #"bean_color": "#FF6F00",
        #"bean_mean_color": "#009D91",
        'bean_show_median':False,
        }
Iothers=[RepeatedMeasures_df.Iothers_WB[RepeatedMeasures_df.Group == id] for id in RepeatedMeasures_df.Group.unique()]
Resid = [mdf.resid[RepeatedMeasures_df.Group == id] for id in RepeatedMeasures_df.Group.unique()]
Text=stats.ttest_ind(Resid[0], Resid[1])
print(name + ' p-val: ' + str(Text[1]))

ax = fig.add_subplot(3,3,9)
sm.graphics.beanplot(Resid, ax=ax, labels=labels, plot_opts=plot_opts,  \
                     jitter=True)
ax.set_xlabel("Group")
ax.set_ylabel("Residuals \n (Iothers ~ Age + Sex + FD)")
ax.set_title("Iothers_WholeBrain")
#ax.text(.5,.5,"p value: {:.3f}".format(Text[1]))  
#SE ME PONE EL TEXTO EN CUALQUIER LADO. NDEAH!

    # plt.show()
#filename = '/Users/angeles/Desktop/Iothers_plot1000.jpg'
#plt.savefig(filename, dpi=1000)


SZ_RepeatedMeasures_df= RepeatedMeasures_df[(RepeatedMeasures_df.Group == "SZ")]    
#Model doesnt work with nan values (2 subjects have nan ATPdose2)    
SZ_RepeatedMeasures_df=SZ_RepeatedMeasures_df.dropna(axis = 0, how ='any')    
#  

md = smf.mixedlm("Iothers_WB ~ Age_demeaned + Sex + FD_demeaned + ATPdose", \
                 data= SZ_RepeatedMeasures_df, groups=SZ_RepeatedMeasures_df["ID"])  
mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
print(mdf.summary())

fig = plt.figure()
x=SZ_RepeatedMeasures_df.TPANSS
y= mdf.resid
ax=sns.regplot(x, y)
ax.set_xlabel("PANSS total")
ax.set_ylabel("Residuals \n (Iothers ~ Age + Sex + FD + APdose)")
ax.set_title("Iothers vs PANSS")


md = smf.mixedlm("Iothers_WB ~ Age_demeaned + Sex + FD_demeaned + TPANSS", \
                 data= SZ_RepeatedMeasures_df, groups=SZ_RepeatedMeasures_df["ID"])  
mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
print(mdf.summary())

fig = plt.figure()
x=SZ_RepeatedMeasures_df.ATPdose
y= mdf.resid
ax=sns.regplot(x, y)
ax.set_xlabel("AP Dose")
ax.set_ylabel("Residuals \n (Iothers ~ Age + Sex + FD + TPANSS)")
ax.set_title("Iothers vs AP dose")




name = 'Iself_WB'
model = smf.ols((name +' ~ Age_demeaned + Sex + FDmax_demeaned '), data=Iselfs_df).fit()
fig = plt.figure(figsize=(20,20),dpi=100)
#plt.title("My title")
plot_opts = {
        #"cutoff_val": 0.3,
        #"cutoff_type": "abs",
        "label_fontsize": "large",
        "label_rotation": 0,
        #"bean_color": "#FF6F00",
        #"bean_mean_color": "#009D91",
        'bean_show_median':False,
        }

Iself=[Iselfs_df.Iself_WB[Iselfs_df.Group == id] for id in Iselfs_df.Group.unique()]
Resid = [model.resid[Iselfs_df.Group == id] for id in Iselfs_df.Group.unique()]
Text=stats.ttest_ind(Resid[0], Resid[1])
print(name + 'p-val: ' + str(Text[1]))

ax = fig.add_subplot(3,3,9)
sm.graphics.beanplot(Iself, ax=ax, labels=labels, plot_opts=plot_opts, \
                     jitter=True)
ax.set_xlabel("Group")
ax.set_ylabel("Residuals \n (Iself ~ Age + Sex + FD)")
ax.set_title("Iself_WholeBrain")
#ax.text(.5,.5,"p value: {:.3f}".format(Text[1]))  
#SE ME PONE EL TEXTO EN CUALQUIER LADO. NDEAH!

name = 'Iself_WB'
model = smf.ols((name +' ~ Age_demeaned + Sex + FDmax_demeaned + deltaPANSS'), data=SZ_only).fit()  #

fig = plt.figure()
x=SZ_only.deltaATP
y= model.resid
ax=sns.regplot(x, y)
ax.set_xlabel("delta AP Dose")
ax.set_ylabel("Residuals \n (Iself ~ Age + Sex + FD + deltaPANSS)")
ax.set_title("Iself vs delta AP dose")


SZ_only28=SZ_only[~SZ_only.ATPdose2.isna()]  

name = 'Iself_WB'
model = smf.ols((name +' ~ Age_demeaned + Sex + FDmax_demeaned + deltaATP'), data=SZ_only28).fit()  #

fig = plt.figure()
x=SZ_only28.deltaPANSS
y= model.resid
ax=sns.regplot(x, y)
ax.set_xlabel("delta PANSS")
ax.set_ylabel("Residuals \n (Iself ~ Age + Sex + FD + deltaATP)")
ax.set_title("Iself vs delta PANSS")




name='DevfromHealth_value'
md = smf.mixedlm(name+" ~ Age_demeaned + Sex + FD_demeaned", \
                 data= RepeatedMeasures_df, groups=RepeatedMeasures_df["ID"])
mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
print(mdf.summary())
fig = plt.figure(figsize=(20,20),dpi=100)

#plt.title("My title")
plot_opts = {
        #"cutoff_val": 0.3,
        #"cutoff_type": "abs",
        "label_fontsize": "large",
        "label_rotation": 0,
        #"bean_color": "#FF6F00",
        #"bean_mean_color": "#009D91",
        'bean_show_median':False,
        }

#Dev=[RepeatedMeasures_df.DevfromHealth_value[RepeatedMeasures_df.Group == id] for id in RepeatedMeasures_df.Group.unique()]

Resid = [mdf.resid[RepeatedMeasures_df.Group == id] for id in RepeatedMeasures_df.Group.unique()]
Text=stats.ttest_ind(Resid[0], Resid[1])
print(name + ' p-val: ' + str(Text[1]))

ax = fig.add_subplot(3,3,9)
sm.graphics.beanplot(Resid, ax=ax, labels=labels, plot_opts=plot_opts,  \
                     jitter=True)
ax.set_xlabel("Group")
ax.set_ylabel("Residuals \n (DevfromHealthyFC ~ Age + Sex + FD)")
ax.set_title("Dev from healthy FC")



SZ_RepeatedMeasures_df= RepeatedMeasures_df[(RepeatedMeasures_df.Group == "SZ")]
  
#Model doesnt work with nan values (2 subjects have nan ATPdose2)    
SZ_RepeatedMeasures_df=SZ_RepeatedMeasures_df.dropna(axis = 0, how ='any')    


md = smf.mixedlm("DevfromHealth_value ~ Age_demeaned + Sex + FD_demeaned + TPANSS ", \
                 data= SZ_RepeatedMeasures_df, groups=SZ_RepeatedMeasures_df["ID"])  
mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
print(mdf.summary())

fig = plt.figure()
x=SZ_RepeatedMeasures_df.ATPdose
y= mdf.resid
ax=sns.regplot(x, y)
ax.set_xlabel("AP dose")
ax.set_ylabel("Residuals \n (DevHFC~ Age + Sex + FD + TPANSS)")
ax.set_title("DevHFC vs AP dose")


md = smf.mixedlm("DevfromHealth_value ~ Age_demeaned + Sex + FD_demeaned + ATPdose ", \
                 data= SZ_RepeatedMeasures_df, groups=SZ_RepeatedMeasures_df["ID"])  
mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
print(mdf.summary())

fig = plt.figure()
x=SZ_RepeatedMeasures_df.TPANSS
y= mdf.resid
ax=sns.regplot(x, y)
ax.set_xlabel("TPANSS")
ax.set_ylabel("Residuals \n (DevHFC~ Age + Sex + FD + APdose)")
ax.set_title("DevHFC vs PANSS total")




