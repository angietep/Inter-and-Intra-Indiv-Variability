#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 12:49:29 2022

@author: angeles
"""


#%% Import and define

import pandas as pd
import scipy.io as sio   #to import matlab data
import os
import numpy as np
#import matlab.engine
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats import multitest
import statsmodels.formula.api as smf
#from statannot import add_stat_annotation
from scipy import stats
from plotnine import * #to plot with ggplot
import patchworklib as pw  #to make subplots
# statmodels functions to get VIF 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def getDevfHC_corrmean(*args): #(FCtrt,FCtrtHC) corr(FC_subji_Sess1, meanFC_group_Sess1)

    FCtrt = args[0]
        
    maskut=np.triu_indices(FCtrt.shape[0],k=1)
        
    DevFromHealth_S1=np.zeros(FCtrt.shape[2]//2)
    DevFromHealth_S2=np.zeros(FCtrt.shape[2]//2)

    if len(args)==2:   
        FCtrtHC= args[1]

        idxG1=[x for x in np.arange(0,FCtrtHC.shape[2],2)]
        idxG2=[x for x in np.arange(1,FCtrtHC.shape[2],2)]    
        
        FC_HCS1 = FCtrtHC[:,:, idxG1]
        FC_HCS2 = FCtrtHC[:,:, idxG2]
        
        refFC_groupS1=np.mean(FC_HCS1,axis=2) #mean FC sess 1
        refFC_groupS2=np.mean(FC_HCS2,axis=2) #mean FC sess 2 

        for i,subject in enumerate(np.arange(0,FCtrt.shape[2],2)):
           
            FC_sess1 = FCtrt[:,:,subject]
            FC_sess2 = FCtrt[:,:,subject+1]
               
            DevFromHealth_S1[i]=1- np.corrcoef(FC_sess1[maskut],refFC_groupS1[maskut])[0,1]
            DevFromHealth_S2[i]=1- np.corrcoef(FC_sess2[maskut],refFC_groupS2[maskut])[0,1]
    
    elif len(args)==1:
        
        for i,subject in enumerate(np.arange(0,FCtrt.shape[2],2)):
                
                FC_sess1 = FCtrt[:,:,subject]
                FC_sess2 = FCtrt[:,:,subject+1]

                idxG1=[x for x in np.arange(0,FCtrt.shape[2],2) if x != subject] 
                idxG2=[x for x in np.arange(1,FCtrt.shape[2],2) if x != subject+1]    
               
                FC_S1 = FCtrt[:,:, idxG1]
                FC_S2 = FCtrt[:,:, idxG2]
                refFC_groupS1=np.mean(FC_S1,axis=2) #mean FC sess 1
                refFC_groupS2=np.mean(FC_S2,axis=2) #mean FC sess 2     
              
                DevFromHealth_S1[i]=1- np.corrcoef(FC_sess1[maskut],refFC_groupS1[maskut])[0,1]
                DevFromHealth_S2[i]=1- np.corrcoef(FC_sess2[maskut],refFC_groupS2[maskut])[0,1]
                
    return DevFromHealth_S1, DevFromHealth_S2

def getDevfHC_meancorr(*args): #(FCtrt,FCtrtHC) #mean(corr(FC_subji_sess1,FC_subjk_sess1))
 #Iothers against HC group
    FCtrt = args[0]
        
    maskut=np.triu_indices(FCtrt.shape[0],k=1)
        
    DevFromHealth_S1=np.zeros(FCtrt.shape[2]//2)
    DevFromHealth_S2=np.zeros(FCtrt.shape[2]//2)

    if len(args)==2: #FCtrt vs FCtrtHC
        FCtrtHC= args[1]
        
        corrS1=np.zeros(FCtrtHC.shape[2]//2)
        corrS2=np.zeros(FCtrtHC.shape[2]//2)
        
        for i,subject in enumerate(np.arange(0,FCtrt.shape[2],2)):

            FC_sess1 = FCtrt[:,:,subject]
            FC_sess2 = FCtrt[:,:,subject+1]
            
            idxG1_HC=[x for x in np.arange(0,FCtrtHC.shape[2],2)]
            idxG2_HC=[x for x in np.arange(1,FCtrtHC.shape[2],2)]    
            
            for k, other in enumerate(idxG1_HC):
                FC_HCother=FCtrtHC[:,:,other]
                corrS1[k]=np.corrcoef(FC_sess1[maskut],FC_HCother[maskut])[0,1]
           
            for k, other in enumerate(idxG2_HC):
                FC_HCother=FCtrtHC[:,:,other]
                corrS2[k]=np.corrcoef(FC_sess2[maskut],FC_HCother[maskut])[0,1] 
                         
            DevFromHealth_S1[i]=1- corrS1.mean()
            DevFromHealth_S2[i]=1- corrS2.mean()
    
    elif len(args)==1:  #FCtrt = FCtrtHC
    
        corrS1=np.zeros(FCtrt.shape[2]//2 - 1)
        corrS2=np.zeros(FCtrt.shape[2]//2 - 1)
        
        for i,subject in enumerate(np.arange(0,FCtrt.shape[2],2)):
                
            FC_sess1 = FCtrt[:,:,subject]
            FC_sess2 = FCtrt[:,:,subject+1]

            idxG1=[x for x in np.arange(0,FCtrt.shape[2],2) if x != subject] 
            idxG2=[x for x in np.arange(1,FCtrt.shape[2],2) if x != subject+1]    
               
            for k, other in enumerate(idxG1):
                FC_other=FCtrt[:,:,other]
                corrS1[k]=np.corrcoef(FC_sess1[maskut],FC_other[maskut])[0,1]
                
            for k, other in enumerate(idxG2):
                FC_other=FCtrt[:,:,other]
                corrS2[k]=np.corrcoef(FC_sess2[maskut],FC_other[maskut])[0,1] 
              
            DevFromHealth_S1[i]=1- corrS1.mean()
            DevFromHealth_S2[i]=1- corrS2.mean()

    return DevFromHealth_S1, DevFromHealth_S2


def getIothers(FCtrt):  #mean(corr(FC_subji_sess1,FC_subjk_sess1))

    maskut=np.triu_indices(FCtrt.shape[0],k=1)
    
    IothersS1=np.zeros(FCtrt.shape[2]//2)
    IothersS2=np.zeros(FCtrt.shape[2]//2)
    corrS1=np.zeros(FCtrt.shape[2]//2 - 1)
    corrS2=np.zeros(FCtrt.shape[2]//2 - 1)

    for i,subject in enumerate(np.arange(0,FCtrt.shape[2],2)):
            
            FC_sess1=FCtrt[:,:,subject]
            FC_sess2=FCtrt[:,:,subject+1]
           
            #idxG1 all sess1 but excluding this subject 
            idxG1=[x for x in np.arange(0,FCtrt.shape[2],2) if x != subject] 
            #idxG2 all sess2 but excluding this subject
            idxG2=[x for x in np.arange(1,FCtrt.shape[2],2) if x != subject+1]    
                      
            for k, other in enumerate(idxG1):
                FC_other=FCtrt[:,:,other]
                corrS1[k]=np.corrcoef(FC_sess1[maskut],FC_other[maskut])[0,1]
            for k, other in enumerate(idxG2):
                FC_other=FCtrt[:,:,other]
                corrS2[k]=np.corrcoef(FC_sess2[maskut],FC_other[maskut])[0,1]
           
            IothersS1[i]=corrS1.mean()
            IothersS2[i]=corrS2.mean()
            
    return IothersS1, IothersS2

def fisher(FC): # Fisher's z inf in diag get assigned = 0
    return np.nan_to_num(np.arctanh(FC),posinf=0)
   
def fisher_inv(FC):
    return np.tanh(FC)

def getIself(FCtrt):
    maskut=np.triu_indices(FCtrt.shape[0],k=1)
    
    Iselfs=np.zeros(FCtrt.shape[2]//2)

    for i,subject in enumerate(np.arange(0,FCtrt.shape[2],2)):
            
            FC_sess1=FCtrt[:,:,subject]
            FC_sess2=FCtrt[:,:,subject+1]
                                 
            Iselfs[i]=np.corrcoef(FC_sess1[maskut],FC_sess2[maskut])[0,1]
            
    return Iselfs
    

#%% Initialize and load data 

os.chdir('/Users/angeles/Documents/GitHub/Iself_Iothers/')

#%behav data
table_CL = pd.read_csv('Data/SubjectsData.csv')

#Functional connectomes
FC=sio.loadmat('Data/FCtrtDiCER1_SZCL_SCh400Nt7TianS2_vox275.mat')
FCtrt_SZ_CL_nofish=np.array(FC['FCtrt_SZ_CL'])
FCtrt_SZ_CL= fisher(FCtrt_SZ_CL_nofish)

FC=sio.loadmat('Data/FCtrtDiCER1_HCCL_SCh400Nt7TianS2_vox275.mat')
FCtrt_HC_CL_nofish=np.array(FC['FCtrt_HC_CL'])
FCtrt_HC_CL= fisher(FCtrt_HC_CL_nofish)

#lists of subjects from FCs
subjHC_CL=['C01','C02','C03','C04','C05','C07','C11','C15','C16', \
           'C18','C19','C20','C21','C24','C25','C27','C28','C30', \
           'C31','C32','C36','C41','C42','C43','C46','C47','C48', \
           'C50','C51','C52','C53','C57']
    
subjSZ_CL=['1401','1402','1405','1407','1413','1418','1420','1421', \
           '1423','1424','1425','1426','1428','1430','1433','1434', \
           '1435','1436','1437','1441','1442','1443','1445','1446', \
           '1449','1457','1458','1461','1465','1466']
 

# SUBNETWORKS: Schaefer400 7Networks + 32 subcort Tian

ATLAS={'VIS': np.ix_(np.concatenate((np.arange(0,31,1),np.arange(200,230,1))),np.concatenate((np.arange(0,31,1),np.arange(200,230,1)))), #[1:31,201:230];  MATLAB idx
       'SOM': np.ix_(np.concatenate((np.arange(31,68,1),np.arange(230,270,1))),np.concatenate((np.arange(31,68,1),np.arange(230,270,1)))), #[32:68,231:270];
       'DORSATT': np.ix_(np.concatenate((np.arange(68,91,1),np.arange(270,293,1))),np.concatenate((np.arange(68,91,1),np.arange(270,293,1)))), #[69:91,271:293]; 
       'VENTATT': np.ix_(np.concatenate((np.arange(91,113,1),np.arange(293,318,1))),np.concatenate((np.arange(91,113,1),np.arange(293,318,1)))), #[92:113,294:318]; 
       'LIMBIC': np.ix_(np.concatenate((np.arange(113,126,1),np.arange(318,331,1))),np.concatenate((np.arange(113,126,1),np.arange(318,331,1)))), #[114:126,319:331];
       'FP': np.ix_(np.concatenate((np.arange(126,148,1),np.arange(331,361,1))),np.concatenate((np.arange(126,148,1),np.arange(331,361,1)))),# [127:148,332:361];  
       'DMN': np.ix_(np.concatenate((np.arange(148,200,1),np.arange(361,400,1))),np.concatenate((np.arange(148,200,1),np.arange(361,400,1)))), #[149:200,362:400];
       'subcort': np.ix_(np.concatenate((np.arange(400,416,1),np.arange(416,432,1))),np.concatenate((np.arange(400,416,1),np.arange(416,432,1)))), #[401:416,417:432]; 
       }

HC_df= table_CL.loc[table_CL['ID'].isin(subjHC_CL)].reset_index()
SZ_df=table_CL.loc[table_CL['ID'].isin(subjSZ_CL)].reset_index()
del table_CL

HC_df.loc[:, ('FDmax')] = HC_df.loc[:,("FD1", "FD2")].max(axis=1)
HC_df.loc[:,('Group')] = 'HC'

SZ_df.loc[:,"FDmax"] = SZ_df.loc[:,("FD1", "FD2")].max(axis=1)
SZ_df.loc[:,'Group'] = 'SZ'

#%% get Iselfs, Iothers and DevfHC (Whole Brain)  

print("DevfHC_corrmean")
[DevFromHealth_S1_HC, DevFromHealth_S2_HC] =getDevfHC_corrmean(FCtrt_HC_CL)              
[DevFromHealth_S1_SZ, DevFromHealth_S2_SZ] =getDevfHC_corrmean(FCtrt_SZ_CL, FCtrt_HC_CL)              
print("DevfHC meancorr")
[DevFromHealth_meancorr_S1_HC, DevFromHealth_meancorr_S2_HC] =getDevfHC_meancorr(FCtrt_HC_CL)              
[DevFromHealth_meancorr_S1_SZ, DevFromHealth_meancorr_S2_SZ] =getDevfHC_meancorr(FCtrt_SZ_CL, FCtrt_HC_CL)    
print("Iothers")
[Iothers_S1_HC, Iothers_S2_HC] =getIothers(FCtrt_HC_CL)              
[Iothers_S1_SZ, Iothers_S2_SZ] =getIothers(FCtrt_SZ_CL)          
print("Iselfs")
Iselfs_SZ =getIself(FCtrt_SZ_CL)          
Iselfs_HC =getIself(FCtrt_HC_CL)          

HC_df.loc[:,'DevFromHealth_S1'] = DevFromHealth_meancorr_S1_HC.tolist() 
HC_df.loc[:,'DevFromHealth_S2'] = DevFromHealth_meancorr_S2_HC.tolist()
SZ_df.loc[:,'DevFromHealth_S1'] = DevFromHealth_meancorr_S1_SZ.tolist()
SZ_df.loc[:,'DevFromHealth_S2'] = DevFromHealth_meancorr_S2_SZ.tolist()
del DevFromHealth_S1_HC, DevFromHealth_S2_HC, DevFromHealth_S1_SZ, DevFromHealth_S2_SZ
del DevFromHealth_meancorr_S1_HC, DevFromHealth_meancorr_S1_SZ, DevFromHealth_meancorr_S2_HC, DevFromHealth_meancorr_S2_SZ

HC_df.loc[:,'Iothers_S1_WB'] = Iothers_S1_HC.tolist() #Whole Brain
HC_df.loc[:,'Iothers_S2_WB'] = Iothers_S2_HC.tolist()        
SZ_df.loc[:,'Iothers_S1_WB'] = Iothers_S1_SZ.tolist()
SZ_df.loc[:,'Iothers_S2_WB'] = Iothers_S2_SZ.tolist()
del Iothers_S1_HC, Iothers_S2_HC, Iothers_S1_SZ, Iothers_S2_SZ 

HC_df.loc[:,'Iself_WB'] = Iselfs_HC.tolist()        
SZ_df.loc[:,'Iself_WB'] = Iselfs_SZ.tolist()
del Iselfs_HC, Iselfs_SZ
#%% get Iothers and Iselfs subnetworks 

for idx, network in enumerate(ATLAS.keys()):
    print (f'Subnetwork {network}')
    FCtrt_HC_ntw=FCtrt_HC_CL[ATLAS[network]]
    FCtrt_SZ_ntw=FCtrt_SZ_CL[ATLAS[network]]
    #Plot para verificar FC subnetworks
    #plt.figure()
    #plt.imshow(FC_HC_ntw[:,:,0])

    name_s1='Iothers_S1_'+network
    name_s2='Iothers_S2_'+network
    name_Iself='Iself_'+network
    
    #HC
    [Iothers_S1_HC, Iothers_S2_HC] =getIothers(FCtrt_HC_ntw)   
    Iself_HC_ntw=getIself(FCtrt_HC_ntw)
    #SZ
    [Iothers_S1_SZ, Iothers_S2_SZ] =getIothers(FCtrt_SZ_ntw)              
    Iself_SZ_ntw=getIself(FCtrt_SZ_ntw)
       
    HC_df.loc[:,name_s1] = Iothers_S1_HC.tolist()
    HC_df.loc[:,name_s2] = Iothers_S2_HC.tolist()
    HC_df.loc[:,name_Iself] = Iself_HC_ntw.tolist()
    SZ_df.loc[:,name_Iself] = Iself_SZ_ntw.tolist()
    SZ_df.loc[:,name_s1] = Iothers_S1_SZ.tolist()
    SZ_df.loc[:,name_s2] = Iothers_S2_SZ.tolist()
    
    del Iself_HC_ntw, Iself_SZ_ntw, Iothers_S1_HC, Iothers_S2_HC, Iothers_S1_SZ, Iothers_S2_SZ, name_s1, name_s2, name_Iself, FCtrt_HC_ntw, FCtrt_SZ_ntw
del idx, network

#%% Organize DataFrames

#CONCATENATE DF

SZandHC_df = pd.concat([HC_df,SZ_df])
SZandHC_df=SZandHC_df.reset_index()

SZandHC_df.loc[:,"deltaPANSS"] = abs(SZandHC_df.TPANSS2-SZandHC_df.TPANSS)
SZandHC_df.loc[:,"deltaTP"] = abs(SZandHC_df.TP2-SZandHC_df.TP)
SZandHC_df.loc[:,"deltaTN"] = abs(SZandHC_df.TN2-SZandHC_df.TN)
SZandHC_df.loc[:,"deltaTG"] = abs(SZandHC_df.TG2-SZandHC_df.TG)

SZandHC_df.loc[:,"deltaATP"] = abs(SZandHC_df.ATPdose2-SZandHC_df.ATPdose)

#SESS1 - SESS2
SZandHC_df.loc[:,"deltaDevfHC_signed"] = SZandHC_df.DevFromHealth_S1-SZandHC_df.DevFromHealth_S2
SZandHC_df.loc[:,"deltaPANSS_signed"] = SZandHC_df.TPANSS-SZandHC_df.TPANSS2
SZandHC_df.loc[:,"deltaTP_signed"] = SZandHC_df.TP-SZandHC_df.TP2
SZandHC_df.loc[:,"deltaTN_signed"] = SZandHC_df.TN-SZandHC_df.TN2
SZandHC_df.loc[:,"deltaTG_signed"] = SZandHC_df.TG-SZandHC_df.TG2
SZandHC_df.loc[:,"deltaATP_signed"] = SZandHC_df.ATPdose-SZandHC_df.ATPdose2


SZandHC_df.rename(columns = {'DaysBetweenSessions':'DBS'}, inplace = True)
 

SZandHC_df = SZandHC_df.drop(columns=['MATRICS1', 'MATRICS2', 'IQ', 'Education','index','level_0'])
SZandHC_df = SZandHC_df.drop(SZandHC_df[SZandHC_df.DBS.isna()].index)

# SZ only

SZ_only = SZandHC_df[(SZandHC_df.Group == 'SZ')] 
SZ_only = SZ_only.drop(SZ_only[SZ_only.deltaATP.isna()].index)


#%% Create Repeated Measures DFs

#####################
### Repeated measures HC vs SZ
#####################
"""
#Dos IOTHERS/DevfHC POR SUJETO: 
    #ID     Yvar
    #C01    IothersS1
    #C01    IothersS2
    #C02    IothersS1
    #C02    IothersS2
    # ....
"""
Iothers_rep = pd.melt(SZandHC_df, id_vars=["ID","Age", "DBS", "Sex", "Group"], \
                             value_vars=["Iothers_S1_WB","Iothers_S2_WB"], \
                             var_name = "Sess", value_name= "Iothers_WB" )    
    
DevHealth_rep = pd.melt(SZandHC_df, id_vars=["ID"], 
                             value_vars=["DevFromHealth_S1","DevFromHealth_S2"], 
                             var_name = "Sess", value_name= "DevfromHealth_value" )    
    
FD_rep = pd.melt(SZandHC_df, id_vars=["ID"], 
                                 value_vars=["FD1","FD2"], 
                                 var_name = "FDsess", value_name= "FD" )

RepeatedMeasures_df = pd.concat([Iothers_rep, DevHealth_rep["DevfromHealth_value"] ,
                                 FD_rep["FD"]],axis=1)


for idx, network in enumerate(ATLAS.keys()):
    name_s1='Iothers_S1_'+network
    name_s2='Iothers_S2_'+network
 
    Iothers_subnet_df= pd.melt(SZandHC_df, id_vars=["ID"], \
                             value_vars=[name_s1,name_s2], \
                             var_name = "Sess", value_name= "Iothers_"+network )
    
    RepeatedMeasures_df = pd.concat([RepeatedMeasures_df,Iothers_subnet_df["Iothers_"+network]],axis=1)
 
    

del Iothers_rep, DevHealth_rep, FD_rep, Iothers_subnet_df

#################
#### SZ ONLY
#################

Iothers_rep = pd.melt(SZ_only, id_vars=["ID","Age", "DBS", "Sex", "Group"], \
                             value_vars=["Iothers_S1_WB","Iothers_S2_WB"], \
                             var_name = "Sess", value_name= "Iothers_WB" )
    
DevHealth_rep = pd.melt(SZ_only, id_vars=["ID"], 
                             value_vars=["DevFromHealth_S1","DevFromHealth_S2"], 
                             var_name = "Sess", value_name= "DevfromHealth_value" )    
    
FD_rep = pd.melt(SZ_only, id_vars=["ID"], 
                                 value_vars=["FD1","FD2"], 
                                 var_name = "FDsess", value_name= "FD" )
    
PANSS_rep = pd.melt(SZ_only, id_vars=["ID"], \
                                 value_vars=["TPANSS","TPANSS2"], \
                                 var_name = "TPANSSsess", value_name= "TPANSS" )

TP_rep = pd.melt(SZ_only, id_vars=["ID"], \
                                 value_vars=["TP","TP2"], \
                                 var_name = "TPsess", value_name= "TP" )
TN_rep = pd.melt(SZ_only, id_vars=["ID"], \
                                 value_vars=["TN","TN2"], \
                                 var_name = "TNsess", value_name= "TN" )

TG_rep = pd.melt(SZ_only, id_vars=["ID"], \
                             value_vars=["TG","TG2"], \
                             var_name = "TGsess", value_name= "TG" )

ATP_rep = pd.melt(SZ_only, id_vars=["ID"], \
                                 value_vars=["ATPdose","ATPdose2"], \
                                 var_name = "ATPsess", value_name= "ATPdose" )
    
    
SZ_RepeatedMeasures_df = pd.concat([Iothers_rep, DevHealth_rep["DevfromHealth_value"] ,
                                 FD_rep["FD"], PANSS_rep["TPANSS"],
                                 TN_rep["TN"], TP_rep["TP"], TG_rep["TG"],
                                 ATP_rep["ATPdose"]],axis=1)

for idx, network in enumerate(ATLAS.keys()):
    name_s1='Iothers_S1_'+network
    name_s2='Iothers_S2_'+network
 
    Iothers_subnet_df= pd.melt(SZ_only, id_vars=["ID"], \
                             value_vars=[name_s1,name_s2], \
                             var_name = "Sess", value_name= "Iothers_"+network )
    
    SZ_RepeatedMeasures_df = pd.concat([SZ_RepeatedMeasures_df,Iothers_subnet_df["Iothers_"+network]],axis=1)
 

del Iothers_rep, DevHealth_rep, FD_rep, PANSS_rep, 
del TP_rep, TN_rep, TG_rep, ATP_rep, Iothers_subnet_df


#%% Standardize dataframes

Numvars_z = SZandHC_df.drop(columns=["Group","ID","Sex"]
                           +list(SZandHC_df.filter(regex='^ATP')) #ATPdose, ATPdose2
                           +list(SZandHC_df.filter(regex='^delta')) # deltaPANSS,deltaATP
                           +list(SZandHC_df.filter(regex='^T')) #TPANSS, TN, TG, TP
                           )
deltaDevfHC=SZandHC_df["deltaDevfHC_signed"]
Numvars_z = Numvars_z.join(deltaDevfHC).apply(stats.zscore)
                                                        
SZandHC_df_zscore = pd.concat([SZandHC_df[["Group","ID","Sex"]], Numvars_z], axis=1)
del Numvars_z


Numvars_z = SZ_only.drop(columns=["Group","ID","Sex"]).apply(stats.zscore)
SZ_only_zscore = pd.concat([SZ_only[["Group","ID","Sex"]], Numvars_z], axis=1)
del Numvars_z


Numvars_z = RepeatedMeasures_df.drop(columns=["Group","ID","Sex","Sess"]).apply(stats.zscore)
RepeatedMeasures_df_zscore = pd.concat([RepeatedMeasures_df[["Group","ID","Sex","Sess"]], Numvars_z], axis=1)
del Numvars_z

Numvars_z = SZ_RepeatedMeasures_df.drop(columns=["Group","ID","Sex","Sess"]).apply(stats.zscore)
SZ_RepeatedMeasures_df_zscore = pd.concat([SZ_RepeatedMeasures_df[["Group","ID","Sex","Sess"]], Numvars_z], axis=1)
del Numvars_z

#%% Results table: mean + std Iselfs & Iothers

mIself_SZ = SZ_df.Iself_WB.mean()
stdIself_SZ = SZ_df.Iself_WB.std()

mIself_HC = HC_df.Iself_WB.mean()
stdIself_HC = HC_df.Iself_WB.std()

mDevfHC_S1_SZ = SZ_df.DevFromHealth_S1.mean()
stdDevfHC_S1_SZ = SZ_df.DevFromHealth_S1.std()

#Similarly for Iothers ....


stats.ttest_ind(SZ_df.Iself_WB,HC_df.Iself_WB)
stats.ttest_ind(SZ_df.DevFromHealth_S2,HC_df.DevFromHealth_S2)

#%% MODEL: Iself 
    
# Make text file with model summaries (including Group as variable)    
uncorrected_p=np.zeros(9)
uncorrected_p_PANSS=np.zeros(9)
uncorrected_p_ATP=np.zeros(9)

f = open("Results/Iselfs_modelresults.txt", "w")

name = 'Iself_WB'
model = smf.ols((name +' ~ Age + Sex + FDmax + DBS + Group'), data=SZandHC_df_zscore).fit()
uncorrected_p[0]=model.pvalues["Group[T.SZ]"]
print(model.summary(), file = f)

model = smf.ols((name +' ~ Age + Sex + FDmax + DBS + deltaATP + deltaPANSS'), data=SZ_only_zscore).fit()  #
uncorrected_p_PANSS[0]=model.pvalues["deltaPANSS"]
uncorrected_p_ATP[0]=model.pvalues["deltaATP"]
print(model.summary(), file = f) 

model = smf.ols((name +' ~ Age + Sex + FDmax + DBS + deltaATP + deltaTP + deltaTN + deltaTG'), data=SZ_only_zscore).fit()  #
print(model.summary(), file = f) 


for idx, network in enumerate(ATLAS.keys()):
    name = 'Iself_' + network
    model = smf.ols( (name +' ~ Age + Sex + FDmax + DBS + Group'), data=SZandHC_df_zscore).fit()
    uncorrected_p[idx+1]=model.pvalues["Group[T.SZ]"]
    print(model.summary(), file = f)
    
    model = smf.ols( (name +' ~ Age + Sex + FDmax + DBS + deltaATP + deltaPANSS'), data=SZ_only_zscore).fit()
    uncorrected_p_PANSS[idx+1]=model.pvalues["deltaPANSS"]
    uncorrected_p_ATP[idx+1]=model.pvalues["deltaATP"]
    print(model.summary(), file = f) 
    
    model = smf.ols( (name +' ~ Age + Sex + FDmax + DBS + deltaATP + deltaTP + deltaTN + deltaTG'), data=SZ_only_zscore).fit()
    print(model.summary(), file = f) 

f.close()

p_corr=multitest.fdrcorrection(uncorrected_p[1:], alpha=0.05, method='indep', is_sorted=False)[1]
p_corrPANSS=multitest.fdrcorrection(uncorrected_p_PANSS[1:], alpha=0.05, method='indep', is_sorted=False)[1]
p_corrATP=multitest.fdrcorrection(uncorrected_p_ATP[1:], alpha=0.05, method='indep', is_sorted=False)[1]

#%%     # Plot residuals (model without Group as variable)

name = 'Iself_WB'
model = smf.ols((name +' ~ Age + Sex + FDmax + DBS '), data=SZandHC_df_zscore).fit()
fig = plt.figure(figsize=(35,40))#,dpi=300)

#plt.title("My title")
plot_opts = {
        #"cutoff_val": 0.3,
        #"cutoff_type": "abs",
        "label_fontsize": "xx-large",
        "label_rotation": 0,
        #"bean_color": "#FF6F00",
        #"bean_mean_color": "#009D91",
        'bean_show_median':False,
        'jitter_marker_size':8, #default 4
        'violin_lw': 3, #default 1
        }
labels = ["HC","FEP"]

Resid = [model.resid[SZandHC_df_zscore.Group == id] for id in SZandHC_df_zscore.Group.unique()]
Text=stats.ttest_ind(Resid[0], Resid[1])
print(name + 'p-val: ' + str(Text[1]))

meanHC = Resid[0].mean()
meanSZ = Resid[1].mean()
errHC = Resid[0].std()/len(Resid[0])
errSZ = Resid[1].std()/len(Resid[1])

ax = fig.add_subplot(3,3,1)
sm.graphics.beanplot(Resid, ax=ax, labels=labels, plot_opts=plot_opts, jitter=True)

#plt.errorbar((1,2), (meanHC,meanSZ), (errHC,errSZ), ecolor='r', linestyle='None', marker='^',mec='r', mfc='r')

ax.set_xlabel("Group", fontsize=30)
ax.set_ylabel("Residuals", fontsize=30)
ax.set_title("Iself_WholeBrain", fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=25)

#ax.text(.5,.5,"p value: {:.3f}".format(Text[1]))  
#SE ME PONE EL TEXTO EN CUALQUIER LADO. NDEAH!

for idx, network in enumerate(ATLAS.keys()):
    name = 'Iself_' + network
    model= smf.ols( (name +' ~ Age + Sex + FDmax + DBS '), data=SZandHC_df_zscore).fit()
       
    Resid = [model.resid[SZandHC_df_zscore.Group == id ] for id in SZandHC_df_zscore.Group.unique()]   
    Text=stats.ttest_ind(Resid[0], Resid[1])
    print(name + ' p-val: ' + str(Text[1]))
    meanHC = Resid[0].mean()
    meanSZ = Resid[1].mean()
    errHC = Resid[0].std()/len(Resid[0])
    errSZ = Resid[1].std()/len(Resid[1])

    
    ax = fig.add_subplot(3,3,idx+2)
    sm.graphics.beanplot(Resid, ax=ax, labels=labels, plot_opts=plot_opts, jitter=True)
    #plt.errorbar((1,2), (meanHC,meanSZ), (errHC,errSZ), ecolor='r', linestyle='None', marker='^',mec='r', mfc='r')

    ax.set_xlabel("Group", fontsize=30)
    ax.set_ylabel("Residuals", fontsize=30)
    ax.set_title(name, fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=25)


filename = '/Users/angeles/Desktop/Iself_plot1000.jpg'
#plt.savefig(filename, dpi=1000)
plt.show()

#%% MODEL: Iothers

uncorrected_p_GROUP=np.zeros(9)
uncorrected_p_PANSS=np.zeros(9)
uncorrected_p_ATP=np.zeros(9)

#model: smf.mixedlm (Yvar ~ Age + ... + Group, data = Iothers_df, groups=Iothers_df["ID"])
    
f = open("Results/Iothers_modelsresults.txt", "w")

print("MODEL: Iothers_WB ~ Age + Sex + FD + Group + (1|subject) \n", file = f)
md = smf.mixedlm("Iothers_WB ~ Age + Sex + FD + Group", \
                 data= RepeatedMeasures_df_zscore, groups=RepeatedMeasures_df_zscore["ID"])
mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
print(mdf.summary(),file=f)
uncorrected_p_GROUP[0]=mdf.pvalues["Group[T.SZ]"]


print("MODEL: Iothers_WB ~ Age + Sex + FD + TPANSS + ATPdose + (1|subject) \n", file = f)
md = smf.mixedlm("Iothers_WB ~ Age + Sex + FD + TPANSS + ATPdose", \
                 data= SZ_RepeatedMeasures_df_zscore, groups=SZ_RepeatedMeasures_df_zscore["ID"])  
mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
print(mdf.summary(), file =f )
uncorrected_p_PANSS[0]=mdf.pvalues["TPANSS"]
uncorrected_p_ATP[0]=mdf.pvalues["ATPdose"]

print("MODEL: Iothers_WB ~ Age + Sex + FD + TP + TN + TG + ATPdose + (1|subject) \n", file = f)
md = smf.mixedlm("Iothers_WB ~ Age + Sex + FD + TP + TN + TG + ATPdose", \
                 data= SZ_RepeatedMeasures_df_zscore, groups=SZ_RepeatedMeasures_df_zscore["ID"])  
mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
print(mdf.summary(), file =f )

for idx, network in enumerate(ATLAS.keys()):
    name = 'Iothers_' + network
    
    print("MODEL: "+name+" ~ Age + Sex + FD + Group + (1|subject) \n", file = f)
    md = smf.mixedlm("Iothers_"+network+" ~ Age + Sex + FD + Group", \
                 data= RepeatedMeasures_df_zscore, groups=RepeatedMeasures_df_zscore["ID"])
    mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
    print(mdf.summary(),file=f)
    uncorrected_p_GROUP[idx+1]=mdf.pvalues["Group[T.SZ]"]

    print("MODEL: "+name+" ~ Age + Sex + FD + TPANSS + ATPdose + (1|subject) \n", file = f)
    md = smf.mixedlm("Iothers_"+network+" ~ Age + Sex + FD + TPANSS + ATPdose", \
                     data= SZ_RepeatedMeasures_df_zscore, groups=SZ_RepeatedMeasures_df_zscore["ID"])  
    mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
    print(mdf.summary(), file =f )
    uncorrected_p_PANSS[idx+1]=mdf.pvalues["TPANSS"]
    uncorrected_p_ATP[idx+1]=mdf.pvalues["ATPdose"]
    
    print("MODEL: "+name+" ~ Age + Sex + FD + TP + TN + TG + ATPdose + (1|subject) \n", file = f)
    md = smf.mixedlm("Iothers_"+network+" ~ Age + Sex + FD + TP + TN + TG + ATPdose", \
                     data= SZ_RepeatedMeasures_df_zscore, groups=SZ_RepeatedMeasures_df_zscore["ID"])  
    mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
    print(mdf.summary(), file =f )

f.close()

p_corrGROUP=multitest.fdrcorrection(uncorrected_p_GROUP[1:], alpha=0.05, method='indep', is_sorted=False)[1]
p_corrPANSS=multitest.fdrcorrection(uncorrected_p_PANSS[1:], alpha=0.05, method='indep', is_sorted=False)[1]
p_corrATP=multitest.fdrcorrection(uncorrected_p_ATP[1:], alpha=0.05, method='indep', is_sorted=False)[1]


#%%     # Plot residuals (model without Group as variable)

name = 'Iothers_WB'
md = smf.mixedlm(name+" ~ Age + Sex + FD ", \
               data= RepeatedMeasures_df_zscore, groups=RepeatedMeasures_df_zscore["ID"])    
mdf = md.fit()

fig = plt.figure(figsize=(35,40))#,dpi=100)
#plt.title("My title")
plot_opts = {
        #"cutoff_val": 0.3,
        #"cutoff_type": "abs",
        "label_fontsize": "xx-large",
        "label_rotation": 0,
        #"bean_color": "#FF6F00",
        #"bean_mean_color": "#009D91",
        'bean_show_median':False,
        'jitter_marker_size':8, #default 4
        'violin_lw': 3, #default 1
        }
labels = ["HC","FEP"]

Resid = [mdf.resid[RepeatedMeasures_df_zscore.Group == id] for id in RepeatedMeasures_df_zscore.Group.unique()]
Text=stats.ttest_ind(Resid[0], Resid[1])
print(name + ' p-val: ' + str(Text[1]))

meanHC = Resid[0].mean()
meanSZ = Resid[1].mean()
errHC = Resid[0].std()/len(Resid[0])
errSZ = Resid[1].std()/len(Resid[1])

ax = fig.add_subplot(3,3,1)
sm.graphics.beanplot(Resid, ax=ax, labels=labels, plot_opts=plot_opts, jitter=True)
#plt.errorbar((1,2), (meanHC,meanSZ), (errHC,errSZ), ecolor='r', linestyle='None', marker='^',mec='r', mfc='r')

ax.set_xlabel("Group", fontsize=30)
ax.set_ylabel("Residuals", fontsize=30)
ax.set_title("Iothers_WholeBrain", fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=25)

for idx, network in enumerate(ATLAS.keys()):
    name = 'Iothers_' + network
    md= smf.mixedlm( (name +' ~ Age + Sex + FD '), \
                   data= RepeatedMeasures_df_zscore, groups=RepeatedMeasures_df_zscore["ID"])    
    mdf = md.fit()
    
    Resid = [mdf.resid[RepeatedMeasures_df_zscore.Group == id ] for id in RepeatedMeasures_df_zscore.Group.unique()]   
    Text=stats.ttest_ind(Resid[0], Resid[1])
    print(name + ' p-val: ' + str(Text[1]))
    
    meanHC = Resid[0].mean()
    meanSZ = Resid[1].mean()
    errHC = Resid[0].std()/len(Resid[0])
    errSZ = Resid[1].std()/len(Resid[1])
    
    ax = fig.add_subplot(3,3,idx+2)
    sm.graphics.beanplot(Resid, ax=ax, labels=labels, plot_opts=plot_opts, jitter=True)
    #plt.errorbar((1,2), (meanHC,meanSZ), (errHC,errSZ), ecolor='r', linestyle='None', marker='^',mec='r', mfc='r')

    ax.set_xlabel("Group", fontsize=30)
    ax.set_ylabel("Residuals", fontsize=30)
    ax.set_title(name, fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=25)


    
filename = '/Users/angeles/Desktop/Iothers_plot1000.jpg'
#plt.savefig(filename, dpi=1000)
plt.show()
#%% MODEL: DevfHC and deltaDevfHC
 # Mixed effects: two measures per subject
    
f = open("Results/DevfHC_modelsresults.txt", "w")

print("MODEL: DevfromHealth_value ~ Age + FD + Sex + Group + (1|subject) \n", file = f)
md = smf.mixedlm("DevfromHealth_value ~ Age + Sex + FD + Group", \
                 data= RepeatedMeasures_df_zscore, groups=RepeatedMeasures_df_zscore["ID"])
mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
print(mdf.summary(),file=f)


print("MODEL: DevfromHealth_value ~ Age + Sex + FD + TPANSS + ATPdose + (1|subject) \n", file = f)
md = smf.mixedlm("DevfromHealth_value ~ Age + Sex + FD + TPANSS + ATPdose ", \
                 data= SZ_RepeatedMeasures_df_zscore, groups=SZ_RepeatedMeasures_df_zscore["ID"])  
mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
print(mdf.summary(), file =f )


print("MODEL: DevfromHealth_value ~ Age + Sex + FD + TP + TN + TG + ATPdose + (1|subject) \n", file = f)
md = smf.mixedlm("DevfromHealth_value ~ Age + Sex + FD + TP + TN + TG + ATPdose ", \
                 data= SZ_RepeatedMeasures_df_zscore, groups=SZ_RepeatedMeasures_df_zscore["ID"])  
mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"
print(mdf.summary(), file =f )

f.close()


f = open("Results/deltaDevfHC_modelsresults.txt", "w")

name = 'deltaDevfHC_signed'
model = smf.ols((name +' ~ Age + Sex + FDmax + Group'), data=SZandHC_df_zscore).fit()
print(model.summary(), file = f)

model = smf.ols((name +' ~ Age + Sex + FDmax + deltaATP_signed + deltaPANSS_signed'), data=SZ_only_zscore).fit()  #
print(model.summary(), file = f) 

model = smf.ols((name +' ~ Age + Sex + FDmax + deltaATP_signed + deltaTP_signed + deltaTN_signed + deltaTG_signed'), data=SZ_only_zscore).fit()  #
print(model.summary(), file = f) 
    
f.close()


#%% PLOT: deltaDevfHC fitted vs actual ---deltaPANSS
name = 'deltaDevfHC_signed'
model = smf.ols((name +' ~ Age + Sex + FDmax + deltaATP_signed + deltaPANSS_signed'), data=SZ_only).fit()  #

df= pd.DataFrame(model.fittedvalues,columns=['deltaDevfHC_fitted'])
df['deltaDevfHC_actual']=SZ_only.deltaDevfHC_signed
df['deltaPANSS_signed']=SZ_only.deltaPANSS_signed


p = ( ggplot (df) +
 aes(x="deltaDevfHC_fitted", y="deltaDevfHC_actual") +
 labs(x='fitted deltaDevfHC', y='actual deltaDevfHC') +
 geom_point(aes(fill = 'deltaPANSS_signed'), colour='black', size = 4) +
 geom_abline(intercept = 0, slope = 1) +
 theme(text = element_text(size = 16)) 
 )
p.draw(show=True)

#%% PLOT: Iothers fitted vs actual ---TotalPANSS and APdose

md = smf.mixedlm("Iothers_WB ~ Age + Sex + FD + TPANSS + ATPdose", \
                 data= SZ_RepeatedMeasures_df, groups=SZ_RepeatedMeasures_df["ID"])  
mdf = md.fit()  #method=["lbfgs"]  method="bfgs" or method="cg"

df= pd.DataFrame(mdf.fittedvalues,columns=['IothersWB_fitted'])
df['IothersWB_actual']=SZ_RepeatedMeasures_df.Iothers_WB
df['TotalPANSS']=SZ_RepeatedMeasures_df.TPANSS
df['APdose']=SZ_RepeatedMeasures_df.ATPdose


p = ( ggplot (df) +
 aes(x="IothersWB_fitted", y="IothersWB_actual") +
 labs(x='fitted Iothers (WB)', y='actual Iothers (WB)') +
 geom_point(aes(fill = 'TotalPANSS'), colour='black', size = 4) +
 geom_abline(intercept = 0, slope = 1) +
 theme(text = element_text(size = 16)) 
 )
p.draw(show=True)


p = ( ggplot (df) +
 aes(x="IothersWB_fitted", y="IothersWB_actual") +
 labs(x='fitted Iothers (WB)', y='actual Iothers (WB)') +
 geom_point(aes(fill = 'APdose'), colour='black', size = 4) +
 geom_abline(intercept = 0, slope = 1) + 
 theme(text = element_text(size = 16))     
 )
p.draw(show=True)

#%% EXTRA PLOTS: Significant correlations

p = ( ggplot (SZ_RepeatedMeasures_df) +
 aes(x="TPANSS", y="Iothers_WB") +
 labs(x="PANSS Total Score") +
 geom_smooth(method= 'lm') +
 geom_point()
 )
p.draw(show=True)


p = ( ggplot (SZ_RepeatedMeasures_df) +
 aes(x="ATPdose", y="Iothers_WB") +
 labs(x="AP dose (CPZ eq)") +
 geom_smooth(method= 'lm') +
 geom_point()
 )
p.draw(show=True)


p = ( ggplot (SZ_RepeatedMeasures_df) +
 aes(x="TN", y="DevfromHealth_value") +
 labs() +
 geom_smooth(method= 'lm') +
 geom_point()
 )
p.draw(show=True)



p = ( ggplot (SZ_only) +
 aes(x="deltaPANSS_signed", y="deltaDevfHC_signed") +
 labs(y="deltaDevfHC") +
 geom_smooth(method= 'lm') +
 geom_point()
 )
p.draw(show=True)


p = ( ggplot (SZ_only) +
 aes(x="deltaTN_signed", y="deltaDevfHC_signed") +
 labs() +
 geom_smooth(method= 'lm') +
 geom_point()
 )
p.draw(show=True)


#%% EXTRA PLOTS: corr Residuals vs clinical variables
labels = ["HC","FEP"]

name = 'Iothers_WB'
md = smf.mixedlm(name+" ~ Age + Sex + FD", \
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



md = smf.mixedlm("Iothers_WB ~ Age + Sex + FD +  ATPdose", \
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


md = smf.mixedlm("Iothers_WB ~ Age + Sex + FD + TPANSS", \
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
model = smf.ols((name +' ~ Age + Sex + FDmax + DBS '), data=SZandHC_df).fit()
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

Iself=[SZandHC_df.Iself_WB[SZandHC_df.Group == id] for id in SZandHC_df.Group.unique()]
Resid = [model.resid[SZandHC_df.Group == id] for id in SZandHC_df.Group.unique()]
Text=stats.ttest_ind(Resid[0], Resid[1])
print(name + 'p-val: ' + str(Text[1]))

ax = fig.add_subplot(3,3,9)
sm.graphics.beanplot(Iself, ax=ax, labels=labels, plot_opts=plot_opts, \
                     jitter=True)
ax.set_xlabel("Group")
ax.set_ylabel("Residuals \n (Iself ~ Age + Sex + FD + DBS)")
ax.set_title("Iself_WholeBrain")


name = 'Iself_WB'
model = smf.ols((name +' ~ Age + Sex + FDmax + DBS + deltaPANSS'), data=SZ_only).fit()  #

fig = plt.figure()
x=SZ_only.deltaATP
y= model.resid
ax=sns.regplot(x, y)
ax.set_xlabel("delta AP Dose")
ax.set_ylabel("Residuals \n (Iself ~ Age + Sex + FD + DBS + deltaPANSS)")
ax.set_title("Iself vs delta AP dose")


name = 'Iself_WB'
model = smf.ols((name +' ~ Age + Sex + FDmax + DBS + deltaATP'), data=SZ_only).fit()  #

fig = plt.figure()
x=SZ_only.deltaPANSS
y= model.resid
ax=sns.regplot(x, y)
ax.set_xlabel("delta PANSS")
ax.set_ylabel("Residuals \n (Iself ~ Age + Sex + FD + DBS + deltaATP)")
ax.set_title("Iself vs delta PANSS")




name='DevfromHealth_value'
md = smf.mixedlm(name+" ~ Age + Sex + FD", \
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
ax.set_ylabel("Residuals \n (DevfromHealthyFC ~ Age + Sex + FD )")
ax.set_title("Dev from healthy FC")



md = smf.mixedlm("DevfromHealth_value ~ Age + Sex + FD + TPANSS ", \
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


md = smf.mixedlm("DevfromHealth_value ~ Age + Sex + FD  + ATPdose ", \
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


#%% EXTRA: deltaPANSS vs DBS

p = ( ggplot (SZ_only) +
 aes(x="DBS", y="deltaPANSS") +
 labs(x="Days Between Sessions", y="deltaPANSS (Total score)") +
 geom_smooth(method= 'lm') +
 geom_point()
 )
p.draw(show=True)

(r, pval)=stats.pearsonr(SZ_only.DBS, SZ_only.deltaPANSS)


p = ( ggplot (SZ_only) +
 aes(x="DBS", y="deltaATP") +
 labs(x="Days Between Sessions", y="deltaAP") +
 geom_smooth(method= 'lm') +
 geom_point()
 )
p.draw(show=True)

(r,pval)=stats.pearsonr(SZ_only.DBS, SZ_only.deltaATP)



def compute_vif(considered_features):
    
    X = SZ_only[considered_features]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1
    
    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='intercept']
    return vif


considered_features = ['deltaPANSS', 'deltaATP', 'DBS','Age','FDmax']
vif = compute_vif(considered_features).sort_values('VIF', ascending=False)





#%% EXTRA: Iothers per subject plots
p = ( ggplot (SZ_RepeatedMeasures_df) +
 aes(x="ID", y="Iothers_WB", color="Sex") +
 geom_point() +
 geom_boxplot()
 )
p.draw(show=True)



#%% EXTRA: Clinical heterogeneity?

df_sess1=SZ_only[["TP","TN","TG","TPANSS"]]
df_sess2=SZ_only[["TP2","TN2","TG2","TPANSS2"]]
df_repmes=SZ_RepeatedMeasures_df[["TP","TN","TG","Sess"]]


# set figure size

plt.figure(figsize=(10,7))


# Generate a mask to onlyshow the upper triangle
mask = np.tril(np.ones_like(df_sess2.corr(), dtype=bool))
# generate heatmap
sns.heatmap(df_sess2.corr(), annot=True, mask=mask, vmin=-1, vmax=1)

# Generate a mask to onlyshow the bottom triangle
mask = np.triu(np.ones_like(df_sess1.corr(), dtype=bool))
# generate heatmap
sns.heatmap(df_sess1.corr(), annot=True, mask=mask, vmin=-1, vmax=1, cbar= False)


plt.title('Pearson\'s correlations of PANSS subscales \n Session 1 bottom / Session 2 top')
plt.show()


# compute the vif for all given features
def compute_vif(considered_features):
    
    X = df_sess2[considered_features]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1
    
    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='intercept']
    return vif

# features to consider removing
considered_features = ['TP2', 'TN2', 'TG2']


# compute vif 
vif = compute_vif(considered_features).sort_values('VIF', ascending=False)



#%%         Pairplots 
def corrfunc(x, y, **kws):
    (r, p) = stats.pearsonr(x, y)  
    ax = plt.gca()
    print(ax.get_position().ymax)
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.05, .8), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.05, .6), xycoords=ax.transAxes)
  
sns.set(font_scale=2)
g = sns.pairplot(df_repmes, kind="reg", diag_kind="hist", hue = "Sess", markers=["o", "D"])
#g.map(corrfunc)

ax=plt.gca()

new_title = ''
g._legend.set_title(new_title)
# replace labels
new_labels = ['Session 1', 'Session 2']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)




#plt.show()    
    

p = stats.pearsonr(SZ_only.TP, SZ_only.TN)
print (f"TP-TN: {p}")
p = stats.pearsonr(SZ_only.TP, SZ_only.TG)
print (f"TP-TG: {p}")
p = stats.pearsonr(SZ_only.TP, SZ_only.TPANSS)
print (f"TP-Total: {p}")
p = stats.pearsonr(SZ_only.TN, SZ_only.TG)
print (f"TN-TG: {p}")
p = stats.pearsonr(SZ_only.TN, SZ_only.TPANSS)
print (f"TN-Total: {p}")
p = stats.pearsonr(SZ_only.TG, SZ_only.TPANSS)
print (f"TG-Total: {p}")

p = stats.pearsonr(SZ_only.TP2, SZ_only.TN2)
print (f"TP2-TN2: {p}")
p = stats.pearsonr(SZ_only.TP2, SZ_only.TG2)
print (f"TP2-TG2: {p}")
p = stats.pearsonr(SZ_only.TP2, SZ_only.TPANSS2)
print (f"TP2-Total2: {p}")
p = stats.pearsonr(SZ_only.TN2, SZ_only.TG2)
print (f"TN2-TG2: {p}")
p = stats.pearsonr(SZ_only.TN2, SZ_only.TPANSS2)
print (f"TN2-Total2: {p}")
p = stats.pearsonr(SZ_only.TG2, SZ_only.TPANSS2)
print (f"TG2-Total2: {p}")




