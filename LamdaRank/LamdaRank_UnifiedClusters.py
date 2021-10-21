#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shahab Sotudian
"""



import numpy as np
import pandas as pd
import math
import six
from six.moves import range
from scipy.stats import rankdata
import sys
import time
from sklearn import preprocessing
from scipy.stats import spearmanr
import random
import copy
import xgboost as xgb
from patsy import dmatrices
from numpy import asarray
from numpy import savetxt
from sklearn.model_selection import train_test_split
import time
from scipy.stats import spearmanr

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@                         Functions
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def dcg_at_k(r, k):
    """
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider

    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):

    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def Performance_Metrics(S_in):
    RANKS=S_in[S_in['RANK_True']==1]
    ACC_All=S_in[S_in['dockq']>0.23]
    Med_All=S_in[S_in['dockq']>0.49]
    # Spearman
    Spearman=spearmanr(S_in['RANK_Pred'], S_in['RANK_True'])[0]
    Spearman_C=spearmanr(S_in['RANK_ClusPro'], S_in['RANK_True'])[0]

    # Highest Quality
    T_1_star= int(RANKS['RANK_Pred']<=1)
    T_5_star= int(RANKS['RANK_Pred']<=5)
    T_10_star= int(RANKS['RANK_Pred']<=10)
    
    CT_1_star= int(RANKS['RANK_ClusPro']<=1)
    CT_5_star= int(RANKS['RANK_ClusPro']<=5)
    CT_10_star= int(RANKS['RANK_ClusPro']<=10)

    # ACC
    if ACC_All.empty:
        ACCT_1=ACCT_5=ACCT_10=ACCCT_1=ACCCT_5=ACCCT_10=0
        Have_Acc=0
    else:
        Have_Acc=1
        ACCT_1=int(np.where( sum(x<=1 for x in ACC_All['RANK_Pred'])  > 0.5 , 1, 0))
        ACCT_5=int(np.where( sum(x<=5 for x in ACC_All['RANK_Pred'])  > 0.5 , 1, 0))
        ACCT_10= int(np.where( sum(x<=10 for x in ACC_All['RANK_Pred'])  > 0.5 , 1, 0))
        
        ACCCT_1=int(np.where( sum(x<=1 for x in ACC_All['RANK_ClusPro'])  > 0.5 , 1, 0))
        ACCCT_5=int(np.where( sum(x<=5 for x in ACC_All['RANK_ClusPro'])  > 0.5 , 1, 0))
        ACCCT_10=int(np.where( sum(x<=10 for x in ACC_All['RANK_ClusPro'])  > 0.5 , 1, 0))
    
    
    # Med
    if Med_All.empty:
        MedT_1=MedT_5=MedT_10=MedCT_1=MedCT_5=MedCT_10=0
        Have_Med=0
    else:
        Have_Med=1
        MedT_1=int(np.where( sum(x<=1 for x in Med_All['RANK_Pred'])  > 0.5 , 1, 0))
        MedT_5=int(np.where( sum(x<=5 for x in Med_All['RANK_Pred'])  > 0.5 , 1, 0))
        MedT_10=int(np.where( sum(x<=10 for x in Med_All['RANK_Pred'])  > 0.5 , 1, 0))
        
        MedCT_1=int(np.where( sum(x<=1 for x in Med_All['RANK_ClusPro'])  > 0.5 , 1, 0))
        MedCT_5=int(np.where( sum(x<=5 for x in Med_All['RANK_ClusPro'])  > 0.5 , 1, 0))
        MedCT_10=int(np.where( sum(x<=10 for x in Med_All['RANK_ClusPro'])  > 0.5 , 1, 0))

    return [Spearman,Spearman_C,T_1_star,CT_1_star,T_5_star,CT_5_star,T_10_star,CT_10_star,ACCT_1,ACCCT_1,ACCT_5,ACCCT_5,ACCT_10,ACCCT_10,Have_Acc,MedT_1,MedCT_1,MedT_5,MedCT_5,MedT_10,MedCT_10,Have_Med]


# =========-------------------------------------------------------------------
#  Data Preprocessing  -   
# =========-------------------------------------------------------------------    



Which_Type='ANTIBODY' # 'ANTIBODY'  'ENZYME'   'OTHERS'
Output_Type ='dockq' # ['dockq','fnat','lrms','irms']
Num_Features= 30   # Total number of features

if Which_Type == 'ANTIBODY':
    Test_Proteins= ['3rvw','4dn4','4g6j','3hi6','3l5w','2vxt','2w9e','4g6m','3g6d','3eo1','3v6z','3hmx','3eoa','3mxw']
    
    Train_Proteins=['1mlc','1iqd','2jel','1nca','1ahw','1e6j','1kxq','1wej','1dqj','2fd6','2i25','1jps','2hmi','1k4c',
                    '1i9r','1bj1','1bgx','1qfw','2vis','1nsn','1bvk','1fsk','1vfb']
    
    
    Features_Ordered= ['mincomp_fa_rep','mincomp_score','mincomp_total_score','piper_fa_rep','piper_score','piper_total_score',
                       'movedcomp_total_score','movedcomp_score','var_mem_Elec','var_mem_Born','kurt_mem_Avdw','SPPS','skew_mem_Rvdw',
                       'kurt_mem_Born','avg_mem_Elec','mincomp_fa_dun','PREMIN_COMP_Torsions','movedcomp_fa_dun','POSTMIN_COMP_Torsions',
                       'PREMIN_SING_Torsions','POSTMIN_SING_Torsions','mincomp_rama_prepro','cen_Elec','movedcomp_rama_prepro','size',
                       'piper_fa_dun','mincomp_fa_atr','cen_Born','avg_mem_Born','piper_rama_prepro','mincomp_fa_sol',
                       'piper_fa_intra_sol_xover4','mincomp_omega','POSTMIN_SING_Impropers','piper_yhh_planarity','movedcomp_omega',
                       'movedcomp_fa_intra_rep','mincomp_pro_close','var_mem_Avdw','POSTMIN_SING_Bonded','piper_omega',
                       'movedcomp_pro_close','PREMIN_COMP_Impropers','mincomp_fa_intra_rep','PREMIN_SING_Impropers','POSTMIN_COMP_Impropers',
                       'PREMIN_COMP_Bonded','PREMIN_SING_Bonded','POSTMIN_COMP_Bonded','PREMIN_COMP_Angles','piper_pro_close',
                       'POSTMIN_SING_Angles','movedcomp_fa_intra_sol_xover4','PREMIN_SING_Angles','POSTMIN_COMP_Angles',
                       'mincomp_fa_intra_sol_xover4','skew_mem_Born','avg_mem_dist','movedcomp_yhh_planarity','cen_Avdw',
                       'mincomp_yhh_planarity','avg_mem_Avdw','var_mem_dist','movedcomp_ref','mincomp_ref','piper_fa_atr',
                       'movedcomp_fa_atr','var_mem_DARS','movedcomp_fa_sol','avg_mem_DARS','piper_fa_sol','var_mem_Rvdw','cen_DARS',
                       'piper_dslf_fa13','piper_ref','SPAR','var_mem_Teng','movedcomp_dslf_fa13','cen_Rvdw','movedcomp_hbond_lr_bb',
                       'mincomp_hbond_lr_bb','movedcomp_hbond_bb_sc','piper_hbond_bb_sc','movedcomp_hbond_sr_bb','mincomp_hbond_sr_bb',
                       'avg_mem_Rvdw','mincomp_hbond_bb_sc','piper_hbond_sr_bb','movedcomp_hbond_sc','mincomp_fa_elec','piper_hbond_sc',
                       'movedcomp_fa_elec','SPAS','mincomp_hbond_sc','cen_Teng','piper_fa_elec','avg_mem_Teng','piper_hbond_lr_bb','piper_time']
    
    DATA_ALL = pd.read_csv('/home/Final_DATA_Antibody.csv')

elif Which_Type == 'ENZYME':
    Test_Proteins= ['2gaf','4hx3','3a4s','2yvj','4fza','3fn1','3pc8','3lvk','3k75','4iz7','3vlb','4h03','3h11','1jtd','4lw4','2a1a']
    
    Train_Proteins=['3sgq','1fq1','2o3b','1d6r','2pcc','2mta','1kkl','1oc0','1mah','1m10','1jzd','1cgi','1ijk','1f51','2ayo','7cei',
                    '1jiw','2uuy','2oul','1us7','1jk9','1tmq','2sni','1avx','1gxd','2oob','1hia','1dfj','2nz8','2j0t','2z0e','1nw9',
                    '1ay7','1jtg','2oor','2ot3','1jwh','1bvn','1oph','1ezu','1e6e','1acb','1f6m','2abz','1f34','1jmo','1gl1','1udi',
                    '1r6q','1zm4','1eaw','1ewy','2b42','1buh','1pxv','2sic','2a9k','4cpa','1z5y','1gla','2o8v','1zli','1r0r','1fle',
                    '1oyv','1ppe','1wdw','1yvb','1clv','2ido']
    
    Features_Ordered= ['movedcomp_fa_rep','movedcomp_score','movedcomp_total_score','piper_lk_ball_wtd','POSTMIN_COMP_VWD03',
                       'POSTMIN_SING_VWD03','mincomp_lk_ball_wtd','var_mem_Born','mincomp_hbond_lr_bb','piper_pro_close',
                       'movedcomp_hbond_lr_bb','movedcomp_lk_ball_wtd','movedcomp_pro_close','piper_hbond_lr_bb','var_mem_Elec',
                       'POSTMIN_COMP_Angles','POSTMIN_COMP_Impropers','piper_dslf_fa13','movedcomp_dslf_fa13','mincomp_pro_close',
                       'POSTMIN_SING_Bonded','movedcomp_omega','piper_fa_rep','size','PREMIN_SING_Bonded','PREMIN_COMP_Bonded',
                       'POSTMIN_SING_Impropers','POSTMIN_COMP_Bonded','mincomp_fa_rep','mincomp_hbond_sr_bb','piper_hbond_sr_bb',
                       'piper_total_score','piper_score','piper_omega','POSTMIN_SING_Angles','avg_mem_DARS','cen_DARS',
                       'movedcomp_hbond_sr_bb','PREMIN_SING_Impropers','PREMIN_COMP_Impropers','PREMIN_SING_Angles','mincomp_omega',
                       'PREMIN_COMP_Angles','movedcomp_fa_dun','SPPS','mincomp_fa_dun','mincomp_total_score','mincomp_score',
                       'mincomp_fa_intra_rep','movedcomp_yhh_planarity','piper_rama_prepro','mincomp_rama_prepro',
                       'movedcomp_rama_prepro','piper_yhh_planarity','kurt_mem_dist','movedcomp_hbond_sc','cen_Elec',
                       'mincomp_yhh_planarity','mincomp_hbond_sc','mincomp_dslf_fa13','var_mem_dist','movedcomp_fa_intra_rep',
                       'movedcomp_hbond_bb_sc','mincomp_hbond_bb_sc','kurt_mem_Born','avg_mem_dist','avg_mem_Elec','PREMIN_SING_VWD03',
                       'PREMIN_COMP_VWD03','SPAR','PREMIN_COMP_Torsions','PREMIN_SING_Torsions','piper_fa_dun','mincomp_p_aa_pp',
                       'movedcomp_p_aa_pp','POSTMIN_COMP_Torsions','mincomp_fa_elec','piper_hbond_bb_sc','POSTMIN_SING_Torsions',
                       'var_mem_DARS','movedcomp_fa_sol','movedcomp_fa_elec','piper_fa_atr','piper_fa_elec','mincomp_fa_atr',
                       'piper_fa_sol','mincomp_fa_sol','movedcomp_fa_atr','SPAS','piper_hbond_sc','piper_fa_intra_sol_xover4',
                       'movedcomp_time','movedcomp_fa_intra_sol_xover4','mincomp_fa_intra_sol_xover4','piper_fa_intra_rep',
                       'piper_ref','movedcomp_ref','mincomp_ref','skew_mem_Born','kurt_mem_Teng','skew_mem_Teng','skew_mem_DARS',
                       'skew_mem_dist','cen_Teng','mincomp_time','cen_Rvdw','var_mem_Teng','var_mem_Rvdw','avg_mem_Avdw','avg_mem_Rvdw']
    
    DATA_ALL = pd.read_csv('/home/Final_DATA_Enzyme.csv')    

elif Which_Type == 'OTHERS':
    Test_Proteins=['3szk','1m27','2x9a','baad','3aad','3s9d','3daw','cp57','3h2v','3bx7',
                   '3biw','3p57','4m76','3aaa','3f1p','bp57','2gtp','1rke','3r9a','3l89',
                   '1exb','4jcv']
    
    Train_Proteins=['1rv6','1eer','1e4k','2fju','1z0k','1ofu','1a2k','3cph','1ibr','1ib1',
                    '1k5d','1k74','1syx','3d5s','1i4d','1rlb','1he8','1r8s','1pvh','1fcc',
                    '3bp8','1azs','2hqs','1gpw','2c0l','2i9b','2hle','1de4','1qa9','1fqj',
                    '1h1v','1e96','1j2j','1n2c','1t6b','2hrk','1kac','1y64','2b4j','1ghq',
                    '2g77','1ml0','1gcq','1efn','1ffw','1bkd','1zhi','1klu','1xd3','2j7p',
                    '2cfh','1hcf','2vdb','1atn','1grn','1lfd','1b6c','1ak4','1wq1','1ira',
                    '2btf','1sbb','1xqs','1mq8','1xu1','1i2m','1he1','1s1q','2oza','1zhh',
                    '1ktz','1fak','1fc2','1kxp','1gp2','2ajf','1akj','1h9d','2a5t']
    
    Features_Ordered=['mincomp_score','mincomp_total_score','mincomp_fa_rep','piper_total_score',
                      'piper_score','piper_fa_rep','cen_Avdw','avg_mem_Avdw','var_mem_Teng',
                      'var_mem_Avdw','avg_mem_Teng','mincomp_lk_ball_wtd','POSTMIN_COMP_Impropers',
                      'size','POSTMIN_SING_Impropers','piper_p_aa_pp','mincomp_p_aa_pp',
                      'movedcomp_lk_ball_wtd','mincomp_pro_close','movedcomp_p_aa_pp','piper_rama_prepro',
                      'mincomp_rama_prepro','movedcomp_rama_prepro','PREMIN_COMP_Angles','PREMIN_SING_Angles',
                      'piper_yhh_planarity','cen_DARS','movedcomp_hbond_bb_sc','avg_mem_DARS','POSTMIN_COMP_VWD03',
                      'POSTMIN_COMP_Angles','POSTMIN_SING_Angles','SPPS','mincomp_hbond_bb_sc','piper_hbond_bb_sc',
                      'movedcomp_hbond_sc','mincomp_yhh_planarity','mincomp_fa_dun','mincomp_hbond_sc','movedcomp_pro_close',
                      'piper_pro_close','movedcomp_ref','mincomp_ref','piper_ref','movedcomp_yhh_planarity','SPAR',
                      'movedcomp_fa_dun','piper_fa_elec','movedcomp_fa_elec','piper_hbond_sc','movedcomp_fa_atr',
                      'piper_fa_intra_sol_xover4','movedcomp_fa_sol','mincomp_fa_elec','movedcomp_omega','piper_fa_atr',
                      'piper_fa_sol','mincomp_fa_intra_sol_xover4','SPAS','movedcomp_fa_intra_sol_xover4','mincomp_fa_atr',
                      'mincomp_fa_intra_rep','movedcomp_dslf_fa13','piper_fa_intra_rep','POSTMIN_SING_VWD03','PREMIN_COMP_VWD03',
                      'mincomp_fa_sol','PREMIN_SING_VWD03','piper_dslf_fa13','POSTMIN_SING_Torsions','PREMIN_SING_Torsions',
                      'PREMIN_COMP_Torsions','POSTMIN_COMP_Torsions','piper_fa_dun','movedcomp_fa_intra_rep','piper_hbond_sr_bb',
                      'movedcomp_hbond_sr_bb','mincomp_hbond_sr_bb','mincomp_dslf_fa13','mincomp_omega','PREMIN_COMP_Impropers',
                      'piper_omega','movedcomp_hbond_lr_bb','PREMIN_SING_Impropers','mincomp_hbond_lr_bb','piper_hbond_lr_bb',
                      'skew_mem_DARS','cen_Elec','avg_mem_Elec','avg_mem_Born','cen_Born','piper_time','kurt_mem_Teng','skew_mem_Teng',
                      'mincomp_time','movedcomp_time']
    
    DATA_ALL = pd.read_csv('/home/Others_Protein_DATA.csv')













# =========-------------------------------------------------------------------
#  Unify Number of clusters

All_Proteins = Test_Proteins+Train_Proteins
Num_Clus=[]
for i in range(len(All_Proteins)):
    X_features= DATA_ALL[DATA_ALL['pdbid'].isin([All_Proteins[i]])]
    X_Relevance= DATA_ALL[DATA_ALL['pdbid'].isin([All_Proteins[i]])]
    print('Query: ', i , 'Num clusters: ', len(X_Relevance))
    Num_Clus.append(len(X_Relevance))
print( " # =========------------------------------------------------------------------- ")
print( "Min Number of clusters: ",min(Num_Clus))
print( "Average Number of clusters: ",np.mean(Num_Clus))
print( " # =========------------------------------------------------------------------- ")
del X_features, X_Relevance, i
# Remove extra clusters
for i in range(len(All_Proteins)):
    X_features = DATA_ALL[DATA_ALL['pdbid'].isin([All_Proteins[i]])]
    X_features = X_features.nlargest( min(Num_Clus) , Output_Type)   # We used min(Num_Clus)    Return ordered lis + may need to change **********
    if i == 0:
        Unified_DATA_ALL = X_features
    else:
        Unified_DATA_ALL = pd.concat([Unified_DATA_ALL, X_features], ignore_index=True)
            
del i, X_features,        
    

Unified_DATA_ALL = Unified_DATA_ALL.reset_index(drop=True)







X=Unified_DATA_ALL.iloc[:][Features_Ordered[0:Num_Features]]
Y=pd.DataFrame(Unified_DATA_ALL.iloc[:][Output_Type])   # ['dockq','fnat','lrms','irms']

DATA_ALL = copy.deepcopy(Unified_DATA_ALL)
del Unified_DATA_ALL

# Normalazation
Normalization_scaler = preprocessing.MinMaxScaler()#MinMaxScaler  StandardScaler
Normalized_X=pd.DataFrame(   Normalization_scaler.fit_transform(X.values)    )
Normalized_X.columns=X.columns
del Normalization_scaler





# Final data 

X_Train=Normalized_X[DATA_ALL['pdbid'].isin(Train_Proteins)]
Y_Train=Y[DATA_ALL['pdbid'].isin(Train_Proteins)]

X_Test=Normalized_X[DATA_ALL['pdbid'].isin(Test_Proteins)]
Y_Test=Y[DATA_ALL['pdbid'].isin(Test_Proteins)]


dtrain = xgb.DMatrix(X_Train, label=Y_Train)
dtest = xgb.DMatrix(X_Test, label=Y_Test)



# # Grouping
# X_groups=np.array([Num_Drugs for i in range(int(len(TrainX)/Num_Drugs))]).flatten()

# dtrain.set_group(group=X_groups) # Set the query_id values to DMatrix data structure




# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@                         Main Algorithm ApproxNDCG
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

"""
rank:pairwise: Use LambdaMART to perform pairwise ranking where the pairwise loss is minimized

rank:ndcg: Use LambdaMART to perform list-wise ranking where Normalized Discounted Cumulative Gain (NDCG) is maximized

rank:map: Use LambdaMART to perform list-wise ranking where Mean Average Precision (MAP) is maximized

rank:map ranknet
rank:pairwise LamdaRank
rank:ndcg LamdaMart 
"""


# Grid SEarch CV

iter_ = 0 
best_error = 0
best_iter = 0
best_model = None

col_sample_rates = [ 0.7, 1]
subsamples = [1]
etas = [0.01, 0.1 , 0.3]
max_depths = [3, 6, 9]
reg_alphas = [0] # [0, 0.1 , 10] Inactive
reg_lambdas = [1] #[1, 0.1 , 10] Inactive
ntrees = [100, 1000]
CV_Results=[]
total_models = len(col_sample_rates)*len(subsamples)*len(etas)*len(max_depths)*len(reg_alphas)*len(reg_lambdas)*len(ntrees)

# CV_X_train, CV_X_test, CV_y_train, CV_y_test = train_test_split(X_Train, Y_Train, test_size=0.30, random_state=42)
Predicted_NDCGatk = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

for col_sample_rate in col_sample_rates:
    for subsample in subsamples:
        for eta in etas:
            for max_depth in max_depths:
                for reg_alpha in reg_alphas:
                    for reg_lambda in reg_lambdas:
                        for ntree in ntrees:

                            params = {
                                 'booster': 'gbtree',
                                 'colsample_bytree': col_sample_rate,
                                 'eta': eta,
                                 'eval_metric': 'ndcg',
                                 'max_depth': max_depth,
                                 'nthread': 4,
                                 'objective': 'rank:pairwise',
                                 'reg_alpha': reg_alpha,
                                 'reg_lambda': reg_lambda,
                                 'seed': 12345,
                                 #'silent': 0,
                                 'subsample': subsample,
                                 'eval_metric':'ndcg'}
                            Learned_Model = xgb.train(params, dtrain, ntree)

                            NDCG_all=[0,0,0,0,0,0,0,0,0,0] 
                            for i in Test_Proteins:
                                dtest_CoMPlex = xgb.DMatrix( Normalized_X[DATA_ALL['pdbid'].isin([i])] , label=  Y[DATA_ALL['pdbid'].isin([i])]  )
                                ypred = pd.DataFrame(Learned_Model.predict(dtest_CoMPlex))
                                 
                                ypred.columns=['Score_Prediction']
                                ypred=ypred.reset_index(drop=True) 
                                Y_True=Y[DATA_ALL['pdbid'].isin([i])]
                                Y_True=Y_True.reset_index(drop=True) 
                                S_in = pd.concat([ypred,Y_True], axis=1)
                                S_in['RANK_True']= S_in['dockq'].rank(method = 'first',ascending=False)
                                S_in['RANK_Pred']= S_in['Score_Prediction'].rank(method = 'first',ascending=False)
                                S_in['RANK_ClusPro']= range(1,S_in.shape[0]+1,1)
                                
                                Concat = np.array(S_in.values)
                                sorted_array = Concat[np.argsort(Concat[:, 3])]
                                RGT= sorted_array[:,1]
                                # NDCG @ k
                                NDCG_all = np.add(NDCG_all, [ndcg_at_k(RGT, 1),ndcg_at_k(RGT, 2),ndcg_at_k(RGT, 3),ndcg_at_k(RGT, 4),ndcg_at_k(RGT, 5),ndcg_at_k(RGT, 6),ndcg_at_k(RGT, 7),ndcg_at_k(RGT, 8),ndcg_at_k(RGT, 9),ndcg_at_k(RGT, 10)])  
                            
                            Predicted_NDCG=NDCG_all/len(Test_Proteins)           
                            print(Predicted_NDCG)
                            Predicted_NDCGatk=np.vstack([Predicted_NDCGatk,Predicted_NDCG])    
                             
            
                            CV_Results.append([iter_, col_sample_rate, subsample,eta,max_depth,reg_alpha,reg_lambda,ntree])
                            iter_ += 1





# # save to csv file



SAVE_Name_NDCG_RESULTS="NDCG_RESULTS_" + Which_Type + '_NFeature_' + str(Num_Features) +'.csv'

SAVE_Name_Parameters_Table_= "Parameters_Table_" + Which_Type + '_NFeature_' + str(Num_Features) + '.csv'

# save to csv file
savetxt(SAVE_Name_NDCG_RESULTS, Predicted_NDCGatk, delimiter=',')

savetxt(SAVE_Name_Parameters_Table_, CV_Results, delimiter=',')


    
