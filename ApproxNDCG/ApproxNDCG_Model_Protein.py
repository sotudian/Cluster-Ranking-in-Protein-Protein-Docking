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
start_time = time.time()


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@                         Functions
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

Which_Type='ANTIBODY' # 'ANTIBODY'  'ENZYME'   'OTHERS'
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
    
    DATA_ALL = pd.read_csv('/home//Final_DATA_Enzyme.csv')    

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




X=DATA_ALL.iloc[:][Features_Ordered[0:Num_Features]]
Y=pd.DataFrame(DATA_ALL.iloc[:]['dockq'])   # ['dockq','fnat','lrms','irms']



# Normalazation
Normalization_scaler = preprocessing.MinMaxScaler()#MinMaxScaler  StandardScaler
Normalized_X=pd.DataFrame(   Normalization_scaler.fit_transform(X.values)    )
Normalized_X.columns=X.columns




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


def Pi_Hat(Big_X_features,x,Theta, Alpha):
    PI=1
    PI_Hat=1
    for i in range(len(Big_X_features)):
        S_x_y= np.inner( Theta, x ) - np.inner( Theta, Big_X_features[i,:] )
        # print(S_x_y)
        if S_x_y != 0:
            PI_Hat= PI_Hat + np.exp(-Alpha*S_x_y)/(1+np.exp(-Alpha*S_x_y))
        # if S_x_y<0:
        #     PI=PI+1
    return PI_Hat
       

def Gradient_Update(Big_X_Relevance,Big_X_features,Theta, Alpha):
        # Part1  
    N_n= 1/dcg_at_k(sorted(Big_X_Relevance, reverse=True), len(Big_X_Relevance))
    if N_n == float('inf'):          # Revise N_n=0
        N_n=1
    Eq_41_Sumation= np.zeros(len(Big_X_features[1,:]))



    for Ind_x in range(len(Big_X_features)):
        global S_x_y
        x= Big_X_features[Ind_x,:]
        rx=Big_X_Relevance[Ind_x]

        # Part2 Eq_42
        Eq_42= np.zeros(len(Big_X_features[1,:]))
        for i in range(len(Big_X_features)):
            S_x_y= np.inner( Theta, x ) - np.inner( Theta, Big_X_features[i,:] )
            if S_x_y != np.array([0.]):
               
                Grad_Diff= x - Big_X_features[i,:]  # This is for linear score functions
                P1=np.exp(Alpha*S_x_y)/((1+np.exp(Alpha*S_x_y))**2)
                Eq_42=Eq_42+ (P1*Grad_Diff)

        Eq_42=-Alpha*Eq_42  
           
        # Part3 Eq_43        
        PP1= -(((2**rx)-1)/(math.log2(1+Pi_Hat(Big_X_features,x,Theta, Alpha)))**2)
        PP2= 1/((1+Pi_Hat(Big_X_features,x,Theta, Alpha))*np.log(2))
        Eq_43=PP1*PP2
   
   
        # Part 4 Eq_42_43

        Eq_41_Sumation= Eq_41_Sumation + (Eq_43 * Eq_42)
     
    # Final gradient
    return  N_n*Eq_41_Sumation  
       

def iter_lines(lines, has_targets=True, one_indexed=True, missing=0.0):
    """Transforms an iterator of lines to an iterator of LETOR rows.
    Each row is represented by a (x, y, qid, comment) tuple.
    Parameters
    ----------
    lines : iterable of lines
        Lines to parse.
    has_targets : bool, optional
        Whether the file contains targets. If True, will expect the first token
        of every line to be a real representing the sample's target (i.e.
        score). If False, will use -1 as a placeholder for all targets.
    one_indexed : bool, optional
        Whether feature ids are one-indexed. If True, will subtract 1 from each
        feature id.
    missing : float, optional
        Placeholder to use if a feature value is not provided for a sample.
    Yields
    ------
    x : array of floats
        Feature vector of the sample.
    y : float
        Target value (score) of the sample, or -1 if no target was parsed.
    qid : object
        Query id of the sample. This is currently guaranteed to be a string.
    comment : str
        Comment accompanying the sample.
    """
    for line in lines:
                   
        data, _, comment = line.rstrip().partition('#')
        toks = data.split()

        num_features = 0
        x = np.repeat(missing, 8)
        y = -1.0
        if has_targets:
            # print(toks)
            # print("#########################")
            y = float(toks[0])
            toks = toks[1:]

        qid = _parse_qid_tok(toks[0])

        for tok in toks[1:]:
            fid, _, val = tok.partition(':')
            fid = int(fid)
            val = float(val)
            if one_indexed:
                fid -= 1
            assert fid >= 0
            while len(x) <= fid:
                orig = len(x)
                x.resize(len(x) * 2)
                x[orig:orig * 2] = missing

            x[fid] = val
            num_features = max(fid + 1, num_features)

        assert num_features > 0
        x.resize(num_features)

        yield (x, y, qid, comment)


   
def read_dataset(source, has_targets=True, one_indexed=True, missing=0.0):
    """Parses a LETOR dataset from `source`.
    Parameters
    ----------
    source : string or iterable of lines
        String, file, or other file-like object to parse.
    has_targets : bool, optional
        See `iter_lines`.
    one_indexed : bool, optional
        See `iter_lines`.
    missing : float, optional
        See `iter_lines`.
    Returns
    -------
    X : array of arrays of floats
        Feature matrix (see `iter_lines`).
    y : array of floats
        Target vector (see `iter_lines`).
    qids : array of objects
        Query id vector (see `iter_lines`).
    comments : array of strs
        Comment vector (see `iter_lines`).
    """
    if isinstance(source, six.string_types):
        source = source.splitlines()

    max_width = 0
    xs, ys, qids, comments = [], [], [], []
    it = iter_lines(source, has_targets=has_targets,
                    one_indexed=one_indexed, missing=missing)
    for x, y, qid, comment in it:
        xs.append(x)
        ys.append(y)
        qids.append(qid)
        comments.append(comment)
        max_width = max(max_width, len(x))

    assert max_width > 0
    X = np.ndarray((len(xs), max_width), dtype=np.float64)
    X.fill(missing)
    for i, x in enumerate(xs):
        X[i, :len(x)] = x
    ys = np.array(ys) if has_targets else None
    qids = np.array(qids)
    comments = np.array(comments)

    return (X, ys, qids, comments)
   

def _parse_qid_tok(tok):
    assert tok.startswith('qid:')
    return tok[4:]

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


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@                         Main Algorithm ApproxNDCG
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# =========-------------------------------------------------------------------
#  Data Preprocessing  -   Train
# =========-------------------------------------------------------------------    
   
# Load data 

# X_Train=Normalized_X[DATA_ALL['pdbid'].isin(Train_Proteins)]
# Y_Train=Y[DATA_ALL['pdbid'].isin(Train_Proteins)]

# X_Train=X_Train.reset_index(drop=True)
# Y_Train=Y_Train.reset_index(drop=True)




Num_Queries=len(Train_Proteins)
Num_Features= Normalized_X.shape[1]



Weight_Y=1
Y=Y*Weight_Y  
Num_itr=20
Learning_Rate = 0.1
Alpha=10

# Initialize Theta  
Theta_t_1 = np.zeros([(Num_itr+1),Num_Features])
#Theta_t_1[0,:]=np.random.random([1,Num_Features])
Theta_t_1[0,:]=np.ones([1,Num_Features])


for t in range(Num_itr):    
    print('**** Iteration ', t+1, '  *******************************')
    THETA= Theta_t_1[t,:]
    for i in range(Num_Queries):
        Big_X_features= Normalized_X[DATA_ALL['pdbid'].isin([Train_Proteins[i]])]
        Big_X_Relevance= Y[DATA_ALL['pdbid'].isin([Train_Proteins[i]])]

        # Compute gradient for i-th query
        Delta_Theta=  Gradient_Update(np.array(Big_X_Relevance),np.array(Big_X_features),THETA, Alpha)
      
        THETA = THETA + (Learning_Rate * Delta_Theta)

    Theta_t_1[(t+1),:]= THETA
    # Shuffle queries
    random.seed(t)
    random.shuffle(Train_Proteins)  
   



# =========-------------------------------------------------------------------
#  Data Preprocessing  -   Validation
# =========-------------------------------------------------------------------    

# with open("/home/vali.txt") as validationfile:
#     VX, Vy, Vqids, Vc = read_dataset(validationfile, has_targets=True , one_indexed=True)
   


# =========-------------------------------------------------------------------
#  Data Preprocessing  -   Test
# =========-------------------------------------------------------------------  
   
# X_Test=Normalized_X[DATA_ALL['pdbid'].isin(Test_Proteins)]
# Y_Test=Y[DATA_ALL['pdbid'].isin(Test_Proteins)]
# X_Test=X_Test.reset_index(drop=True)
# Y_Test=Y_Test.reset_index(drop=True)


Num_Test_Queries=len(Test_Proteins)

 
Predicted_NDCGatk = np.array([0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
for pp in range(Num_itr):    
    NDCG_all=[0,0,0,0,0,0,0,0,0,0,0]    
    for tt in range(Num_Test_Queries):
        Test_X_features= Normalized_X[DATA_ALL['pdbid'].isin([Test_Proteins[tt]])]
        Pred_Q= np.matmul(np.array(Test_X_features)  , Theta_t_1[pp]  )     # Prediction in iteration pp-th
        Ground_Q = np.array( Y[DATA_ALL['pdbid'].isin([Test_Proteins[tt]])] )
        Pred_Ranks=(rankdata(-Pred_Q, method='ordinal'))    # Decreasing order
        Pred_Ranks = np.expand_dims(Pred_Ranks, axis=1)
        Concat= np.concatenate((Ground_Q,Pred_Ranks), axis=1)
        sorted_array = Concat[np.argsort(Concat[:, 1])]
        RGT= sorted_array[:,0]
        # NDCG @ k
        NDCG_all = np.add(NDCG_all, [pp, ndcg_at_k(RGT, 1),ndcg_at_k(RGT, 2),ndcg_at_k(RGT, 3),ndcg_at_k(RGT, 4),ndcg_at_k(RGT, 5),ndcg_at_k(RGT, 6),ndcg_at_k(RGT, 7),ndcg_at_k(RGT, 8),ndcg_at_k(RGT, 9),ndcg_at_k(RGT, 10)])  
       

    Predicted_NDCG=NDCG_all/Num_Test_Queries
    print(Predicted_NDCG)
    Predicted_NDCGatk=np.vstack([Predicted_NDCGatk,Predicted_NDCG])


print("--- %s seconds ---" % (time.time() - start_time))

# save numpy array as csv file
from numpy import asarray
from numpy import savetxt


# save to csv file



SAVE_Name_Theta= Which_Type + '_NFeature_' + str(Num_Features) +'_Learned_Thetas_itr'+str(Num_itr)+'_Alpha'+str(Alpha)+'.csv'

SAVE_Name_Results= Which_Type + '_NFeature_' + str(Num_Features) + '_Results_itr'+str(Num_itr)+'_Alpha'+str(Alpha)+'.csv'

# save to csv file
savetxt(SAVE_Name_Theta, Theta_t_1, delimiter=',')

savetxt(SAVE_Name_Results, Predicted_NDCGatk, delimiter=',')
