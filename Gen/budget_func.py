#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 00:50:53 2019

@author: mounir
"""

import sys
sys.path.insert(0, "../")
sys.path.insert(0, "../../equivalent_decision_trees/")

import numpy as np
from lib_tree import extract_rule, error_vote_rf


def EvCost(clf,type_rf = False,EV_type = 'depth',worst_path_case=False,flow_weighted_costs=True,on_data = False,X=None,y=None,parallel_comp = False):
    ### Return a number of binary test units###
    if not type_rf:
        dtree = clf
        if flow_weighted_costs:
    
            mean_depth = 0
            max_depth = 0
            
            if EV_type != 'depth':
                print('Only depth can be flow weighted')
            if not on_data:
                
                if worst_path_case:
                    return EvCost(dtree,EV_type = 'depth',worst_path_case=True,flow_weighted_costs=False,on_data = False,X=None,y=None)
    
                N = sum(dtree.tree_.value[0].reshape(-1))
                
                
                leaves = np.where(dtree.tree_.feature == -2)[0]
                for l in leaves :
                    phis,taus,s = extract_rule(dtree,l)
                    depth = np.array(phis).size
                    
                    if depth > max_depth :
                        max_depth = depth
                        
                    mean_depth += depth*sum(dtree.tree_.value[l].reshape(-1))/N
            else :
                N = y.size
    
                for i in dtree.apply(X):
                    phis,taus,s = extract_rule(dtree,i)
                    depth = np.array(phis).size
    
                    if depth > max_depth :
                        max_depth = depth
                        
                    mean_depth += depth/N 
                    
            if worst_path_case :   
                return max_depth
            else:   
                return mean_depth
        else:
            if EV_type == 'depth':
                if worst_path_case:
                    return dtree.tree_.max_depth
                else:
                    return EvCost(dtree,EV_type = 'depth',worst_path_case=False,flow_weighted_costs=True,on_data = False,X=None,y=None)
    
            elif EV_type == 'nodes_size':            
                return dtree.tree_.node_count   
    else:
        rf = clf
        
        max_parallel_ev = 0
        sum_ev = 0
        for dt in rf.estimators_ :
            ev = EvCost(dt,EV_type = EV_type,worst_path_case=worst_path_case,flow_weighted_costs=flow_weighted_costs,on_data = on_data,X=X,y=y)
            sum_ev += ev
            if ev > max_parallel_ev:
                max_parallel_ev = ev
        if parallel_comp:                        
            return max_parallel_ev
        else:
            return sum_ev

                
            
def FeatAcq(clf,L_c,type_rf = False,worst_path_case=False,flow_weighted_costs=True,on_data = False,X=None,y=None,feature_groups=None,feature_groups_costs=None):

    feat_use = np.zeros(L_c.size)
    grouped_feats = False
    
    if (feature_groups is None and feature_groups_costs is not None ) :
        print('Warning : missing variable : feature_groups !')
    if (feature_groups is not None and feature_groups_costs is None) :
        print('Warning : missing costs for groups of features!')
    if feature_groups is not None and feature_groups_costs is not None:
        if len(set(feature_groups)) > len(feature_groups_costs):
            print('Warning : incoherence in the number of features groups !')
        else:
            grouped_feats = True
            feat_group_use = np.zeros(feature_groups_costs.size)

    
    if not type_rf : 
        dtree = clf
        n_feats = dtree.tree_.n_features
    else:
        rf = clf
        n_feats = rf.estimators_[0].tree_.n_features  
        
    if n_feats != L_c.size:
        print('Erreur sur le nb de features')
    else:
        if not type_rf :
            if flow_weighted_costs:
                ### estimating proba according to data distr. of the k-th feature to be used by the tree ###
                if worst_path_case: 
                    ### ###
                    print('WARNING : Asking both flow weighted cost and worst case cost !')
                    ### ###
                    return FeatAcq(clf,L_c,type_rf = type_rf,worst_path_case=True,flow_weighted_costs=False,on_data = False,X=None,y=None)
 
                if not on_data:
                    N = sum(dtree.tree_.value[0].reshape(-1))
                    
                    leaves = np.where(dtree.tree_.feature == -2)[0]
                    for l in leaves :
                        phis,taus,s = extract_rule(dtree,l)
                        for k in range(L_c.size):
                            feat_use[k] += (k in phis)*sum(dtree.tree_.value[l].reshape(-1))/N
                else :
                        
                    N = y.size
                    
                    for i in dtree.apply(X):
                        phis,taus,s = extract_rule(dtree,i)
                          
                        for k in range(L_c.size):
                            feat_use[k] += (k in phis)*1/N     


            else:
                leaves = np.where(dtree.tree_.feature == -2)[0]
                feat_use_l = np.zeros((leaves.size,L_c.size))
    
                w_path = 0
                w_path_l = 0
                for j,l in enumerate(leaves) :
                    
                    phis,taus,s = extract_rule(dtree,l)
                    for k in range(L_c.size):
                        if k in phis:
                            feat_use[k] = 1 
                            feat_use_l[j,k]= 1
                            
                    if worst_path_case :
                        
                        if sum(L_c*feat_use_l[j,:]) > w_path:
                            w_path = sum(L_c*feat_use_l[j,:])
                            w_path_l = j
                            
                            
                if worst_path_case :
                     feat_use = feat_use_l[w_path_l,:]
                            
            

            if grouped_feats:
                for k in range(n_feats):
                    if feat_use[k]:
                        feat_group_use[feature_groups[k]] = 1
                        
                return sum(L_c*feat_use)+sum(feature_groups_costs*feat_group_use), feat_use
            
            return sum(L_c*feat_use), feat_use
        
        else:
            ### RF case ###
            if flow_weighted_costs:
                ### estimating proba according to data distr. of the k-th feature to be used by the rf ###
                if worst_path_case: 
                    ### ###
                    print('WARNING : Asking both flow weighted cost and worst case cost !')
                    ### ###
                    return FeatAcq(clf,L_c,type_rf = type_rf,worst_path_case=True,flow_weighted_costs=False,on_data = False,X=None,y=None)
                
                if not on_data:
                ### ###
                    print('Hard to know the flow weighted FA cost in RF case without data...')
                
                    f_b = np.zeros((rf.n_estimators,L_c.size))
                    for k,dt in enumerate(rf.estimators_):
                        w,f_b[k,:] = FeatAcq(dt,L_c,type_rf = False,worst_path_case=worst_path_case,flow_weighted_costs=True,on_data = False,X=None,y=None)
                    
                    feat_use =np.mean(f_b[k],axis=0)
                ### ###

                else :
                    N = y.size
                    
                    dt_apply = np.zeros(rf.n_estimators,dtype=object)
                    for k,dt in enumerate(rf.estimators_):
                        dt_apply[k] = dt.apply(X)
                        
                    for j in range(N):
                        feat_use_trees = np.zeros((rf.n_estimators,L_c.size))
                        for z,dt in enumerate(rf.estimators_):

                            phis,taus,s = extract_rule(dt,dt_apply[z][j])
                            for k in range(L_c.size):
                                feat_use_trees[z,k] = (k in phis)
                                
                        feat_use_bool = ( np.sum(feat_use_trees,axis=0) != 0 )
                        for k in range(L_c.size):
                            feat_use[k] += feat_use_bool[k]*1/N     
            else:
                feat_use_trees = np.zeros((rf.n_estimators,L_c.size))
                
                for k,dt in enumerate(rf.estimators_):
                    w,feat_use_trees[k,:] = FeatAcq(dt,L_c,type_rf = False,worst_path_case=worst_path_case,flow_weighted_costs=False,on_data = False,X=None,y=None)
                
                feat_use = ( np.sum(feat_use_trees,axis=0) != 0 )
                
                if worst_path_case :
                ### ###
                     print('Hard to know worst case FA cost in RF case...')
                ### ###
                            
            if grouped_feats:
                for k in range(n_feats):
                    if feat_use[k]:
                        feat_group_use[feature_groups[k]] = 1
                        
                return sum(L_c*feat_use)+sum(feature_groups_costs*feat_group_use), feat_use 
               
            return sum(L_c*feat_use), feat_use            
    
def TreeError(dtree,on_data = False,X=None,y=None):
        if not on_data:
            N = sum(dtree.tree_.value[0].reshape(-1))
            nb_err = 0
            leaves = np.where(dtree.tree_.feature == -2)[0]
            for l in leaves :
                nb_err += sum(dtree.tree_.value[l].reshape(-1))-np.amax(dtree.tree_.value[l])
            return nb_err/N
        else :
            return 1 - dtree.score(X,y)

    
def RFError(rf,on_data = False,X=None,y=None,tree_vote_only=True):
        if not on_data:
            tree_err = np.zeros(rf.n_estimators)
            for k,tr in enumerate(rf.estimators_):
                tree_err[k]=TreeError(tr,on_data=on_data,X=X,y=y)

            return np.mean(tree_err)

        else :
            ###
            if tree_vote_only:
                return error_vote_rf(rf,X,y)
            ###
            
            else:
                p = np.zeros(rf.n_classes_)
                w = np.zeros(y.size)
                for c in range(rf.n_classes_):
                    p[c]=sum(y==c)/y.size
                    w[y==c] = 1 - p[c]
                
                return 1 - rf.score(X,y,sample_weight=w)
    
    
def LagrangianCost(dtree,L_c,alpha,on_data=False,X=None,y=None):
        fa,fu = FeatAcq(dtree,L_c,on_data=on_data,X=X,y=y)
        return TreeError(dtree,on_data=on_data,X=X,y=y) + alpha*fa
  
def LagrangianCostRF(rf,L_c,alpha,on_data=False,X=None,y=None):
        L = 0
        for t in rf.estimators_:
            L+= LagrangianCost(t,L_c,alpha,on_data=on_data,X=X,y=y)
        return L/rf.n_estimators
    
# =============================================================================
# 
# =============================================================================
class BudgetFunction:
    def __init__(self,FA,EV,FA_costs,EV_K,EV_type='depth',type_rf=False,worst_path_case = False,flow_weighted_costs=False,on_data=False,X=None,y=None,feature_groups=None,feature_groups_costs=None):
        self.FA = FA
        self.EV = EV
        self.FA_costs = FA_costs
        self.EV_K = EV_K
        self.EV_type = EV_type
        self.type_rf = type_rf
        self.worst_path_case = worst_path_case
        self.flow_weighted_costs = flow_weighted_costs 
        self.on_data = on_data
        
        self.feature_groups=feature_groups
        self.feature_groups_costs=feature_groups_costs
        
        self.X = X
        self.y = y
        
    def Compute(self,indiv):
        if self.on_data and self.X is None:
            print('WARNING : Assessing data are missing -> assessing on tree values')
            self.on_data = False
        
        X = self.X
        y = self.y
        
        FA_value = 0
        EV_value = 0
        
        if self.FA :
            FA_value,b = FeatAcq(indiv.clf,self.FA_costs,type_rf = self.type_rf,worst_path_case = self.worst_path_case, flow_weighted_costs=self.flow_weighted_costs,on_data = self.on_data,X=X,y=y,feature_groups=self.feature_groups,feature_groups_costs=self.feature_groups_costs)            
        if self.EV :
            EV_value = self.EV_K*EvCost(indiv.clf,type_rf = self.type_rf,EV_type = self.EV_type,worst_path_case = self.worst_path_case,flow_weighted_costs=self.flow_weighted_costs,on_data = self.on_data,X=X,y=y)        

            
        budgetcost = FA_value + EV_value

        return budgetcost
# =============================================================================
# 
# =============================================================================
