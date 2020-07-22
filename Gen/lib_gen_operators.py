#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 00:55:48 2019

@author: mounir
"""

import sys
sys.path.insert(0, "../")
sys.path.insert(0, "../../equivalent_decision_trees/")

import genetic
from lib_tree import extract_rule, cut_into_leaf2,fill_with_samples,depth_array
from eq_trees import eq_rec_tree, eqtree_rec_rf
from lib_eq import CoherentFusionDecisionTree

import numpy as np
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# =============================================================================
# 
# =============================================================================

def RandomOnePrune(dtree,policy=None,depth_thresh=2):

    if policy is None:
        all_inds = np.arange(0,dtree.tree_.feature.size)
        depths = depth_array(dtree,all_inds)
        mx_d = max(depths)

        inds1 = np.where(dtree.tree_.feature != -2 )[0]
        inds2 = np.where(depths >= depth_thresh)[0]
        
        inds = list(set(list(inds1)).intersection(set(list(inds2))))

        if len(inds) != 0:
            node = np.random.choice(inds)
            cut_into_leaf2(dtree,node)

        return dtree
    
    elif policy == 'exp' :

        all_inds = np.arange(0,dtree.tree_.feature.size)
        depths = depth_array(dtree,all_inds)
        mx_d = max(depths)
        
        inds1 = np.where(dtree.tree_.feature != -2 )[0]
        inds2 = np.where(depths >= depth_thresh)[0]
        t = np.random.exponential(mx_d/2)
        inds3 = np.where(depths >= int(mx_d -t))[0]
        
        inds = set(list(inds1)).intersection(set(list(inds2)))
        inds = inds.intersection(set(list(inds3)))
        inds = list(inds)


        if len(inds) != 0:
            node = np.random.choice(inds)
            cut_into_leaf2(dtree,node)

        return dtree
    else:
        all_inds = np.arange(0,dtree.tree_.feature.size)
        depths = depth_array(dtree,all_inds)
        mx_d = max(depths)
        
        inds1 = np.where(dtree.tree_.feature != -2 )[0]
        inds2 = np.where(depths > depth_thresh)[0]
        
        inds = list(set(list(inds1)).intersection(set(list(inds2))))

        if len(inds) != 0:
            node = np.random.choice(inds)
            cut_into_leaf2(dtree,node)
        
        return dtree 
    
def RandomPrune(dtree,N,policy=None,depth_thresh=2):
    for i in range(N):
        dtree = RandomOnePrune(dtree,policy=policy,depth_thresh=depth_thresh)
    return dtree


           
def GenerateEqTrees(dtree,nb, max_depth = None,from_depth = None, on_subtrees = False, subtrees_nodes = None, finishing_features = list(), smallest_tree = False, X_source = None, y_source = None):
    L_trees = np.zeros(nb,dtype=object)

    for k in range(nb):
        new_dt = copy.deepcopy(dtree)
        new_dt = eq_rec_tree(new_dt,0, max_depth = max_depth, from_depth = from_depth, on_subtrees = on_subtrees, subtrees_nodes = subtrees_nodes, finishing_features = finishing_features, 
                             smallest_tree = smallest_tree)

        L_trees[k] = new_dt

        
    return L_trees

# =============================================================================
#   special DT
# =============================================================================
def ReprodIndividualsFromRF(list_indiv,max_id,options):

    list_indiv = list(list_indiv)
    rf = RandomForestClassifier(n_estimators=len(list_indiv))
    trees = list()
    for indiv in list_indiv :
        trees.append(indiv.clf)
        
    rf.estimators_ = trees
    rf.n_classes_ = trees[0].n_classes_
    rf.classes_ = trees[0].classes_
    
    new_dt = eqtree_rec_rf(rf,0,max_depth = options['max_depth'],smallest_tree=False)

    new_id = max_id +1

    indiv3 = genetic.individual(new_dt,new_id,type_rf=False,alpha= options['alpha'],evaluate_on_data= options['on_data'],X= options['X'],y= options['y'])
    
    return indiv3
 
def ReprodCrossOver(list_indiv,max_id,options):
    list_indiv = list(list_indiv)

    if len(list_indiv) > 2 :
        print('Warning : more than 2 parents..')
    elif len(list_indiv) < 2 :
        print('Warning : parent missing..')
    else :
        new_id = max_id +1
        ###
        node = np.random.randint(list_indiv[0].clf.tree_.node_count)
        ###
        new_dt = CoherentFusionDecisionTree(list_indiv[0].clf, node, list_indiv[1].clf)

        indiv3 = genetic.individual(new_dt,new_id,type_rf=False,alpha= options['alpha'],evaluate_on_data= options['on_data'],X= options['X'],y= options['y'])

    
    return indiv3
        
        
def PruningSimpleMutation(indiv,options,min_depth=1):
    nb = options['nb']
    dtree = copy.deepcopy(indiv.clf)
    dtree = RandomPrune(dtree,nb,policy=None,depth_thresh=min_depth)
    indiv.clf = dtree 
    return indiv

 

def EqTreeMutation(indiv,options):

    dtree = copy.deepcopy(indiv.clf)
    dtree = eq_rec_tree(dtree,0)

    indiv.clf = dtree 
    return indiv

# =============================================================================
#     SPECIAL RF
# =============================================================================

def RFReprodCrossOver(list_indiv,max_id,options):
    list_indiv = list(list_indiv)

    if len(list_indiv) > 2 :
        print('Warning : more than 2 parents..')
    elif len(list_indiv) < 2 :
        print('Warning : parent missing..')
    else :

        clfp1 = list_indiv[0].clf
        clfp2 = list_indiv[1].clf
        clfchild = copy.deepcopy(clfp1)
        new_id = max_id +1
        
        for k,e in enumerate(clfp1.estimators_):
            
            ###
            node = np.random.randint(e.tree_.node_count)
            ###
            clfchild.estimators_[k] = CoherentFusionDecisionTree(e, node, clfp2.estimators_[k])

        indiv3 = genetic.individual(clfchild,new_id,type_rf=True,alpha= options['alpha'],evaluate_on_data= options['on_data'],X= options['X'],y= options['y'])

    
    return indiv3
            
def RFReprodExchTrees(list_indiv,max_id,options):
 
    list_indiv = list(list_indiv)
    
    clfp1 = list_indiv[0].clf
    clfp2 = list_indiv[1].clf
    clfchild = copy.deepcopy(clfp1)
  
    if options['cross_shuffle'] :
        n1 = clfp1.n_estimators//2  
        n2 = clfp2.n_estimators//2  
        clf1_p = np.random.choice(clfp1.estimators_,size=n1,replace=False)
        clf2_p = np.random.choice(clfp2.estimators_,size=n2,replace=False)

        clfchild.estimators_ = list(clf1_p) + list(clf2_p)
    else:

        inds = np.random.randint(0,clfp2.n_estimators,clfp2.n_estimators//2)
        for j in inds:
            clfchild.estimators_[j] = clfp2.estimators_[j]
    
    clfchild.n_estimators = len(clfchild.estimators_)
    new_id = max_id +1

    indiv3 = genetic.individual(clfchild,new_id,type_rf=True,alpha=options['alpha'],evaluate_on_data=options['on_data'],X=options['X'],y=options['y'])
    
    return indiv3

def RFPruningSimpleMutation(indiv,options,min_depth=1):
 
    nb = options['nb']
    rf = copy.deepcopy(indiv.clf)
    for k,dtree in enumerate(rf.estimators_):
        rf.estimators_[k] = RandomPrune(dtree,nb,policy='exp',depth_thresh=min_depth)

    indiv.clf = rf    
    return indiv
    
def RFEqTreeMutation(indiv,options):
    
    for k,dtree in enumerate(indiv.clf.estimators_):
        indiv.clf.estimators_[k] = eq_rec_tree(dtree,0)

    return indiv
# =============================================================================
# 
# =============================================================================
            
