#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 09:26:41 2019

@author: mounir
"""
import numpy as np
import copy
import time
import os,sys
sys.path.insert(0, "../")
sys.path.insert(0, "../../equivalent_decision_trees/")

from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from budget_func import BudgetFunction,TreeError,RFError
import lib_gen_operators as lib_gen
import genetic

from sklearn.model_selection import StratifiedKFold

from lib_tree import fill_with_samples, error_vote_rf

from sklearn.datasets import make_classification

# =============================================================================
# 
# =============================================================================
SEED = 0

N_EST = 10
MIN_DEPTH = 2
MAX_DEPTH = 6
BOUND_DEPTH = 6

alpha = 0.1
size_init = 40

N_iter = 15
ratio_death = 0.6
ratio_mut = 0.8
ratio_rep = 0.25
t_nat = 3

N_rep = 5

# =============================================================================
#   ALGORITHM INPUTS
# =============================================================================
SIZE = 300
N_FEATS = 20
N_RED = 5

### labeled data ###
X,y = make_classification(n_samples = SIZE, n_features=N_FEATS, n_informative=N_FEATS-N_RED, n_redundant = N_RED)

i_tr_0 = np.random.choice(np.arange(SIZE),SIZE//3,replace=False)
i_tr_1 = np.random.choice(np.arange(SIZE),SIZE//3,replace=False)
i_test = np.random.choice(np.arange(SIZE),SIZE//3,replace=False)

X_train_init = X[i_tr_0]
y_train_init = y[i_tr_0]

X_train_budget = X[i_tr_1]
y_train_budget = y[i_tr_1]

X_test = X[i_test]
y_test = y[i_test]

### feature costs ###
L_c = np.random.exponential(size=N_FEATS)

# =============================================================================
#  GENETIC ALGORITHM
# =============================================================================

hists = np.zeros(N_rep,dtype=object)

clf_init = RandomForestClassifier(n_estimators = N_EST,max_depth=MAX_DEPTH,random_state=SEED)
clf_init.fit(X_train_init,y_train_init)


budget = BudgetFunction(1,1,L_c,0.005,EV_type='depth',type_rf=True,flow_weighted_costs=True,on_data=True,X=X_train_budget,y=y_train_budget)


for i in range(N_rep):

    ind_init = genetic.individual(clf_init,0,alpha,type_rf=True,evaluate_on_data=True,X=X_train_budget,y=y_train_budget)
    init_value = ind_init.compute_value(budget)
    init_depth = ind_init.compute_depth()
    init_model_size = ind_init.compute_model_size()

    pop = genetic.population(alpha,type_rf = True,evaluate_on_data=True,X_source=X_train_init,y_source=y_train_init,X=X_train_budget,y=y_train_budget,budget_function=budget,min_depth=MIN_DEPTH)

    for dt in ind_init.clf.estimators_:
        fill_with_samples(dt,X_train_budget,y_train_budget,refill = 1)
    
    pop.GenerateInitialFromOne(ind_init,size_init,max_depth=BOUND_DEPTH,smallest_tree=False)

    for r in pop.list_indiv :
        for dt in r.clf.estimators_:
            fill_with_samples(dt,X_train_budget,y_train_budget,refill = 1)


    hist = pop.Launch(N_iter,ratio_death,ratio_mut,ratio_rep,t_nat)
    hists[i] = hist

    vs = np.array(pop.values)
    best_indiv =  pop.list_indiv[np.argmin(vs)]

    # =============================================================================
    #
    # =============================================================================

    best_indiv.X = X_test
    best_indiv.y = y_test
    best_indiv.compute_value(budget)
    
    print('Alpha', alpha)

    print('Init val :', init_value)
    print('Fin val :', best_indiv.value)

    print('Erreur init: ',ind_init.emp_error)
    print('Erreur fin: ', best_indiv.emp_error)

    # =============================================================================
    #
    # =============================================================================

