#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:01:29 2019

@author: mounir
"""

import sys
sys.path.insert(0, "../")
sys.path.insert(0, "../../equivalent_decision_trees/")


from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from budget_func import BudgetFunction,TreeError,RFError
import lib_gen_operators as lib_gen

from lib_tree import depth_tree

import numpy as np
import time
import copy
        

class individual:

    def __init__(self,clf,new_id,alpha,type_rf=False,evaluate_on_data=False,X=None,y=None,args=None):
        self.id = new_id

        self.alpha = alpha
        self.evaluate_on_data = evaluate_on_data
        self.X = X
        self.y = y
        
        self.age = 0
        self.type_rf = type_rf
        self.clf = clf
        self.max_depth = self.clf.max_depth
        self.value = 0
        
        self.budget = 0
        self.emp_error = 0
        self.depth = 0
        
        self.ancestors = list()
        self.immunity = 0
        
        self.dead = False
        
        if type_rf :
            self.values = np.zeros(clf.n_estimators)

            self.model_size = model_size(clf,type_rf = True)

        else:
            self.values = None
            self.model_size = model_size(clf,type_rf = False)

    def copy(self,new_id):

        new_indiv = individual(copy.deepcopy(self.clf),new_id,self.alpha,self.type_rf,self.evaluate_on_data,self.X,self.y)

        new_indiv.value = self.value
        new_indiv.budget = self.budget
        new_indiv.emp_error = self.emp_error
        new_indiv.max_depth = self.max_depth
        new_indiv.depth = self.depth
        new_indiv.depths = self.depths
        new_indiv.ancestors = self.ancestors
        new_indiv.immunity = self.immunity
        
        return new_indiv
    

    def GenerateEqIndiv(self,max_id,size=5,max_depth = None, from_depth = None, on_subtrees = False, subtrees_nodes = None, finishing_features = list(),smallest_tree=False,budget_function=None):

        list_indiv = list()

        print('Generating intial pop..')
        if not self.type_rf:
            dt_init = self.clf

            print('Creating equivalent trees...')
            if smallest_tree:
                L_trees = lib_gen.GenerateEqTrees(dt_init,size,max_depth=max_depth,from_depth=from_depth,
                                                  on_subtrees = on_subtrees, subtrees_nodes = subtrees_nodes, finishing_features = finishing_features, 
                                                  smallest_tree = True,X_source=self.X,y_source=self.y)
            else:
                L_trees = lib_gen.GenerateEqTrees(dt_init,size,max_depth=max_depth,from_depth=from_depth,
                                                  on_subtrees = on_subtrees, subtrees_nodes = subtrees_nodes, finishing_features = finishing_features, 
                                                  smallest_tree = False,X_source=self.X,y_source=self.y)
            print('    Computing values ...')
            for i in range(size):
                max_id +=  1
                new_indiv = individual(L_trees[i],max_id,type_rf=False,alpha=self.alpha,evaluate_on_data=self.evaluate_on_data,X=self.X,y=self.y)
                new_indiv.compute_value(budget_function)
                new_indiv.compute_depth()
                new_indiv.compute_model_size()
                
                new_indiv.ancestors = [new_indiv.id]
                
                list_indiv.append(new_indiv)

        else:
            rf_init = self.clf
            
            generated_trees = np.zeros((size,rf_init.n_estimators),dtype=object)

            print('Creating equivalent trees...')
            for k,t in enumerate(rf_init.estimators_) :
                if smallest_tree:
                    generated_trees[:,k] = lib_gen.GenerateEqTrees(t,size,max_depth=max_depth,from_depth=from_depth,
                                                  on_subtrees = on_subtrees, subtrees_nodes = subtrees_nodes, finishing_features = finishing_features, 
                                                  smallest_tree = True,X_source=self.X,y_source=self.y)
                else:
                    generated_trees[:,k] = lib_gen.GenerateEqTrees(t,size,max_depth=max_depth,from_depth=from_depth,
                                                  on_subtrees = on_subtrees, subtrees_nodes = subtrees_nodes, finishing_features = finishing_features, 
                                                  smallest_tree = False,X_source=self.X,y_source=self.y)

            print('    Computing values ...')
            for i in range(size):
                rf = copy.deepcopy(rf_init)
                rf.estimators_ = generated_trees[i]
                rf.n_estimators = len(rf.estimators_)

                max_id +=  1

                new_indiv = individual(rf,max_id,type_rf=True,alpha=self.alpha,evaluate_on_data=self.evaluate_on_data,X=self.X,y=self.y)
                new_indiv.compute_value(budget_function)
                new_indiv.compute_depth()
                new_indiv.compute_model_size()
                
                new_indiv.ancestors = [new_indiv.id]
                
                list_indiv.append(new_indiv)

        return list_indiv
        
    def compute_value(self,func):
        self.budget = func.Compute(self)
        if not self.type_rf:
            self.emp_error = TreeError(self.clf,self.evaluate_on_data,self.X,self.y)
            self.value = self.emp_error + self.alpha*self.budget
            return self.value
        else:
            self.emp_error = RFError(self.clf,self.evaluate_on_data,self.X,self.y)
            self.value = self.emp_error + self.alpha*self.budget
            return self.value 

    def compute_depth(self):
        
        self.max_depth = self.clf.max_depth

        if not self.type_rf :
            self.depth = depth_tree(self.clf)    
        else:
            d = np.zeros(self.clf.n_estimators)
            
            for k,dt in enumerate(self.clf.estimators_):
                d[k] = depth_tree(dt)
            self.depth = np.mean(d)
            self.depths = d
            
        return self.depth
    

    def compute_model_size(self):
        if not self.type_rf :
            self.model_size = sum(self.clf.tree_.feature != -2)
        else:
            S = 0
            for dt in self.clf.estimators_ :
                S = S + model_size(dt,type_rf = False)
            self.model_size
            
        return self.model_size


# =============================================================================
# 
# =============================================================================

def model_size(clf,type_rf=False):
    if not type_rf :
        return sum(clf.tree_.feature != -2)
    else:
        S = 0
        for dt in clf.estimators_ :
            S = S + model_size(dt,type_rf = False)
        return S
    
class Constraints:
    
    def __init__(self,on_err=False,on_budg=False,on_worst_case=False,on_model_size=False,costs_B=[1,1,1],cost_model_size_B=1000):  

        if not (on_err or on_budg or on_worst_case or on_model_size):
            print('WARNING : Please precise which cost is subject to hard budget.')
            
        self.on_err = on_err
        self.on_budg = on_budg
        self.on_worst_case = on_worst_case
        self.on_model_size = on_model_size
        self.budget_constants_vector = np.array(costs_B+[cost_model_size_B])
        
    def check_budget(self,indiv):
        
        verified_budgets = np.zeros(4,dtype=bool)
        
        if self.on_err :
            verified_budgets[0] = ( indiv.emp_error < self.budget_constants_vector[0] )
        if self.on_budg :
            verified_budgets[1] = ( indiv.budget < self.budget_constants_vector[1] )
        if self.on_worst_case :
            indiv_copy = indiv.copy()
            wc_budget = BudgetFunction(1,0,indiv_copy.FA_costs,0,EV_type='depth',type_rf=indiv_copy.type_rf)
            v = indiv_copy.compute_value(wc_budget)
            verified_budgets[2] = ( v < self.budget_constants_vector[2] )
        if self.on_model_size :
            verified_budgets[3] = ( indiv.model_size < self.budget_constants_vector[3] )
        
        tests = np.array([self.on_err,self.on_budg,self.on_worst_case,self.on_model_size]).astype(int)
        
        if sum(verified_budgets) == sum(tests) :
            return True,verified_budgets
        else:
            return False,verified_budgets
# =============================================================================
# 
# =============================================================================

class MutationFunctions:
    def __init__(self):
        self.MUT_FUNC_LIST = [lib_gen.PruningSimpleMutation,lib_gen.EqTreeMutation,lib_gen.TransferMutation,lib_gen.RandomSplitMutation]
        
class ReproductionFunctions:
    def __init__(self):
        self.REP_FUNC_LIST = [lib_gen.ReprodIndividualsFromRF,lib_gen.ReprodCrossOver]
        
class RFMutationFunctions:
    def __init__(self):
        self.MUT_FUNC_LIST = [lib_gen.RFPruningSimpleMutation,lib_gen.RFEqTreeMutation]
        
class RFReproductionFunctions:
    def __init__(self):
        self.REP_FUNC_LIST = [lib_gen.RFReprodExchTrees,lib_gen.RFReprodCrossOver]
        
# =============================================================================
#         
# =============================================================================
        
class EvolutionnaryIndivOrCoupleOperator:
    
    def __init__(self,clf_type_rf,operator='Mutation',version=0,dict_args=dict(),cpy_before_mut=False,budget_function=None):  
        self.clf_type_rf = clf_type_rf
        self.operator = operator
        self.version = version
        self.dict_args = dict_args
        self.cpy_before_mut = cpy_before_mut
        
        self.budget_function = budget_function

    def apply(self,indiv,max_id=0,nb=1,min_depth=1):
        DEBUG_ = False
        
        if self.operator == 'Reproduction' or self.operator == 'reproduction' :
            
            childs = list()
            
            if self.clf_type_rf :
                REP_FUNC_LIST = RFReproductionFunctions().REP_FUNC_LIST
            else:
                REP_FUNC_LIST = ReproductionFunctions().REP_FUNC_LIST
                
            for i in range(nb):
                
                child = REP_FUNC_LIST[self.version](indiv,max_id,self.dict_args)
                child.compute_value(self.budget_function)
                child.compute_depth()
                child.compute_model_size()
                                
                ###
                anc = list()
                for ind in indiv:
                    anc += ind.ancestors
                child.ancestors = list(set(anc))
                ###
                
                childs.append(child)
            
            return childs
            
        elif self.operator == 'Mutation' or self.operator == 'mutation' or self.operator == 'Mutate' or self.operator == 'mutate' :
            
            if self.clf_type_rf :
                MUT_FUNC_LIST = RFMutationFunctions().MUT_FUNC_LIST
            else:
                MUT_FUNC_LIST = MutationFunctions().MUT_FUNC_LIST
                
            mutants = list()
            
            for i in range(nb):
                if not indiv.immunity:
                    if self.cpy_before_mut :
                        ####
                        new_indiv = indiv.copy(max_id)
                        ####
                        mutant = MUT_FUNC_LIST[self.version](new_indiv,self.dict_args,min_depth=min_depth)
                    else:
                        mutant = MUT_FUNC_LIST[self.version](indiv,self.dict_args,min_depth=min_depth)
                        
                    
                    mutant.compute_value(self.budget_function)
                    mutant.compute_depth()
                    mutant.compute_model_size()
                    
                    if DEBUG_ :
                        print('Value : '+str(indiv.value)+'->'+str(mutant.value))
                        print('Error : '+str(indiv.emp_error)+'->'+str(mutant.emp_error))
                        print('Budget : '+str(indiv.budget)+'->'+str(mutant.budget))
                        print('Depths : '+str(indiv.depths)+'->'+str(mutant.depths))
                    ###
                    mutant.ancestors = indiv.ancestors
                    ###
                    
                    mutants.append(mutant)
                
            return mutants
            
        elif self.operator == 'Death' or self.operator == 'death':
            if not indiv.immunity:
                indiv.dead = True
                del indiv
                return 0
        
        elif self.operator == 'Copy' or self.operator == 'copy':
            list_copies = list()
            for k in range(int(nb)):
                new_indiv = indiv.copy(max_id)
                list_copies.append(new_indiv)
                max_id += 1
            
            return list_copies
        
        
            
class SelectionPopOperator:
    def __init__(self,type_='nb',version=0,args=None):
        self.type = type_
        self.args = args
        self.version = version
        
    def apply(self,pop):
        
        if self.type == 'nb':
            if self.version == 0:
                N = int(self.args)
                indexes = np.argsort(pop.values)[:N]
                return indexes
            elif self.version == 1:
                N = int(self.args)
                indexes = np.argsort(pop.values)[-N:]
                return indexes
            elif self.version == 2:
                print('exponential policy')
                N = int(self.args)
                inds = pop.size*np.random.exponential(0.5,N)
                inds = inds[inds < pop.size]
                inds = inds.astype(int)
                
                inds = set(inds)
                inds = list(inds)
                indexes = np.argsort(pop.values)[inds]
                return indexes
            else:
                N = int(self.args)
                indexes = np.random.choice(np.arange(pop.size),N)
                return indexes   
        
class EvolutionnarySelectedPopOperator:
    
    def __init__(self,clf_type_rf,indexes,operator='Mutation',version=0,dict_args=dict(),cpy_before_mut=True,budget_function=None):  
        self.clf_type_rf = clf_type_rf
        self.operator = operator
        self.version = version
        self.dict_args = dict_args
        
        self.indexes = indexes

        self.cpy_before_mut = cpy_before_mut
        self.budget_function = budget_function
        
    def apply(self,pop,hard_constraints=None,min_depth=1):
        
        if self.operator == 'Mutation' or self.operator == 'mutation' or self.operator == 'Mutate' or self.operator == 'mutate' :
            mutants = list()
            
            for i in self.indexes:
                indiv = pop.list_indiv[i]
                mut = EvolutionnaryIndivOrCoupleOperator(self.clf_type_rf,'mutation',self.version,self.dict_args,self.cpy_before_mut,budget_function=self.budget_function)
                mutants += mut.apply(indiv,max_id=pop.max_id,nb=1,min_depth=min_depth)
                
                if not self.cpy_before_mut:
                    pop.values[i] = indiv.value                
                
                if hard_constraints is not None:
                    for m in mutants:
                        b,v = hard_constraints.check_budget(m)
                        if b :
                            m.immunity = 1            
                
            if self.cpy_before_mut:
                pop.add_individuals(mutants)
                    
        elif self.operator == 'Copy' or self.operator == 'copy' :
            copies = list()

            for i in self.indexes:
                indiv = pop.list_indiv[i]
                c = EvolutionnaryIndivOrCoupleOperator(self.clf_type_rf,'copy',self.version,self.dict_args)
                ###
                copies += c.apply(indiv,max_id=pop.max_id,nb=5) 
                ###        
                
            pop.add_individuals(copies)
                
        elif self.operator == 'Death' or self.operator == 'death':
            pop.remove_individuals(self.indexes)

                
        elif self.operator == 'Reproduction' or self.operator == 'reproduction':
            reproducers = np.array(pop.list_indiv)[self.indexes]
        
            ### This can be done in several ways ###

            children = list()

            for j,rep in enumerate(reproducers):
                partners = np.array(list(set(reproducers)-set([rep])))
                sel_partners = list(np.random.choice(partners,size=self.dict_args['nat'],replace=False))
  
                r = EvolutionnaryIndivOrCoupleOperator(self.clf_type_rf,'reproduction',self.version,self.dict_args,budget_function=self.budget_function)                  
                children += r.apply([rep]+sel_partners,max_id=pop.max_id,nb = int(self.dict_args['nat']) )
                
                if hard_constraints is not None:
                    for c in children:
                        b,v = hard_constraints.check_budget(c)
                        if b :
                            c.immunity = 1    

            pop.add_individuals(children)

   
        pop.size = len(pop.list_indiv)
        
        return pop
# =============================================================================
# 
# =============================================================================

        
class population:
    
    def __init__(self,alpha,type_rf=False,evaluate_on_data=False,X_source=None,y_source=None,X=None,y=None,budget_function=BudgetFunction(0,0,0,0),hard_constraints=None,min_depth=1):
        self.budget_function = budget_function
        self.FA_costs = budget_function.FA_costs
        
        self.alpha = alpha
        self.evaluate_on_data = evaluate_on_data

        self.X_source = X_source
        self.y_source = y_source
        self.X = X
        self.y = y
                
        self.min_depth = min_depth
        self.mean_age = 0
        
        self.depths = list()
        self.budgets = list()
        self.emp_errors = list()
        self.max_depths = list()

        self.max_id = 0
        self.size = 0
        self.clf_starter = None
        self.type_rf = type_rf
        self.values = list()
        self.list_indiv = list()
        
        self.hard_constraints = hard_constraints

    def remove_individuals(self,indexes):
        ind_to_keep = list(set(np.arange(0,self.size)) - set(indexes))
        self.list_indiv = list(np.array(self.list_indiv)[ind_to_keep])
        self.values = list(np.array(self.values)[ind_to_keep])
        self.emp_errors = list(np.array(self.emp_errors)[ind_to_keep])
        self.budgets = list(np.array(self.budgets)[ind_to_keep])
        self.max_depths = list(np.array(self.max_depths)[ind_to_keep])
        
        self.depths = list(np.array(self.depths)[ind_to_keep])
        
        self.size = len(self.list_indiv)
        
        return 1
    
    def add_individuals(self,list_new_indivs):
        
        costs = list()
        errors = list()
        budgets = list()
        max_depths = list()
        depths = list()
        
        for I in list_new_indivs:
            costs.append(I.value)
            errors.append(I.emp_error)
            budgets.append(I.budget)
            max_depths.append(I.max_depth)
            depths.append(I.depth)

        self.list_indiv += list_new_indivs
        self.values += costs
        self.emp_errors += errors
        self.budgets += budgets
        self.max_depths += max_depths 
        
        self.depths += depths
        
        self.size = len(self.list_indiv)

        return 1
          

    def GenerateInitialFromOne(self,indiv_init,size, max_depth = None, from_depth = None, on_subtrees = False, subtrees_nodes = None, finishing_features = list(),smallest_tree=False):

        print('Generating intial pop..')
        indivs_to_add = indiv_init.GenerateEqIndiv(1,size,max_depth = max_depth, from_depth = from_depth, on_subtrees = on_subtrees, subtrees_nodes = subtrees_nodes, finishing_features = finishing_features, smallest_tree=smallest_tree,budget_function= self.budget_function)
        all_indivs = indivs_to_add+[indiv_init]
        
        indiv_init.compute_value(self.budget_function)
        self.add_individuals(all_indivs)
        
        if self.hard_constraints is not None:
            self.check_all_hard_constr()
            
        return 1
        
    
        
    def Launch(self,N_iter,ratio_death,ratio_mut,ratio_rep,nat,max_depth=6,MAX_SIZE=None):
        hist = history(N_iter+1,q=0.1)

        for i in range(N_iter):
            vs = np.array(self.values)
            best_indiv = self.list_indiv[np.argmin(vs)]         
            hist.sizes[hist.it] = self.size
            hist.best_scores[hist.it] = best_indiv.value
            hist.mean_scores[hist.it] = np.mean(vs)
            hist.med_scores[hist.it] = np.median(vs)
            hist.quant_scores[hist.it] = np.quantile(vs,hist.q)
            hist.best_ages[hist.it] = best_indiv.age
            hist.mean_ages[hist.it] = self.mean_age
            hist.best_ids[hist.it] = best_indiv.id
            hist.best_depth[hist.it] = best_indiv.depth
            hist.mean_depth[hist.it] = np.mean(self.depths)
            hist.med_depth[hist.it] = np.median(self.depths)
            #hist.err[hist.it] = 1 - best_indiv.clf.score(self.X,self.y)
            hist.err[hist.it] = best_indiv.emp_error
            hist.budget[hist.it] = best_indiv.budget
            hist.it += 1
        
            #print('Iteration n ',i+1)
            print('It. / Size Pop / Value / Err / Depth',i+1,self.size,best_indiv.value,best_indiv.emp_error,best_indiv.depth)
            #print('Best Value : ',best_indiv.value)
            #print('Best indiv error : ',best_indiv.emp_error)
            #print('Best indiv depths : ',best_indiv.depths)
            #print('Size Pop : ',self.size)
            
            self.NextGen(ratio_death,ratio_mut,ratio_rep,nat,max_depth=max_depth)
            
        vs = np.array(self.values)
        best_indiv = self.list_indiv[np.argmin(vs)]         
        hist.sizes[hist.it] = self.size
        hist.best_scores[hist.it] = best_indiv.value
        hist.mean_scores[hist.it] = np.mean(vs)
        hist.med_scores[hist.it] = np.median(vs)
        hist.quant_scores[hist.it] = np.quantile(vs,hist.q)
        hist.best_ages[hist.it] = best_indiv.age
        hist.mean_ages[hist.it] = self.mean_age
        hist.best_ids[hist.it] = best_indiv.id  
        hist.best_depth[hist.it] = best_indiv.depth
        hist.mean_depth[hist.it] = np.mean(self.depths)
        hist.med_depth[hist.it] = np.median(self.depths)
        #hist.err[hist.it] = 1 - best_indiv.clf.score(self.X,self.y)
        hist.err[hist.it] = best_indiv.emp_error
        hist.budget[hist.it] = best_indiv.budget
          
        return hist
    
    
    def NextGen(self,ratio_death,ratio_mut,ratio_rep,nat,min_size =3,nb_best=1,nb_ev=5,rand_constr=None, max_depth=None, prof=None):

        options = dict()
        options['FA_costs'] = self.FA_costs
        options['alpha'] = self.alpha
        options['on_data'] = self.evaluate_on_data
        options['max_depth'] = max_depth
        options['X'] = self.X
        options['y'] = self.y
        
        options['nb'] = 5
        options['nat'] = nat
        
        options['cross_shuffle'] = False
        
        
        selector = SelectionPopOperator(type_='nb',version=0,args=ratio_rep*self.size+1)
        indexes_rep = selector.apply(self)
        rep = EvolutionnarySelectedPopOperator(self.type_rf,indexes_rep,operator='reproduction',version=0,dict_args=options,budget_function=self.budget_function)
        rep.apply(self,hard_constraints = self.hard_constraints)


        selector = SelectionPopOperator(type_='nb',version=2,args=ratio_mut*self.size)
        indexes_mut = selector.apply(self)

        mut = EvolutionnarySelectedPopOperator(self.type_rf,indexes_mut,operator='mutation',version=0,dict_args=options,budget_function=self.budget_function)
        mut.apply(self,hard_constraints = self.hard_constraints,min_depth=self.min_depth)

        if self.size*(1-ratio_death) > min_size:
            
            selector = SelectionPopOperator(type_='nb',version=1,args=ratio_death*self.size)
            indexes_kill = selector.apply(self)        
            death = EvolutionnarySelectedPopOperator(self.type_rf,indexes_kill,operator='death')
            death.apply(self)     

        ma = 0
        for indiv in self.list_indiv:
            indiv.age +=1
            ma += indiv.age
            
        self.mean_age = ma/self.size

        return 1

    def check_all_hard_constr(self):
        if self.hard_constraints is not None:
            for indiv in self.list_indiv:
                b,v = self.hard_constraints.check_budget(indiv)
                if b :
                    indiv.immunity = 1            

# =============================================================================
#     
# =============================================================================

class history:
    def __init__(self,size,q=0.1):
        self.it = 0
        self.sizes = np.zeros(size)
        self.best_scores = np.zeros(size)
        self.mean_scores = np.zeros(size)
        self.med_scores = np.zeros(size)
        self.best_ages = np.zeros(size)
        self.mean_ages = np.zeros(size)
        self.best_ids = np.zeros(size)
        self.best_depth = np.zeros(size)
        self.mean_depth = np.zeros(size)
        self.med_depth = np.zeros(size)
        self.quant_scores = np.zeros(size)
        self.err = np.zeros(size)
        self.budget = np.zeros(size)
        self.q = q

