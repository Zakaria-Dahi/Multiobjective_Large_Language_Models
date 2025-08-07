#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:14:48 2023

@author: zak
"""

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer
from vidur import main

class fitness_wrapper(ElementwiseProblem):

    def __init__(self,problem_dim):
        self.variables_num = problem_dim[0]
        self.problem_dim = problem_dim
        xl = np.ones(self.variables_num,dtype=np.int64)*20
        xu = np.ones(self.variables_num,dtype=np.int64)*300
        super().__init__(n_var=self.variables_num,n_obj=len(problem_dim),xl = xl,xu = xu,vtype=int)


    def _evaluate(self, x, out, *args, **kwargs):
        fitness_vector = []
        for i in range(len(self.problem_dim)):
            fitness = 1
            for k in range(self.variables_num):
                if i == 0: # plug here the energy function
                    fitness = python -m vidur.main  --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384 \
--random_forrest_execution_time_predictor_config_prediction_max_batch_size 512 \
--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 16384
                if i == 1: # plug here the memory function
                    fitness = fitness * 3
                if i == 2: # plug here the time function
                    fitness = fitness + 2
                if i == 3: # plug here the precision function
                    fitness = fitness / 2
            fitness_vector.append(fitness)
        print(x)
        out["F"] = fitness_vector
