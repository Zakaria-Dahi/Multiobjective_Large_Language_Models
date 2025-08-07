# all algorithms
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
# common to nsga-2, r-nsga-2, nsga-3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
# Addition packages
import numpy as np
from pymoo.core.population import Population



# nsga-2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation


class algs:
    def __init__(self,problem,num_variables):
        self.problem = problem
        self.variables_num = num_variables
    
    def nsga2(self,pop,pc,pm,iter,SeedGen,sampl_config):
        algorithm = NSGA2(pop_size=pop,
                sampling=sampl_config,
                crossover=TwoPointCrossover(prob=pc),
                mutation=BitflipMutation(prob=pm),                   
                eliminate_duplicates=True)
        
        res = minimize(self.problem,
                        algorithm,
                        ('n_gen', iter),
                        seed=SeedGen,
                        verbose=False,
                        save_history=True)
        return res;

        