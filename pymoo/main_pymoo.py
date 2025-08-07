import numpy as np
import math
import random
import matplotlib.pyplot as plt
import json 
import time
import sys
# Import native built-in functions
import sys
import math
import numpy as np
import time
import random
import json
import random
import time

# Import pymoo functions
from algs import algs
from fitness_wrapper import fitness_wrapper 
from pymoo.operators.sampling.rnd import IntegerRandomSampling


# -_-_-_-_-_ Generate the QAOA Ansatz _-_-_-_-_-_-_-_-_

seed_vect = [1] # generate 30 seeds, 1/execution
algorithms = ["nsga2"]
sampl_config = IntegerRandomSampling()

for exe in seed_vect: # browse by execution
    random.seed(exe)
    np.random.seed(seed=1) # fix this to recreate the same QUBO no matter the execution
    for alg in algorithms:
        if alg == "nsga2":
            iter = 20
            pop = 10
            pc = 0.5
            pm = 0.5

        start = time.time()
        dim_prob = [11,11,11,11]
        # _-_-_-_-_ Import Initial Q per Porblem Dimension _-_-_-_-_-_-
        problem = fitness_wrapper(dim_prob)
        algorithm = algs(problem,dim_prob[0])
        if alg =="nsga2":
            pareto = algorithm.nsga2(pop,pc,pm,iter,exe,sampl_config)                                       
        #_-_-_-_-_ Compute the Fitness of the Fronts _-_-_-_-_-
        non_dominated_solutions, non_dominated_fintesses = pareto.opt.get("X", "F")
        # _-_-_-_-_-_-_ Display Results _-_-_-_-_-_-_
        non_dominated_solutions = non_dominated_solutions.astype(int)
        non_dominated_solutions = non_dominated_solutions.tolist()
        non_dominated_fintesses = non_dominated_fintesses.tolist()
        name_results_graphics = "prob_dim_" + str(dim_prob) + "_Algorithm_" + alg + "_Execution_"+ str(exe)
        if len(dim_prob) == 2:
            data = np.array(non_dominated_fintesses)
            x = data[:,0]
            y = data[:,1]
            plt.scatter(x, y)
            plt.xlabel('Metric 1')
            plt.ylabel('Metric 2')
            #plt.show()
            plt.savefig(name_results_graphics+".png")
            plt.close()

        # _-_-_-_-_-_ Display the evolution of the fitness and number of fitness evaluations needed _-_-_-_           
        # Define student_details dictionary
        data ={ "non_dominated_fintesses":non_dominated_fintesses,
                "non_dominated_solutions":non_dominated_solutions,
                "execution_time": time.time() - start} 
        # Convert and write JSON object to file
        with open(name_results_graphics+".json", "w") as outfile: 
            json.dump(data, outfile)
