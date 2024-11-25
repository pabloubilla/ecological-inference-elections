import numpy as np
import pickle
import json
import pandas as pd
import time
import sys
import os

from instance_maker import gen_instance_v3
from EM_full import EM_full, compute_q_list
from EM_mvn_pdf import EM_mvn_pdf, compute_q_mvn_pdf
from EM_simulate import EM_simulate, simulate_Z
from EM_mult import EM_mult, compute_q_multinomial
from EM_mvn_cdf import EM_mvn_cdf, compute_q_mvn_cdf
from EM_algorithm import EM_algorithm, get_p_est

verbose = True
load_bar = True
convergence_value = 0.001
max_iterations = sys.maxsize
step_size = 3000
python_arg = True
HR_unique = False

full = lambda X, b, dict_results, dict_file, p_est: EM_full(X, b, p_est=p_est, max_iterations=max_iterations, convergence_value=convergence_value, verbose=verbose, load_bar=load_bar, dict_results = dict_results, save_dict = True, dict_file = dict_file)
simulate_100 = lambda X, b, dict_results, dict_file, p_est: EM_simulate(X, b, p_est=p_est, max_iterations=max_iterations, convergence_value=convergence_value, simulate=True, samples = 100, step_size = step_size, verbose=verbose, load_bar=load_bar, dict_results = dict_results, save_dict = True, dict_file = dict_file, unique = HR_unique)
simulate_1000 = lambda X, b, dict_results, dict_file, p_est: EM_simulate(X, b, p_est=p_est, max_iterations=max_iterations, convergence_value=convergence_value, simulate=True, samples=1000, step_size = step_size, verbose=verbose, load_bar=load_bar, dict_results = dict_results, save_dict = True, dict_file = dict_file, unique = HR_unique)
cdf = lambda X, b, dict_results, dict_file, p_est: EM_mvn_cdf(X, b, p_est=p_est, max_iterations=max_iterations, convergence_value=convergence_value, verbose=verbose, load_bar=load_bar, dict_results = dict_results, save_dict = True, dict_file = dict_file)
pdf = lambda X, b, dict_results, dict_file, p_est: EM_mvn_pdf(X, b, p_est=p_est, max_iterations=max_iterations, convergence_value=convergence_value, verbose=verbose, load_bar=load_bar, dict_results = dict_results, save_dict = True, dict_file = dict_file)
mult = lambda X, b, dict_results, dict_file, p_est: EM_mult(X, b, p_est=p_est, max_iterations=max_iterations, convergence_value=convergence_value, verbose=verbose, load_bar=load_bar, dict_results = dict_results, save_dict = True, dict_file = dict_file)
EM_methods_dic = {'full': full, 'simulate_100': simulate_100, 
                'simulate_1000': simulate_1000,
                'cdf': cdf, 'pdf': pdf, 'mult': mult}


# EM_methods = [full, simulate_100, simulate_1000, cdf, pdf, mult]
# with_replace_str = "_with_replace"
# EM_method_names = ["full", f"simulate_100{(not HR_unique)*with_replace_str}", f"simulate_1000{(not HR_unique)*with_replace_str}", "cdf", "pdf", "mult"]
# convergence_values = [0.001, 0.0001]
# convergence_names = ['1000', '10000']
# convergence_value = 0.001

### MAIN EXPERIMENT
J_list = [100] # personas
M_list = [50] # mesas
G_list = [2,3,4] # grupos
I_list = [2,3,4,5,10] # candidatos 
lambda_list = [0.5]
seed_list = [i+1 for i in range(20)]
pinit_list = [-1 for i in range(20)]
###

### CONVERGENCE EXPERIMENT
# J_list = [100] # personas
# M_list = [50] # mesas
# G_list = [2,3] # grupos
# I_list = [2,3] # candidatos
# lambda_list = [0.5]
# seed_list = [1 for i in range(20)]
# pinit_list = [i+1 for i in range(20)]
###

### LAMBDA EXPERIMENT
# J_list = [100] # personas
# M_list = [50] # mesas
# G_list = [2,3] # grupos
# I_list = [2,3] # candidatos
# lambda_list = [0.05*i for i in range(0,21)] # as percentage
# seed_list = [i for i in range(20)]
# pinit_list = [-1 for i in range(20)]
###



instances = []

n_instances = len(J_list)*len(M_list)*len(G_list)*len(I_list)*len(seed_list)

for j in J_list:
    for m in M_list:
        for g in G_list:
            for i in I_list:
                for lambda_ in lambda_list:
                    for seed_instance in seed_list:
                        instances.append((j,m,g,i,lambda_, seed_instance))

# function that gets the number of the instance
def get_instance_number(J, M, G, I, lambda_, seed):
    for i, instance in enumerate(instances):
        if instance == (J, M, G, I, lambda_, seed):
            return i
    return -1

# function that prints the run command for the instance
def print_run_instances():
    # create a txt file
    with open('run_instances.txt', 'w') as f:        
        for method in range(5, -1, -1):
            # print()
            f.write(f'\n')
            for J in J_list:
                for M in M_list:
                    for G in G_list:
                        f.write(f'\n')
                        for I in I_list:
                            for lambda_ in lambda_list:
                                for seed in seed_list:
                                    # print(f'# python3 EM_cluster.py {method:d} {J:d} {M:d} {G:d} {I:d} {lambda_:.1f} {seed:d}')
                                    f.write(f'python3 EM_cluster.py {method:d} {J:d} {M:d} {G:d} {I:d} {lambda_:.1f} {seed:d}\n')
    f.close()
# get_instance_number(100, 50, 2, 5, 0.5, 1)

if __name__ == '__main__':
    # print_run_instances()
    # exit()
    if python_arg:
        method_name = sys.argv[1]
        # instance_number = int(sys.argv[2])
        J = int(sys.argv[2])
        M = int(sys.argv[3])
        G = int(sys.argv[4])
        I = int(sys.argv[5])
        lambda_ = float(sys.argv[6])
        seed_instance = int(sys.argv[7])
        seed_pinit = int(sys.argv[8])


        # convergence_value_number = int(sys.argv[3])
        # assert (method_number < len(EM_methods)) and (method_number >= 0), f'Method does not exist, should be int between 0 and {len(EM_methods)-1}' 
        # check that G is an integer greater or equal than 1
        # assert (G >= 1), f'G should be an integer greater or equal than 1'
        # check that I is an integer greater or equal than 1
        # assert (I >= 1), f'I should be an integer greater or equal than 1'
        #assert (convergence_value_number < len(convergence_values)) and (convergence_value_number >= 0), f'Convergence value does not exist, should be int between 0 and {len(convergence_values)-1}'
        # assert (instance_number < len(instances)) and (instance_number >= 0), f'Instance does not exist, should be int between 0 and {len(instances)-1}'
        # J, M, G, I, lambda_, seed = instances[instance_number]
        # method_name = EM_method_names[method_number]
        EM_method = EM_methods_dic[method_name]
        # convergence_value = convergence_values[convergence_value_number]
        # method_name = EM_method_names[method_number] + '_cv' + convergence_names[convergence_value_number]
        method_name = method_name + '_cv' + str(int(1/convergence_value))
        
    print('-'*70)
    # print('instancia ',instance_number,': ',f"J = {J}, M = {M}, G = {G}, I = {I}, lambda = {int(100*lambda_)}%, seed = {seed}")
    print(f"J = {J}, M = {M}, G = {G}, I = {I}, lambda = {int(100*lambda_)}%, seed_instance = {seed_instance}, seed_pinit = {seed_pinit}")
    print('método ',method_name)
    print('-'*70)
    # # generate folder for method if it doesn't exist
    # method_folder = f"results/{method_name}"
    # if not os.path.exists(method_folder):
    #     os.makedirs(method_folder)
    # # generate folder for instance if it doesn't exist
    # instance_folder = f"{method_folder}/J{J}_M{M}_G{G}_I{I}_lambda{lambda_}"
    # if not os.path.exists(instance_folder):
    #     os.makedirs(instance_folder)

    # generate folder for instance if it doesn't exist
    instance_folder = f"results/J{J}_M{M}_G{G}_I{I}_lambda{int(100*lambda_)}"
    if not os.path.exists(instance_folder):
        os.makedirs(instance_folder)
    # generate folder for method_instance if it doesn't exist
    method_instance_folder = f"{instance_folder}/{method_name}"
    if not os.path.exists(method_instance_folder):
        os.makedirs(method_instance_folder)

    # gen instance
    name_of_instance = f"J{J}_M{M}_G{G}_I{I}_L{int(100*lambda_)}_seed{seed_instance}"

    # create folder instances if it doesn't exist
    if not os.path.exists("instances"):
        os.makedirs("instances")

    gen_instance_v3(G, I, M, J, lambda_ = lambda_, seed = seed_instance, name = name_of_instance, terminar = False)
    # load instance with json  
    with open(f"instances/{name_of_instance}.json", 'r') as f:
        data = json.load(f)
    X = np.array(data["n"])
    b = np.array(data["b"])
    p = np.array(data["p"])
    dict_results = {}
    dict_results['X'] = X
    dict_results['b'] = b
    dict_results['p'] = p
    dict_results['J'] = J
    dict_results['M'] = M
    dict_results['G'] = G
    dict_results['I'] = I
    dict_results['lambda_'] = lambda_
    dict_results['seed'] = seed_instance
    dict_results['method'] = method_name
    dict_results['convergence_value'] = convergence_value

    print(p)

    dict_file_name = f"{method_instance_folder}/{seed_instance}"
    if seed_pinit == -1:
        p_est_start = None
    else:
        # draw from Dirichlet
        np.random.seed(seed_pinit)
        p_est_start = np.random.dirichlet([1]*I, size=G)
        dict_file_name += f"_pinit{seed_pinit}"
    dict_file_name += ".pickle"


    # if (I > 3) and (method_number == 0):
    #     print(f"The code has been aborted since: method_name = {method_name} and I = {I}")
    #     exit()
        
    # run EM
    results = EM_method(X, b, dict_results, dict_file = dict_file_name,
                         p_est = p_est_start)

# for j in range(5, -1, -1):
#     print()
#     for i in range(0, 240):
#         print(f'# python3 EM_cluster.py {j:d} {i:d} 0')


# /opt/homebrew/bin/python3.9 EM_cluster.py 2 0 0
# /opt/homebrew/bin/python3.9 EM_cluster.py 5 0 0
# python3 EM_cluster.py method_number J M G I lambda_ seed
# python3 EM_cluster.py 5 100 50 2 2 0.5 1
