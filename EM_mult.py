import numpy as np
import time
from tqdm import tqdm
from EM_algorithm import EM_algorithm, get_p_est
import json

import matplotlib.pyplot as plt

def compute_q_multinomial(n,p,b):
    p_cond = (b @ p)[:,None,:] - p[None,...] 
    q = p[None,...]*np.expand_dims(n, axis=1) / p_cond
    q[np.isnan(q)] = 0
    q = q/np.sum(q, axis=2)[:,:,None]
    q[np.isnan(q)] = 0
    return q

# def compute_q_multinomial(n,p,b):
#     p_cond = (b @ p)[:,None,:] - p[None,...] 
#     q = np.expand_dims(n, axis=1) / p_cond
#     # print('r')
#     # print(p_cond)
#     # print('q')
#     # print(q/np.sum(q, axis=2)[:,:,None])
#     # print('p')
#     # print(p[None,...])
#     # exit()
#     q[np.isnan(q)] = 0
#     q = p[None,...]*q/np.sum(q, axis=2)[:,:,None]
#     #q[np.isnan(q)] = 0
#     return q

def compute_p(q, b):
    num = np.sum(np.multiply(q,b[...,None]),axis=0)
    dem = np.sum(b,axis=0)[...,None]
    return num / dem

# (g,i) compute estimate of p using EM algorithm with parameters X and b 
def EM_mult(X, b, p_est = None, convergence_value = 0.0001, max_iterations = 100, 
                p_method = 'group_proportional', load_bar = True, verbose = True,
                dict_results = {}, save_dict = False, dict_file = None):
    if p_est is None:
        p_est = get_p_est(X, b, p_method)
    return EM_algorithm(X, b, p_est, compute_q_multinomial, convergence_value, max_iterations, load_bar, verbose,
                        dict_results, save_dict, dict_file)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    # Example
    # instances/J100_M50_G2_I2_L50_seed1.json
    # read

    ll_lists = []
    q_lists = []

    for s in range(1,21):
        with open(f'instances/J100_M50_G3_I4_L50_seed{s}.json') as f:
            data = json.load(f)

        X = np.array(data['n'])
        b = np.array(data['b'])

        

        print(X.shape)
        print(b.shape)

        p_est = get_p_est(X, b, 'uniform')


        p_est, i, run_time, ll_list, q_list = EM_mult(X, b, p_est, max_iterations = 100,
                                        p_method = 'group_proportional', load_bar = True, verbose = True,
                                        dict_results = {}, save_dict = False, dict_file = None)
        
        ll_lists.append(ll_list)
        q_lists.append(q_list)




    fig = plt.figure(figsize=(5,15))
    for i, ll_list in enumerate(ll_lists):
        plt.plot(ll_list, label=f'seed {i+1}', alpha=0.5)
    plt.title('ll_list')
    plt.legend()
    plt.show()

    ll_array = np.array(ll_lists)
    q_array = np.array(q_lists)

    # # plot average
    # ll_avg = np.mean(ll_array, axis=0)
    # plt.plot(ll_avg)
    # plt.title('ll_avg')
    # plt.show()

    # delta ll (llt - ll_{t-1})
    ll_delta = ll_array[:,1:] - ll_array[:,:-1]

    # print if any is lower than 0
    number_of_negatives = np.sum(ll_delta < 0)
    # print smalles delta
    print(f'Smallest delta in ll_delta: {np.min(ll_delta)}')
    print(f'Number of negative values in ll_delta: {number_of_negatives}')
    # plot
    plt.plot(ll_delta.T, alpha=0.7)
    plt.title('ll_delta')
    plt.ylim(-.1,.1)
    # hline
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.show()



    # Q delta
    q_delta = q_array[:,1:] - q_array[:,:-1]
    # plot
    plt.plot(q_delta.T, alpha=0.7)
    plt.title('q_delta')
    # hline
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.show()
