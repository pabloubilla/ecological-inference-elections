import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from group_opt import new_group_matrix, optimze_age_groups_v2, bootstrap
from EM_mult import EM_mult
from matplotlib.colors import ListedColormap
import textwrap
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def load_data(votes_path, electors_path, candidates_path, groups_path):
    votes = pd.read_csv(votes_path)
    electors = pd.read_csv(electors_path)
    with open(candidates_path, 'rb') as f:
        candidatos = pickle.load(f)
    with open(groups_path, 'rb') as f:
        grupos = pickle.load(f)
    return votes, electors, candidatos, grupos

def merge_data(votes, electors, grupos):
    llave_mesa = ['CIRCUNSCRIPCION ELECTORAL', 'MESA']
    return votes.merge(electors[llave_mesa + grupos], on=llave_mesa, how='inner')

def group_heatmap(df_circ, grupos, cmap='Blues', trunc=0.7, output_path=None):


    # sample 30 rows
    df_circ = df_circ.sample(50, random_state=123)

    fig, ax = plt.subplots(1, 1, figsize=(9,2.5), sharey=True)
    cbar_ax = fig.add_axes([1.01, 0.3, 0.03, 0.4])


    group_proportion = df_circ[grupos].T / np.sum(df_circ[grupos].T)
    im = sns.heatmap(group_proportion, ax=ax, xticklabels=False, yticklabels=True,
                        cmap=cmap, cbar=True, cbar_ax=cbar_ax, vmin=0, vmax=trunc,
                        cbar_kws={"shrink": 0.3})
    if trunc < 1:
        n_trunc = int(trunc*10)
        cbar_ax.set_yticks([0.1, 0.3, 0.5, 0.7])
        cbar_ax.set_yticklabels(['0.1', '0.3', '0.5'] + [f' > {0.1*n_trunc:.1f}'])
    cbar_ax.set_title('Proportion of voters', fontsize=10, pad=10)


    #     ax[index].set_yticklabels(grupos)
    ax.set_ylabel('Age Range')
    ax.set_xlabel('Sampled Ballot-boxes in district: Puente Alto')
    #     ax[index].tick_params(left=True)
            # else:
        #     ax[index].tick_params(left=False)

    fig.subplots_adjust(wspace=0.15, right=0.95)

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
    plt.show()




def plot_probabilities(df, candidatos, grupos, circs, candidates_used, group_combinations_list, output_path=None):
    candidate_label = ['A', 'B', 'C']
    markers = ['o', 'v', 's']
    linestyles = ['-', '--', '-.']

    # fig = plt.figure(constrained_layout=True, figsize=(12, 7))
    fig = plt.figure(constrained_layout=True, figsize=(10, 4))

    subfigs = fig.subfigures(nrows=2, ncols=2, width_ratios=[10, 1])

    for i, circ in enumerate(circs):
        x = df.loc[df['CIRCUNSCRIPCION ELECTORAL'] == circ, candidatos].values
        b = df.loc[df['CIRCUNSCRIPCION ELECTORAL'] == circ, grupos].values
        M = len(x)
        subfigs[i,0].suptitle(f'\nDistrict: {circ} (B = {M})', fontsize = 10)
        axs = subfigs[i,0].subplots(nrows=1, ncols=2, sharey=True)

        for j, group_combinations in enumerate(group_combinations_list):
            b_new, group_names = new_group_matrix(b, group_combinations)
            results = {}
            em = EM_mult(x, b_new, max_iterations=10000, dict_results=results, verbose=False, load_bar=False)
            ll = (results['Q'] - results['E_log_q']) / M
            p_est = results['p_est']

            for c_ix, c in enumerate(candidates_used):
                axs[j].plot(p_est[:, c], label=candidate_label[c_ix], marker=markers[c_ix], linestyle=linestyles[c_ix])

            axs[j].set_xticks(range(len(group_names)), group_names)
            axs[j].text(0.5, 0.9, f'Mean Log-likelihood: {np.round(ll, 3)}',
                        horizontalalignment='center', verticalalignment='center', transform=axs[j].transAxes,
                        bbox=dict(facecolor='white', alpha=0.5))
            axs[0].set_ylim([-0.02, 1.02])
            if j == 0:
                axs[j].set_ylabel('Probability')

    axs_label = subfigs[0,1].subplots(nrows=1, ncols=1)
    for c_ix, c in enumerate(candidates_used):
        axs_label.plot([], [], label=candidate_label[c_ix], marker=markers[c_ix], linestyle=linestyles[c_ix])
    axs_label.set_frame_on(False)
    axs_label.xaxis.set_visible(False)
    axs_label.yaxis.set_visible(False)
    axs_label.legend(loc='lower center', bbox_to_anchor = (0.5,-.8), title = 'Candidate')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()




def plot_ll_vs_std(df, candidatos, grupos, circs, group_combinations_list, output_path = None):
    colors = ['green', 'orange', 'purple']
    size_list = [40, 90, 140, 200, 260]

    df_groups = []

    for circ in circs:
        print(circ)
        opt_ll = -np.inf
        x = df.loc[df['CIRCUNSCRIPCION ELECTORAL'] == circ, candidatos].values
        b = df.loc[df['CIRCUNSCRIPCION ELECTORAL'] == circ, grupos].values
        M = len(x)
        # iterate backwards group_cobinations_list
        below_thres = False
        for group_combinations in group_combinations_list[::-1]:
        # group_combinations = optimze_age_groups(b, age_groups = 8, min_std = min_std, max_corr = max_corr)
            b_new, _ = new_group_matrix(b, group_combinations) 
            results = {}
            em = EM_mult(x, b_new, max_iterations = 10000, dict_results = results, verbose = False, load_bar =False)
            ll = results['Q'] - results['E_log_q']
            p_bootstrap = bootstrap(x, b_new, S = 20, seed = 123)
            p_std = np.max(np.std(p_bootstrap, axis = 0))
            if not below_thres:
                if p_std <= 0.05:
                    below_thres = True
                    df_groups.append([circ, M, group_combinations, ll, p_std, 1])
            df_groups.append([circ, M, group_combinations, ll, p_std, 0])    

    df_groups = pd.DataFrame(df_groups, columns = ['Circunscripcion', 'M', 'group_combinations', 'll', 'p_std', 'alphas'])

    # Some formatting for a nice plot
    df_groups['G'] = df_groups['group_combinations'].apply(len) # number of groups
    df_groups['ll-M'] = df_groups['ll']/df_groups['M'] # average ll
    df_groups['Circunscripcion-M'] = df_groups['Circunscripcion'] + ' ' + df_groups['M'].astype(str)
    df_groups['District'] = df_groups['Circunscripcion'].apply(lambda x: x.title())  #.apply(lambda x: x.split(' ')[0])
    df_groups['District'] = df_groups['District'].astype(str) + ' (B = ' + df_groups['M'].astype(str) +')'
    df_groups['# Agg Groups'] = df_groups['G']

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot scatter
    scatter = sns.scatterplot(
        data=df_groups, 
        x='ll-M', 
        y='p_std',
        hue='District', 
        size='# Agg Groups', 
        sizes=size_list, 
        legend='brief',  # Enable legend generation
        palette=colors,
        alpha=1, 
        ax=ax
    )

    # Set axis labels
    ax.set_xlabel('Mean Log-likelihood', fontsize=12)
    ax.set_ylabel('Max std(p)', fontsize=12)

    # Set axis limits
    ax.set_ylim([-0.005, 0.55])
    ax.set_xlim([-8, 0.01])

    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Get handles and labels for the legend
    h,l = ax.get_legend_handles_labels()
    l1 = ax.legend(h[:len(circs)+1],l[:len(circs)+1], loc='upper left', fontsize = 12)
    l2 = ax.legend(h[len(circs)+1:],l[len(circs)+1:], loc='center left', fontsize = 12)
    ax.add_artist(l1) 

    # Show and save the plot
    plt.show()
    plt.savefig('images/ll_std_example.pdf', bbox_inches='tight')
    plt.close()



def main():
    # File paths
    votes_path = '2021_11_Presidencial/output/2021_11_Presidencial_VOTOS.csv'
    electors_path = '2021_11_Presidencial/output/2021_11_Presidencial_ELECTORES.csv'
    candidates_path = '2021_11_Presidencial/output/CANDIDATOS.pickle'
    groups_path = '2021_11_Presidencial/output/GRUPOS.pickle'

    # Load data
    votes, electors, candidatos, grupos = load_data(votes_path, electors_path, candidates_path, groups_path)
    df = merge_data(votes, electors, grupos)

    # # Plot heatmaps
    circ = 'PUENTE ALTO'
    electors_circ = electors[electors['CIRCUNSCRIPCION ELECTORAL'] == circ]
    group_heatmap(electors_circ, grupos, cmap='Blues', output_path='images/group_distribution_example_poster.pdf')

    # # Plot probabilities
    circs = ['EL GOLF', 'SALTOS DEL LAJA']
    group_combinations_list = [[[0], [1], [2], [3], [4], [5], [6], [7]],
                                [[0, 1], [2, 3, 4], [5, 6, 7]]]
    candidates_used = [2, 3, 5]
    plot_probabilities(df, candidatos, grupos, circs, candidates_used, group_combinations_list, 
                       output_path='images/group_probabilities_example.pdf')

    # # Plot ll vs std
    # circs = ['CANCURA', 'EL BELLOTO', 'PROVIDENCIA']
    # group_combinations_list = [[[0,1,2,3,4,5,6,7]],
    #                 [[0,1,2,3],[4,5,6,7]],
    #                 [[0,1],[2,3],[4,5],[6,7]],
    #                 [[0,1],[2],[3],[4],[5],[6,7]],
    #                 [[0],[1],[2],[3],[4],[5],[6],[7]]]
    # plot_ll_vs_std(df, candidatos, grupos, circs, group_combinations_list, 
    #                output_path = 'images/ll_vs_std_example.pdf')

if __name__ == '__main__':
    main()


