import pickle
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# function that read the instances results and return a dataframe
def read_results():

    with_replace = False

    J_list = [100] # personas
    M_list = [50] # mesas
    G_list = [2,3,4] # grupos
    I_list = [2,3,4,5,10] # candidatos 
    # G_list = [2]
    # I_list = [2]
    lambda_list = [0.5]
    seed_list = [i+1 for i in range(20)]
    # seed_list = [1]
    
    instances = []
    n_instances = len(J_list)*len(M_list)*len(G_list)*len(I_list)*len(seed_list)
    # EM_method_names = ["full", "simulate_100", "simulate_1000", "cdf", "pdf", "mult"]
    if with_replace:
        EM_method_names = ["full", "simulate_100_with_replace", "simulate_1000_with_replace", "cdf", "pdf", "mult"]
    else:
        EM_method_names = ["full", "simulate_100", "simulate_1000", "cdf", "pdf", "mult"]

    df_list = []
    for J in J_list:
        for M in M_list:
            for G in G_list:
                for I in I_list:
                    for lambda_ in lambda_list:
                            for method in EM_method_names:
                                for seed in seed_list:
                                    # if 'simulate' in method:
                                        # J = 100
                                        # M = 50
                                        # G = 2
                                        # I = 2
                                        # lambda_ = 0.5
                                        # method = 'cdf'
                                        # seed = 1
                                    try:
                                        pickle_path = f'results/J{J}_M{M}_G{G}_I{I}_lambda{int(100*lambda_)}/{method}_cv1000/{seed}.pickle'
                                        # pickle_path = f'/Users/charlesthraves/Dropbox (MIT)/Research/SERVEL/Mesas Outlier 2/Codigos/Pablo/elections-model/results/J{J}_M{M}_G{G}_I{I}_lambda{int(100*lambda_)}/{method}_cv1000/{seed}.pickle'
                                        # check if path exists
                                        # try:
                                        with open(pickle_path, 'rb') as f:
                                            data = pickle.load(f)
                                        
                                        p = data['p']
                                
                                        p_est = data['p_est']
                                        mean_error = np.mean(np.abs(p-p_est))
                                        max_error = np.max(np.abs(p-p_est))
                                        time = data['time']
                                    
                                        if 'simulate' in method:
                                            simulate_time = data['simulation_time']
                                        else:
                                            simulate_time = np.nan
                                        iter = data['iterations']
                                        end = data['end']
                                        df_list.append([J,M,G,I,lambda_,method,seed,time,simulate_time,iter,end,mean_error,max_error])
                                    except:
                                        print(f'Error in {pickle_path}')
                                        continue
                                        
                                    #     print(f'Error in {pickle_path}')
                                    #     continue
    
    df = pd.DataFrame(df_list, columns=['J','M','G','I','lambda','method','seed','time','simulate_time','iter','end','mean_error','max_error'])                                  
    return df

# function that prints the running time table
def latex_table_time():

    # choose method order
    with_replace = False
    with_replace_str = '_with_replace' if with_replace else ''
    method_order = ['full', f'simulate_100{with_replace_str}', f'simulate_1000{with_replace_str}', 'cdf', 'pdf', 'mult']
    
    # decimal digits
    decimals_method = [1, 1, 1, 2, 2, 4]

    # get the data of results
    df = read_results()

    # replace time for nan if end = -1 
    df.loc[df['end'] == 0, 'time'] = np.nan
    df['total_time'] = df['time'] + df['simulate_time'].fillna(0)

    ### include the simulation time ##
    # grab the average time with J,M,G,I as row and method as column
    df_time = df.groupby(['J','M','G','I','method']).mean().reset_index().pivot(index=['J','M','G','I'], columns='method', values='total_time').reset_index()
    df_time[['J', 'M', 'G', 'I']] = df_time[['J', 'M', 'G', 'I']].astype(int)

    # round to 4 decimals for methods
    # df_time[method_order] = df_time[method_order].round(3)
    # round to decimals_method respectively
    df_time[method_order] = df_time[method_order].apply(lambda x: np.round(x, decimals_method[method_order.index(x.name)]))

    for method, decimals in zip(method_order, decimals_method):
        df_time[method] = df_time[method].apply(lambda x: f'{x:.{decimals}f}')

    # df_time[method_order] = df_time[method_order].applymap(lambda x: "{:.3f}".format(x))

    # add separator for thousands and put between brackets for those that have a comma
    # df_time[method_order] = df_time[method_order].applymap(lambda x: f'{float(x):,.3f}' if float(x) >= 1000 else x) #.replace(',', ' ')

    # display floats with 3 decimals even if they are 0'seed

    # put numbers with commas between brackets
    # df_time[method_order] = df_time[method_order].applymap(lambda x: f'{{{x}}}' if ',' in x else x)

    # replace 'nan' with '-' 
    df_time = df_time.replace('nan', '{-}')



    # reorder columns
    J = 100
    df_time_latex = pd.DataFrame()
    # change order of columns and use english notation
    df_time_latex[['C', 'G'] + method_order] = df_time.loc[df_time['J'] == J, ['I', 'G'] + method_order]
    # sort by C and G
    df_time_latex = df_time_latex.sort_values(['C', 'G'])


    df_time_latex.loc[df_time_latex['C'].duplicated(),['C']] = ''

    # add empty columns between 'C' and methods
    df_time_latex.insert(2, 'empty1', '')

    # transform to latex
    print(df_time_latex.to_latex(index=False, escape=False, column_format='cccccccccc', float_format=":.%4f"))

    return 0

def latex_table_simulation_time():
    method_order_sim = ['simulate_timesimulate_100_with_replace', 'timesimulate_100_with_replace', 'simulate_timesimulate_1000_with_replace', 'timesimulate_1000_with_replace']
    df = read_results()

    # grab the average time with J,M,G,I as row and method as column
    df_sim = df.loc[df['method'].str.contains('simulate_1')].groupby(['J','M','G','I','method']).mean().reset_index().pivot(index=['J','M','G','I'], columns='method', values=['time', 'simulate_time']).reset_index()

    df_sim[['J', 'M', 'G', 'I']] = df_sim[['J', 'M', 'G', 'I']].astype(int)
    df_sim.columns = [f'{col[0]}{col[1]}' for col in df_sim.columns]

    # round to 4 decimals for methods
    df_sim[method_order_sim] = df_sim[method_order_sim].round(3)
    df_sim[method_order_sim] = df_sim[method_order_sim].applymap(lambda x: "{:.3f}".format(x))

    # add separator for thousands and put between brackets for those that have a comma
    df_sim[method_order_sim] = df_sim[method_order_sim].applymap(lambda x: f'{float(x):,.3f}' if float(x) >= 1000 else x) #.replace(',', ' ')

    # display floats with 3 decimals even if they are 0'seed

    # put numbers with commas between brackets
    # df_sim[method_order] = df_sim[method_order].applymap(lambda x: f'{{{x}}}' if ',' in x else x)

    # replace 'nan' with '-' 
    df_sim = df_sim.replace('nan', '{-}')

    # reorder columns
    J = 100
    df_time_sim_latex = pd.DataFrame()

    # change order of columns and use english notation
    df_time_sim_latex[['C', 'G'] + method_order_sim] = df_sim.loc[df_sim['J'] == J, ['I', 'G'] + method_order_sim]
    # sort by C and G
    df_time_sim_latex = df_time_sim_latex.sort_values(['C', 'G'])


    df_time_sim_latex.loc[df_time_sim_latex['C'].duplicated(),['C']] = ''

    # add empty columns between 'C' and methods
    df_time_sim_latex.insert(2, 'empty1', '')

    # transform to latex
    print(df_time_sim_latex.to_latex(index=False, escape=False, float_format=":.%4f"))
    return 0

def latex_table_error():
    with_replace = False
    with_replace_str = '_with_replace' if with_replace else ''
    method_order = ['full', f'simulate_100{with_replace_str}', f'simulate_1000{with_replace_str}', 'cdf', 'pdf', 'mult']
    
    df = read_results()
    
    # check this case for the mean error (when it did'nt finish)
    df.loc[df['end'] == 0,'mean_error'] = np.nan

    # CG   

    # grab the average time with J,M,G,I as row and method as column
    df_error = df.groupby(['J','M','G','I','method']).mean().reset_index().pivot(index=['J','M','G','I'], columns='method', values='mean_error').reset_index()
    df_error[['J', 'M', 'G', 'I']] = df_error[['J', 'M', 'G', 'I']].astype(int)
    # 
    # round to 4 decimals for methods
    df_error[method_order] = df_error[method_order].round(3)
    df_error[method_order] = df_error[method_order].applymap(lambda x: "{:.3f}".format(x))

    # add separator for thousands and put between brackets for those that have a comma
    df_error[method_order] = df_error[method_order].applymap(lambda x: f'{float(x):,.3f}' if float(x) >= 1000 else x) #.replace(',', ' ')

    # display floats with 3 decimals even if they are 0'seed

    # put numbers with commas between brackets
    # df_error[method_order] = df_error[method_order].applymap(lambda x: f'{{{x}}}' if ',' in x else x)

    # replace 'nan' with '-' 
    df_error = df_error.replace('nan', '{-}')



    # reorder columns
    J = 100
    df_error_latex = pd.DataFrame()
    # change order of columns and use english notation
    df_error_latex[['C', 'G'] + method_order] = df_error.loc[df_error['J'] == J, ['I', 'G'] + method_order]
    # sort by C and G
    df_error_latex = df_error_latex.sort_values(['C', 'G'])


    df_error_latex.loc[df_error_latex['C'].duplicated(),['C']] = ''

    # add empty columns between 'C' and methods
    df_error_latex.insert(2, 'empty1', '')

    # transform to latex
    print(df_error_latex.to_latex(index=False, escape=False, column_format='cccccccccc', float_format=":.%4f"))

    # # replace I by C
    # df_error['C'] = df_time['I']

def latex_table_error_v2():
    with_replace = False
    with_replace_str = '_with_replace' if with_replace else ''
    method_order = ['full', f'simulate_100{with_replace_str}', f'simulate_1000{with_replace_str}', 'cdf', 'pdf', 'mult']
    
    df = read_results()
    # CG   

    # grab the average time with J,M,G,I as row and method as column
    df_error = df.groupby(['J','M','G','I','method']).mean().reset_index().pivot(index=['J','M','G','I'], columns='method', values='mean_error').reset_index()
    df_error[['J', 'M', 'G', 'I']] = df_error[['J', 'M', 'G', 'I']].astype(int)
    # 
    # round to 4 decimals for methods
    df_error[method_order] = df_error[method_order].round(3)
    df_error[method_order] = df_error[method_order].applymap(lambda x: "{:.3f}".format(x))

    # add separator for thousands and put between brackets for those that have a comma
    df_error[method_order] = df_error[method_order].applymap(lambda x: f'{float(x):,.3f}' if float(x) >= 1000 else x) #.replace(',', ' ')

    # display floats with 3 decimals even if they are 0'seed

    # put numbers with commas between brackets
    # df_error[method_order] = df_error[method_order].applymap(lambda x: f'{{{x}}}' if ',' in x else x)

    # replace 'nan' with '-' 
    df_error = df_error.replace('nan', '{-}')



    # reorder columns
    J = 100
    df_error_latex = pd.DataFrame()
    # change order of columns and use english notation
    df_error_latex[['G', 'C'] + method_order] = df_error.loc[df_error['J'] == 100, ['G', 'I'] + method_order]
    # sort by C and G
    df_error_latex = df_error_latex.sort_values(['G', 'C'])


    df_error_latex.loc[df_error_latex['G'].duplicated(),['G']] = ''

    # add empty columns between 'C' and methods
    df_error_latex.insert(2, 'empty1', '')

    # transform to latex
    print(df_error_latex.to_latex(index=False, escape=False, column_format='cccccccccc', float_format=":.%4f"))

    # # replace I by C
    # df_error['C'] = df_time['I']
    return 0


def plot_EM_error():

    ## TODO: bigotes, align tick labels
    ## 

    with_replace = False
    with_replace_str = '_with_replace' if with_replace else ''
    method_order = ['full', f'simulate_100{with_replace_str}', f'simulate_1000{with_replace_str}', 'cdf', 'pdf', 'mult']

    method_names = ['EXACT', 'H&R 100', 'H&R2 1000', 'MVN CDF', 'MVN PDF', 'MULT']

    df = read_results()

    # Set unfinished cases' mean error to NaN
    df.loc[df['end'] == 0, 'mean_error'] = np.nan

    # Compute the mean error grouped by J, M, G, I, and method
    df_error = df.groupby(['J', 'M', 'G', 'I', 'method'])['mean_error'].mean().reset_index()


    # Round values to 3 decimals
    df_error['mean_error'] = df_error['mean_error'].round(3)


    # Replace NaN values with a placeholder for visualization purposes
    # df_error['mean_error'] = df_error['mean_error'].fillna(-1)

    # Filter for a specific J value
    J = 100
    df_filtered = df_error[df_error['J'] == J]

    # Rename columns for better readability
    df_filtered = df_filtered.rename(columns={'I': 'C'})

    # remove nans
    df_filtered = df_filtered.dropna(subset=['mean_error'])


    
    
    # Custom function for filtering methods per subplot
    def filtered_barplot(data, x, y, order, **kwargs):
        # Filter methods that have NaN values for this specific subplot
        data = data.dropna(subset=[y])  # Remove rows where `y` is NaN
        valid_order = [method for method in order if method in data[x].values]
        sns.barplot(
            data=data,
            x=x,
            y=y,
            order=valid_order,  # Use filtered method order
            **kwargs
        )

    # Create a FacetGrid with seaborn
    g = sns.FacetGrid(
        df_filtered,
        col="C",
        row="G",
        margin_titles=True,
        sharey=True,
        sharex=False,
        height=2,
        aspect=1.2,
    )

    # Map a barplot to each subplot
    g.map_dataframe(
        sns.barplot,
        x="method",
        y="mean_error",
        order=method_order,
        palette="Set1",  # Assign unique colors for methods
    )


    # Create a custom legend
    method_colors = sns.color_palette("Set1", len(method_order))
    legend_labels = method_order


    # for ax in g.axes.flat:
    #     for container in ax.containers:
    #         ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=8)


    # Add a legend
    handles = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(method_colors, method_order)
    ]



    # g.figure.legend(
    #     handles=handles,
    #     title="Methods",
    #     loc="right",
    #     bbox_to_anchor=(1.1, 0.5),
    #     # bigger,
    #     ncol=1,

    # )

    # Customize axes labels and titles
    g.set_axis_labels("", "")
    g.set_xticklabels(rotation=90, ha="right")
    # x tick labels to be method_order
    g.set(xticks=range(len(method_order)), xticklabels=method_names)


    # # Modify margin titles and place them inside a box
    for ax, col_name in zip(g.axes[0], g.col_names):
        ax.text(
            0.5,
            1.05,  # Adjust vertical position above the plot
            f"C = {col_name}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor="black",
                    pad=5),
            fontsize=10,
            # bold
            fontweight='bold'
        )

    for ax, row_name in zip(g.axes[:, -1], g.row_names):
        ax.text(
            1.05,  # Adjust horizontal position to the left of the plot
            0.5,
            f"G = {row_name}",
            ha="center",
            va="center",
            rotation=-90,
            transform=ax.transAxes,
            # box filled with white 
            bbox=dict(facecolor="white", edgecolor="black",
                pad=5),
            fontsize=10,
            # bold
            fontweight='bold'
        )


    # Iterate over all axes in the grid
    for row_idx, row_axes in enumerate(g.axes):
        for col_idx, ax in enumerate(row_axes):
            # Only modify axes in columns greater than 3
            if col_idx > 2:
                # Get current ticks and labels
                ticks = ax.get_xticks()
                labels = [label.get_text() for label in ax.get_xticklabels()]
                
                # Find and remove the "EXACT" tick
                if "EXACT" in labels:
                    exact_idx = labels.index("EXACT")  # Find the index of "EXACT"
                    labels[exact_idx] = ""  # Set it to an empty string
                    
                    # Update ticks and labels
                    ax.set_xticks(ticks)
                    ax.set_xticklabels(labels)
                    
                    # Hide the corresponding tick line
                    for tickline in ax.xaxis.get_ticklines():
                        if tickline.get_xdata()[0] == ticks[exact_idx]:
                            tickline.set_visible(False)

    # ALSO REMOVE ALL TICKS FOR PLOTS THAT ARE NOT IN THE BORDER (LAST ROW)
    for row_idx, row_axes in enumerate(g.axes):
        for col_idx, ax in enumerate(row_axes):
            if row_idx != 2:
                ax.set_xticks([])
                ax.set_xticklabels([])



    # # x axis name
    # g.set_xlabels("E-Step Method", fontsize=10)


    # # y axis name
    # g.set_ylabels("MAE", fontsize=10)

    # g.set_axis_labels("E-Step Method", "MAE")

    # Optional: Adjust the fontsize for the labels
    g.fig.text(0.5, -0.01, "E-Step Method", ha='center', fontsize=10)
    g.fig.text(-0.01, 0.5, "MAE", va='center', rotation='vertical', fontsize=10)



    # Remove default margin titles
    g.set_titles(col_template='', row_template='')

    # Adjust layout
    g.figure.tight_layout()
    g.figure.subplots_adjust(top=0.85)  # Adjust layout for legend and title

    # Save and show the plot
    plt.savefig("images/EM_error.pdf", bbox_inches="tight")



    # # Create a FacetGrid with seaborn
    # g = sns.FacetGrid(df_filtered, col="C", row="G", margin_titles=True, sharey=True, sharex=True)

    # # Map a barplot to each subplot
    # g.map_dataframe(
    #     sns.barplot,
    #     x="method",
    #     y="mean_error",
    #     order=method_order,
    #     # all bars the same color
    #     palette='tab10',
    # )

    # # Adjust the axes and titles
    # g.set_axis_labels("Method", "Mean Error")
    # g.set_titles(col_template="G={col_name}", row_template="C={row_name}")
    # g.set_xticklabels(rotation=45, ha="right")

    # # Add a legend
    # for ax in g.axes.flat:
    #     for label in ax.get_xticklabels():
    #         label.set_rotation(45)

    # # Show the plot
    # plt.tight_layout()
    # plt.savefig('images/EM_error.png')
    # plt.show()


def plot_EM_error_boxplot():

    with_replace = False
    with_replace_str = '_with_replace' if with_replace else ''
    method_order = ['full', f'simulate_100{with_replace_str}', f'simulate_1000{with_replace_str}', 'cdf', 'pdf', 'mult']
    method_names = ['EXACT', 'H&R 100', 'H&R2 1000', 'MVN CDF', 'MVN PDF', 'MULT']

    df = read_results()

    # Set unfinished cases' mean error to NaN
    df.loc[df['end'] == 0, 'mean_error'] = np.nan

    # Filter for a specific J value
    J = 100
    df_filtered = df[df['J'] == J]

    # Rename columns for better readability
    df_filtered = df_filtered.rename(columns={'I': 'C'})

    # Remove NaN values
    df_filtered = df_filtered.dropna(subset=['mean_error'])

    # Create a FacetGrid with seaborn
    g = sns.FacetGrid(
        df_filtered,
        col="C",
        row="G",
        margin_titles=True,
        sharey=True,
        sharex=False,
        height=2,
        aspect=1.2,
    )

    # Map a boxplot to each subplot
    g.map_dataframe(
        sns.boxplot,
        x="method",
        y="mean_error",
        order=method_order,
        palette="Set1",  # Assign unique colors for methods
        showfliers=True  # Show outliers to include full data distribution
    )

    # Customizing x-axis labels
    g.set_axis_labels("", "")
    g.set_xticklabels(rotation=90, ha="right")
    g.set(xticks=range(len(method_order)), xticklabels=method_names)

    # Modify margin titles
    for ax, col_name in zip(g.axes[0], g.col_names):
        ax.text(
            0.5,
            1.05,
            f"C = {col_name}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor="black", pad=5),
            fontsize=10,
            fontweight='bold'
        )

    for ax, row_name in zip(g.axes[:, -1], g.row_names):
        ax.text(
            1.05,
            0.5,
            f"G = {row_name}",
            ha="center",
            va="center",
            rotation=-90,
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor="black", pad=5),
            fontsize=10,
            fontweight='bold'
        )

    # Iterate over all axes in the grid
    for row_idx, row_axes in enumerate(g.axes):
        for col_idx, ax in enumerate(row_axes):
            # Only modify axes in columns greater than 3
            if col_idx > 2:
                # Get current ticks and labels
                ticks = ax.get_xticks()
                labels = [label.get_text() for label in ax.get_xticklabels()]
                
                # Find and remove the "EXACT" tick
                if "EXACT" in labels:
                    exact_idx = labels.index("EXACT")  # Find the index of "EXACT"
                    labels[exact_idx] = ""  # Set it to an empty string
                    
                    # Update ticks and labels
                    ax.set_xticks(ticks)
                    ax.set_xticklabels(labels)
                    
                    # Hide the corresponding tick line
                    for tickline in ax.xaxis.get_ticklines():
                        if tickline.get_xdata()[0] == ticks[exact_idx]:
                            tickline.set_visible(False)

    # ALSO REMOVE ALL TICKS FOR PLOTS THAT ARE NOT IN THE BORDER (LAST ROW)
    for row_idx, row_axes in enumerate(g.axes):
        for col_idx, ax in enumerate(row_axes):
            if row_idx != 2:
                ax.set_xticks([])
                ax.set_xticklabels([])

    # Set axis labels
    g.fig.text(0.5, -0.01, "E-Step Method", ha='center', fontsize=10)
    g.fig.text(-0.01, 0.5, "MAE", va='center', rotation='vertical', fontsize=10)

    # for every xtick make sure its centered
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('center')

    # Remove default margin titles
    g.set_titles(col_template='', row_template='')

    # Adjust layout
    g.figure.tight_layout()
    g.figure.subplots_adjust(top=0.85)

    # Save and show the plot
    plt.savefig("images/EM_error_boxplot.pdf", bbox_inches="tight")
    plt.show()


   

def plot_EM_time():
    ## TODO: 10 al lado, numeros completos (sin potencia) 
    with_replace = False
    with_replace_str = '_with_replace' if with_replace else ''
    method_order = ['full', f'simulate_100{with_replace_str}', f'simulate_1000{with_replace_str}', 'cdf', 'pdf', 'mult']
    method_names = ['EXACT', 'H&R 100', 'H&R 1000', 'MVN CDF', 'MVN PDF', 'MULT']

    df = read_results()

    # Set unfinished cases' time to NaN
    df.loc[df['end'] == 0, 'time'] = np.nan

    # Compute the mean time grouped by J, M, G, I, and method
    df_time = df.groupby(['J', 'M', 'G', 'I', 'method'])['time'].mean().reset_index()

    # Round values to 3 decimals
    df_time['time'] = df_time['time'].round(3)

    # Remove NaN values
    df_time = df_time.dropna(subset=['time'])

    # Filter for a specific J value
    J = 100
    df_filtered = df_time[df_time['J'] == J]

    # Rename column for better readability
    df_filtered = df_filtered.rename(columns={'I': 'C'})

    df_filtered['C'] = pd.Categorical(df_filtered['C'], categories=[2, 3, 4, 5, 10], ordered=True)


    # Plot settings
    g_values = df_filtered['G'].unique()
    fig, axes = plt.subplots(1, len(g_values), figsize=(15, 5), sharey=True)

    # linestyles
    markers = ['o' ,'s']
    lines = ['-', '--', '-.']

    for i, g in enumerate(g_values):
        ax = axes[i]
        df_g = df_filtered[df_filtered['G'] == g]
        
        for idx, (method, method_label) in enumerate(zip(method_order, method_names)):
            df_method = df_g[df_g['method'] == method]
            ax.plot(
                df_method['C'], 
                df_method['time'], 
                marker=markers[idx % len(markers)],  # Assign marker cyclically if needed
                linestyle=lines[idx % len(lines)],   # Assign line style cyclically if needed
                label=method_label
            )
        
        ax.set_title(f'G = {g}')
        ax.set_xlabel('C')
        if i == 0:
            ax.set_ylabel('Mean Time [s]')
        
        # Only show legend in the last plot
        if i == len(g_values) - 1:
            ax.legend()

    # log scale
    for ax in axes:
        ax.set_yscale('log')

    # 2,3,4,5,10 ticks for C
    for ax in axes:
        ax.set_xticks([2, 3, 4, 5, 10])


    plt.tight_layout()
    plt.savefig("images/EM_time.pdf", bbox_inches="tight")
    plt.show()


# main
if __name__ == '__main__':
    table_type = sys.argv[1]
    if table_type == 'time':
        latex_table_time()
        plot_EM_time()
    elif table_type == 'simulation_time':
        latex_table_simulation_time()
    elif table_type == 'error':
        latex_table_error()
        # plot_EM_error()
        plot_EM_error_boxplot()
    
# python3 analize_EM.py time
# python3 analize_EM.py simulation_time
# python3 analize_EM.py error