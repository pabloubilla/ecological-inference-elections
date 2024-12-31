import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# ======================== MAP DATA ========================
# Map File Path
map_dir = os.path.join('maps', 'regiones_chile', 'Regional.shp')

# Read Map
gdf = gpd.read_file(map_dir)

# Match Region Names
region_mapping = {
    'Región de Arica y Parinacota': "DE ARICA Y PARINACOTA",
    'Región de Tarapacá': "DE TARAPACA",
    'Región de Antofagasta': "DE ANTOFAGASTA",
    'Región de Atacama': "DE ATACAMA",
    'Región de Coquimbo': "DE COQUIMBO",
    'Región de Valparaíso': "DE VALPARAISO",
    'Región Metropolitana de Santiago': "METROPOLITANA DE SANTIAGO",
    "Región del Libertador Bernardo O'Higgins": "DEL LIBERTADOR GENERAL BERNARDO O'HIGGINS",
    'Región del Maule': "DEL MAULE",
    'Región de Ñuble': "DE NUBLE",
    'Región del Bío-Bío': "DEL BIOBIO",
    'Región de La Araucanía': "DE LA ARAUCANIA",
    'Región de Los Ríos': "DE LOS RIOS",
    'Región de Los Lagos': "DE LOS LAGOS",
    'Región de Aysén del Gral.Ibañez del Campo': "DE AYSEN DEL GENERAL CARLOS IBANEZ DEL CAMPO",
    'Región de Magallanes y Antártica Chilena': "DE MAGALLANES Y DE LA ANTARTICA CHILENA",
    'Zona sin demarcar': None
}





# remove x lower than 1e7 * -0.85 and y higher than 1e6 * -5
# gdf = gdf[(gdf['geometry'].bounds['maxx'] < 1e6 * -0.85) & (gdf['geometry'].bounds['miny'] > 1e6 * -5)]

# Apply Mapping
gdf['Matched_Region'] = gdf['Region'].map(region_mapping)

# ======================== HEATMAP DATA ========================
# Load Electors Data
output_folder = 'output/'
use_candidate_names = False



electors_df = pd.read_csv(f'2021_11_Presidencial/{output_folder}/2021_11_Presidencial_ELECTORES.csv')
# in CIRCUNSCRIPCION ELECTORAL, replace '.' with ''
electors_df['CIRCUNSCRIPCION ELECTORAL'] = electors_df['CIRCUNSCRIPCION ELECTORAL'].str.replace('.', '')

# %%
def age_group_to_index(age_group):
    index_list = []
    start_numbers = [18, 20, 30, 40, 50, 60, 70, 80]
    end_numbers = [19, 29, 39, 49, 59, 69, 79]
    # check if there's a plus
    if '+' in age_group:
        age_group = age_group.split('+')
        age_group[0] = int(age_group[0])
        # see where is located in start_numbers
        for i in range(len(start_numbers)):
            if age_group[0] == start_numbers[i]:
                index_list.extend(range(i,len(start_numbers)))
                break
        return index_list
    # separate by - and transform to int
    age_group = age_group.split('-')
    age_group = list(map(int, age_group))
    # get the index of the start and end numbers
    for i in range(len(start_numbers)):
        if age_group[0] == start_numbers[i]:
            index_list.append(i)
            break
    for i in range(len(end_numbers)):
        if age_group[1] == end_numbers[i]:
            # if not in list, append all indexes before
            if i not in index_list:
                index_list.extend(range(index_list[0]+1,i+1))
            break
    return index_list

# %%
regions = ["DE ARICA Y PARINACOTA",
           "DE TARAPACA", 
           "DE ANTOFAGASTA",
           "DE ATACAMA",
           "DE COQUIMBO",
           "DE VALPARAISO",
           "METROPOLITANA DE SANTIAGO",
           "DEL LIBERTADOR GENERAL BERNARDO O'HIGGINS",
           "DEL MAULE",
           "DE NUBLE",
           "DEL BIOBIO",
           "DE LA ARAUCANIA",
           "DE LOS RIOS",
           "DE LOS LAGOS",
           "DE AYSEN DEL GENERAL CARLOS IBANEZ DEL CAMPO",
           "DE MAGALLANES Y DE LA ANTARTICA CHILENA"]

# %%
region_labels = ['AP', 'TA', 'AN', 'AT', 'CO', 'VA', 'RM', 'LI', 'ML', 'NB', 'BI', 'AR', 'LR', 'LL', 'AI', 'MA']

# %%
# get all folders in output_file
# regions = [f for f in os.listdir(output_file) if os.path.isdir(os.path.join(output_file, f))]

AGE_GROUPS = ['18-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']

# read output/CANDIDATOS.pickle
with open(f'2021_11_Presidencial/{output_folder}/CANDIDATOS.pickle', 'rb') as f:
    candidatos = pickle.load(f)


boric_heatmap = np.zeros((len(AGE_GROUPS), len(regions)))
kast_heatmap = np.zeros((len(AGE_GROUPS), len(regions)))
parisi_heatmap = np.zeros((len(AGE_GROUPS), len(regions)))

# loop over regions
for r in regions:
    # print(r)
    region_path = os.path.join(f'2021_11_Presidencial/{output_folder}', r)
    # get all districts in region file
    districts = [f for f in os.listdir(region_path) if os.path.isdir(os.path.join(region_path, f))]

    tota_electors_in_region = np.zeros(len(AGE_GROUPS))
    total_weighted_in_region = np.zeros(len(AGE_GROUPS))    

    # weighted_probs_in_region = np.zeros(len(AGE_GROUPS))
    weighted_boric = np.zeros(len(AGE_GROUPS))
    weighted_kast = np.zeros(len(AGE_GROUPS))
    weighted_parisi = np.zeros(len(AGE_GROUPS))
    for d in districts:

        # print(d)
        # get GRUPOS.pickle from district
        district_path = os.path.join(region_path, d)
        
        try:
            with open(os.path.join(district_path, 'GRUPOS.pickle'), 'rb') as f:
                grupos_d = pickle.load(f)
            probs = pd.read_csv(os.path.join(district_path, f'P_{d}.csv'), index_col=0)
            df_district = pd.read_csv(os.path.join(district_path, f'{d}.csv'))
        except FileNotFoundError:
            print(f'File not found {d}')
            continue
        

        
        # which column is GABRIEL BORIC and JOSE ANTONIO KAST
        idx_boric = candidatos.index('GABRIEL BORIC')
        idx_kast = candidatos.index('JOSE ANTONIO KAST')
        idx_parisi = candidatos.index('FRANCO ALDO PARISI')

        # define matrices
        probs_matrix = probs.values
        electors_matrix_agg = df_district[grupos_d].values
        electors_agg_sum = electors_matrix_agg.sum(axis=0)

        district_fixed_name = d.replace('.', '') # shouldn't need to do this, but there's 3 mismatched names
        electors_district = electors_df[electors_df['CIRCUNSCRIPCION ELECTORAL'] == district_fixed_name]
        electors_matrix = electors_district[AGE_GROUPS].values
        
        # get sum of electors (each column)
        electors_sum = electors_matrix.sum(axis=0)

        # add to total electors in region
        tota_electors_in_region += electors_sum

        # get index from probs
        index = probs.index
        # get index_list
        group_indexes = [age_group_to_index(i) for i in index]

        # weighted electors sum
        electors_weighted_sum = np.zeros(len(AGE_GROUPS))
        for i in range(len(group_indexes)):
            total_macro_group = np.sum(electors_sum[group_indexes[i]])
            if total_macro_group == 0:
                print(grupos_d)
                print('group_indexes', group_indexes)
                print('district', d)
                print('region', r) 
                print('age_group', index[i])
                print('--')
            else:
                electors_weighted_sum[group_indexes[i]] = electors_sum[group_indexes[i]] / total_macro_group
        total_weighted_in_region += electors_weighted_sum

        # create new empty prob matrix with original age groups
        new_probs_matrix = np.zeros((len(AGE_GROUPS), probs_matrix.shape[1]))
        # loop over group_indexes so that grouped indexes have the same probability
        for i in range(len(group_indexes)):
            new_probs_matrix[group_indexes[i]] = probs_matrix[i]
    
        # probs boric
        probs_boric = new_probs_matrix[:, idx_boric]
        # probs kast
        probs_kast = new_probs_matrix[:, idx_kast]
        # probs paris
        probs_parisi = new_probs_matrix[:, idx_parisi]

        # use weighted sum of electors
        weighted_boric += electors_weighted_sum * probs_boric
        weighted_kast += electors_weighted_sum * probs_kast
        weighted_parisi += electors_weighted_sum * probs_parisi

        # weighted_boric += electors_sum * probs_boric
        # weighted_kast += electors_sum * probs_kast
        # weighted_parisi += electors_sum * probs_parisi
        # weighted_probs_in_region += electors_sum * probs_matrix.sum(axis=1)
        
    # get final probabilities for region
    prob_region_boric = weighted_boric / total_weighted_in_region
    prob_region_kast = weighted_kast / total_weighted_in_region
    prob_region_parisi = weighted_parisi / total_weighted_in_region


    # orininal way 
    # prob_region_boric = weighted_boric / tota_electors_in_region
    # prob_region_kast   = weighted_kast / tota_electors_in_region
    # prob_region_parisi = weighted_parisi / tota_electors_in_region

    # add to heatmaps
    boric_heatmap[:, regions.index(r)] = prob_region_boric
    kast_heatmap[:, regions.index(r)] = prob_region_kast
    parisi_heatmap[:, regions.index(r)] = prob_region_parisi

# Create DataFrames for Heatmap Data
boric_df = pd.DataFrame(boric_heatmap.T, columns=AGE_GROUPS, index=regions)
kast_df = pd.DataFrame(kast_heatmap.T, columns=AGE_GROUPS, index=regions)

# Merge with GeoDataFrame
gdf = gdf.merge(boric_df.add_prefix('Boric_'), left_on='Matched_Region', right_index=True, how='left')
gdf = gdf.merge(kast_df.add_prefix('Kast_'), left_on='Matched_Region', right_index=True, how='left')

# ======================== PLOT HEATMAP ========================
def plot_candidate_heatmaps(candidate_prefix, title_prefix):
    fig, axes = plt.subplots(1, len(AGE_GROUPS), figsize=(20, 7))
    for i, age_group in enumerate(AGE_GROUPS):
        column = f'{candidate_prefix}_{age_group}'
        ax = axes[i]
        gdf.plot(column=column, cmap='YlOrRd', legend=False, edgecolor='black', ax = ax)
        ax.set_title(f'{age_group}', fontsize=6)
        ax.set_axis_off()
        ax.axis('off')
        # # aspect equal
        ax.set_aspect('equal')
        # restrict axis to avoid plt.xlim(1e7 * -0.85, gdf['geometry'].bounds['maxx'].max())
        ax.set_xlim(1e7 * -0.85, gdf['geometry'].bounds['maxx'].max())
    # no white space
    plt.tight_layout()

    plt.show()

# Plot Heatmaps for Each Candidate
plot_candidate_heatmaps('Boric', 'Support for Gabriel Boric')
# plot_candidate_heatmaps('Kast', 'Support for José Antonio Kast')

