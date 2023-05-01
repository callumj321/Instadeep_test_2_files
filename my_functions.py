import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
acid_list = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','X','B','U','O']

def data_prep(file_loc,identifier):
    # Concatenates datafiles into a dataframe.

    # Parameters:
    #   file_loc (str): Path to the folder containing the training split data.
    #   identifier (str): Dev, test or train - select the split data to read.

    # Returns:
    #   data_concat (dataframe): A dataframe with all data from the split

    data_files = glob.glob(file_loc+identifier+'/*')
    data_files_in = [pd.read_csv(data_file) for data_file in data_files]
    data_concat = pd.concat(data_files_in,ignore_index=True)
    data_concat['split'] = identifier
    return data_concat

def plot_hist_comp(df,feat,bins=100,splits=['train','dev','test']):
    # Plots a histogram showing the breakdown of the entire dataset into splits.

    # Parameters:
    #   df (pandas - dataframe): The dataframe of all labelled data for all splits.
    #   feat (str): The feature of the dataframe to analyse.
    #   bins (int): The number of bins to split the data into for the histogram:
    #   splits (list): The names of the data splits. For the sake of this exercide these will always be the same.

    # Returns:
    #   plt (matplotlib - plot): Returns a plot of the data with appropriate titles and legend.

    df[feat].hist(bins=100,label='All data')
    for split in splits:
        df.query('split == @split')[feat].hist(bins=100,label=split)
    plt.title('Distribution of '+str(feat)+' for the entire dataset')
    plt.xlabel(feat)
    plt.ylabel('# of sequences')
    plt.legend()
    return plt

def val_count_hist(df,feat,bins=500,splits=['train','dev','test'],xlim=[0,1000],ylim=[0,1000]):
    # Plots a histogram showing the breakdown of the entire dataset into splits. Includes a value count for string data.

    # Parameters:
    #   df (pandas - dataframe): The dataframe of all labelled data for all splits.
    #   feat (str): The feature of the dataframe to analyse.
    #   bins (int): The number of bins to split the data into for the histogram:
    #   splits (list): The names of the data splits. For the sake of this exercide these will always be the same.
    #   xlim (array): Tells the plot where to start and stop the window of the graph being displayed on the x-axis.
    #   ylim (array): Tells the plot where to start and stop the window of the graph being displayed on the y-axis.

    # Returns:
    #   plt (matplotlib - plot): Returns a plot of the data with appropriate titles and legend.

    df[feat].value_counts().hist(bins=bins,label='All data')
    for split in splits:
        df.query('split == @split')[feat].value_counts().hist(bins=bins,label=split)
    plt.title('Distribution of '+str(feat)+' for the entire dataset')
    plt.xlabel(str(feat)+' size')
    plt.ylabel('# of '+str(feat)+' of size x')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    return plt

def seq_threshold_check(df,feat,threshold):
    # Finds and reports the percentage of the dataset with values under a threshold.

    # Parameters:
    #   df (pandas - dataframe): The dataframe of all labelled data for all splits.
    #   feat (str): The feature of the dataframe to analyse.
    #   Threshold (int): Report the amount of the dataset with values lower than this amount.
    
    per_of_seq = int(np.around(len(df.query("{0} < @threshold".format(feat))[feat].values)/len(df.index),decimals=2)*100)
    max_of_feat = df[feat].values.max()
    print(str(per_of_seq)+'% of feature '+str(feat)+' are less than '+str(threshold))
    print('Feature max: '+str(max_of_feat)+'\n')

def val_count_seq_threshold_check(df,feat,threshold):
    # Finds and reports the percentage of the dataset with values under a threshold. Includes a value count for string data.

    # Parameters:
    #   df (pandas - dataframe): The dataframe of all labelled data for all splits.
    #   feat (str): The feature of the dataframe to analyse.
    #   Threshold (int): Report the amount of the dataset with values lower than this amount.

    per_of_seq = int(np.around(sum(df[feat].value_counts() < threshold)/len(df[feat].value_counts().keys()),decimals=2)*100)
    max_of_feat = df[feat].value_counts().max()
    print(str(per_of_seq)+'% of feature '+str(feat)+' are less than '+str(threshold))
    print('Largest group: '+str(max_of_feat)+'\n')

def build_AA_index_lookup():
    # Creates a dictionary of amino acid letter codes that return a numeric substitution.

    # Returns:
    #   acid_lookup (dict.) - A dictionary of amino acid letter keys corresponding to numeric substitutions.

    acid_lookup = {}
    i = 1
    for AA in acid_list:
        acid_lookup[AA] = i
        i += 1
    return acid_lookup