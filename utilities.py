'''A collection of helper procedures for the 
PCA demonstration'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def components_table(pca, original):
    '''Organize the pca components_ attribute in a dataframe
    with labels for the orignal features and for the PCs'''

    table = pd.DataFrame(index=["PC{}".format(i + 1) for i in range(pca.n_components)],
                        columns=original.columns, data=pca.components_)
    return table


def biplot(original, reduced_data, pca):
    '''
    Produce a biplot that shows a scatterplot of the reduced
    data and the projections of the original features.
    
    original: original data, before transformation.
               Needs to be a pandas dataframe with valid column names
    reduced_data: the reduced data (the first two dimensions are plotted)
    pca: pca object that contains the components_ attribute

    return: a matplotlib AxesSubplot object (for any additional customization)
    
    This procedure is inspired by the script:
    https://github.com/teddyroland/python-biplot
    '''

    fig, ax = plt.subplots(figsize = (12,6))
    # scatterplot of the reduced data    
    ax.scatter(x=reduced_data[:, 0], y=reduced_data[:, 1], 
        facecolors='b', edgecolors='b', s=70, alpha=0.5)
    
    feature_vectors = pca.components_.T

    # we use scaling factors to make the arrows easier to see
    arrow_size, text_pos = 2.0, 2.5,

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], 
                  head_width=0.1, head_length=0.2, linewidth=1, color='red')
        ax.text(v[0]*text_pos, v[1]*text_pos, original.columns[i], color='black', 
                 ha='center', va='center', fontsize=18)

    ax.set_xlabel("PC 1", fontsize=14)
    ax.set_ylabel("PC 2", fontsize=14)
    ax.set_title("PC plane with original feature projections.", fontsize=16);
    return ax
