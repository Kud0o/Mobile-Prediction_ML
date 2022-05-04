import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


"""
    the figure I did draw is wrong  
"""

def Correlation(X):
    corr = X.corr()
    #top_feature = corr.index[abs(corr['Value'] > 0.5)]
    # Correlation plot
    plt.subplots(figsize=(12, 8))
    #top_corr = R[top_feature].corr()
    sns.heatmap(corr, annot=True)
    plt.show()