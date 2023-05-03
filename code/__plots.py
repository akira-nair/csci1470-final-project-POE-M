#!/usr/bin/env python
'''
File        :   __plots.py
Author      :   Akira Nair, Christine Jeong, Sedong Hwang
Description :   Plots
'''

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc = {"figure.dpi":300})

def plot_convergence(history):
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5))
    sns.lineplot(history.history['loss'], ax = ax[0])
    sns.lineplot(history.history['accuracy'], ax = ax[1])
    fig.savefig("models/convergence.png")
    