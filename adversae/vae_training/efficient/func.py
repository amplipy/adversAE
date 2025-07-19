import numpy as np
from matplotlib import pyplot as plt


def plot_func(x, y, title=None, xlabel=None, ylabel=None):
    """
    Plot a function with given x and y values.
    
    Parameters:
    - x: x-axis values
    - y: y-axis values
    - title: Title of the plot
    - xlabel: Label for the x-axis
    - ylabel: Label for the y-axis
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Function Curve', color='blue')
    
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    
    plt.grid()
    plt.legend()
    plt.show()