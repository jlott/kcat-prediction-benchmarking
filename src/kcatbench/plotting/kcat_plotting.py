import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_model_comparison(df, model_x, model_y, log_scale=True):
    """
    Creates a scatter plot comparing kcat predictions from two models.
    
    Args:
        df (pd.DataFrame): The dataframe containing the results.
        model_x (str): Column name for the model on the X axis.
        model_y (str): Column name for the model on the Y axis.
        log_scale (bool): If True, plots on log10 scale.
    """
    plt.figure(figsize=(8, 8))
    sns.set_style("whitegrid")
    
    plot_data = df[[model_x, model_y]].dropna()
    plot_data = plot_data[~plot_data.isin([np.inf, -np.inf]).any(axis=1)]
    
    if len(plot_data) == 0:
        print("No valid overlapping data points found.")
        return

    sns.scatterplot(
        data=plot_data, 
        x=model_x, 
        y=model_y, 
        alpha=0.2, 
        edgecolor="k",
        s=10
    )
    
    data_min = min(plot_data[model_x].min(), plot_data[model_y].min())
    data_max = max(plot_data[model_x].max(), plot_data[model_y].max())
    
    if log_scale:
        pad_factor = 2.0
        lower_limit = data_min / pad_factor
        upper_limit = data_max * pad_factor
    else:
        pad = (data_max - data_min) * 0.05
        lower_limit = data_min - pad
        upper_limit = data_max + pad

    print(lower_limit)
    print(upper_limit)
        
    plt.plot([lower_limit, upper_limit], [lower_limit, upper_limit], 'r--', label='Perfect Agreement', lw=2)

    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(f"{model_x} $k_{{cat}}$ ($s^{{-1}}$) [log scale]", fontsize=12)
        plt.ylabel(f"{model_y} $k_{{cat}}$ ($s^{{-1}}$) [log scale]", fontsize=12)
    else:
        plt.xlabel(f"{model_x} $k_{{cat}}$ ($s^{{-1}}$)", fontsize=12)
        plt.ylabel(f"{model_y} $k_{{cat}}$ ($s^{{-1}}$)", fontsize=12)
        
    plt.xlim(lower_limit, upper_limit)
    plt.ylim(lower_limit, upper_limit)
    
    # plt.gca().set_aspect('equal', adjustable='box')
    
    plt.title(f"Comparison: {model_x} vs {model_y}", fontsize=14)
    plt.legend()
    
    plt.tight_layout()
    plt.show()