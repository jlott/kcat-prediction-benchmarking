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
        alpha=0.6, 
        edgecolor="k",
        s=80
    )
    
    min_val = min(plot_data[model_x].min(), plot_data[model_y].min())
    max_val = max(plot_data[model_x].max(), plot_data[model_y].max())
    
    if log_scale:
        min_val = min_val if min_val > 0 else 1e-4
        
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Agreement', lw=2)

    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(f"{model_x} $k_{{cat}}$ ($s^{{-1}}$) [log scale]", fontsize=12)
        plt.ylabel(f"{model_y} $k_{{cat}}$ ($s^{{-1}}$) [log scale]", fontsize=12)
    else:
        plt.xlabel(f"{model_x} $k_{{cat}}$ ($s^{{-1}}$)", fontsize=12)
        plt.ylabel(f"{model_y} $k_{{cat}}$ ($s^{{-1}}$)", fontsize=12)
        
    plt.title(f"Comparison: {model_x} vs {model_y}", fontsize=14)
    plt.legend()
    
    plt.tight_layout()
    plt.show()