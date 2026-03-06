import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from pathlib import Path
from kcatbench.util import ROOT_DIR

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Source: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)



def plot_model_comparison(df, model_x, model_y, model_x_name, model_y_name, log_scale=True, gridsize=50, save=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.set_style("ticks")
    
    plot_data = df[[model_x, model_y]].dropna()
    plot_data = plot_data[~plot_data.isin([np.inf, -np.inf]).any(axis=1)]

    if log_scale:
        plot_data = plot_data[(plot_data[model_x] > 0) & (plot_data[model_y] > 0)]
    
    if len(plot_data) == 0:
        print("No valid overlapping data points found.")
        return
    
    x_vals = plot_data[model_x]
    y_vals = plot_data[model_y]
    
    if log_scale:
        log_x = np.log10(x_vals)
        log_y = np.log10(y_vals)
        
        r, _ = pearsonr(log_x, log_y)
        rmse = np.sqrt(mean_squared_error(log_x, log_y))
        stats_text = (f"Pearson $r = {r:.2f}$\n")
    else:
        r, _ = pearsonr(x_vals, y_vals)
        rmse = np.sqrt(mean_squared_error(x_vals, y_vals))
        stats_text = (f"$N = {len(plot_data)}$\n"
                      f"Pearson $r = {r:.2f}$\n"
                      f"RMSE = {rmse:.2f}")


    data_min = min(plot_data[model_x].min(), plot_data[model_y].min())
    data_max = max(plot_data[model_x].max(), plot_data[model_y].max())
    
    if log_scale:
        pad_factor = 2.0

        safe_min = data_min if data_min > 1e-10 else 1e-4

        lower_limit = safe_min / pad_factor
        upper_limit = data_max * pad_factor
    else:
        pad = (data_max - data_min) * 0.05
        lower_limit = data_min - pad
        upper_limit = data_max + pad

    # sns.scatterplot(
    #     data=plot_data, 
    #     x=model_x, 
    #     y=model_y, 
    #     alpha=0.2, 
    #     edgecolor="k",
    #     s=10
    # )

    hb = ax.hexbin(
        x_vals, 
        y_vals, 
        gridsize=gridsize, 
        cmap='inferno_r',
        mincnt=1,     
        xscale='log' if log_scale else 'linear',
        yscale='log' if log_scale else 'linear',
        edgecolors='none' 
    )

    confidence_ellipse(np.log10(x_vals), np.log10(y_vals), ax, edgecolor='red', linewidth=2)
    
    cb = plt.colorbar(
        hb, 
        label='Count', 
        shrink=0.5,     
        aspect=20,      
        pad=0.05      
    )
    

    plt.plot([lower_limit, upper_limit], [lower_limit, upper_limit], 
             color='black',     
             linestyle='--',      
             alpha=0.6,        
             linewidth=1.0, 
             label='Perfect Agreement')

    plt.text(0.05, 0.95, stats_text, 
             transform=plt.gca().transAxes, 
             fontsize=11, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(f"{model_x_name} $k_{{cat}}$ ($s^{{-1}}$) [log scale]", fontsize=12)
        plt.ylabel(f"{model_y_name} $k_{{cat}}$ ($s^{{-1}}$) [log scale]", fontsize=12)
    else:
        plt.xlabel(f"{model_x_name} $k_{{cat}}$ ($s^{{-1}}$)", fontsize=12)
        plt.ylabel(f"{model_y_name} $k_{{cat}}$ ($s^{{-1}}$)", fontsize=12)
        
    plt.xlim(lower_limit, upper_limit)
    plt.ylim(lower_limit, upper_limit)
    
    plt.gca().set_box_aspect(1)
    sns.despine()

    plt.gca().minorticks_off()
    cb.ax.minorticks_off()
    
    plt.title(f"{model_x_name} vs {model_y_name}", fontsize=14)
    # plt.legend()
    
    plt.tight_layout()

    if save:
        plt.savefig(
            str(ROOT_DIR / "data" / "results" / f"{model_x_name}_vs_{model_y_name}.png"), 
            dpi=300,             
            bbox_inches='tight', 
            transparent=False   
        )
    plt.show()