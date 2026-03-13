import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from pathlib import Path
from upsetplot import from_contents, UpSet
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



def plot_model_comparison(df:pd.DataFrame, model_x:str, model_y:str, model_x_name:str, model_y_name:str, log_scale=True, gridsize=50, save=False, vmax=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.set_style("ticks")

    plot_data = df[[model_x, model_y]].dropna().copy()

    # 2. FORCE list extraction BEFORE any math or comparisons happen
    for col in [model_x, model_y]:
        # A simpler, more aggressive lambda to pull the first item
        plot_data[col] = plot_data[col].apply(
            lambda x: x[0] if type(x) in [list, np.ndarray, tuple] else x
        )
        
        # Force the column to be numeric floats
        plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')
    
    plot_data = plot_data[[model_x, model_y]].dropna()
    plot_data = plot_data[~plot_data.isin([np.inf, -np.inf]).any(axis=1)]

    if log_scale:
        plot_data = plot_data[(plot_data[model_x] > 0) & (plot_data[model_y] > 0)]
    
    if len(plot_data) == 0:
        print("No valid overlapping data points found.")
        return
    
    x_vals = plot_data[model_x]
    y_vals = plot_data[model_y]

    data_min = min(plot_data[model_x].min(), plot_data[model_y].min())
    data_max = max(plot_data[model_x].max(), plot_data[model_y].max())
    
    if log_scale:
        plot_x = np.log10(x_vals)
        plot_y = np.log10(y_vals)
        
        r, _ = pearsonr(plot_x, plot_y)
        rmse = np.sqrt(mean_squared_error(plot_x, plot_y))
        stats_text = (f"Pearson $r = {r:.2f}$\n"
                      f"$N = {len(plot_data)}$")
        
        pad_factor = 2.0
        safe_min = data_min if data_min > 1e-10 else 1e-4
        lower_limit = np.log10(safe_min / pad_factor)
        upper_limit = np.log10(data_max * pad_factor)
    else:
        plot_x = x_vals
        plot_y = y_vals
        r, _ = pearsonr(plot_x, plot_y)
        rmse = np.sqrt(mean_squared_error(plot_x, plot_y))
        stats_text = (f"$N = {len(plot_data)}$\n"
                      f"Pearson $r = {r:.2f}$\n"
                      f"RMSE = {rmse:.2f}")
                      
        pad = (data_max - data_min) * 0.05
        lower_limit = data_min - pad
        upper_limit = data_max + pad

    hb = ax.hexbin(
        plot_x, 
        plot_y, 
        gridsize=gridsize, 
        cmap='inferno_r',
        mincnt=1,     
        edgecolors='none',
        extent=[lower_limit, upper_limit, lower_limit, upper_limit],
        vmax=vmax
    )

    confidence_ellipse(plot_x, plot_y, ax, edgecolor='red', linestyle='--', linewidth=1)
    
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
        plt.xlabel(f"{model_x_name} $k_{{cat}}$ ($s^{{-1}}$) [log scale]", fontsize=14)
        plt.ylabel(f"{model_y_name} $k_{{cat}}$ ($s^{{-1}}$) [log scale]", fontsize=14)

        log_formatter = FuncFormatter(lambda x, pos: f"$10^{{{x:g}}}$")
        
        ax.xaxis.set_major_formatter(log_formatter)
        ax.yaxis.set_major_formatter(log_formatter)
    else:
        plt.xlabel(f"{model_x_name} $k_{{cat}}$ ($s^{{-1}}$)", fontsize=12)
        plt.ylabel(f"{model_y_name} $k_{{cat}}$ ($s^{{-1}}$)", fontsize=12)
        
    plt.xlim(lower_limit, upper_limit)
    plt.ylim(lower_limit, upper_limit)
    
    plt.gca().set_box_aspect(1)
    sns.despine()

    n_std_val = 3.0 
    
    # 2. Calculate the exact standard deviations and means
    cov_matrix = np.cov(plot_x, plot_y)
    std_x = np.sqrt(cov_matrix[0, 0])
    std_y = np.sqrt(cov_matrix[1, 1])
    mean_x = np.mean(plot_x)
    mean_y = np.mean(plot_y)
    
    # 3. Calculate the exact bounding box edges
    x_bounds = [mean_x - (n_std_val * std_x), mean_x + (n_std_val * std_x)]
    y_bounds = [mean_y - (n_std_val * std_y), mean_y + (n_std_val * std_y)]
    
    # 4. Inject these explicitly as minor ticks
    ax.set_xticks(x_bounds, minor=True)
    ax.set_yticks(y_bounds, minor=True)
    
    # 5. Style only the minor ticks to be red, thicker, and point inward
    ax.tick_params(which='minor', color='red', length=8, width=2, direction='in')

    cb.ax.minorticks_off()
    
    plt.title(f"{model_x_name} vs {model_y_name}", fontsize=16, fontweight='bold')
    
    plt.tight_layout()

    if save:
        plt.savefig(
            str(ROOT_DIR / "data" / "results" / f"{model_x_name}_vs_{model_y_name}.png"), 
            dpi=300,             
            bbox_inches='tight', 
            transparent=False   
        )
    plt.show()

def get_performance_subsets(df, models, percentage=10, subset_type='best'):
    
    if subset_type not in ['best', 'worst']:
        raise ValueError("subset_type must be either 'best' or 'worst'")
        
    model_sets = {}
    
    for model in models:
        mod_col = f"{model}_kcat"

        clean_df = df.copy()

        clean_df[mod_col] = clean_df[mod_col].apply(
            lambda x: x[0] if type(x) in [list, np.ndarray, tuple] else x
        )
        
        # Force the column to be numeric floats
        clean_df[mod_col] = pd.to_numeric(clean_df[mod_col], errors='coerce')
        
        # 1. Filter out invalid/zero values to safely calculate log10
        valid_mask = (
            (clean_df['experimental_kcat'] > 0) & 
            (clean_df[mod_col] > 0) & 
            clean_df['experimental_kcat'].notna() & 
            clean_df[mod_col].notna()
        )
        clean_df = clean_df[valid_mask]
        
        if len(clean_df) == 0:
            print(f"Warning: No valid data found for {model}.")
            model_sets[model] = set()
            continue
            
        # 2. Calculate the absolute error in log10 space
        # Error = |log10(model) - log10(experimental)|
        errors = np.abs(np.log10(clean_df[mod_col]) - np.log10(clean_df['experimental_kcat']))
        
        # 3. Determine the threshold based on the requested subset
        if subset_type == 'best':
            # For the "best" 10%, we want the 10th percentile of errors
            # and we keep everything smaller than or equal to that threshold.
            threshold = np.percentile(errors, percentage)
            subset_mask = errors <= threshold
            
        elif subset_type == 'worst':
            # For the "worst" 10%, we want the 90th percentile of errors
            # and we keep everything greater than or equal to that threshold.
            threshold = np.percentile(errors, 100 - percentage)
            subset_mask = errors >= threshold
            
        # 4. Extract the IDs and convert to a Python set for easy comparison
        subset_ids = clean_df.loc[subset_mask, 'ID'].tolist()
        model_sets[model] = set(subset_ids)
        
    return model_sets

def plot_model_upset(model_sets, display_names, title="Model Agreement on Top 10% Predictions", save_path=None):
    """
    Creates an UpSet plot from a dictionary of overlapping sets.
    """
    if not model_sets or all(len(s) == 0 for s in model_sets.values()):
        print("Error: No data in model_sets to plot.")
        return

    # --- THE FIX ---
    # Temporarily disable Pandas Copy-on-Write to prevent upsetplot from crashing
    original_cow = pd.options.mode.copy_on_write
    pd.options.mode.copy_on_write = False

    plot_ready_sets = {}
    for old_name, reaction_set in model_sets.items():
        # Get the new name if it exists, otherwise use the old one
        new_name = display_names.get(old_name, old_name) 
        plot_ready_sets[new_name] = reaction_set
    
    try:
        # Convert the dictionary into the upsetplot format
        upset_data = from_contents(plot_ready_sets)
        
        # Configure the UpSet plot
        upset = UpSet(
            upset_data, 
            subset_size='count', 
            show_counts=False, 
            sort_by='cardinality',
            sort_categories_by='cardinality',
            facecolor="darkblue",
            element_size=40
        )

        upset.style_subsets(min_degree=6, max_degree=6, facecolor="red")
        upset.style_subsets(min_degree=5, max_degree=5, facecolor=(1.0, 0.0, 0.0, 0.7))
        
        # Create the figure and render the plot
        fig = plt.figure(figsize=(10, 6))

        axes_dict = upset.plot(fig=fig)
    
        # The set names live on the y-axis of the 'matrix' plot
        axes_dict['matrix'].tick_params(axis='y', labelsize=15)  # <-- Increase this number to make it bigger

        # upset.plot(fig=fig)
        
        # Add the title
        plt.suptitle(title, fontsize=24, fontweight='bold')
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
            
        plt.show()
        
    finally:
        # Restore the original pandas setting so we don't mess up the rest of your script
        pd.options.mode.copy_on_write = original_cow