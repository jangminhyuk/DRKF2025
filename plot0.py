#!/usr/bin/env python3
"""
Violin plots and theta effect plots for MSE analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os
from matplotlib.patches import Rectangle

plt.rcParams.update({
    'font.size': 18,           # Base font size
    'axes.titlesize': 22,      # Title font size
    'axes.labelsize': 26,      # Axis label font size
    'xtick.labelsize': 20,     # X-axis tick label size
    'ytick.labelsize': 20,     # Y-axis tick label size
    'legend.fontsize': 20,     # Legend font size
    'figure.titlesize': 24     # Figure title size
})

# plt.style.use('ggplot')  # or 'classic' for pure Matplotlib


# def get_color_for_filter(filt, i):
#     """Color mapping for filters."""
#     if filt == 'drkf_neurips':
#         return 'purple'
#     elif filt == 'drkf_inf':
#         return 'blue'
#     elif filt == 'drkf_finite':
#         return 'blue'
#     elif filt == 'drkf_finite_cdc':
#         return 'brown'
#     elif filt == 'drkf_inf_cdc':
#         return 'brown'
#     elif filt == 'risk':
#         return 'orange'
#     elif filt == 'risk_seek':
#         return 'darkviolet'
#     elif filt == 'bcot':
#         return 'red'
#     elif filt == 'finite':
#         return 'black'  # Same color as inf
#     elif filt == 'inf':
#         return 'black'
#     else:
#         # Use tab10 colormap for other methods
#         colors = plt.cm.tab10(np.linspace(0, 1, 10))
#         return colors[i % len(colors)]

def get_color_for_filter(filt, i):
    """Soft academic color mapping for filters (Set2-inspired palette)."""
    # Define a soft academic palette (muted but distinct)
    bright_palette = [
            "#1f77b4",  # strong blue
            "#ff7f0e",  # vivid orange
            "#2ca02c",  # rich green
            "#d62728",  # deep red
            "#9467bd",  # purple
            "#8c564b",  # brown
            "#e377c2",  # pink
            "#7f7f7f",  # gray
            "#bcbd22",  # olive
            "#17becf"   # cyan
        ]
    
    # Map specific filters to consistent soft colors
    color_map = {
        'drkf_neurips': bright_palette[0],
        'drkf_inf': bright_palette[3],
        'drkf_finite': bright_palette[3],
        'drkf_finite_cdc': bright_palette[2],
        'drkf_inf_cdc': bright_palette[2],
        'risk': bright_palette[1],
        'risk_seek': bright_palette[4],
        'bcot': bright_palette[5],
        'finite': bright_palette[7],
        'inf': bright_palette[7],
    }
    
    return color_map.get(filt, bright_palette[i % len(bright_palette)])


def load_data(file_path):
    """Load pickled data from file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def create_violin_plot(all_results, raw_experiments_data, optimal_results, dist, filters, filter_labels):
    """Create violin plot for each method showing MSE distribution using best parameters"""
    
    # Collect MSE data for each filter using their optimal robust parameters
    mse_data = {filt: [] for filt in filters}
    
    for filt in filters:
        if filt in ['finite', 'inf']:
            # Non-robust methods: use data from first robust parameter
            robust_val = list(raw_experiments_data.keys())[0]
            experiments = raw_experiments_data[robust_val]
            for exp in experiments:
                if filt in exp:
                    mse_values = [np.mean(sim['mse']) for sim in exp[filt]]
                    mse_data[filt].extend(mse_values)
        else:
            # Robust methods: use data from optimal robust parameter
            optimal_theta = optimal_results[filt]['robust_val']
            if optimal_theta in raw_experiments_data:
                experiments = raw_experiments_data[optimal_theta]
                for exp in experiments:
                    if filt in exp:
                        mse_values = [np.mean(sim['mse']) for sim in exp[filt]]
                        mse_data[filt].extend(mse_values)
    
    # Determine y-axis clipping value FIRST to handle outliers (especially BCOT)
    y_clip = None
    if 'finite' in mse_data and mse_data['finite']:
        finite_kf_max = max(mse_data['finite'])
        y_clip = finite_kf_max * 1.5
        print(f"Clipping violin plot y-axis at {y_clip:.4f} (1.5x time-varying KF max)")
    elif 'inf' in mse_data and mse_data['inf']:
        # Fallback to time-invariant KF if time-varying not available
        inf_kf_max = max(mse_data['inf'])
        y_clip = inf_kf_max * 1.5
        print(f"Clipping violin plot y-axis at {y_clip:.4f} (1.5x time-invariant KF max)")

    # Create the violin plot
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Prepare data for violin plot - only include filters with data
    active_filters = [filt for filt in filters if mse_data[filt]]
    violin_data = [mse_data[filt] for filt in active_filters]
    violin_labels = [filter_labels[filt] for filt in active_filters]
    
    # Create violin plot
    parts = ax.violinplot(violin_data, positions=range(len(active_filters)), showmeans=True, showmedians=True)
    
    # Customize violin plot colors using global color function
    for i, (pc, filt) in enumerate(zip(parts['bodies'], active_filters)):
        color = get_color_for_filter(filt, i)
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    # Set labels and title
    ax.set_xticks(range(len(active_filters)))
    ax.set_xticklabels(violin_labels, rotation=45, ha='right')
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.grid(True, alpha=0.3)
    
    # Apply y-axis clipping to hide outliers
    if y_clip is not None:
        ax.set_ylim(top=y_clip)
        # Also set bottom limit to 0 for better visualization
        ax.set_ylim(bottom=0)
        # Set 5 y-ticks from 0 to y_clip
        ax.set_yticks(np.linspace(0, y_clip, 5))
    
    plt.tight_layout()
    
    # Ensure results directory exists
    results_path = "./results/estimation/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, f'violin_plot_{dist}_estimation.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Violin plot saved as: {output_path}")

def create_regret_violin_plot(all_results, raw_experiments_data, optimal_regret_results, dist, filters, filter_labels):
    """Create violin plot for each method showing regret distribution using best regret parameters"""
    
    # First, we need to reconstruct regret values from MSE and baseline MSE for each raw experiment
    regret_data = {filt: [] for filt in filters}
    
    for filt in filters:
        if filt in ['finite', 'inf']:
            # Non-robust methods: use data from first robust parameter
            robust_val = list(raw_experiments_data.keys())[0]
            experiments = raw_experiments_data[robust_val]
            # Get baseline MSE from this parameter set
            baseline_mse_vals = []
            for exp in experiments:
                if 'mmse_baseline' in exp:
                    baseline_mse_vals.extend([np.mean(sim['mse']) for sim in exp['mmse_baseline']])
            
            for exp in experiments:
                if filt in exp and 'mmse_baseline' in exp:
                    mse_values = [np.mean(sim['mse']) for sim in exp[filt]]
                    baseline_values = [np.mean(sim['mse']) for sim in exp['mmse_baseline']]
                    regret_values = [mse - baseline for mse, baseline in zip(mse_values, baseline_values)]
                    regret_data[filt].extend(regret_values)
        else:
            # Robust methods: use data from optimal regret parameter
            optimal_theta = optimal_regret_results[filt]['robust_val']
            if optimal_theta in raw_experiments_data:
                experiments = raw_experiments_data[optimal_theta]
                for exp in experiments:
                    if filt in exp and 'mmse_baseline' in exp:
                        mse_values = [np.mean(sim['mse']) for sim in exp[filt]]
                        baseline_values = [np.mean(sim['mse']) for sim in exp['mmse_baseline']]
                        regret_values = [mse - baseline for mse, baseline in zip(mse_values, baseline_values)]
                        regret_data[filt].extend(regret_values)
    
    # Determine y-axis clipping value FIRST to handle outliers (especially BCOT)
    y_clip_top = None
    if 'finite' in regret_data and regret_data['finite']:
        finite_regret_max = max(regret_data['finite'])
        if finite_regret_max > 0:  # Only clip if regret is positive
            y_clip_top = finite_regret_max * 1.5
            print(f"Clipping regret violin plot y-axis at {y_clip_top:.4f} (1.5x time-varying KF regret max)")
    elif 'inf' in regret_data and regret_data['inf']:
        # Fallback to time-invariant KF
        inf_regret_max = max(regret_data['inf'])
        if inf_regret_max > 0:
            y_clip_top = inf_regret_max * 1.5
            print(f"Clipping regret violin plot y-axis at {y_clip_top:.4f} (1.5x time-invariant KF regret max)")

    # Create the violin plot
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Prepare data for violin plot
    violin_data = [regret_data[filt] for filt in filters if regret_data[filt]]
    violin_labels = [filter_labels[filt] for filt in filters if regret_data[filt]]
    
    # Create violin plot
    parts = ax.violinplot(violin_data, positions=range(len(violin_labels)), showmeans=True, showmedians=True)
    
    # Customize violin plot colors using global color function
    active_filters = [filt for filt in filters if regret_data[filt]]
    for i, (pc, filt) in enumerate(zip(parts['bodies'], active_filters)):
        color = get_color_for_filter(filt, i)
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    
    # Set labels and title
    ax.set_xticks(range(len(violin_labels)))
    ax.set_xticklabels(violin_labels, rotation=45, ha='right')
    ax.set_ylabel('Regret MSE')
    ax.grid(True, alpha=0.3)
    
    # Apply y-axis clipping to hide outliers
    if y_clip_top is not None:
        ax.set_ylim(top=y_clip_top)
        # Set 5 y-ticks from 0 to y_clip_top
        ax.set_yticks(np.linspace(0, y_clip_top, 5))
    
    plt.tight_layout()
    
    # Ensure results directory exists
    results_path = "./results/estimation/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, f'regret_violin_plot_{dist}_estimation.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Regret violin plot saved as: {output_path}")

def create_theta_effect_plot(all_results, dist, filters, filter_labels):
    """Create plot showing effect of robust parameter theta on averaged MSE"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract robust values and sort them
    robust_vals = sorted(all_results.keys())
    
    # Define markers for each method
    markers = ['o', 's', '^', 'D', 'v', '<', '>', '>', 'o', 'o', '+', 'x']
    
    # Use global color function for consistent coloring
    
    # Define letter labels (A) to (L) - expanded to accommodate more filters
    letter_labels = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)']
    
    # Plot each filter
    for i, filt in enumerate(filters):
        # Determine line style based on filter type (finite versions get dotted lines)
        if filt in ['finite', 'drkf_finite', 'drkf_finite_cdc']:
            linestyle = ':'  # Dotted line for TV (finite) versions
        else:
            linestyle = '-'  # Solid line for SS (inf) and other methods
            
        if filt in ['finite', 'inf']:
            # For non-robust methods, plot horizontal line
            mse_vals = [all_results[robust_vals[0]]['mse'][filt]] * len(robust_vals)
            mse_stds = [all_results[robust_vals[0]]['mse_std'][filt]] * len(robust_vals)
            label = f"{letter_labels[i]} {filter_labels[filt]}"  # Remove (Non-robust) and add letter
        else:
            # For robust methods, plot actual theta effect
            # Safely collect data points where the filter has results
            theta_vals = []
            mse_vals = []
            mse_stds = []
            for rv in robust_vals:
                if filt in all_results[rv]['mse']:
                    theta_vals.append(rv)
                    mse_vals.append(all_results[rv]['mse'][filt])
                    mse_stds.append(all_results[rv]['mse_std'][filt])
            
            # Skip this filter if no data points available
            if not mse_vals:
                print(f"Warning: No data available for filter '{filt}' - skipping from theta effect plot")
                continue
                
            # Use collected theta values instead of all robust_vals
            robust_vals_filtered = theta_vals
            label = f"{letter_labels[i]} {filter_labels[filt]}"  # Add letter label
        
        # Plot without error bars
        # For non-robust methods, draw horizontal line without markers
        if filt in ['finite', 'inf']:
            ax.plot(robust_vals, mse_vals, 
                    marker='None', 
                    color=get_color_for_filter(filt, i),
                    linestyle=linestyle,
                    linewidth=2,
                    label=label)
        else:
            ax.plot(robust_vals_filtered, mse_vals, 
                    marker=markers[i % len(markers)], 
                    color=get_color_for_filter(filt, i),
                    linestyle=linestyle,
                    linewidth=2,
                    markersize=8,
                    label=label)
            
    
    # Let matplotlib auto-scale to the actual plotted data
    
    # Customize plot
    ax.set_xlabel('θ')
    ax.set_ylabel('Average MSE', fontsize=28)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Clip y-axis to handle BCOT outliers - use 1.5x finite KF MSE as reference
    if 'finite' in filters:
        first_robust_val = sorted(all_results.keys())[0]
        if 'finite' in all_results[first_robust_val]['mse']:
            finite_kf_mse = all_results[first_robust_val]['mse']['finite']
            y_clip_max = finite_kf_mse * 1.5
            ax.set_ylim(top=y_clip_max)
            print(f"Clipping theta effect plot y-axis at {y_clip_max:.4f} (1.5x time-varying KF)")
    
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    
    plt.tight_layout()
    
    # Ensure results directory exists
    results_path = "./results/estimation/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, f'theta_effect_{dist}_estimation.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Theta effect plot saved as: {output_path}")

def create_regret_theta_effect_plot(all_results, dist, filters, filter_labels):
    """Create plot showing effect of robust parameter theta on regret (MSE difference from MMSE baseline)"""
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Extract robust values and sort them
    robust_vals = sorted(all_results.keys())
    
    # Define markers for each method
    markers = ['o', 's', '^', 'D', 'v', '<', '>', '>', 'o', 'o', '+', 'x']
    
    # Use global color function for consistent coloring
    
    # Define letter labels (A) to (L) - expanded to accommodate more filters
    letter_labels = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)']
    
    # Plot each filter
    for i, filt in enumerate(filters):
        # Determine line style based on filter type (finite versions get dotted lines)
        if filt in ['inf', 'drkf_inf', 'drkf_inf_cdc']:
            linestyle = ':'  # Dotted line for SS (infinite) versions
        else:
            linestyle = '-'  # Solid line for TV (finite) and other methods
            
        if filt in ['finite', 'inf']:
            # For non-robust methods, plot horizontal line
            regret_vals = [all_results[robust_vals[0]]['regret'][filt]] * len(robust_vals)
            regret_stds = [all_results[robust_vals[0]]['regret_std'][filt]] * len(robust_vals)
            label = f"{letter_labels[i]} {filter_labels[filt]}"
        else:
            # For robust methods, plot actual theta effect
            # Safely collect data points where the filter has results
            theta_vals = []
            regret_vals = []
            regret_stds = []
            for rv in robust_vals:
                if filt in all_results[rv]['regret']:
                    theta_vals.append(rv)
                    regret_vals.append(all_results[rv]['regret'][filt])
                    regret_stds.append(all_results[rv]['regret_std'][filt])
            
            # Skip this filter if no data points available
            if not regret_vals:
                print(f"Warning: No data available for filter '{filt}' - skipping from regret theta effect plot")
                continue
                
            # Use collected theta values instead of all robust_vals
            robust_vals_filtered = theta_vals
            label = f"{letter_labels[i]} {filter_labels[filt]}"
        
        # Plot without error bars
        # For non-robust methods, draw horizontal line without markers
        cl = 0.2 #confidence level
        if filt in ['finite', 'inf']:
            ax.plot(robust_vals, regret_vals, 
                    marker='None', 
                    color=get_color_for_filter(filt, i),
                    linestyle=linestyle,
                    linewidth=2.5,
                    label=label)
            # ax.fill_between(robust_vals,
            #     np.array(regret_vals) - cl*np.array(regret_stds),
            #     np.array(regret_vals) + cl*np.array(regret_stds),
            #     color=get_color_for_filter(filt, i),
            #     alpha=0.15
            # )
        else:
            ax.plot(robust_vals_filtered, regret_vals, 
                    marker=markers[i % len(markers)], 
                    markerfacecolor='white',
                    markeredgecolor=get_color_for_filter(filt, i),
                    color=get_color_for_filter(filt, i),
                    markeredgewidth=1.2,
                    linestyle=linestyle,
                    linewidth=2.5,
                    markersize=12,
                    label=label)  
            # ax.fill_between(robust_vals_filtered,
            #     np.array(regret_vals) - cl*np.array(regret_stds),
            #     np.array(regret_vals) + cl*np.array(regret_stds),
            #     color=get_color_for_filter(filt, i),
            #     alpha=0.15
            # )
                
        
        
    # Let matplotlib auto-scale to the actual plotted data
    
    # Customize plot
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel('Average Regret MSE', fontsize=28, labelpad=15)
    ax.set_xscale('log')
    # ax.grid(True, alpha=0.3)
    ax.grid(True, which='major', linestyle='--', linewidth=1.0, alpha=0.4)
    ax.tick_params(axis='both', which='major', width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', width=1.0, length=4)

    
    # Clip y-axis to handle BCOT outliers and set reasonable lower bound
    y_clip_max = None
    if 'finite' in filters:
        first_robust_val = sorted(all_results.keys())[0]
        if 'finite' in all_results[first_robust_val]['regret']:
            finite_kf_regret = all_results[first_robust_val]['regret']['finite']
            if finite_kf_regret > 0:  # Only clip positive regret
                y_clip_max = finite_kf_regret * 1.5
                ax.set_ylim(top=y_clip_max)
                print(f"Clipping regret theta effect plot y-axis at {y_clip_max:.4f} (1.5x time-varying KF regret)")
    
    # Set lower bound slightly below 0 to avoid showing way below
    if y_clip_max is not None:
        y_clip_min = -0.1 * y_clip_max  # Show 10% below zero relative to clipped max
    else:
        y_clip_min = -0.5  # Default small negative range
    ax.set_ylim(bottom=y_clip_min)
    print(f"Setting regret theta effect plot y-axis bottom at {y_clip_min:.4f}")
    
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, frameon=False)
    
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.12, left=0.12, top=0.7, right=0.98)


    
    # Ensure results directory exists
    results_path = "./results/estimation/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, f'regret_theta_effect_{dist}_estimation.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Regret theta effect plot saved as: {output_path}")

def create_optimal_comparison_plot(optimal_results, dist, filter_labels):
    """Create bar plot comparing optimal MSE for each method"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    filters = list(optimal_results.keys())
    mse_vals = [optimal_results[filt]['mse'] for filt in filters]
    mse_stds = [optimal_results[filt]['mse_std'] for filt in filters]
    labels = [filter_labels[filt] for filt in filters]
    
    # Use global color function for consistent coloring
    
    # Create bar plot with custom colors
    bar_colors = [get_color_for_filter(filt, i) for i, filt in enumerate(filters)]
    bars = ax.bar(range(len(filters)), mse_vals, yerr=mse_stds, 
                  color=bar_colors, alpha=0.7, capsize=4)
    
    # Add optimal theta values as text on bars
    for i, (bar, filt) in enumerate(zip(bars, filters)):
        height = bar.get_height()
        theta_val = optimal_results[filt]['robust_val']
        ax.text(bar.get_x() + bar.get_width()/2., height + mse_stds[i],
                f'θ={theta_val}',
                ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(range(len(filters)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Optimal Mean Squared Error (MSE)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set 5 y-ticks starting from 0
    y_max = max(mse_vals) + max(mse_stds)
    ax.set_yticks(np.linspace(0, y_max * 1.1, 5))
    
    plt.tight_layout()
    
    # Ensure results directory exists
    results_path = "./results/estimation/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, f'optimal_comparison_{dist}_estimation.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Optimal comparison plot saved as: {output_path}")

def create_optimal_regret_comparison_plot(optimal_regret_results, dist, filter_labels):
    """Create bar plot comparing optimal regret for each method"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    filters = list(optimal_regret_results.keys())
    regret_vals = [optimal_regret_results[filt]['regret'] for filt in filters]
    regret_stds = [optimal_regret_results[filt]['regret_std'] for filt in filters]
    labels = [filter_labels[filt] for filt in filters]
    
    # Use global color function for consistent coloring
    
    # Create bar plot with custom colors
    bar_colors = [get_color_for_filter(filt, i) for i, filt in enumerate(filters)]
    bars = ax.bar(range(len(filters)), regret_vals, yerr=regret_stds, 
                  color=bar_colors, alpha=0.7, capsize=4)
    
    # Add optimal theta values as text on bars
    for i, (bar, filt) in enumerate(zip(bars, filters)):
        height = bar.get_height()
        theta_val = optimal_regret_results[filt]['robust_val']
        # Position text above or below bar depending on regret sign
        text_y = height + regret_stds[i] if height >= 0 else height - regret_stds[i]
        va = 'bottom' if height >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., text_y,
                f'θ={theta_val}',
                ha='center', va=va, fontsize=8)
    
    
    ax.set_xticks(range(len(filters)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Optimal Regret MSE')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set 5 y-ticks starting from 0 
    y_min = min(regret_vals) - max(regret_stds)
    y_max = max(regret_vals) + max(regret_stds) 
    ax.set_yticks(np.linspace(max(0, y_min), y_max * 1.1, 5))
    
    plt.tight_layout()
    
    # Ensure results directory exists
    results_path = "./results/estimation/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, f'optimal_regret_comparison_{dist}_estimation.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Optimal regret comparison plot saved as: {output_path}")

def main(dist):
    """Main function to create all plots"""
    
    results_path = "./results/estimation/"
    
    # Load results
    try:
        all_results = load_data(os.path.join(results_path, f'overall_results_{dist}_estimation.pkl'))
        optimal_results = load_data(os.path.join(results_path, f'optimal_results_{dist}_estimation.pkl'))
        optimal_regret_results = load_data(os.path.join(results_path, f'optimal_regret_results_{dist}_estimation.pkl'))
        raw_experiments_data = load_data(os.path.join(results_path, f'raw_experiments_{dist}_estimation.pkl'))
    except FileNotFoundError as e:
        print(f"Error: Could not find results file. Make sure you've run main5.py first.")
        print(f"Missing file: {e}")
        return
    
    # Get filters from the loaded results to match main5.py execution list
    available_filters = ['finite', 'inf', 'risk', 'risk_seek', 'drkf_neurips', 'bcot', 'drkf_finite_cdc', 'drkf_inf_cdc', 'drkf_finite', 'drkf_inf']
    # Only use filters that have results in the data
    filters = [f for f in available_filters if f in optimal_results]
    filter_labels = {
        'finite': "Time-varying KF",
        'inf': "Steady-state KF",
        'risk': "Risk-Sensitive Filter (risk-averse)",
        'risk_seek': "Risk-Sensitive Filter (risk-seeking)", 
        'drkf_neurips': "Time-varying DRKF [5]",
        'bcot': "Time-varying DRKF [10]",
        'drkf_finite_cdc': "Time-varying DRKF [12]",
        'drkf_inf_cdc': "Steady-state DRKF [12]",
        'drkf_finite': "Time-varying DRKF (ours)",
        'drkf_inf': "Steady-state DRKF (ours)"
    }
    
    print(f"Creating plots for {dist} distribution...")
    
    # Create all plots (original MSE-based plots)
    create_violin_plot(all_results, raw_experiments_data, optimal_results, dist, filters, filter_labels)
    create_theta_effect_plot(all_results, dist, filters, filter_labels)
    create_optimal_comparison_plot(optimal_results, dist, filter_labels)
    
    # Create regret-based plots
    create_regret_violin_plot(all_results, raw_experiments_data, optimal_regret_results, dist, filters, filter_labels)
    create_regret_theta_effect_plot(all_results, dist, filters, filter_labels)
    create_optimal_regret_comparison_plot(optimal_regret_results, dist, filter_labels)
    
    print(f"All plots (MSE and Regret) created successfully for {dist} distribution!")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # MSE-based rankings
    sorted_optimal = sorted(optimal_results.items(), key=lambda item: item[1]['mse'])
    print(f"\nRanking by optimal MSE ({dist} distribution):")
    for i, (filt, info) in enumerate(sorted_optimal, 1):
        print(f"{i:2d}. {filter_labels[filt]:<30} MSE: {info['mse']:.4f} (±{info['mse_std']:.4f}) θ: {info['robust_val']}")
    
    # Show improvement over baseline
    baseline_mse = sorted_optimal[0][1]['mse']  # Best performing method
    print(f"\nMSE comparison to best method ({filter_labels[sorted_optimal[0][0]]}):")
    for filt, info in sorted_optimal[1:]:
        improvement = ((info['mse'] - baseline_mse) / baseline_mse) * 100
        print(f"  {filter_labels[filt]:<30} +{improvement:5.1f}% worse")
    
    # Regret-based rankings
    sorted_optimal_regret = sorted(optimal_regret_results.items(), key=lambda item: item[1]['regret'])
    print(f"\nRanking by optimal Regret ({dist} distribution):")
    for i, (filt, info) in enumerate(sorted_optimal_regret, 1):
        print(f"{i:2d}. {filter_labels[filt]:<30} Regret: {info['regret']:.4f} (±{info['regret_std']:.4f}) θ: {info['robust_val']}")
    
    # Show regret comparison - baseline is always 0 (MMSE baseline has perfect regret)
    print(f"\nRegret comparison to MMSE Baseline (Regret = 0):")
    for filt, info in sorted_optimal_regret:
        regret_value = info['regret']
        print(f"  {filter_labels[filt]:<30} Regret: {regret_value:+6.4f}")
    
    # Show differences between MSE-optimal and Regret-optimal theta values
    print(f"\nComparison of optimal θ values (MSE vs Regret optimization):")
    print(f"{'Method':<30} {'MSE-optimal θ':<15} {'Regret-optimal θ':<15} {'Same?':<10}")
    print("-" * 70)
    for filt in filters:
        if filt in optimal_results and filt in optimal_regret_results:
            mse_theta = optimal_results[filt]['robust_val']
            regret_theta = optimal_regret_results[filt]['robust_val']
            same = "Yes" if mse_theta == regret_theta else "No"
            print(f"{filter_labels[filt]:<30} {str(mse_theta):<15} {str(regret_theta):<15} {same:<10}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create plots from main5.py results")
    parser.add_argument('--dist', default="normal", type=str,
                        help="Distribution type (normal or quadratic)")
    
    args = parser.parse_args()
    main(args.dist)