#!/usr/bin/env python3
"""
Plot showing the effect of N_data (EM training data size) on optimal regret MSE for each filter.
X-axis: N_data values [10, 20, 30, 40, 50]
Y-axis: Optimal Regret MSE
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os

plt.rcParams.update({
    'font.size': 18,           # Base font size
    'axes.titlesize': 22,      # Title font size
    'axes.labelsize': 26,      # Axis label font size
    'xtick.labelsize': 20,     # X-axis tick label size
    'ytick.labelsize': 20,     # Y-axis tick label size
    'legend.fontsize': 20,     # Legend font size
    'figure.titlesize': 24     # Figure title size
})

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

def create_ndata_effect_plot(all_ndata_results, dist, filters, filter_labels):
    """Create plot showing effect of N_data on optimal regret MSE for each filter"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract N_data values and sort them
    N_data_values = sorted(all_ndata_results.keys())
    
    # Define markers for each method
    markers = ['o', 's', '^', 'D', 'v', '<', '>', '>', 'o', 'o', '+', 'x']
    
    # Define fixed letter labels for each specific filter (one-to-one mapping)
    filter_letter_map = {
        'finite': '(A)',
        'inf': '(B)', 
        'risk': '(C)',
        'risk_seek': '(D)',
        'drkf_neurips': '(E)',
        'bcot': '(F)',
        'drkf_finite_cdc': '(G)',
        'drkf_inf_cdc': '(H)',
        'drkf_finite': '(I)',
        'drkf_inf': '(J)'
    }
    
    # Plot each filter
    for i, filt in enumerate(filters):
        # Determine line style based on filter type (finite versions get dotted lines)
        if filt in ['inf', 'drkf_inf', 'drkf_inf_cdc']:
            linestyle = ':'  # Dotted line for SS (infinite) versions
        else:
            linestyle = '-'  # Solid line for TV (finite) and other methods
        
        # Collect data points for this filter across N_data values
        ndata_vals = []
        regret_vals = []
        regret_stds = []
        
        for ndata in N_data_values:
            if ndata in all_ndata_results:
                optimal_regret_results = all_ndata_results[ndata]['optimal_regret_results']
                if filt in optimal_regret_results:
                    ndata_vals.append(ndata)
                    regret_vals.append(optimal_regret_results[filt]['regret'])
                    regret_stds.append(optimal_regret_results[filt]['regret_std'])
        
        # Skip this filter if no data points available
        if not regret_vals:
            print(f"Warning: No data available for filter '{filt}' - skipping from N_data effect plot")
            continue
        
        letter_label = filter_letter_map.get(filt, '(?)')  # Default to '(?)' if filter not in map
        label = f"{letter_label} {filter_labels[filt]}"
        
        # Plot with markers and error bars (optional)
        if filt in ['finite', 'inf']:
            # For non-robust methods, use simple line without markers
            ax.plot(ndata_vals, regret_vals, 
                    marker='None', 
                    color=get_color_for_filter(filt, i),
                    linestyle=linestyle,
                    linewidth=2.5,
                    label=label)
        else:
            # For robust methods, use markers
            ax.plot(ndata_vals, regret_vals, 
                    marker=markers[i % len(markers)], 
                    markerfacecolor='white',
                    markeredgecolor=get_color_for_filter(filt, i),
                    color=get_color_for_filter(filt, i),
                    markeredgewidth=1.2,
                    linestyle=linestyle,
                    linewidth=2.5,
                    markersize=12,
                    label=label)
        
        # Optional: Add error bars (confidence intervals)
        # cl = 0.2  # confidence level
        # ax.fill_between(ndata_vals,
        #     np.array(regret_vals) - cl*np.array(regret_stds),
        #     np.array(regret_vals) + cl*np.array(regret_stds),
        #     color=get_color_for_filter(filt, i),
        #     alpha=0.15
        # )
    
    # Customize plot
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Average Regret MSE', fontsize=28, labelpad=15)
    ax.grid(True, which='major', linestyle='--', linewidth=1.0, alpha=0.4)
    ax.tick_params(axis='both', which='major', width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', width=1.0, length=4)
    
    # Set x-axis to show only the N_data values we have
    ax.set_xticks(N_data_values)
    
    # Set x-axis limits with appropriate padding based on the range
    x_min, x_max = min(N_data_values), max(N_data_values)
    x_range = x_max - x_min
    
    if x_min <= 5:  # For small values, use fixed padding
        padding = 1
    else:  # For larger ranges, use proportional padding
        padding = max(1, x_range * 0.05)
    
    ax.set_xlim(max(0, x_min - padding), x_max + padding)
    
    # Clip y-axis to preserve scale without BCOT outliers
    # Calculate reasonable upper bound excluding BCOT at N=5
    reasonable_regret_values = []
    for ndata in N_data_values:
        if ndata in all_ndata_results:
            optimal_regret_results = all_ndata_results[ndata]['optimal_regret_results']
            for filt in filters:
                if filt in optimal_regret_results:
                    # Skip BCOT at N=5 for scale calculation
                    if filt == 'bcot' and ndata == 5:
                        continue
                    reasonable_regret_values.append(optimal_regret_results[filt]['regret'])
    
    # Set fixed y-axis limits for regret plot
    ax.set_ylim(bottom=-0.2, top=3.0)
    
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, frameon=False)
    
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.12, left=0.12, top=0.7, right=0.98)
    
    # Ensure results directory exists
    results_path = f"./results/ndata_study/{dist}/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, f'ndata_effect_regret_{dist}.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"N_data effect plot saved as: {output_path}")

def create_ndata_effect_mse_plot(all_ndata_results, dist, filters, filter_labels):
    """Create plot showing effect of N_data on optimal MSE for each filter"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract N_data values and sort them
    N_data_values = sorted(all_ndata_results.keys())
    
    # Define markers for each method
    markers = ['o', 's', '^', 'D', 'v', '<', '>', '>', 'o', 'o', '+', 'x']
    
    # Define fixed letter labels for each specific filter (one-to-one mapping)
    filter_letter_map = {
        'finite': '(A)',
        'inf': '(B)', 
        'risk': '(C)',
        'risk_seek': '(D)',
        'drkf_neurips': '(E)',
        'bcot': '(F)',
        'drkf_finite_cdc': '(G)',
        'drkf_inf_cdc': '(H)',
        'drkf_finite': '(I)',
        'drkf_inf': '(J)'
    }
    
    # Plot each filter
    for i, filt in enumerate(filters):
        # Determine line style based on filter type (infinite versions get dotted lines)
        if filt in ['inf', 'drkf_inf', 'drkf_inf_cdc']:
            linestyle = ':'  # Dotted line for SS (infinite) versions
        else:
            linestyle = '-'  # Solid line for TV (finite) and other methods
        
        # Collect data points for this filter across N_data values
        ndata_vals = []
        mse_vals = []
        mse_stds = []
        
        for ndata in N_data_values:
            if ndata in all_ndata_results:
                optimal_results = all_ndata_results[ndata]['optimal_results']
                if filt in optimal_results:
                    ndata_vals.append(ndata)
                    mse_vals.append(optimal_results[filt]['mse'])
                    mse_stds.append(optimal_results[filt]['mse_std'])
        
        # Skip this filter if no data points available
        if not mse_vals:
            print(f"Warning: No data available for filter '{filt}' - skipping from N_data MSE effect plot")
            continue
        
        letter_label = filter_letter_map.get(filt, '(?)')  # Default to '(?)' if filter not in map
        label = f"{letter_label} {filter_labels[filt]}"
        
        # Plot with markers
        if filt in ['finite', 'inf']:
            # For non-robust methods, use simple line without markers
            ax.plot(ndata_vals, mse_vals, 
                    marker='None', 
                    color=get_color_for_filter(filt, i),
                    linestyle=linestyle,
                    linewidth=2.5,
                    label=label)
        else:
            # For robust methods, use markers
            ax.plot(ndata_vals, mse_vals, 
                    marker=markers[i % len(markers)], 
                    markerfacecolor='white',
                    markeredgecolor=get_color_for_filter(filt, i),
                    color=get_color_for_filter(filt, i),
                    markeredgewidth=1.2,
                    linestyle=linestyle,
                    linewidth=2.5,
                    markersize=12,
                    label=label)
    
    # Customize plot
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Average MSE', fontsize=28, labelpad=15)
    ax.set_yscale('log')  # Use log scale for MSE as in original plots
    ax.grid(True, which='major', linestyle='--', linewidth=1.0, alpha=0.4)
    ax.tick_params(axis='both', which='major', width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', width=1.0, length=4)
    
    # Set x-axis to show only the N_data values we have
    ax.set_xticks(N_data_values)
    
    # Set x-axis limits with appropriate padding based on the range
    x_min, x_max = min(N_data_values), max(N_data_values)
    x_range = x_max - x_min
    
    if x_min <= 5:  # For small values, use fixed padding
        padding = 1
    else:  # For larger ranges, use proportional padding
        padding = max(1, x_range * 0.05)
    
    ax.set_xlim(max(0, x_min - padding), x_max + padding)
    
    # Clip y-axis to preserve scale without BCOT outliers
    # Calculate reasonable upper bound excluding BCOT at N=5
    reasonable_mse_values = []
    for ndata in N_data_values:
        if ndata in all_ndata_results:
            optimal_results = all_ndata_results[ndata]['optimal_results']
            for filt in filters:
                if filt in optimal_results:
                    # Skip BCOT at N=5 for scale calculation
                    if filt == 'bcot' and ndata == 5:
                        continue
                    reasonable_mse_values.append(optimal_results[filt]['mse'])
    
    if reasonable_mse_values:
        max_reasonable = max(reasonable_mse_values)
        # Add 10% padding to the reasonable max
        y_upper_limit = max_reasonable * 1.1
        # Set both upper and lower bounds for MSE plot
        ax.set_ylim(bottom=7e-2, top=y_upper_limit)
    
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, frameon=False)
    
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.12, left=0.12, top=0.7, right=0.98)
    
    # Ensure results directory exists
    results_path = f"./results/ndata_study/{dist}/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, f'ndata_effect_mse_{dist}.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"N_data MSE effect plot saved as: {output_path}")

def load_combined_ndata_results(dist, ndata_values=None):
    """Load and combine N_data results from multiple experiments
    
    Args:
        dist: Distribution type ('normal' or 'quadratic')
        ndata_values: List of specific N_data values to load (e.g., [2, 5, 10, 20, 30, 40, 50])
                     If None, loads all available N_data values
    """
    results_path = f"./results/ndata_study/{dist}/"
    
    # Try to load individual N_data files and combined files
    all_ndata_results = {}
    
    # First, try to load the main combined file
    main_file = os.path.join(results_path, f'all_ndata_results_{dist}.pkl')
    if os.path.exists(main_file):
        try:
            main_results = load_data(main_file)
            # Filter to only include requested N_data values if specified
            if ndata_values is not None:
                filtered_results = {k: v for k, v in main_results.items() if k in ndata_values}
                all_ndata_results.update(filtered_results)
                print(f"Loaded main N_data results (filtered): {sorted(filtered_results.keys())}")
            else:
                all_ndata_results.update(main_results)
                print(f"Loaded main N_data results (all): {sorted(main_results.keys())}")
        except Exception as e:
            print(f"Error loading main file {main_file}: {e}")
    
    # Determine which N_data values to load
    if ndata_values is None:
        # Auto-discover all available N_data values
        additional_ndata_values = []
        if os.path.exists(results_path):
            import glob
            ndata_folders = glob.glob(os.path.join(results_path, "N_data_*"))
            for folder in ndata_folders:
                try:
                    ndata_val = int(os.path.basename(folder).split('_')[-1])
                    additional_ndata_values.append(ndata_val)
                except ValueError:
                    continue
        print(f"Auto-discovered N_data values: {sorted(additional_ndata_values)}")
    else:
        # Use explicitly provided N_data values
        additional_ndata_values = ndata_values
        print(f"Loading specified N_data values: {sorted(additional_ndata_values)}")
    
    # Try to load each specified N_data value
    for ndata in additional_ndata_values:
        ndata_folder = os.path.join(results_path, f"N_data_{ndata}")
        if os.path.exists(ndata_folder):
            try:
                # Load individual result files from this N_data folder
                overall_file = os.path.join(ndata_folder, f'overall_results_{dist}_N_{ndata}.pkl')
                optimal_file = os.path.join(ndata_folder, f'optimal_results_{dist}_N_{ndata}.pkl')
                optimal_regret_file = os.path.join(ndata_folder, f'optimal_regret_results_{dist}_N_{ndata}.pkl')
                raw_file = os.path.join(ndata_folder, f'raw_experiments_{dist}_N_{ndata}.pkl')
                
                if all(os.path.exists(f) for f in [overall_file, optimal_file, optimal_regret_file, raw_file]):
                    all_results = load_data(overall_file)
                    optimal_results = load_data(optimal_file)
                    optimal_regret_results = load_data(optimal_regret_file)
                    raw_experiments_data = load_data(raw_file)
                    
                    # Add this N_data to the combined results
                    all_ndata_results[ndata] = {
                        'all_results': all_results,
                        'optimal_results': optimal_results,
                        'optimal_regret_results': optimal_regret_results,
                        'raw_experiments_data': raw_experiments_data
                    }
                    print(f"Loaded additional N_data = {ndata} results")
                else:
                    print(f"Some files missing for N_data = {ndata}, skipping")
            except Exception as e:
                print(f"Error loading N_data = {ndata}: {e}")
    
    # Try to load from alternative locations if needed
    # Look for any additional combined result files
    import glob
    additional_files = glob.glob(os.path.join(results_path, f'*ndata*{dist}*.pkl'))
    for file_path in additional_files:
        if 'all_ndata_results' not in file_path:  # Skip main file we already loaded
            try:
                additional_results = load_data(file_path)
                if isinstance(additional_results, dict):
                    # Check if this looks like N_data results
                    for key, value in additional_results.items():
                        if isinstance(key, (int, float)) and isinstance(value, dict):
                            if key not in all_ndata_results:
                                all_ndata_results[key] = value
                                print(f"Loaded N_data = {key} from {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error loading additional file {file_path}: {e}")
    
    if not all_ndata_results:
        raise FileNotFoundError(f"No N_data results found in {results_path}. Make sure you've run main0_numsample.py first.")
    
    return all_ndata_results

def create_combined_ndata_effect_plot(normal_data, quadratic_data, filters, filter_labels):
    """Create side-by-side plots showing N_data effect on regret for normal and quadratic distributions"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), gridspec_kw={'wspace': 0.25})
    
    # Function to create N_data effect plot for a single distribution
    def plot_distribution_ndata_effect(ax, all_ndata_results, dist_name):
        # Extract N_data values and sort them
        N_data_values = sorted(all_ndata_results.keys())
        
        # Define markers for each method
        markers = ['o', 's', '^', 'D', 'v', '<', '>', '>', 'o', 'o', '+', 'x']
        
        # Define fixed letter labels for each specific filter (one-to-one mapping)
        filter_letter_map = {
            'finite': '(A)',
            'inf': '(B)', 
            'risk': '(C)',
            'risk_seek': '(D)',
            'drkf_neurips': '(E)',
            'bcot': '(F)',
            'drkf_finite_cdc': '(G)',
            'drkf_inf_cdc': '(H)',
            'drkf_finite': '(I)',
            'drkf_inf': '(J)'
        }
        
        # Plot each filter
        for i, filt in enumerate(filters):
            # Determine line style based on filter type (finite versions get dotted lines)
            if filt in ['inf', 'drkf_inf', 'drkf_inf_cdc']:
                linestyle = ':'  # Dotted line for SS (infinite) versions
            else:
                linestyle = '-'  # Solid line for TV (finite) and other methods
            
            # Collect data points for this filter across N_data values
            ndata_vals = []
            regret_vals = []
            regret_stds = []
            
            for ndata in N_data_values:
                if ndata in all_ndata_results:
                    optimal_regret_results = all_ndata_results[ndata]['optimal_regret_results']
                    if filt in optimal_regret_results:
                        ndata_vals.append(ndata)
                        regret_vals.append(optimal_regret_results[filt]['regret'])
                        regret_stds.append(optimal_regret_results[filt]['regret_std'])
            
            # Skip this filter if no data points available
            if not regret_vals:
                print(f"Warning: No data available for filter '{filt}' in {dist_name} - skipping")
                continue
            
            letter_label = filter_letter_map.get(filt, '(?)')  # Default to '(?)' if filter not in map
            label = f"{letter_label} {filter_labels[filt]}"
            
            # Plot with markers
            if filt in ['finite', 'inf']:
                # For non-robust methods, use simple line without markers
                ax.plot(ndata_vals, regret_vals, 
                        marker='None', 
                        color=get_color_for_filter(filt, i),
                        linestyle=linestyle,
                        linewidth=2.5,
                        label=label)
            else:
                # For robust methods, use markers
                ax.plot(ndata_vals, regret_vals, 
                        marker=markers[i % len(markers)], 
                        markerfacecolor='white',
                        markeredgecolor=get_color_for_filter(filt, i),
                        color=get_color_for_filter(filt, i),
                        markeredgewidth=1.2,
                        linestyle=linestyle,
                        linewidth=2.5,
                        markersize=12,
                        label=label)
        
        # Customize plot
        ax.set_xlabel('Number of samples')
        ax.set_ylabel('Average Regret MSE', fontsize=28, labelpad=15)
        ax.grid(True, which='major', linestyle='--', linewidth=1.0, alpha=0.4)
        ax.tick_params(axis='both', which='major', width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', width=1.0, length=4)
        
        # Set x-axis to show only the N_data values we have
        ax.set_xticks(N_data_values)
        
        # Set x-axis limits with appropriate padding based on the range
        x_min, x_max = min(N_data_values), max(N_data_values)
        x_range = x_max - x_min
        
        if x_min <= 5:  # For small values, use fixed padding
            padding = 1
        else:  # For larger ranges, use proportional padding
            padding = max(1, x_range * 0.05)
        
        ax.set_xlim(max(0, x_min - padding), x_max + padding)
        
        # Clip y-axis to preserve scale without BCOT outliers
        # Calculate reasonable upper bound excluding BCOT at N=5
        reasonable_regret_values = []
        for ndata in N_data_values:
            if ndata in all_ndata_results:
                optimal_regret_results = all_ndata_results[ndata]['optimal_regret_results']
                for filt in filters:
                    if filt in optimal_regret_results:
                        # Skip BCOT at N=5 for scale calculation
                        if filt == 'bcot' and ndata == 5:
                            continue
                        reasonable_regret_values.append(optimal_regret_results[filt]['regret'])
        
        # Set fixed y-axis limits for regret plot
        ax.set_ylim(bottom=-0.2, top=3.0)
    
    # Plot normal distribution on left
    plot_distribution_ndata_effect(ax1, normal_data, 'normal')
    
    # Plot quadratic distribution on right  
    plot_distribution_ndata_effect(ax2, quadratic_data, 'quadratic')
    
    # Add subplot labels a) and b)
    ax1.text(0.5, -0.25, 'a)', transform=ax1.transAxes, fontsize=24, ha='center', va='top')
    ax2.text(0.5, -0.25, 'b)', transform=ax2.transAxes, fontsize=24, ha='center', va='top')
    
    # Create a shared legend at the top in two rows
    handles, labels = ax1.get_legend_handles_labels()
    # Calculate number of columns to create 2 rows
    ncol = (len(labels) + 1) // 2  # Round up division to get columns for 2 rows
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.98), loc='upper center', ncol=ncol, frameon=False, fontsize=21)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82, bottom=0.25)
    
    # Ensure results directory exists
    results_path = "./results/ndata_study/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, 'combined_ndata_effect_regret_normal_quadratic.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined N_data effect plot saved as: {output_path}")

def create_combined_ndata_effect_mse_plot(normal_data, quadratic_data, filters, filter_labels):
    """Create side-by-side plots showing N_data effect on MSE for normal and quadratic distributions"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), gridspec_kw={'wspace': 0.25})
    
    # Function to create N_data effect MSE plot for a single distribution
    def plot_distribution_ndata_mse_effect(ax, all_ndata_results, dist_name):
        # Extract N_data values and sort them
        N_data_values = sorted(all_ndata_results.keys())
        
        # Define markers for each method
        markers = ['o', 's', '^', 'D', 'v', '<', '>', '>', 'o', 'o', '+', 'x']
        
        # Define fixed letter labels for each specific filter (one-to-one mapping)
        filter_letter_map = {
            'finite': '(A)',
            'inf': '(B)', 
            'risk': '(C)',
            'risk_seek': '(D)',
            'drkf_neurips': '(E)',
            'bcot': '(F)',
            'drkf_finite_cdc': '(G)',
            'drkf_inf_cdc': '(H)',
            'drkf_finite': '(I)',
            'drkf_inf': '(J)'
        }
        
        # Plot each filter
        for i, filt in enumerate(filters):
            # Determine line style based on filter type (finite versions get dotted lines)
            if filt in ['finite', 'drkf_finite', 'drkf_finite_cdc']:
                linestyle = ':'  # Dotted line for TV (finite) versions
            else:
                linestyle = '-'  # Solid line for SS (inf) and other methods
            
            # Collect data points for this filter across N_data values
            ndata_vals = []
            mse_vals = []
            mse_stds = []
            
            for ndata in N_data_values:
                if ndata in all_ndata_results:
                    optimal_results = all_ndata_results[ndata]['optimal_results']
                    if filt in optimal_results:
                        ndata_vals.append(ndata)
                        mse_vals.append(optimal_results[filt]['mse'])
                        mse_stds.append(optimal_results[filt]['mse_std'])
            
            # Skip this filter if no data points available
            if not mse_vals:
                print(f"Warning: No data available for filter '{filt}' in {dist_name} - skipping")
                continue
            
            letter_label = filter_letter_map.get(filt, '(?)')  # Default to '(?)' if filter not in map
            label = f"{letter_label} {filter_labels[filt]}"
            
            # Plot with markers
            if filt in ['finite', 'inf']:
                # For non-robust methods, use simple line without markers
                ax.plot(ndata_vals, mse_vals, 
                        marker='None', 
                        color=get_color_for_filter(filt, i),
                        linestyle=linestyle,
                        linewidth=2.5,
                        label=label)
            else:
                # For robust methods, use markers
                ax.plot(ndata_vals, mse_vals, 
                        marker=markers[i % len(markers)], 
                        markerfacecolor='white',
                        markeredgecolor=get_color_for_filter(filt, i),
                        color=get_color_for_filter(filt, i),
                        markeredgewidth=1.2,
                        linestyle=linestyle,
                        linewidth=2.5,
                        markersize=12,
                        label=label)
        
        # Customize plot
        ax.set_xlabel('Number of samples')
        ax.set_ylabel('Average MSE', fontsize=28, labelpad=15)
        # ax.set_yscale('log')  # Use log scale for MSE as in original plots
        ax.grid(True, which='major', linestyle='--', linewidth=1.0, alpha=0.4)
        ax.tick_params(axis='both', which='major', width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', width=1.0, length=4)
        
        # Set x-axis to show only the N_data values we have
        ax.set_xticks(N_data_values)
        
        # Set x-axis limits with appropriate padding based on the range
        x_min, x_max = min(N_data_values), max(N_data_values)
        x_range = x_max - x_min
        
        if x_min <= 5:  # For small values, use fixed padding
            padding = 1
        else:  # For larger ranges, use proportional padding
            padding = max(1, x_range * 0.05)
        
        ax.set_xlim(max(0, x_min - padding), x_max + padding)
        
        # Clip y-axis to preserve scale without BCOT outliers
        # Calculate reasonable upper bound excluding BCOT at N=5
        reasonable_mse_values = []
        for ndata in N_data_values:
            if ndata in all_ndata_results:
                optimal_results = all_ndata_results[ndata]['optimal_results']
                for filt in filters:
                    if filt in optimal_results:
                        # Skip BCOT at N=5 for scale calculation
                        if filt == 'bcot' and ndata == 5:
                            continue
                        reasonable_mse_values.append(optimal_results[filt]['mse'])
        
        if reasonable_mse_values:
            max_reasonable = max(reasonable_mse_values)
            # Add 10% padding to the reasonable max
            y_upper_limit = max_reasonable * 1.1
            # Set both upper and lower bounds for MSE plot
            ax.set_ylim(bottom=7e-2, top=y_upper_limit)
    
    # Plot normal distribution on left
    plot_distribution_ndata_mse_effect(ax1, normal_data, 'normal')
    
    # Plot quadratic distribution on right
    plot_distribution_ndata_mse_effect(ax2, quadratic_data, 'quadratic')
    
    # Add subplot labels a) and b)
    ax1.text(0.5, -0.25, 'a)', transform=ax1.transAxes, fontsize=24, ha='center', va='top')
    ax2.text(0.5, -0.25, 'b)', transform=ax2.transAxes, fontsize=24, ha='center', va='top')
    
    # Create a shared legend at the top in two rows
    handles, labels = ax1.get_legend_handles_labels()
    # Calculate number of columns to create 2 rows
    ncol = (len(labels) + 1) // 2  # Round up division to get columns for 2 rows
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.98), loc='upper center', ncol=ncol, frameon=False, fontsize=21)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82, bottom=0.25)
    
    # Ensure results directory exists
    results_path = "./results/ndata_study/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, 'combined_ndata_effect_mse_normal_quadratic.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined N_data effect MSE plot saved as: {output_path}")

def create_combined_plots():
    """Create combined plots comparing normal and quadratic distributions"""
    
    # Determine which N_data values to load (same as main function)
    ndata_values = [5, 10, 20, 30, 40, 50]
    
    # Load data for both distributions
    try:
        # Normal distribution data
        normal_data = load_combined_ndata_results('normal', ndata_values)
        
        # Quadratic distribution data
        quadratic_data = load_combined_ndata_results('quadratic', ndata_values)
    except FileNotFoundError as e:
        print(f"Error: Could not find results files for both distributions.")
        print(f"Missing file: {e}")
        print("Make sure you've run main0_numsample.py for both normal and quadratic distributions.")
        return
    
    # Get filters from the loaded results
    available_filters = ['finite', 'inf', 'risk', 'risk_seek', 'drkf_neurips', 'bcot', 'drkf_finite_cdc', 'drkf_inf_cdc', 'drkf_finite', 'drkf_inf']
    
    # Use filters that are available in both distributions
    filters = []
    for f in available_filters:
        found_normal = False
        found_quadratic = False
        for ndata_res in normal_data.values():
            if f in ndata_res['optimal_results'] or f in ndata_res['optimal_regret_results']:
                found_normal = True
                break
        for ndata_res in quadratic_data.values():
            if f in ndata_res['optimal_results'] or f in ndata_res['optimal_regret_results']:
                found_quadratic = True
                break
        if found_normal and found_quadratic:
            filters.append(f)
    
    filter_labels = {
        'finite': "Time-varying KF",
        'inf': "Steady-state KF",
        'risk': "Risk-Sensitive Filter (risk-averse)",
        'risk_seek': "Risk-Sensitive Filter (risk-seeking)", 
        'drkf_neurips': "Time-varying DRKF [16]",
        'bcot': "Time-varying DRKF [19]",
        'drkf_finite_cdc': "Time-varying DRKF [21]",
        'drkf_inf_cdc': "Steady-state DRKF [21]",
        'drkf_finite': "Time-varying DRKF (ours)",
        'drkf_inf': "Steady-state DRKF (ours)"
    }
    
    print("Creating combined N_data effect plots for normal vs quadratic distributions...")
    
    # Create the combined N_data effect plots
    create_combined_ndata_effect_plot(normal_data, quadratic_data, filters, filter_labels)
    create_combined_ndata_effect_mse_plot(normal_data, quadratic_data, filters, filter_labels)
    
    print("Combined plots created successfully!")

def main(dist):
    """Main function to create N_data effect plots
    
    Args:
        dist: Distribution type ('normal' or 'quadratic')
    """
    
    # ============================================================================
    # EDIT THIS LIST TO SPECIFY WHICH N_DATA VALUES TO PLOT
    # ============================================================================
    # Explicitly specify which N_data experiments to combine and plot
    # Edit this list as needed:
    ndata_values = [5, 10, 20, 30, 40, 50]
    
    
    print(f"Using N_data values: {ndata_values}")
    
    # Load combined N_data results from multiple sources
    try:
        all_ndata_results = load_combined_ndata_results(dist, ndata_values)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Get filters from the first N_data result to match main0_numsample.py execution list
    first_ndata = list(all_ndata_results.keys())[0]
    available_filters = ['finite', 'inf', 'risk', 'risk_seek', 'drkf_neurips', 'bcot', 'drkf_finite_cdc', 'drkf_inf_cdc', 'drkf_finite', 'drkf_inf']
    #available_filters = ['finite', 'inf', 'risk', 'risk_seek', 'drkf_neurips', 'drkf_finite_cdc', 'drkf_inf_cdc', 'drkf_finite', 'drkf_inf']
    # Only use filters that have results in at least one N_data
    filters = []
    for f in available_filters:
        found = False
        for ndata_res in all_ndata_results.values():
            if f in ndata_res['optimal_results'] or f in ndata_res['optimal_regret_results']:
                found = True
                break
        if found:
            filters.append(f)
    
    filter_labels = {
        'finite': "Time-varying KF",
        'inf': "Steady-state KF",
        'risk': "Risk-Sensitive Filter (risk-averse)",
        'risk_seek': "Risk-Sensitive Filter (risk-seeking)", 
        'drkf_neurips': "Time-varying DRKF [16]",
        'bcot': "Time-varying DRKF [19]",
        'drkf_finite_cdc': "Time-varying DRKF [21]",
        'drkf_inf_cdc': "Steady-state DRKF [21]",
        'drkf_finite': "Time-varying DRKF (ours)",
        'drkf_inf': "Steady-state DRKF (ours)"
    }
    
    print(f"Creating N_data effect plots for {dist} distribution...")
    print(f"Available filters: {filters}")
    print(f"N_data values: {sorted(all_ndata_results.keys())}")
    
    # Create N_data effect plots
    create_ndata_effect_plot(all_ndata_results, dist, filters, filter_labels)
    create_ndata_effect_mse_plot(all_ndata_results, dist, filters, filter_labels)
    
    print(f"N_data effect plots created successfully for {dist} distribution!")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("N_DATA EFFECT SUMMARY STATISTICS")
    print("="*80)
    
    N_data_values = sorted(all_ndata_results.keys())
    
    print(f"\nEffect of N_data on Optimal Regret ({dist} distribution):")
    print(f"{'Filter':<35} " + " ".join([f"N={n:<8}" for n in N_data_values]))
    print("-" * (35 + 10 * len(N_data_values)))
    
    for filt in filters:
        regret_str = f"{filter_labels[filt]:<35}"
        for ndata in N_data_values:
            if ndata in all_ndata_results:
                opt_regret = all_ndata_results[ndata]['optimal_regret_results']
                if filt in opt_regret:
                    regret_val = opt_regret[filt]['regret']
                    regret_str += f" {regret_val:<8.4f}"
                else:
                    regret_str += f" {'N/A':<8}"
            else:
                regret_str += f" {'N/A':<8}"
        print(regret_str)
    
    print(f"\nEffect of N_data on Optimal MSE ({dist} distribution):")
    print(f"{'Filter':<35} " + " ".join([f"N={n:<8}" for n in N_data_values]))
    print("-" * (35 + 10 * len(N_data_values)))
    
    for filt in filters:
        mse_str = f"{filter_labels[filt]:<35}"
        for ndata in N_data_values:
            if ndata in all_ndata_results:
                opt_mse = all_ndata_results[ndata]['optimal_results']
                if filt in opt_mse:
                    mse_val = opt_mse[filt]['mse']
                    mse_str += f" {mse_val:<8.4f}"
                else:
                    mse_str += f" {'N/A':<8}"
            else:
                mse_str += f" {'N/A':<8}"
        print(mse_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create N_data effect plots from main0_numsample.py results")
    parser.add_argument('--dist', default="normal", type=str,
                        help="Distribution type (normal or quadratic)")
    parser.add_argument('--combined', action='store_true',
                        help="Create combined plots comparing normal vs quadratic distributions")
    
    args = parser.parse_args()
    
    if args.combined:
        create_combined_plots()
    else:
        main(args.dist)