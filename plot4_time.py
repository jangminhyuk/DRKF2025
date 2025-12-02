#!/usr/bin/env python3
"""
Computation time comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
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
#         return 'black'
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

def create_computation_time_plot():
    """Create computation time vs time horizon plot"""
    
    results_path = "./results/time/"
    
    try:
        # Load results
        avg_execution_times = load_data(os.path.join(results_path, 'avg_execution_times.pkl'))
        filter_labels = load_data(os.path.join(results_path, 'filter_labels.pkl'))
        filter_max_T = load_data(os.path.join(results_path, 'filter_max_T.pkl'))
        
        # Update filter labels to match plot5_1.py format if needed
        updated_filter_labels = {
            'finite': "Time-varying KF",
            'inf': "Steady-state KF",
            'risk': "Risk-Sensitive Filter (risk-averse)",
            'risk_seek': "Risk-Sensitive Filter (risk-seeking)", 
            'drkf_neurips': "Time-varying DRKF [9]",
            'bcot': "Time-varying DRKF [12]",
            'drkf_finite_cdc': "Time-varying DRKF [14]",
            'drkf_inf_cdc': "Steady-state DRKF [14]",
            'drkf_finite': "Time-varying DRKF (ours)",
            'drkf_inf': "Steady-state DRKF (ours)"
        }
        
        # Use updated labels where available, keep original otherwise
        for key in filter_labels:
            if key in updated_filter_labels:
                filter_labels[key] = updated_filter_labels[key]
                
    except FileNotFoundError as e:
        print(f"Error: Could not find results file. Make sure you've run main5_time.py first.")
        print(f"Missing file: {e}")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Define markers for each method (matching plot1_with_MPC.py MPC cost theta effect style)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', '>', 'o', 'o', '+', 'x']

    # Define standard filter order and corresponding letters (matching plot1_with_MPC.py)
    standard_filter_order = ['finite', 'inf', 'risk', 'risk_seek', 'drkf_neurips', 'bcot', 'drkf_finite_cdc', 'drkf_inf_cdc', 'drkf_finite', 'drkf_inf']
    letter_labels_map = {
        'finite': '(A)', 'inf': '(B)', 'risk': '(C)', 'risk_seek': '(D)', 
        'drkf_neurips': '(E)', 'bcot': '(F)', 'drkf_finite_cdc': '(G)', 
        'drkf_inf_cdc': '(H)', 'drkf_finite': '(I)', 'drkf_inf': '(J)'
    }
    
    # Get all unique T values across all filters
    all_T_values = set()
    for filter_name in avg_execution_times:
        all_T_values.update(avg_execution_times[filter_name].keys())
    sorted_T_values = sorted(all_T_values)
    
    # Plot each filter
    filter_names = list(filter_labels.keys())
    for i, filter_name in enumerate(filter_names):
        if filter_name not in avg_execution_times or not avg_execution_times[filter_name]:
            continue
            
        # Extract T values and corresponding mean times for this filter
        T_vals = []
        mean_times = []
        
        for T in sorted_T_values:
            if T in avg_execution_times[filter_name]:
                T_vals.append(T)
                mean_times.append(avg_execution_times[filter_name][T]['mean'])
        
        if not T_vals:  # Skip if no data
            continue
            
        # Determine line style based on filter type (finite versions get dotted lines)
        if filter_name in ['inf', 'drkf_inf', 'drkf_inf_cdc']:
            linestyle = ':'  # Dotted line for SS (infinite) versions
        else:
            linestyle = '-'  # Solid line for TV (fininte) and other methods
            
        # Create label with correct letter prefix based on filter name
        letter_prefix = letter_labels_map.get(filter_name, f'({chr(65+i)})')  # Fallback to position-based if not found
        label = f"{letter_prefix} {filter_labels[filter_name]}"
        
        # Get marker index based on standard order position
        marker_idx = standard_filter_order.index(filter_name) if filter_name in standard_filter_order else i
        
        # Plot the data
        ax.plot(T_vals, mean_times,
                marker=markers[marker_idx % len(markers)],
                markerfacecolor='white',
                markeredgecolor=get_color_for_filter(filter_name, marker_idx),
                color=get_color_for_filter(filter_name, marker_idx),
                markeredgewidth=1.2,
                linestyle=linestyle,
                linewidth=2.5,
                markersize=12,
                label=label)
        
    # Customize plot
    ax.set_xlabel('T', fontsize=26)
    ax.set_ylabel('Computation Time (sec.)', fontsize=28)
    ax.set_yscale('log')  # Use log scale for y-axis

    ax.grid(True, which='major', linestyle='--', linewidth=1.0, alpha=0.4)
    ax.tick_params(axis='both', which='major', width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', width=1.0, length=4)

    
    # Set reasonable axis limits
    ax.set_xlim(left=0, right=105)
    
    # Legend
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, frameon=False)
    
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.12, left=0.12, top=0.7, right=0.98)

    
    # Save the plot
    output_path = os.path.join(results_path, 'computation_time_comparison.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Computation time plot saved as: {output_path}")
    
    return avg_execution_times, filter_labels, filter_max_T

def print_summary_statistics(avg_execution_times, filter_labels, filter_max_T):
    """Print detailed summary statistics"""
    
    print("\n" + "="*80)
    print("COMPUTATION TIME COMPARISON RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nFilter Methods Analyzed:")
    for filter_name, label in filter_labels.items():
        max_T_tested = max(avg_execution_times[filter_name].keys()) if avg_execution_times[filter_name] else 0
        timeout_status = f"(TIMEOUT at T={filter_max_T[filter_name]})" if filter_max_T[filter_name] < 100 else "(Completed all T)"
        print(f"  • {label}: Tested up to T = {max_T_tested} {timeout_status}")
    
    # Find common T values for fair comparison
    common_T_values = None
    for filter_name in filter_labels:
        filter_T_values = set(avg_execution_times[filter_name].keys()) if avg_execution_times[filter_name] else set()
        if common_T_values is None:
            common_T_values = filter_T_values
        else:
            common_T_values = common_T_values.intersection(filter_T_values)
    
    common_T_sorted = sorted(common_T_values) if common_T_values else []
    print(f"\nCommon time horizons tested by all filters: {len(common_T_sorted)} values")
    if common_T_sorted:
        print(f"Range: T = {min(common_T_sorted)} to T = {max(common_T_sorted)}")
    
    # Performance ranking at different T values
    if common_T_sorted:
        print(f"\nPerformance Rankings at Selected Time Horizons:")
        selected_T = [T for T in [10, 20, 30, 50] if T in common_T_sorted]
        
        for T in selected_T:
            print(f"\n  At T = {T}:")
            filter_times = []
            for filter_name in filter_labels:
                if T in avg_execution_times[filter_name]:
                    mean_time = avg_execution_times[filter_name][T]['mean']
                    filter_times.append((filter_name, mean_time))
            
            # Sort by execution time
            filter_times.sort(key=lambda x: x[1])
            
            for rank, (filter_name, exec_time) in enumerate(filter_times, 1):
                print(f"    {rank}. {filter_labels[filter_name]}: {exec_time:.4f}s")
    
    # Overall average performance ranking
    print(f"\nOverall Average Performance Ranking:")
    overall_avg = {}
    for filter_name in filter_labels:
        if avg_execution_times[filter_name]:
            all_times = [stats['mean'] for stats in avg_execution_times[filter_name].values()]
            overall_avg[filter_name] = np.mean(all_times)
            T_count = len(all_times)
            print(f"  {filter_labels[filter_name]}: {overall_avg[filter_name]:.4f}s average (across {T_count} time horizons)")
        else:
            overall_avg[filter_name] = float('inf')
            print(f"  {filter_labels[filter_name]}: No data available")
    
    # Sort and display ranking
    sorted_filters = sorted(overall_avg.items(), key=lambda x: x[1])
    print(f"\nFinal Ranking by Average Computation Time:")
    for rank, (filter_name, avg_time) in enumerate(sorted_filters, 1):
        if avg_time != float('inf'):
            print(f"  {rank}. {filter_labels[filter_name]}: {avg_time:.4f}s average")
        else:
            print(f"  {rank}. {filter_labels[filter_name]}: Failed to complete")
    
    # Scalability analysis
    print(f"\nScalability Analysis (Time Growth with Horizon):")
    for filter_name in filter_labels:
        if len(avg_execution_times[filter_name]) >= 3:  # Need at least 3 points
            T_vals = sorted(avg_execution_times[filter_name].keys())
            times = [avg_execution_times[filter_name][T]['mean'] for T in T_vals]
            
            # Simple growth rate calculation (last/first)
            growth_factor = times[-1] / times[0] if times[0] > 0 else float('inf')
            T_growth = T_vals[-1] / T_vals[0] if T_vals[0] > 0 else float('inf')
            
            print(f"  {filter_labels[filter_name]}: {growth_factor:.2f}x time increase for {T_growth:.1f}x horizon increase")
    
    # Timeout analysis
    timeout_filters = [name for name, max_T in filter_max_T.items() if max_T < 100]
    if timeout_filters:
        print(f"\nTimeout Analysis (Filters exceeding 30s):")
        for filter_name in timeout_filters:
            timeout_T = filter_max_T[filter_name]
            if timeout_T in avg_execution_times[filter_name]:
                timeout_time = avg_execution_times[filter_name][timeout_T]['mean']
                print(f"  • {filter_labels[filter_name]}: Exceeded 30s at T={timeout_T} ({timeout_time:.2f}s)")
            else:
                print(f"  • {filter_labels[filter_name]}: Exceeded 30s at T={timeout_T}")
    else:
        print(f"\nNo filters exceeded the 30-second timeout threshold.")
    
    print("\n" + "="*80)

def main():
    """Main function to create plots and print summary"""
    
    print("Creating computation time comparison plot...")
    
    # Create the plot and get data
    results = create_computation_time_plot()
    if results is None:
        return
    
    avg_execution_times, filter_labels, filter_max_T = results
    
    # Print comprehensive summary
    print_summary_statistics(avg_execution_times, filter_labels, filter_max_T)
    
    print("\nComputation time analysis completed successfully!")

if __name__ == "__main__":
    main()