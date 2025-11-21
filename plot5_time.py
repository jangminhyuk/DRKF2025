#!/usr/bin/env python3
"""
System dimension scalability plots.
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

def load_data(file_path):
    """Load pickled data from file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def create_scalability_plot():
    """Create system dimension scalability plot."""
    
    results_path = "./results/scalability/"
    
    try:
        avg_execution_times = load_data(os.path.join(results_path, 'drkf_inf_scalability_avg.pkl'))
        metadata = load_data(os.path.join(results_path, 'drkf_inf_scalability_metadata.pkl'))
    except FileNotFoundError as e:
        print(f"Error: Could not find results file. Make sure you've run main6_time.py first.")
        print(f"Missing file: {e}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract dimensions and corresponding mean times
    dimensions = sorted(avg_execution_times.keys())
    mean_times = [avg_execution_times[n]['mean'] for n in dimensions]  # Keep in seconds
    
    # Plot the data without error bars
    ax.plot(dimensions, mean_times,
            marker='o',
            color='black',  # Use black color
            linestyle='-',
            linewidth=2,
            markersize=8)
    
    # Customize plot
    ax.set_xlabel('System Dimension', fontsize=28)
    ax.set_ylabel('Computation Time (sec.)', fontsize=28)
    ax.set_yscale('log')  # Use log scale for y-axis (matching plot_time.py)
    ax.grid(True, alpha=0.3)
    
    # Set integer ticks for x-axis
    ax.set_xticks(dimensions)
    ax.set_xlim(left=min(dimensions)-0.5, right=max(dimensions)+0.5)
    
    # Set specific y-axis ticks for 10^-1, 10^0, 10^1
    ax.set_yticks([0.1, 1.0, 10.0])
    ax.set_yticklabels(['$10^{-1}$', '$10^{0}$', '$10^{1}$'])
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(results_path, 'system_dimension_scalability.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"System dimension scalability plot saved as: {output_path}")
    
    return avg_execution_times, metadata

def print_scalability_summary(avg_execution_times, metadata):
    """Print detailed scalability summary"""
    
    print("\n" + "="*80)
    print("SYSTEM DIMENSION SCALABILITY RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nExperiment Parameters:")
    print(f"  • Filter: {metadata['filter']} (Steady-State DRKF)")
    print(f"  • Fixed time horizon: T = {metadata['T']}")
    print(f"  • Robust parameter: θ = {metadata['robust_val']}")
    print(f"  • System dimensions tested: {metadata['dimensions_tested']}")
    
    print(f"\nExecution Time Results:")
    print(f"{'Dimension (n)':<15} {'Avg Time (s)':<15} {'Std Dev (s)':<15} {'Min (s)':<12} {'Max (s)':<12} {'Samples':<8}")
    print("-" * 85)
    
    dimensions = sorted(avg_execution_times.keys())
    for n in dimensions:
        stats = avg_execution_times[n]
        mean_s = stats['mean']
        std_s = stats['std']
        min_s = stats['min']
        max_s = stats['max']
        print(f"{n:<15} {mean_s:<15.4f} {std_s:<15.4f} {min_s:<12.4f} {max_s:<12.4f} {stats['count']:<8}")
    
    # Scalability analysis
    print(f"\nScalability Analysis:")
    times_s = [avg_execution_times[n]['mean'] for n in dimensions]
    
    if len(dimensions) >= 2:
        # Calculate growth rates
        print(f"  Growth rates between consecutive dimensions:")
        for i in range(1, len(dimensions)):
            n_prev, n_curr = dimensions[i-1], dimensions[i]
            time_prev, time_curr = times_s[i-1], times_s[i]
            
            dim_ratio = n_curr / n_prev
            time_ratio = time_curr / time_prev
            
            print(f"    n={n_prev} to n={n_curr}: {dim_ratio:.1f}x dimension → {time_ratio:.2f}x time")
    
    # Estimate computational complexity
    if len(dimensions) >= 3:
        # Simple polynomial fit to log-log data
        log_dims = np.log(dimensions)
        log_times = np.log([avg_execution_times[n]['mean'] for n in dimensions])
        coeffs = np.polyfit(log_dims, log_times, 1)
        complexity_exponent = coeffs[0]
        
        print(f"\nEstimated Computational Complexity:")
        print(f"  Time ∝ n^{complexity_exponent:.2f}")
        
        if complexity_exponent < 1.5:
            complexity_class = "Sub-quadratic (excellent scalability)"
        elif complexity_exponent < 2.5:
            complexity_class = "Approximately quadratic"
        elif complexity_exponent < 3.5:
            complexity_class = "Approximately cubic"
        else:
            complexity_class = "Higher-order polynomial (poor scalability)"
        
        print(f"  Classification: {complexity_class}")
        
        # Extrapolate to larger dimensions
        print(f"\nExtrapolated execution times for larger systems:")
        test_dimensions = [25, 30, 40, 50]
        for n_test in test_dimensions:
            if n_test > max(dimensions):
                predicted_time = np.exp(coeffs[1]) * (n_test ** coeffs[0])  # Keep in seconds
                print(f"    n={n_test}: ~{predicted_time:.3f} s")
    
    # Practical limits analysis
    print(f"\nPractical Performance Assessment:")
    real_time_threshold = 0.1  # 0.1s threshold for real-time applications
    
    fast_dimensions = [n for n in dimensions if avg_execution_times[n]['mean'] < real_time_threshold]
    slow_dimensions = [n for n in dimensions if avg_execution_times[n]['mean'] >= real_time_threshold]
    
    if fast_dimensions:
        print(f"  Real-time suitable (< {real_time_threshold}s): n ≤ {max(fast_dimensions)}")
    
    if slow_dimensions:
        print(f"  Approaching limits (≥ {real_time_threshold}s): n ≥ {min(slow_dimensions)}")
    
    # Overall performance summary
    min_time_s = min(times_s)
    max_time_s = max(times_s)
    min_dim = dimensions[times_s.index(min_time_s)]
    max_dim = dimensions[times_s.index(max_time_s)]
    
    print(f"\nPerformance Range:")
    print(f"  Fastest: n={min_dim}, {min_time_s:.4f}s")
    print(f"  Slowest: n={max_dim}, {max_time_s:.4f}s")
    print(f"  Range: {max_time_s/min_time_s:.1f}x variation across dimensions")
    
    # Memory and computational considerations
    print(f"\nComputational Considerations:")
    print(f"  State space size scales as O(n²) for covariance matrices")
    print(f"  Observation space varies with dimension (ny = max(2, n//2))")
    print(f"  Matrix operations complexity: O(n³) for inversions, O(n²) for multiplications")
    
    if complexity_exponent > 0:
        efficiency = 3.0 / complexity_exponent  # Compare to O(n³) theoretical worst case
        print(f"  Algorithm efficiency: {efficiency:.1f}x better than O(n³) worst case")
    
    print("\n" + "="*80)

def main():
    """Main function to create plots and print summary"""
    
    print("Creating system dimension scalability plot...")
    
    # Create the plot and get data
    results = create_scalability_plot()
    if results is None:
        return
    
    avg_execution_times, metadata = results
    
    # Print comprehensive summary
    print_scalability_summary(avg_execution_times, metadata)
    
    print("\nSystem dimension scalability analysis completed successfully!")

if __name__ == "__main__":
    main()