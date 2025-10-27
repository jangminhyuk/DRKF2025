#!/usr/bin/env python3
"""
3D surface plots for regret MSE analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pickle
import argparse
import os
from scipy.interpolate import griddata

plt.rcParams.update({
    'font.size': 14,           # Base font size
    'axes.titlesize': 16,      # Title font size
    'axes.labelsize': 14,      # Axis label font size
    'xtick.labelsize': 12,     # X-axis tick label size
    'ytick.labelsize': 12,     # Y-axis tick label size
    'legend.fontsize': 12,     # Legend font size
    'figure.titlesize': 18     # Figure title size
})

def load_data(file_path):
    """Load pickled data from file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def create_3d_surface_plot(regret_data, dist):
    """Create 3D surface plot of average regret MSE."""
    df = pd.DataFrame(regret_data)
    
    print(f"Theta_w range: {df['theta_w'].min():.3f} to {df['theta_w'].max():.3f}")
    print(f"Theta_v range: {df['theta_v'].min():.3f} to {df['theta_v'].max():.3f}")
    print(f"Regret range: {df['avg_regret'].min():.6f} to {df['avg_regret'].max():.6f}")
    
    # Create pivot table for proper grid structure
    regret_pivot = df.pivot(index='theta_v', columns='theta_w', values='avg_regret')
    
    # Get unique values and create log-spaced grids
    theta_w_unique = sorted(df['theta_w'].unique())
    theta_v_unique = sorted(df['theta_v'].unique())
    
    # Use log values for axes to enable proper log scaling
    log_theta_w = np.log10(theta_w_unique)
    log_theta_v = np.log10(theta_v_unique)
    
    # Create finer meshgrids for smoother surface using interpolation
    log_theta_w_fine = np.linspace(min(log_theta_w), max(log_theta_w), 50)
    log_theta_v_fine = np.linspace(min(log_theta_v), max(log_theta_v), 50)
    log_theta_w_grid_fine, log_theta_v_grid_fine = np.meshgrid(log_theta_w_fine, log_theta_v_fine)
    
    # Original coarse grid for data
    log_theta_w_grid, log_theta_v_grid = np.meshgrid(log_theta_w, log_theta_v)
    regret_grid = regret_pivot.values
    
    # Interpolate to create smooth surface
    regret_grid_fine = griddata(
        (log_theta_w_grid.flatten(), log_theta_v_grid.flatten()),
        regret_grid.flatten(),
        (log_theta_w_grid_fine, log_theta_v_grid_fine),
        method='cubic'
    )
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create smooth surface plot using interpolated fine grid with enhanced colors
    # Apply power normalization to emphasize lower values more
    from matplotlib.colors import PowerNorm
    
    vmin = np.nanmin(regret_grid_fine)
    vmax = np.nanmax(regret_grid_fine)
    
    surf = ax.plot_surface(log_theta_w_grid_fine, log_theta_v_grid_fine, regret_grid_fine,
                          cmap='plasma', alpha=0.9, linewidth=1.0, antialiased=True,
                          norm=PowerNorm(gamma=0.35, vmin=vmin, vmax=vmax))
    
    # Data points removed for cleaner surface
    
    # Contour lines at bottom removed for cleaner appearance
    
    # Set custom tick labels with specific nice values in scientific notation
    nice_ticks = [0.01, 0.1, 1.0, 10.0]
    nice_log_ticks = [np.log10(t) for t in nice_ticks]
    ax.set_xticks(nice_log_ticks)
    ax.set_xticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'], fontsize=18)
    ax.set_yticks(nice_log_ticks)
    ax.set_yticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'], fontsize=18)
    
    # Set labels with LaTeX formatting and big font
    ax.set_xlabel(r'$\theta_w$', fontsize=24, labelpad=20)
    ax.set_ylabel(r'$\theta_v$', fontsize=24, labelpad=20)
    ax.zaxis.set_rotate_label(False)  # Rotate z-axis label opposite
    ax.set_zlabel('Average Regret MSE', fontsize=28, labelpad=15, rotation=90)
    
    # Move z-axis tick labels further from axis and set font size
    ax.tick_params(axis='z', pad=5, labelsize=18)

    # Add colorbar without title, using same normalization
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cbar.ax.tick_params(labelsize=18)
    
    # Improve viewing angle
    ax.view_init(elev=17, azim=40)
    
    plt.tight_layout()
    
    # Legend removed since no data points
    
    # Save plot BEFORE showing (to ensure proper saving)
    results_path = "./results/3d_data/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, f'3d_regret_surface_{dist}.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.8)
    print(f"3D surface plot saved as: {output_path}")
    
    # Show plot (keep window open for interaction)
    plt.show()

def create_2d_contour_plot(regret_data, dist):
    """Create 2D contour plot of regret MSE vs theta_w and theta_v"""
    
    # Extract data
    theta_w_vals = [d['theta_w'] for d in regret_data]
    theta_v_vals = [d['theta_v'] for d in regret_data]
    regret_vals = [d['avg_regret'] for d in regret_data]
    
    # Create meshgrid for contour plot
    theta_w_unique = sorted(list(set(theta_w_vals)))
    theta_v_unique = sorted(list(set(theta_v_vals)))
    
    # Create regular grid
    theta_w_grid, theta_v_grid = np.meshgrid(theta_w_unique, theta_v_unique)
    
    # Reshape regret data to match grid
    regret_grid = np.zeros_like(theta_w_grid)
    for i, theta_w in enumerate(theta_w_unique):
        for j, theta_v in enumerate(theta_v_unique):
            # Find corresponding regret value
            for d in regret_data:
                if d['theta_w'] == theta_w and d['theta_v'] == theta_v:
                    regret_grid[j, i] = d['avg_regret']
                    break
    
    # Create 2D contour plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create filled contour plot
    contour_filled = ax.contourf(theta_w_grid, theta_v_grid, regret_grid, 
                                levels=20, cmap='viridis', alpha=0.8)
    
    # Add contour lines
    contour_lines = ax.contour(theta_w_grid, theta_v_grid, regret_grid, 
                              levels=20, colors='black', alpha=0.4, linewidths=0.5)
    
    # Add contour labels
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.3f')
    
    # Mark data points
    ax.scatter(theta_w_vals, theta_v_vals, c='red', s=30, alpha=0.8, 
              edgecolors='black', linewidths=0.5, label='Data Points')
    
    # Find and mark minimum regret point
    min_idx = np.argmin(regret_vals)
    min_theta_w = theta_w_vals[min_idx]
    min_theta_v = theta_v_vals[min_idx]
    min_regret = regret_vals[min_idx]
    
    ax.scatter(min_theta_w, min_theta_v, c='white', s=100, 
              edgecolors='black', linewidths=2, marker='*', 
              label=f'Min Regret: {min_regret:.4f}')
    
    # Set labels and title
    ax.set_xlabel('θw (Process Noise Robustness)')
    ax.set_ylabel('θv (Measurement Noise Robustness)')
    ax.set_title(f'2D Regret Contour: DRKF vs Standard KF ({dist.capitalize()} Distribution)')
    
    # Set log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add colorbar
    cbar = fig.colorbar(contour_filled, ax=ax, label='Average Regret MSE')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    results_path = "./results/3d_data/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, f'2d_regret_contour_{dist}.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"2D contour plot saved as: {output_path}")

def create_heatmap_plot(regret_data, dist):
    """Create heatmap of regret MSE vs theta_w and theta_v"""
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(regret_data)
    
    # Create pivot table for heatmap
    heatmap_data = df.pivot(index='theta_v', columns='theta_w', values='avg_regret')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(heatmap_data.values, cmap='viridis', aspect='auto', origin='lower')
    
    # Set tick labels
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_xticklabels([f'{val:.2f}' for val in heatmap_data.columns])
    ax.set_yticklabels([f'{val:.2f}' for val in heatmap_data.index])
    
    # Set labels and title
    ax.set_xlabel('θw (Process Noise Robustness)')
    ax.set_ylabel('θv (Measurement Noise Robustness)')
    ax.set_title(f'Regret Heatmap: DRKF vs Standard KF ({dist.capitalize()} Distribution)')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label='Average Regret MSE')
    
    # Add text annotations for each cell
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            text = ax.text(j, i, f'{heatmap_data.iloc[i, j]:.3f}',
                          ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    results_path = "./results/3d_data/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    output_path = os.path.join(results_path, f'regret_heatmap_{dist}.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap plot saved as: {output_path}")

def print_summary_statistics(regret_data, dist):
    """Print summary statistics of the regret data"""
    
    df = pd.DataFrame(regret_data)
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nRegret MSE Statistics for {dist.capitalize()} Distribution:")
    print(f"  Minimum regret: {df['avg_regret'].min():.6f}")
    print(f"  Maximum regret: {df['avg_regret'].max():.6f}")
    print(f"  Mean regret: {df['avg_regret'].mean():.6f}")
    print(f"  Std regret: {df['avg_regret'].std():.6f}")
    
    # Find optimal theta combination
    min_idx = df['avg_regret'].idxmin()
    optimal_row = df.loc[min_idx]
    print(f"\nOptimal θ combination (minimum regret):")
    print(f"  θw = {optimal_row['theta_w']:.2f}")
    print(f"  θv = {optimal_row['theta_v']:.2f}")
    print(f"  Regret = {optimal_row['avg_regret']:.6f} ± {optimal_row['std_regret']:.6f}")
    
    # Show worst case
    max_idx = df['avg_regret'].idxmax()
    worst_row = df.loc[max_idx]
    print(f"\nWorst θ combination (maximum regret):")
    print(f"  θw = {worst_row['theta_w']:.2f}")
    print(f"  θv = {worst_row['theta_v']:.2f}")
    print(f"  Regret = {worst_row['avg_regret']:.6f} ± {worst_row['std_regret']:.6f}")
    
    # Analyze theta_w effect (averaged over theta_v)
    print(f"\nEffect of θw (averaged over θv):")
    theta_w_effect = df.groupby('theta_w')['avg_regret'].agg(['mean', 'std'])
    for theta_w, stats in theta_w_effect.iterrows():
        print(f"  θw = {theta_w:.2f}: {stats['mean']:.6f} ± {stats['std']:.6f}")
    
    # Analyze theta_v effect (averaged over theta_w)
    print(f"\nEffect of θv (averaged over θw):")
    theta_v_effect = df.groupby('theta_v')['avg_regret'].agg(['mean', 'std'])
    for theta_v, stats in theta_v_effect.iterrows():
        print(f"  θv = {theta_v:.2f}: {stats['mean']:.6f} ± {stats['std']:.6f}")

def main(dist):
    """Main function to create all 3D plots"""
    
    results_path = "./results/3d_data/"
    
    # Load 3D data from main6.py
    try:
        regret_data = load_data(os.path.join(results_path, f'regret_3d_data_{dist}.pkl'))
    except FileNotFoundError as e:
        print(f"Error: Could not find 3D data file. Make sure you've run main6.py first.")
        print(f"Missing file: {e}")
        return
    
    print(f"Creating 3D plots for {dist} distribution...")
    print(f"Loaded {len(regret_data)} data points for plotting")
    
    # Create all plots
    create_3d_surface_plot(regret_data, dist)
    create_2d_contour_plot(regret_data, dist)
    create_heatmap_plot(regret_data, dist)
    
    # Print summary statistics
    print_summary_statistics(regret_data, dist)
    
    print(f"\nAll 3D plots created successfully for {dist} distribution!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create 3D plots from main6.py results")
    parser.add_argument('--dist', default="normal", type=str,
                        help="Distribution type (normal or quadratic)")
    
    args = parser.parse_args()
    main(args.dist)