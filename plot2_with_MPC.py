#!/usr/bin/env python3
"""
Trajectory plots for MPC controller results.
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

def load_data(results_path, dist):
    """Load saved results from experiments."""
    main5_2_file = os.path.join(results_path, f'main5_2_MPC_results_{dist}.pkl')
    if not os.path.exists(main5_2_file):
        raise FileNotFoundError(f"Results file not found: {main5_2_file}")
    
    with open(main5_2_file, 'rb') as f:
        main5_2_results = pickle.load(f)
    
    return main5_2_results

def generate_desired_trajectory(T_total):
    """Generate desired trajectory."""
    Amp = 5.0
    slope = 1.0
    radius = 5.0
    omega = 0.5
    dt = 0.2
    time_steps = int(T_total / dt) + 1
    time = np.linspace(0, T_total, time_steps)

    x_d = Amp * np.sin(omega * time)
    vx_d = Amp * omega * np.cos(omega * time)
    y_d = slope * time
    vy_d = slope * np.ones(time_steps)

    return np.vstack((x_d, vx_d, y_d, vy_d)), time

def extract_trajectory_data(main5_2_results):
    """Extract trajectory data for each filter from main5_2_with_MPC.py results"""
    
    # Define the filters and their order (all 10 filters from main5_2_with_MPC.py)
    filters_order = ['finite', 'inf', 'risk', 'risk_seek', 'drkf_neurips', 'bcot', 
                     'drkf_finite_cdc', 'drkf_inf_cdc', 'drkf_finite', 'drkf_inf']
    filters_order = ['finite', 'inf', 'drkf_neurips', 'bcot', 
                     'drkf_finite_cdc', 'drkf_inf_cdc', 'drkf_finite', 'drkf_inf']
    
    trajectory_data = {}
    
    for filt in filters_order:
        if filt not in main5_2_results:
            print(f"Warning: Filter '{filt}' not found in results, skipping...")
            continue
            
        # Extract trajectory data from main5_2_with_MPC.py results
        results = main5_2_results[filt]['results']
        optimal_theta = main5_2_results[filt].get('optimal_theta', 'N/A')
        
        trajectories = []
        for result in results:  # Each simulation result
            traj = result['state_traj']  # State trajectory
            trajectories.append(traj)
        
        # Convert to numpy array and compute statistics
        trajectories = np.array(trajectories)  # Shape: (num_runs, time_steps, state_dim, 1)
        trajectories = np.squeeze(trajectories, axis=-1)  # Remove last dimension
        
        mean_traj = np.mean(trajectories, axis=0)  # (time_steps, state_dim)
        std_traj = np.std(trajectories, axis=0)    # (time_steps, state_dim)
        
        trajectory_data[filt] = {
            'mean': mean_traj,
            'std': std_traj,
            'optimal_theta': optimal_theta
        }
    
    return trajectory_data, filters_order

def plot_trajectory_subplots(trajectory_data, filters_order, desired_traj, time, dist):
    """Create individual plots for each filter showing 2D X-Y position trajectories"""
    
    # Colors for all 10 filters
    # colors = {
    #     'finite': 'black',             # Same color as inf
    #     'inf': 'black',
    #     'risk': 'orange',
    #     'risk_seek': 'darkviolet',
    #     'drkf_neurips': 'purple',
    #     'bcot': 'red',
    #     'drkf_finite_cdc': 'brown',    # Same color as drkf_inf_cdc
    #     'drkf_inf_cdc': 'brown',
    #     'drkf_finite': 'blue',         # Same color as drkf_inf
    #     'drkf_inf': 'blue'
    # }

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
    colors = {
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

    
    # Filter names for all 10 filters
    filter_names = {
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
    
    # Create results directory
    results_dir = os.path.join("results", "trajectory_tracking_MPC")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    saved_files = []
    
    # Create individual plots for each filter
    for filt in filters_order:
        if filt not in trajectory_data:
            continue
            
        fig, ax = plt.subplots(figsize=(10, 8))
        color = colors[filt]
        
        # Get trajectory data
        mean_traj = trajectory_data[filt]['mean']
        std_traj = trajectory_data[filt]['std']
        optimal_theta = trajectory_data[filt]['optimal_theta']
        
        # Extract X and Y positions
        x_mean = mean_traj[:, 0]  # X position
        y_mean = mean_traj[:, 2]  # Y position
        x_std = std_traj[:, 0]    # X position std
        y_std = std_traj[:, 2]    # Y position std
        
        # Create shaded tube for ±1 standard deviation FIRST (so it appears behind)
        std_mag = 0.5 * (x_std + y_std)  # Average std as radius
        
        # Create uncertainty tube by offsetting perpendicular to trajectory
        dx = np.gradient(x_mean)
        dy = np.gradient(y_mean)
        norms = np.hypot(dx, dy)
        norms[norms == 0] = 1.0  # Avoid division by zero
        dx /= norms
        dy /= norms
        
        # Perpendicular directions
        perp_x = -dy
        perp_y = dx
        
        # Create upper and lower bounds
        upper_x = x_mean + perp_x * std_mag
        upper_y = y_mean + perp_y * std_mag
        lower_x = x_mean - perp_x * std_mag
        lower_y = y_mean - perp_y * std_mag
        
        # Create polygon for shaded tube
        tube_x = np.concatenate([upper_x, lower_x[::-1]])
        tube_y = np.concatenate([upper_y, lower_y[::-1]])
        
        # Plot desired trajectory (black dashed line) FIRST
        ax.plot(desired_traj[0, :], desired_traj[2, :], 'k--', linewidth=2.5, label='Desired Trajectory')
        
        # Plot shaded tube for ±1 standard deviation SECOND
        ax.fill(tube_x, tube_y, color=color, alpha=0.3, label='1-std tube')
        
        # Plot mean trajectory (colored curve) THIRD
        ax.plot(x_mean, y_mean, '-', color=color, linewidth=2.5, label='Mean Trajectory')
        
        # Mark start and end positions with X (no legend)
        ax.scatter(desired_traj[0, 0], desired_traj[2, 0], marker='X', s=150, color='black', linewidth=3)
        ax.scatter(desired_traj[0, -1], desired_traj[2, -1], marker='X', s=150, color='black', linewidth=3)
        
        # Formatting
        ax.set_xlabel('X position (m)', fontsize=28)
        ax.set_ylabel('Y position (m)', fontsize=28)
        
        # Create title
        title_text = filter_names[filt]
        
        ax.set_title(title_text, fontsize=32, pad=15)
        
        # Set specific tick values
        ax.set_xticks([-5, 0, 5])
        ax.set_yticks([0, 5, 10])
        
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Add legend with proper order and large font
        # ax.legend(['Desired Trajectory', '1-std tube', 'Mean Trajectory'], fontsize=22, loc='best')
        
        plt.tight_layout()
        
        # Save individual plot
        save_path = os.path.join(results_dir, f"traj_2d_MPC_{filt}_{dist}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
        saved_files.append(f"traj_2d_MPC_{filt}_{dist}.pdf")
        plt.close(fig)
    
    print(f"\nMPC trajectory plots saved to:")
    for filename in saved_files:
        print(f"- ./results/trajectory_tracking_MPC/{filename}")


def plot_subplots_all_filters(trajectory_data, filters_order, desired_traj, time, dist):
    """Create a subplot figure with all filters in separate subplots"""
    
    # Colors for all 10 filters
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
    colors = {
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
    
    # Filter names for subplot titles
    filter_names = {
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
    
    # Define the complete filter order for alphabet mapping
    complete_filter_order = ['finite', 'inf', 'risk', 'risk_seek', 'drkf_neurips', 'bcot', 
                            'drkf_finite_cdc', 'drkf_inf_cdc', 'drkf_finite', 'drkf_inf']
    
    # Create alphabet mapping based on complete order
    alphabet_mapping = {filt: chr(ord('A') + i) for i, filt in enumerate(complete_filter_order)}
    
    # Calculate available filters
    available_filters = [filt for filt in filters_order if filt in trajectory_data]
    n_filters = len(available_filters)
    
    # Create subplot layout with 4 columns
    if n_filters <= 4:
        rows, cols = 1, 4
    elif n_filters <= 8:
        rows, cols = 2, 4
    else:
        rows, cols = 3, 4  # For 9+ filters
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
    
    # Flatten axes array for easier indexing
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot each filter in its subplot
    for idx, filt in enumerate(available_filters):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        color = colors[filt]
        
        # Get trajectory data
        mean_traj = trajectory_data[filt]['mean']
        std_traj = trajectory_data[filt]['std']
        optimal_theta = trajectory_data[filt]['optimal_theta']
        
        # Extract X and Y positions
        x_mean = mean_traj[:, 0]  # X position
        y_mean = mean_traj[:, 2]  # Y position
        x_std = std_traj[:, 0]    # X position std
        y_std = std_traj[:, 2]    # Y position std
        
        # Create shaded tube for ±1 standard deviation
        std_mag = 0.5 * (x_std + y_std)  # Average std as radius
        
        # Create uncertainty tube by offsetting perpendicular to trajectory
        dx = np.gradient(x_mean)
        dy = np.gradient(y_mean)
        norms = np.hypot(dx, dy)
        norms[norms == 0] = 1.0  # Avoid division by zero
        dx /= norms
        dy /= norms
        
        # Perpendicular directions
        perp_x = -dy
        perp_y = dx
        
        # Create upper and lower bounds
        upper_x = x_mean + perp_x * std_mag
        upper_y = y_mean + perp_y * std_mag
        lower_x = x_mean - perp_x * std_mag
        lower_y = y_mean - perp_y * std_mag
        
        # Create polygon for shaded tube
        tube_x = np.concatenate([upper_x, lower_x[::-1]])
        tube_y = np.concatenate([upper_y, lower_y[::-1]])
        
        # Plot desired trajectory (black dashed line) FIRST
        ax.plot(desired_traj[0, :], desired_traj[2, :], 'k--', linewidth=1.5, alpha=0.8)
        
        # Plot shaded tube for ±1 standard deviation SECOND
        ax.fill(tube_x, tube_y, color=color, alpha=0.3)
        
        # Plot mean trajectory (colored curve) THIRD
        ax.plot(x_mean, y_mean, '-', color=color, linewidth=2)
        
        # Mark start and end positions
        ax.scatter(desired_traj[0, 0], desired_traj[2, 0], marker='X', s=80, color='black', linewidth=2)
        ax.scatter(desired_traj[0, -1], desired_traj[2, -1], marker='X', s=80, color='black', linewidth=2)
        
        # Create title with alphabetical label
        alphabet_label = alphabet_mapping[filt]
        title_text = f"({alphabet_label}) {filter_names[filt]}"
        
        ax.set_title(title_text, fontsize=16, pad=8)
        ax.set_xlabel('X position (m)', fontsize=14)
        ax.set_ylabel('Y position (m)', fontsize=14)
        
        # Set specific tick values
        ax.set_xticks([-5, 0, 5])
        ax.set_yticks([0, 5, 10])
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    # Hide unused subplots
    for idx in range(len(available_filters), len(axes)):
        axes[idx].set_visible(False)
    
    # Title removed to avoid overlap with subplots
    
    # Create a custom legend in the last subplot or in an empty area
    legend_elements = [
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Desired Trajectory'),
        plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.3, label='1-std tube'),
        plt.Line2D([0], [0], color='gray', linewidth=2, label='Mean Trajectory')
    ]
    
    # Add legend to the figure with more space from subplots
    # fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.00), 
    #            ncol=3, fontsize=16, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.95, hspace=0.45, wspace=-0.1)  # Make more room for legend, less for title
    
    # Save subplot figure
    results_dir = os.path.join("results", "trajectory_tracking_MPC")
    save_path = os.path.join(results_dir, f"traj_2d_MPC_subplots_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    
    print(f"MPC subplots trajectory figure saved to: ./results/trajectory_tracking_MPC/traj_2d_MPC_subplots_{dist}.pdf")

def main():
    parser = argparse.ArgumentParser(description='Create trajectory plots for filters from main5_2_with_MPC.py')
    parser.add_argument('--dist', default='normal', choices=['normal', 'quadratic'],
                        help='Distribution type to plot trajectories for')
    parser.add_argument('--time', default=10.0, type=float,
                        help='Total simulation time')
    args = parser.parse_args()
    
    try:
        # Load data
        results_path = "./results/trajectory_tracking_MPC/"
        main5_2_results = load_data(results_path, args.dist)
        
        # Generate desired trajectory
        desired_traj, time = generate_desired_trajectory(args.time)
        
        # Extract trajectory data using optimal parameters
        trajectory_data, filters_order = extract_trajectory_data(main5_2_results)
        
        print(f"Found MPC trajectory data for {len(trajectory_data)} filters: {list(trajectory_data.keys())}")
        
        # Create individual trajectory plots
        plot_trajectory_subplots(trajectory_data, filters_order, desired_traj, time, args.dist)
        
        # Create subplots figure
        plot_subplots_all_filters(trajectory_data, filters_order, desired_traj, time, args.dist)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find MPC results files for distribution '{args.dist}'")
        print(f"Make sure you have run main5_2_with_MPC.py with --dist {args.dist} first")
        print(f"Missing file: {e}")
    except Exception as e:
        print(f"Error creating MPC trajectory plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()