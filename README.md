Distributionally Robust Kalman Filter
====================================================


## Requirements
- Python (>= 3.5)
- numpy (>= 1.17.4)
- scipy (>= 1.6.2)
- matplotlib (>= 3.1.2)
- control (>= 0.9.4)
- **[CVXPY](https://www.cvxpy.org/)**
- **[MOSEK (>= 9.3)](https://www.mosek.com/)** (required by CVXPY for solving optimization problems)
- (pickle5) install if you encounter compatibility issues with pickle
- joblib (>=1.4.2, Used for parallel computation)

### Figure 1

- For Gaussian uncertainties

```
python main0.py --dist normal
```
Then run
```
python plot0.py --dist normal
```

- For U-Quadratic uncertainties

```
python main0.py --dist quadratic
```
Then run
```
python plot0.py --dist quadratic
```

### Figure 2
Effect of number of samples
```
python main0_numsample.py
```
Then run
```
python plot0_numsample.py
```

### Figure 3
3D surface plot
```
python main3.py
```
Then run
```
python plot3.py
```

### Figure 4
Effect of theta on the average MPC cost
```
python main1_with_MPC.py
```
Then run
```
python plot1_with_MPC.py
```
### Figure 5
2D trajectory plot (You need to run main1_with_MPC.py first!)
```
python main2_with_MPC.py
```
Then run
```
python plot2_with_MPC.py
```
### Figure 6
Average offline computation time for various filters
```
python main4_time.py
```
Then run
```
python plot4_time.py
```
### Figure 7, 8
DRKF Sandwich property
```
python ellipses_visualization_new.py
```
### Figure 9
Convergence property
```
python convergence_check.py
```