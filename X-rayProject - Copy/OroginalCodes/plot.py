import matplotlib.pyplot as plt
import numpy as np

def henon_map(x, y, a, b):
    return y + 1 - a * x ** 2, b * x

def henon_map_phase_plot(x, y, a, b, iterations):
    x_vals, y_vals = [], []

    for _ in range(iterations):
        x_vals.append(x)
        y_vals.append(y)
        x, y = henon_map(x, y, a, b)

    return x_vals, y_vals

def henon_map_bifurcation(a_values, iterations, points_per_iteration, x=0.1, y=0.2):
    xy_values = []
    for a in a_values:
        for _ in range(iterations):
            x, y = henon_map(x, y, a, 0.3)
        for _ in range(points_per_iteration):
            x, y = henon_map(x, y, a, 0.3)
            xy_values.append((a, x))
    return xy_values

def henon_map_lyapunov_exponent(x, y, a, b, iterations, delta=1e-9):
    sum_log = 0.0
    lyapunov_vals = []

    for _ in range(iterations):
        x_prime, y_prime = henon_map(x, y, a, b)
        partial_derivative = (henon_map(x + delta, y, a, b)[0] - x_prime) / delta
        sum_log += np.log(np.abs(partial_derivative))
        x, y = x_prime, y_prime
        lyapunov_vals.append(sum_log / (_ + 1))

    return lyapunov_vals

# Set parameters
a_value = 1.4
a_values = np.linspace(0.8, 1.4, 1000)
iterations = 10000
points_per_iteration = 100

# Create subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

# Plotting the phase plot
x_vals, y_vals = henon_map_phase_plot(0.1, 0.2, a_value, 0.3, iterations)
ax1.plot(x_vals, y_vals, '.', markersize=1)
ax1.set_title(f"Henon Map Phase Plot (a={a_value}, b=0.3)")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

# Plotting the bifurcation diagram
bifurcation_data = henon_map_bifurcation(a_values, iterations, points_per_iteration)
ax2.scatter(*zip(*bifurcation_data), s=0.1, marker='.', color='red')
ax2.set_title('Henon Map Bifurcation Diagram')
ax2.set_xlabel('a')
ax2.set_ylabel('x')

# Plotting the Lyapunov exponent
lyapunov_vals = henon_map_lyapunov_exponent(0.1, 0.2, a_value, 0.3, 1000)
ax3.plot(range(1, len(lyapunov_vals) + 1), lyapunov_vals)
ax3.set_title(f"Henon Map Lyapunov Exponent (a={a_value}, b=0.3)")
ax3.set_xlabel("Iterations")
ax3.set_ylabel("Lyapunov Exponent")

plt.tight_layout()
plt.show()