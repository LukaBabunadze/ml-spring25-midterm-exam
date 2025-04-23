import numpy as np
import matplotlib.pyplot as plt

# 1. Define the data
x = np.array([-10, -8, -6, -4, -2,  0.5,  2,  4,  6,  8, 10])
y = np.array([ -7,  -4,  -5,  -1,   0,   1,   2,  3,  3,  5,  6])

# 2. Compute the means
x_mean = np.mean(x)
y_mean = np.mean(y)

# 3. Compute deviations from the mean
x_dev = x - x_mean
y_dev = y - y_mean

# 4. Compute the covariance numerator
cov_xy = np.sum(x_dev * y_dev)

# 5. Compute the components of the denominator (sum of squared deviations)
ss_x = np.sum(x_dev**2)
ss_y = np.sum(y_dev**2)

# 6. Compute Pearson’s r
r = cov_xy / np.sqrt(ss_x * ss_y)

# 7. Compute slope & intercept of the best‐fit line
slope = cov_xy / ss_x
intercept = y_mean - slope * x_mean

# 8. Print the correlation coefficient
print(f"Pearson correlation coefficient (r) = {r:.4f}")

# 9. Plot data + regression line
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, label='Data points')

# draw the regression line over the same x range
x_line = np.linspace(x.min(), x.max(), 100)
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, label=f'Fit: y = {slope:.2f}x + {intercept:.2f}', linewidth=2)

# 10. Reposition axes to cross at (0,0)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# 11. Add grid, labels, legend, title
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Scatter Plot with Regression Line (r = {r:.4f})')
ax.legend()

plt.show()
