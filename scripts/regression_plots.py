import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import norm

def gaussian_fit(data):
    # Compute stats
    mean = np.mean(data)
    std = np.std(data)

    # Histogram setup
    X_plot = np.linspace(data.min(), data.max(), 1000)
    Y_hist, bin_edges = np.histogram(data, bins=300, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Gaussian (normal) fit
    Y_gauss = norm.pdf(X_plot, mean, std)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, Y_hist, width=np.diff(bin_edges), alpha=0.5, label='Data distribution')
    plt.plot(X_plot, Y_gauss, 'r-', linewidth=2, label='Gaussian fit')

    # Add text for mean and std
    plt.text(mean, max(Y_gauss)*0.9, f"μ = {mean:.3f}\nσ = {std:.3f}", 
            fontsize=12, color='red', ha='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.xlabel('target values')
    plt.ylabel('Density')
    plt.title('Distribution of target values with Gaussian fit')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def prediction_plot(y_test, y_pred):
    # Plot y_test vs y_pred for best model
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.6, label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal (y = ŷ)')
    plt.xlabel('Actual target values')
    plt.ylabel('Predicted target values')
    plt.title('Model: Actual vs Predicted target (Best Hyperparameters)')
    plt.legend()
    plt.show()

    
def learning_curve_plot(train_scores, val_scores):
    train_sizes = np.arange(1, len(train_scores) + 1)
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', color='blue', label='Training Loss')
    plt.plot(train_sizes, val_scores, 'o-', color='orange', label='Validation Loss')
    plt.ylabel('Loss (MSE)')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()