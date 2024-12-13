import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import root_scalar


def drift_b(x, theta, m_bar):
    return theta * (m_bar - x)


def drift_B_with_holding(x, c_t, theta, m_bar):
    b = drift_b(x, theta, m_bar)
    return np.where(b + c_t > 0, (b + c_t) / 2, b + c_t)


def diffusion_sigma_with_holding(x, c_t, sigma_bar, theta, m_bar):
    return np.where(drift_b(x, theta, m_bar) + c_t > 0, sigma_bar / 2, sigma_bar)


def compute_c(mu, sigma, theta, m_bar):

    def H(x):
        f = lambda x: norm.pdf(x, loc=mu, scale=sigma)
        F = lambda x: norm.cdf(x, loc=mu, scale=sigma)
        return (
            x
            - (0.5 * theta * sigma**2) * f(x / theta + m_bar)
            - 0.5 * (x - theta * (mu - m_bar)) * F(x / theta + m_bar)
        )

    result = root_scalar(H, method="brentq", bracket=[-10, 10])

    if result.converged:
        return result.root
    else:
        raise ValueError("Root finding did not converge.")


def simulate_one_step(mu, sigma, theta, m_bar, T, M=1000):
    sigma_bar = sigma * np.sqrt(2 * theta)
    Z = np.random.normal(0, 1, M)

    initial_distribution = np.random.normal(mu, sigma, M)
    final_provision = (
        initial_distribution
        + drift_b(initial_distribution, theta, m_bar) * T
        + sigma_bar * np.sqrt(T) * Z
    )

    c_t = compute_c(mu, sigma, theta, m_bar)
    final_distribution = (
        initial_distribution
        + drift_B_with_holding(initial_distribution, c_t, theta, m_bar) * T
        + diffusion_sigma_with_holding(
            initial_distribution, c_t, sigma_bar, theta, m_bar
        )
        * np.sqrt(T)
        * Z
    )

    return final_provision, final_distribution


def plot_onestep(mu, sigma, theta, m_bar, T, M=100000, save_plots=False):
    x_values = np.linspace(-7.5, -2.5, 1000)
    c_t = compute_c(m_bar, sigma, theta, m_bar)
    y_drift_b = drift_b(x_values, theta, m_bar)
    y_drift_B_with_holding = drift_B_with_holding(x_values, c_t, theta, m_bar)

    final_provision, final_distribution = simulate_one_step(
        mu, sigma, theta, m_bar, T, M
    )

    # Compute empirical mean and variance
    mean_provision = np.mean(final_provision)
    var_provision = np.var(final_provision)
    mean_distribution = np.mean(final_distribution)
    var_distribution = np.var(final_distribution)

    # First Plot: Drift Functions
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(x_values, y_drift_b, label="drift_b(x)", color="blue")
    ax1.plot(
        x_values,
        y_drift_B_with_holding,
        label="drift_B_with_holding(x, c_t)",
        color="orange",
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("Function value")
    ax1.set_title("Plot of drift_b(x) and drift_B_with_holding(x, c_t)")
    ax1.legend()
    ax1.grid(True)
    param_text = (
        f"$\\theta$ = {theta}\n"
        f"$\\bar{{m}}$ = {m_bar}\n"
        f"$\\sigma$ = {sigma}\n"
        f"$T$ = {T}"
    )
    ax1.text(
        0.95,
        0.95,
        param_text,
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.7),
    )
    plt.tight_layout()
    plt.show()
    if save_plots:
        fig1.savefig("drift_functions.png")
        print("Drift functions plot saved as 'drift_functions.png'.")

    # Second Plot: Final Distributions
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.hist(
        final_provision,
        bins=30,
        alpha=0.5,
        label="Final Provision",
        density=True,
        color="blue",
    )
    ax2.hist(
        final_distribution,
        bins=30,
        alpha=0.5,
        label="Final Distribution",
        density=True,
        color="orange",
    )
    sns.kdeplot(
        final_provision,
        color="blue",
        label="Density - Final Provision",
        linewidth=2,
        ax=ax2,
    )
    sns.kdeplot(
        final_distribution,
        color="orange",
        label="Density - Final Distribution",
        linewidth=2,
        ax=ax2,
    )
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Density")
    ax2.set_title("Histogram and Density of Final Provision and Final Distribution")
    ax2.legend()
    ax2.grid(True)
    emp_text = (
        f"Empirical Mean (Provision) = {mean_provision:.2f}\n"
        f"Empirical Var (Provision) = {var_provision:.2f}\n"
        f"Empirical Mean (Distribution) = {mean_distribution:.2f}\n"
        f"Empirical Var (Distribution) = {var_distribution:.2f}"
    )
    ax2.text(
        0.95,
        0.95,
        emp_text,
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.7),
    )
    plt.tight_layout()
    plt.show()
    if save_plots:
        fig2.savefig("final_distributions.png")
        print("Final distributions plot saved as 'final_distributions.png'.")
