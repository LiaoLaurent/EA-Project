import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import root_scalar


def drift_b(x, theta, m_bar):
    return theta * (m_bar - x)


def drift_B_with_holding(x, c_t, theta, m_bar):
    b = drift_b(x, theta, m_bar)
    return np.where(b + c_t > 0, (b + c_t) / 2, b + c_t)


def bang_bang_holdings(x, c_t, theta, m_bar):
    return drift_B_with_holding(x, c_t, theta, m_bar) >= 0


def compute_c(agent_distribution, theta, m_bar):
    def fixed_point(x):
        return (
            0.5
            * np.mean(
                np.where(
                    x + drift_b(agent_distribution, theta, m_bar),
                    x + drift_b(agent_distribution, theta, m_bar),
                    0,
                )
            )
            - x
        )

    result = root_scalar(fixed_point, method="brentq", bracket=[-10, 10])

    if result.converged:
        return result.root
    else:
        raise ValueError("Root finding did not converge.")


def compute_Sigma_matrix(agent_distribution, theta, m_bar, sigma_bar):
    c_t = compute_c(agent_distribution, theta, m_bar)
    N = len(agent_distribution)

    A_numerator = bang_bang_holdings(agent_distribution, c_t, theta, m_bar) / (
        1 + bang_bang_holdings(agent_distribution, c_t, theta, m_bar)
    )
    A_denominator = 1 - np.sum(A_numerator) / N
    A = A_numerator / A_denominator

    def Sigma_coefficients(i, j):
        return (sigma_bar * ((i == j) + A[j] / N)) / (
            1 + bang_bang_holdings(agent_distribution[i], c_t, theta, m_bar)
        )

    Sigma_matrix = np.fromfunction(Sigma_coefficients, (N, N), dtype=int)

    return Sigma_matrix


def simulate_trajectory(
    N_agents,
    N_time_steps,
    delta_t,
    theta,
    m_bar,
    sigma_bar,
    T,
    with_mean_equity_holding=True,
    X_0=None,
):
    if X_0 is None:
        x0_mean = m_bar  # Initial population distribution
        x0_std = sigma_bar / np.sqrt(2 * theta)  # Initial population distribution
        X_0 = np.random.normal(x0_mean, x0_std, size=N_agents)

    X_trajectory = np.zeros((N_agents, N_time_steps + 1))
    X_trajectory[:, 0] = X_0

    for t in range(1, N_time_steps + 1):
        # Generate separate Brownian motion for each agent at each time step
        brownian_motion = np.random.normal(0, np.sqrt(delta_t), N_agents)

        if with_mean_equity_holding:
            c_t = compute_c(X_trajectory[:, t - 1], theta, m_bar)
            drift_term = (
                drift_B_with_holding(X_trajectory[:, t - 1], c_t, theta, m_bar)
                * delta_t
            )
            diffusion_term = (
                compute_Sigma_matrix(X_trajectory[:, t - 1], theta, m_bar, sigma_bar)
                @ brownian_motion
            )
        else:
            drift_term = drift_b(X_trajectory[:, t - 1], theta, m_bar) * delta_t
            diffusion_term = sigma_bar * brownian_motion

        dX = drift_term + diffusion_term
        X_trajectory[:, t] = X_trajectory[:, t - 1] + dX

    return X_trajectory


def plot_Nsteps(
    N_agents, N_time_steps, delta_t, theta, m_bar, sigma_bar, T, save_plots=False
):
    # Simulate trajectories
    X_trajectories_with_holding = simulate_trajectory(
        N_agents,
        N_time_steps,
        delta_t,
        theta,
        m_bar,
        sigma_bar,
        T,
        with_mean_equity_holding=True,
    )
    X_trajectories_without_holding = simulate_trajectory(
        N_agents,
        N_time_steps,
        delta_t,
        theta,
        m_bar,
        sigma_bar,
        T,
        with_mean_equity_holding=False,
    )

    final_points_with_holding = X_trajectories_with_holding[:, -1]
    final_points_without_holding = X_trajectories_without_holding[:, -1]

    # Calculate statistics
    mean_with_holding = np.mean(final_points_with_holding)
    std_with_holding = np.std(final_points_with_holding)
    mean_without_holding = np.mean(final_points_without_holding)
    std_without_holding = np.std(final_points_without_holding)

    # First Plot: Trajectories
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    # Display 5 agents trajectories
    trajectories_displayed = min(N_agents, 5)
    for i in range(trajectories_displayed):
        ax1.plot(
            np.linspace(0, T, N_time_steps + 1),
            X_trajectories_with_holding[i],
            color="blue",
            label="With Mutual Holding" if i == 0 else "_nolegend_",
        )
        ax1.plot(
            np.linspace(0, T, N_time_steps + 1),
            X_trajectories_without_holding[i],
            linestyle="--",
            color="gray",
            label="Provision" if i == 0 else "_nolegend_",
        )
    ax1.set_title("Simulation of $X_t$ Trajectories with and without Mutual Holding")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Equity Value $X_t$")
    ax1.legend()
    plt.tight_layout()
    plt.show()
    if save_plots:
        fig1.savefig("trajectories.png")
        print("Trajectories plot saved as 'trajectories.png'.")

    # Second Plot: Histograms
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.hist(
        final_points_with_holding,
        bins=30,
        alpha=0.5,
        color="blue",
        label="With Mean Equity Holding",
        density=True,
    )
    ax2.hist(
        final_points_without_holding,
        bins=30,
        alpha=0.5,
        color="orange",
        label="Provision",
        density=True,
    )
    textstr = (
        f"With Holding:\nMean = {mean_with_holding:.5f}\nStd Dev = {std_with_holding:.5f}\n\n"
        f"Provision:\nMean = {mean_without_holding:.5f}\nStd Dev = {std_without_holding:.5f}"
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    ax2.text(
        0.75,
        0.75,
        textstr,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )
    ax2.set_xlabel("Final Point of Trajectories")
    ax2.set_ylabel("Density")
    ax2.set_title("Histogram of Final Points of Trajectories")
    ax2.legend()
    plt.tight_layout()
    plt.show()
    if save_plots:
        fig2.savefig("histograms.png")
        print("Histograms plot saved as 'histograms.png'.")

    # Third Plot: Density Curves (KDE)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.kdeplot(
        final_points_with_holding,
        color="blue",
        label="Density - With Holding",
        linewidth=2,
        ax=ax3,
    )
    sns.kdeplot(
        final_points_without_holding,
        color="orange",
        label="Density - Provision",
        linewidth=2,
        ax=ax3,
    )
    ax3.set_xlabel("Final Point of Trajectories")
    ax3.set_ylabel("Density")
    ax3.set_title("Density of Final Points of Trajectories")
    ax3.legend()
    ax3.grid(True)
    plt.tight_layout()
    plt.show()
    if save_plots:
        fig3.savefig("density_curves.png")
        print("Density curves plot saved as 'density_curves.png'.")
