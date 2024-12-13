import matplotlib.pyplot as plt
import numpy as np

# Paramètres globaux
np.random.seed(42)
num_simulations = 10000  # Nombre de simulations Monte Carlo
T = 1.0  # Horizon temporel (1 période)
dt = 1.0  # Pas de temps (unique pour une seule période)
n_steps = int(T / dt)  # Nombre d'étapes de temps (1 étape ici)

# Paramètres du modèle avec bornes

# prendre b=lambda*sigma
b = lambda x: np.clip(0.05 * x, -5, 5)  # Drift borné entre -5 et 5
sigma = lambda x: np.clip(
    0.2 * x, 0.01, 10
)  # Volatilité idiosyncratique bornée (ne s'annule pas)
sigma0 = lambda x: np.clip(
    0.1 * x, 0.01, 5
)  # Volatilité du bruit commun bornée (ne s'annule pas)


# Fonction pour simuler les trajectoires (1 période)
def simulate_one_period(x0, num_simulations, strategy=None, c=0.1, noise_type="normal"):
    """
    Simule une seule période pour un ensemble d'agents avec bruit commun.
    noise_type: 'normal' ou 'uniform' pour changer la loi de epsilon et epsilon_0.
    """
    X = np.zeros(num_simulations)
    if noise_type == "normal":
        epsilon_0 = np.random.normal(0, np.sqrt(dt))
        epsilon = np.random.normal(0, 1, num_simulations)
    elif noise_type == "uniform":
        epsilon_0 = np.random.uniform(-1, 1)  # Bruit commun uniforme
        epsilon = np.random.uniform(
            -1, 1, num_simulations
        )  # Bruits idiosyncratiques uniformes

    common_term = sigma0(x0) * epsilon_0
    individual_term = sigma(x0) * epsilon * np.sqrt(dt)
    drift_term = b(x0) * dt
    interaction_term = 0
    if strategy:
        interaction_term = strategy(x0, x0, c) * dt
    X[:] = x0 + drift_term + common_term + individual_term + interaction_term
    return X


# Stratégies des exemples 2.6 et 2.7
def strategy_example_26(x, x_hat, c=0.1):
    return b(x_hat) / np.mean(b(x_hat)) - c


def strategy_example_27(x, x_hat, c=0.1):
    return c * (b(x_hat) / np.mean(b(x_hat)) - 1)


# Simulation avec bruit normal
initial_x = 100  # Valeur initiale
results_26_normal = simulate_one_period(
    initial_x, num_simulations, strategy_example_26, c=0.1, noise_type="normal"
)
results_27_normal = simulate_one_period(
    initial_x, num_simulations, strategy_example_27, c=0.1, noise_type="normal"
)

# Simulation avec bruit uniforme
results_26_uniform = simulate_one_period(
    initial_x, num_simulations, strategy_example_26, c=0.1, noise_type="uniform"
)
results_27_uniform = simulate_one_period(
    initial_x, num_simulations, strategy_example_27, c=0.1, noise_type="uniform"
)

# Calcul des moyennes et variances pour chaque cas
mean_26_normal, var_26_normal = np.mean(results_26_normal), np.var(results_26_normal)
mean_27_normal, var_27_normal = np.mean(results_27_normal), np.var(results_27_normal)

mean_26_uniform, var_26_uniform = np.mean(results_26_uniform), np.var(
    results_26_uniform
)
mean_27_uniform, var_27_uniform = np.mean(results_27_uniform), np.var(
    results_27_uniform
)

# Histogramme pour bruit normal
plt.figure(figsize=(10, 6))
plt.hist(
    results_26_normal,
    bins=30,
    alpha=0.6,
    color="blue",
    label=f"Stratégie unilatérale (2.6)\nÉcart-type: {np.sqrt(var_26_normal):.2f}",
)
plt.axvline(
    mean_26_normal,
    color="darkblue",
    linestyle="--",
    label=f"Moyenne: {mean_26_normal:.2f}",
)
plt.axvline(mean_26_normal + np.sqrt(var_26_normal), color="blue", linestyle=":")
plt.axvline(mean_26_normal - np.sqrt(var_26_normal), color="blue", linestyle=":")

plt.hist(
    results_27_normal,
    bins=30,
    alpha=0.6,
    color="orange",
    label=f"Stratégie à variables séparables (2.7)\nÉcart-type: {np.sqrt(var_27_normal):.2f}",
)
plt.axvline(
    mean_27_normal,
    color="darkorange",
    linestyle="--",
    label=f"Moyenne: {mean_27_normal:.2f}",
)
plt.axvline(mean_27_normal + np.sqrt(var_27_normal), color="orange", linestyle=":")
plt.axvline(mean_27_normal - np.sqrt(var_27_normal), color="orange", linestyle=":")

plt.title("Histogramme des valeurs finales (Bruit normal)", fontsize=14)
plt.xlabel("Valeur finale", fontsize=12)
plt.ylabel("Fréquence", fontsize=12)
plt.legend(loc="upper right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Histogramme pour bruit uniforme
plt.figure(figsize=(10, 6))
plt.hist(
    results_26_uniform,
    bins=30,
    alpha=0.6,
    color="blue",
    label=f"Stratégie unilatérale (2.6)\nÉcart-type: {np.sqrt(var_26_uniform):.2f}",
)
plt.axvline(
    mean_26_uniform,
    color="darkblue",
    linestyle="--",
    label=f"Moyenne: {mean_26_uniform:.2f}",
)
plt.axvline(mean_26_uniform + np.sqrt(var_26_uniform), color="blue", linestyle=":")
plt.axvline(mean_26_uniform - np.sqrt(var_26_uniform), color="blue", linestyle=":")

plt.hist(
    results_27_uniform,
    bins=30,
    alpha=0.6,
    color="orange",
    label=f"Stratégie à variables séparables (2.7)\nÉcart-type: {np.sqrt(var_27_uniform):.2f}",
)
plt.axvline(
    mean_27_uniform,
    color="darkorange",
    linestyle="--",
    label=f"Moyenne: {mean_27_uniform:.2f}",
)
plt.axvline(mean_27_uniform + np.sqrt(var_27_uniform), color="orange", linestyle=":")
plt.axvline(mean_27_uniform - np.sqrt(var_27_uniform), color="orange", linestyle=":")

plt.title("Histogramme des valeurs finales (Bruit uniforme)", fontsize=14)
plt.xlabel("Valeur finale", fontsize=12)
plt.ylabel("Fréquence", fontsize=12)
plt.legend(loc="upper right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
