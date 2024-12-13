import matplotlib.pyplot as plt
import numpy as np

# Paramètres globaux
np.random.seed(42)
num_simulations = 10000  # Nombre de simulations Monte Carlo
T = 1.0  # Horizon temporel (1 période)
dt = 1.0  # Pas de temps (unique pour une seule période)

# Paramètres du modèle avec bornes
b = lambda x: np.clip(0.05 * x, -5, 5)  # Drift borné entre -5 et 5
sigma = lambda x: np.clip(
    0.2 * x, 0.01, 10
)  # Volatilité idiosyncratique bornée (ne s'annule pas)
sigma0 = lambda x: np.clip(
    0.1 * x, 0.01, 5
)  # Volatilité du bruit commun bornée (ne s'annule pas)


# Fonction pour simuler une période
def simulate_one_period(x0, num_simulations, strategy=None, c=0.1):
    """
    Simule une seule période pour un ensemble d'agents avec bruit commun.
    """
    X = np.zeros(num_simulations)
    epsilon_0 = np.random.normal(0, np.sqrt(dt))  # Bruit commun
    epsilon = np.random.normal(0, 1, num_simulations)  # Bruits idiosyncratiques

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
    initial_x, num_simulations, strategy_example_26, c=0.1
)
results_27_normal = simulate_one_period(
    initial_x, num_simulations, strategy_example_27, c=0.1
)
results_null_normal = simulate_one_period(
    initial_x, num_simulations, strategy=None
)  # Stratégie nulle

# Calcul des moyennes et variances pour chaque cas
mean_26_normal, var_26_normal = np.mean(results_26_normal), np.var(results_26_normal)
mean_27_normal, var_27_normal = np.mean(results_27_normal), np.var(results_27_normal)
mean_null_normal, var_null_normal = np.mean(results_null_normal), np.var(
    results_null_normal
)

# Première figure : stratégie 2.6 vs stratégie nulle
plt.figure(figsize=(12, 6))  # Augmentation de la largeur pour un graphe plus "étiré"
plt.hist(
    results_26_normal,
    bins=30,
    alpha=0.6,
    color="blue",
    label="Stratégie unilatérale (2.6)",
)
plt.axvline(
    mean_26_normal,
    color="darkblue",
    linestyle="--",
    label=f"Moyenne (Ex. 2.6): {mean_26_normal:.2f}",
)
plt.axvline(
    mean_26_normal + np.sqrt(var_26_normal),
    color="blue",
    linestyle=":",
    label=f"Écart-type (Ex. 2.6): {np.sqrt(var_26_normal):.2f}",
)
plt.axvline(mean_26_normal - np.sqrt(var_26_normal), color="blue", linestyle=":")

plt.hist(results_null_normal, bins=30, alpha=0.6, color="gray", label="Stratégie nulle")
plt.axvline(
    mean_null_normal,
    color="black",
    linestyle="--",
    label=f"Moyenne (Stratégie nulle): {mean_null_normal:.2f}",
)
plt.axvline(
    mean_null_normal + np.sqrt(var_null_normal),
    color="gray",
    linestyle=":",
    label=f"Écart-type (Stratégie nulle): {np.sqrt(var_null_normal):.2f}",
)
plt.axvline(mean_null_normal - np.sqrt(var_null_normal), color="gray", linestyle=":")

plt.title(
    "Histogramme des valeurs finales : stratégie 2.6 vs stratégie nulle", fontsize=14
)
plt.xlabel("Valeur finale", fontsize=12)
plt.ylabel("Fréquence", fontsize=12)
plt.legend(loc="upper left", fontsize=10)  # Déplacement de la légende
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Deuxième figure : stratégie 2.7 vs stratégie nulle
plt.figure(figsize=(12, 6))  # Augmentation de la largeur pour un graphe plus "étiré"
plt.hist(
    results_27_normal,
    bins=30,
    alpha=0.6,
    color="orange",
    label="Stratégie à variables séparables (2.7)",
)
plt.axvline(
    mean_27_normal,
    color="darkorange",
    linestyle="--",
    label=f"Moyenne (Ex. 2.7): {mean_27_normal:.2f}",
)
plt.axvline(
    mean_27_normal + np.sqrt(var_27_normal),
    color="orange",
    linestyle=":",
    label=f"Écart-type (Ex. 2.7): {np.sqrt(var_27_normal):.2f}",
)
plt.axvline(mean_27_normal - np.sqrt(var_27_normal), color="orange", linestyle=":")

plt.hist(results_null_normal, bins=30, alpha=0.6, color="gray", label="Stratégie nulle")
plt.axvline(
    mean_null_normal,
    color="black",
    linestyle="--",
    label=f"Moyenne (Stratégie nulle): {mean_null_normal:.2f}",
)
plt.axvline(
    mean_null_normal + np.sqrt(var_null_normal),
    color="gray",
    linestyle=":",
    label=f"Écart-type (Stratégie nulle): {np.sqrt(var_null_normal):.2f}",
)
plt.axvline(mean_null_normal - np.sqrt(var_null_normal), color="gray", linestyle=":")

plt.title(
    "Histogramme des valeurs finales : stratégie 2.7 vs stratégie nulle", fontsize=14
)
plt.xlabel("Valeur finale", fontsize=12)
plt.ylabel("Fréquence", fontsize=12)
plt.legend(loc="upper left", fontsize=10)  # Déplacement de la légende
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
