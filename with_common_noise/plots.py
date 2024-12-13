import numpy as np
import matplotlib.pyplot as plt

# Paramètres globaux
np.random.seed(42)
num_simulations = 10000
T = 1.0
dt = 1.0

# Paramètres du modèle avec bornes
b = lambda x: np.clip(0.05 * x, -5, 5)
sigma = lambda x: np.clip(0.2 * x, 0.01, 10)
sigma0 = lambda x: np.clip(0.1 * x, 0.01, 5)


# Simulation d'une période
def simulate_one_period(x0, num_simulations, strategy=None, c=0.1):
    X = np.zeros(num_simulations)
    epsilon_0 = np.random.normal(0, np.sqrt(dt))
    epsilon = np.random.normal(0, 1, num_simulations)
    common_term = sigma0(x0) * epsilon_0
    individual_term = sigma(x0) * epsilon * np.sqrt(dt)
    drift_term = b(x0) * dt
    interaction_term = strategy(x0, x0, c) * dt if strategy else 0
    X[:] = x0 + drift_term + common_term + individual_term + interaction_term
    return X


# Stratégies
def strategy_example_26(x, x_hat, c=0.1):
    return b(x_hat) / np.mean(b(x_hat)) - c


def strategy_example_27(x, x_hat, c=0.1):
    return c * (b(x_hat) / np.mean(b(x_hat)) - 1)


# Données initiales
initial_x = 100
results_26 = simulate_one_period(initial_x, num_simulations, strategy_example_26, c=0.1)
results_27 = simulate_one_period(initial_x, num_simulations, strategy_example_27, c=0.1)
results_null = simulate_one_period(initial_x, num_simulations, strategy=None)

# Création d'un DataFrame pour faciliter la visualisation
import pandas as pd

data = pd.DataFrame(
    {
        "Stratégie 2.6": results_26,
        "Stratégie 2.7": results_27,
        "Stratégie nulle": results_null,
    }
)

# Graphique 1 : Densité cumulée (CDF)
plt.figure(figsize=(12, 6))
colors = ["blue", "orange", "gray"]

for col, color in zip(data.columns, colors):
    sns.ecdfplot(data[col], linewidth=2, label=col, color=color)
    prob = np.mean(data[col] > 100)
    plt.axhline(1 - prob, color=color, linestyle="--", linewidth=1.5)
    plt.text(
        x=150,
        y=1 - prob,
        s=f"P(X > 100) = {prob:.2%}",
        color=color,
        fontsize=10,
        verticalalignment="bottom",
    )

plt.title("Fonctions de distribution cumulée (CDF) des provisions finales", fontsize=14)
plt.xlabel("Valeur finale des provisions", fontsize=12)
plt.ylabel("Probabilité cumulée", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Graphique 2 : Boîtes à moustaches (Boxplots)
plt.figure(figsize=(10, 6))
ax = sns.boxplot(data=data, orient="h", palette="Set2")

# Ajout des annotations pour la médiane et l'IQR
for i, col in enumerate(data.columns):
    median = data[col].median()
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1

    # Annoter la médiane
    ax.text(
        median,
        i,
        f"Médiane: {median:.2f}",
        color="black",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
    )

    # Annoter l'IQR
    ax.text(
        q3 + 5,
        i,
        f"IQR: {iqr:.2f}",
        color="black",
        ha="left",
        va="center",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
    )

plt.title("Distribution des provisions finales : Boxplots", fontsize=14)
plt.xlabel("Valeur finale des provisions", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Graphique 3 : Histogrammes empilés avec moyennes en pointillés
plt.figure(figsize=(12, 6))
means = []

for col, color in zip(data.columns, colors):
    sns.histplot(data[col], bins=30, kde=False, color=color, label=col, alpha=0.6)
    mean_value = data[col].mean()
    means.append(mean_value)
    plt.axvline(
        mean_value,
        color=color,
        linestyle="--",
        linewidth=2,
        label=f"Moyenne ({col}): {mean_value:.2f}",
    )

plt.title(
    "Histogrammes empilés des valeurs finales des provisions avec moyennes", fontsize=14
)
plt.xlabel("Valeur finale", fontsize=12)
plt.ylabel("Fréquence", fontsize=12)
plt.legend(loc="upper left", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
