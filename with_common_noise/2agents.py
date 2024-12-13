import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# Paramètres du modèle
np.random.seed(42)
T = 1.0  # Horizon temporel
n_simulations = 10000  # Nombre de simulations Monte Carlo

# Fonctions b, sigma, sigma0
b = lambda x: np.clip(0.05 * x, -5, 5)
sigma = lambda x: np.clip(0.2 * x, 0.01, 10)
sigma0 = lambda x: np.clip(0.1 * x, 0.01, 5)

# Fonction pour résoudre l'équation (6) par la méthode des racines


def equations(pi, x1_0, x2_0):
    pi12, pi21 = pi
    det_psi = 1 + 0.5 * (pi12 + pi21) - 0.75 * pi12 * pi21

    if det_psi <= 0:
        return [np.inf, np.inf]  # Assurer l'inversibilité

    # Application de l'égalité (5) pour calculer X
    epsilon1 = np.random.normal(0, 1, n_simulations)
    epsilon2 = np.random.normal(0, 1, n_simulations)
    epsilon0 = np.random.normal(0, 1, n_simulations)

    b1, b2 = b(x1_0), b(x2_0)
    sigma1, sigma2 = sigma(x1_0), sigma(x2_0)
    sigma0_1, sigma0_2 = sigma0(x1_0), sigma0(x2_0)

    delta_x1 = (
        (1 + 0.5 * pi21) * (b1 + sigma1 * epsilon1 + sigma0_1 * epsilon0)
        + pi12 * (b2 + sigma2 * epsilon2 + sigma0_2 * epsilon0)
    ) / det_psi

    delta_x2 = (
        (1 + 0.5 * pi12) * (b2 + sigma2 * epsilon2 + sigma0_2 * epsilon0)
        + pi21 * (b1 + sigma1 * epsilon1 + sigma0_1 * epsilon0)
    ) / det_psi

    x1_final = x1_0 + delta_x1
    x2_final = x2_0 + delta_x2

    # Conditions d'optimalité
    eq1 = np.mean(-2 * x1_final * ((1 + 0.5 * pi21) * b1 + pi12 * b2))
    eq2 = np.mean(-2 * x2_final * ((1 + 0.5 * pi12) * b2 + pi21 * b1))

    return [eq1, eq2]


# Résolution de pi
x1_0, x2_0 = 100, 100
initial_guess = [0.5, 0.5]

result = root(equations, initial_guess, args=(x1_0, x2_0))
if result.success:
    pi12_opt, pi21_opt = result.x
    print(f"Solutions optimales: pi12 = {pi12_opt:.3f}, pi21 = {pi21_opt:.3f}")
else:
    raise ValueError("Échec de la résolution des équations.")


# Simulation des trajectoires pour les pi optimaux
def simulate_agents(x1_0, x2_0, n_simulations, pi12, pi21):
    epsilon1 = np.random.normal(0, 1, n_simulations)
    epsilon2 = np.random.normal(0, 1, n_simulations)
    epsilon0 = np.random.normal(0, 1, n_simulations)

    b1, b2 = b(x1_0), b(x2_0)
    sigma1, sigma2 = sigma(x1_0), sigma(x2_0)
    sigma0_1, sigma0_2 = sigma0(x1_0), sigma0(x2_0)

    det_psi = 1 + 0.5 * (pi12 + pi21) - 0.75 * pi12 * pi21

    delta_x1 = (
        (1 + 0.5 * pi21) * (b1 + sigma1 * epsilon1 + sigma0_1 * epsilon0)
        + pi12 * (b2 + sigma2 * epsilon2 + sigma0_2 * epsilon0)
    ) / det_psi

    delta_x2 = (
        (1 + 0.5 * pi12) * (b2 + sigma2 * epsilon2 + sigma0_2 * epsilon0)
        + pi21 * (b1 + sigma1 * epsilon1 + sigma0_1 * epsilon0)
    ) / det_psi

    return x1_0 + delta_x1, x2_0 + delta_x2


x1_final, x2_final = simulate_agents(x1_0, x2_0, n_simulations, pi12_opt, pi21_opt)

# Affichage des histogrammes
plt.figure(figsize=(12, 6))
plt.hist(x1_final, bins=30, alpha=0.7, color="blue", label=r"$X^1_1$")
plt.hist(x2_final, bins=30, alpha=0.7, color="orange", label=r"$X^2_1$")
plt.title("Histogramme des valeurs finales pour les deux agents", fontsize=14)
plt.xlabel("Valeurs finales", fontsize=12)
plt.ylabel("Fréquence", fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()
