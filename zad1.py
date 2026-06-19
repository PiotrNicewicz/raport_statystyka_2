import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

with open('dane1.txt', 'r', encoding='utf-8') as plik:
    dane = [float(linia) for linia in plik]

sigma = 0.2
mu0 = 1.5
alpha = 0.05

n = len(dane)
x_sr = np.mean(dane)

Z = (x_sr - mu0) / (sigma / np.sqrt(n))

print("=" * 50)
print("Liczność próby:", n)
print("Średnia z próby:", round(x_sr, 5))
print("Statystyka Z =", round(Z, 5))
print("=" * 50)

# 1. H1: mi != 1.5

z_kryt = norm.ppf(1 - alpha / 2)
p_value = 2 * (1 - norm.cdf(abs(Z)))

print("\nTEST DWUSTRONNY")
print("H0: mi = 1.5")
print("H1: mi != 1.5")
print("Wartość krytyczna:", round(z_kryt, 4))
print("p-value =", round(p_value, 6))

if abs(Z) > z_kryt:
    print("Odrzucamy H0")
else:
    print("Brak podstaw do odrzucenia H0")

# 2. H1: μ > 1.5

z_kryt = norm.ppf(1 - alpha)
p_value = 1 - norm.cdf(Z)

print("\nTEST PRAWOSTRONNY")
print("H0: mi = 1.5")
print("H1: mi > 1.5")
print("Wartość krytyczna:", round(z_kryt, 4))
print("p-value =", round(p_value, 6))

if Z > z_kryt:
    print("Odrzucamy H0")
else:
    print("Brak podstaw do odrzucenia H0")

# 3. H1: μ < 1.5

z_kryt = norm.ppf(alpha)
p_value = norm.cdf(Z)

print("\nTEST LEWOSTRONNY")
print("H0: mi = 1.5")
print("H1: mi < 1.5")
print("Wartość krytyczna:", round(z_kryt, 4))
print("p-value =", round(p_value, 6))

if Z < z_kryt:
    print("Odrzucamy H0")
else:
    print("Brak podstaw do odrzucenia H0")


x = np.linspace(-4, 4, 500)
y = norm.pdf(x)

fig, ax = plt.subplots(1, 3, figsize=(16, 5))

ax[0].plot(x, y)

z = norm.ppf(1 - alpha / 2)

ax[0].fill_between(
    x, 0, y,
    where=(x <= -z),
    alpha=0.4
)

ax[0].fill_between(
    x, 0, y,
    where=(x >= z),
    alpha=0.4
)

ax[0].axvline(Z, color='red', linestyle='--', label='Z')
ax[0].set_title("H1: mi != 1.5")
ax[0].legend()

ax[1].plot(x, y)

z = norm.ppf(1 - alpha)

ax[1].fill_between(
    x, 0, y,
    where=(x >= z),
    alpha=0.4
)

ax[1].axvline(Z, color='red', linestyle='--', label='Z')
ax[1].set_title("H1: mi > 1.5")
ax[1].legend()

ax[2].plot(x, y)

z = norm.ppf(alpha)

ax[2].fill_between(
    x, 0, y,
    where=(x <= z),
    alpha=0.4
)

ax[2].axvline(Z, color='red', linestyle='--', label='Z')
ax[2].set_title("H1: mi < 1.5")
ax[2].legend()

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("Wpływ poziomu ufności")
print("=" * 50)

print("""
    Poziom ufności = 1 - alpha.

    Jeżeli zwiększamy poziom ufności (np. z 95% do 99%):
    - alpha maleje,
    - obszar krytyczny staje się mniejszy,
    - trudniej odrzucić hipotezę H0,
    - p-value się nie zmienia.

    Jeżeli zmniejszamy poziom ufności (np. z 95% do 90%):
    - alpha rośnie,
    - obszar krytyczny staje się większy,
    - łatwiej odrzucić hipotezę H0,
    - p-value nadal pozostaje takie samo.
    """)