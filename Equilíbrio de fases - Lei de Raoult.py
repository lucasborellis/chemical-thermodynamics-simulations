# Acetonitrila (1) e Nitrometano (2), T[°C] e Psat[kPa]; Orvalho P=(T=90°C); Orvalho T=(P=100kPa)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

T = 90  # Temperatura para orvalho P
P = 100  # Pressão para orvalho T
N = 50 # Número de pontos a serem traçados

# Função para calcular as pressões de saturação (Lei de Raoult)
def P1sat(T):
    return np.exp(14.2724 - 2945.47 / (T + 224))
def P2sat(T):
    return np.exp(14.2043 - 2972.64 / (T + 209))

def orvalho_P(T, N=N):# Função para calcular orvalho P (T cte)
    y1 = np.linspace(0, 1, N)
    y2 = 1 - y1
    P = 1 / (y1 / P1sat(T) + y2 / P2sat(T))  # Pressão total
    x1 = (y1 * P) / P1sat(T)
    return y1, x1, P

def objetivo_orvalho_T(T, y1, P):# Função objetivo para encontrar o orvalho T
    y2 = 1 - y1
    return P - 1 / (y1 / P1sat(T) + y2 / P2sat(T))

def orvalho_T(P, N=N, tol=1e-11, max_iter=1000):# Função para calcular orvalho T (P cte) com precisão ajustada
    y1 = np.linspace(0, 1, N)
    T_orvalho = []
    x1_orvalho = []
    for y1i in y1:
        # Resolve T para cada fração molar y1i, com chute inicial T=90°C
        T = fsolve(objetivo_orvalho_T, 90, args=(y1i, P), xtol=tol, maxfev=max_iter)[0]
        T_orvalho.append(T)
        x1_orvalho.append((y1i * P) / P1sat(T))
    return y1, x1_orvalho, T_orvalho

y1_P, x1_P, P_values = orvalho_P(T)# Cálculo para orvalho P (T cte)
y1_T, x1_T, T_values = orvalho_T(P, tol=1e-11, max_iter=1000)# Cálculo para orvalho T (P cte), ajustando a tolerância

print("\nORVALHO P (T = 90°C):")# Output orvalho P
for i in range(len(P_values)):
    print(f"Iteração{i+1}, x1 = {x1_P[i]:.5f}, y1 = {y1_P[i]:.5f}, P = {P_values[i]:.5f} kPa")

print("\nORVALHO T (P = 100 kPa):")# Output orvalho T
for i in range(len(T_values)):
    print(f"Iteração{i+1}, x1 = {x1_T[i]:.5f}, y1 = {y1_T[i]:.5f}, T = {T_values[i]:.5f} °C")

plt.figure(figsize=(12, 6))# Plotagem dos gráficos
plt.subplot(1, 2, 1)# Gráfico 1: Orvalho P (T cte)
plt.plot(y1_P, P_values, label="Vapor (y1)", marker='o')
plt.plot(x1_P, P_values, label="Líquido (x1)", marker='s')
plt.title(f"Diagrama de Fases L+V - Orvalho P (T = {T}°C)")
plt.xlabel("Fração molar (x1,y1)")
plt.ylabel("Pressão (kPa)")
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2) # Gráfico 2: Orvalho T (P cte)
plt.plot(y1_T, T_values, label="Vapor (y1)", marker='o')
plt.plot(x1_T, T_values, label="Líquido (x1)", marker='s')
plt.title(f"Diagrama de Fases L+V - Orvalho T (P = {P} kPa)")
plt.xlabel("Fração molar (x1,y1)")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()