import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

#[1] Metanol // [2] Acetato de Metila ----------------- Lei de Raoult Modificada
T = 330  # Temperatura fixa em K para orvalho P
P = 202.66  # Pressão constante em kPa para orvalho T
N = 100  # Número de pontos a serem traçados

y1 = np.linspace(0, 1, N) #Gera N valores equidistantes para y1
y2 = 1 - y1

# Funções para calcular as pressões de vapor (Lei de Antoine)
def P1sat(T):
    return np.exp(16.59158 - 3643.31 / (T - 33.424)) # Calcula P1 sat
def P2sat(T):
    return np.exp(14.25326 - 2655.54 / (T - 53.424)) # Calcula P2 sat

# Correlação para Atividade e Coeficientes de Atividade (Gama)
def Atividade(T):
    return 2.771 - 0.00523 * T # Calcula a atividade
def Gama1(A, x2):
    return np.exp(A * (x2 ** 2)) # Calcula o coeficiente de atividade 1
def Gama2(A, x1):
    return np.exp(A * (x1 ** 2)) # Calcula o coeficiente de atividade 2

A = Atividade(T) # Obtendo o valor da função A
P1 = P1sat(T) # Obtendo o valor da função P1sat
P2 = P2sat(T) # Obtendo o valor da função P2sat

# Cálculo do Orvalho P
Orv_P = 1 / (y1 / (P1 * Gama1(A, y2)) + y2 / (P2 * Gama2(A, y1))) # Função Orvalho P
x1 = (y1 * P1 * Gama1(A, y2)) / Orv_P # x1 Orvalho P

# Define a impressão no terminal de seu computador, no formato:
# Orvalho P
# Iteração N, x1:x(N), y1:y(N), P: P(N) kPa 
print("Orvalho P")
for i, yi in enumerate(y1, start=1):
    Orv_P_iteração = Orv_P[np.argwhere(y1 == yi)][0][0]
    print(f"Iteração {i}, x1: {x1[i-1]:.4f}, y1: {yi:.4f}, P: {Orv_P_iteração:.4f} kPa")
#----------------------------------------------------------------------------------------------------
#Orvalho T:
T_orvt= []; y1_orvt = []; x1_orvt = [] #Listas par aarmazenar valores do Orvalho T

#Definição dos parâmetros para orvalho T e função objetivo
def func(T, y1):
    y2 = 1 - y1
    P1 = P1sat(T)
    P2 = P2sat(T)
    Orv_P = 1 / (y1 / (P1 * Gama1(Atividade(T), y2)) + y2 / (P2 * Gama2(Atividade(T), y1)))
    return Orv_P - P  # FUNÇÃO OBJETIVO!!!!!! ----- P - P deve ser igual a 0 - Resolvida na linha 58

# Cálculo do orvalho T usando fsolve
print("\nOrvalho T")
for i, yi in enumerate(y1, start=1):
    Chute_T = 300  # Chute inicial para T
    T_calculado = fsolve(func, Chute_T, args=(yi))[0] #Resolução da Função Objetivo, precisão de 20 casas
    y2_orvt = 1 - yi #y2 para orvalho T
    xi = (yi * P1sat(T_calculado) * Gama1(Atividade(T_calculado), y2_orvt)) / P # Cálculo de x1
    T_orvt.append(T_calculado); y1_orvt.append(yi); x1_orvt.append(xi) # Armazena os resultados
# Define a impressão no terminal de seu computador, no formato:
# Orvalho T
# Iteração N, x1:x(N), y1:y(N), T: T(N) K
    print(f"Iteração {i}, x1: {xi:.4f}, y1: {yi:.4f}, T: {T_orvt[i-1]:.4f} K")

#---------------------------------------------------------------------------------
# Configurações da plotagem do gráfico
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
axs[0].plot(x1, Orv_P, label='P vs x1', color='darkblue')
axs[0].plot(y1, Orv_P, label='P vs y1', color='darkred')
axs[0].set_title('Orvalho P (T=330 K)')
axs[0].set_xlabel('x1, y1')
axs[0].set_ylabel('Pressão (kPa)')
axs[0].grid()
axs[0].legend()
axs[1].plot(y1_orvt, T_orvt, label='T vs y1', color='darkgreen')
axs[1].plot(x1_orvt, T_orvt, label='T vs x1', color='darkgoldenrod')
axs[1].set_title('Orvalho T (P=202.66 kPa)')
axs[1].set_xlabel('y1, x1')
axs[1].set_ylabel('Temperatura (K)')
axs[1].grid()
axs[1].legend()
plt.tight_layout()
plt.show()
