import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#Input (Dados do Fluído - Acetato de Etila)
Tpt = 189.60  #Temperatura no ponto triplo - [K]
Tc = 523.30  #Temperatura no ponto crítico - [K]
Ppt = 1.43e-5  #Pressão no ponto triplo - [bar]
Pc = 38.80  #Pressão no ponto crítico - [bar]
w = 0.36641 #Fator Acêntrico
m = 0.37464 + 1.54226*w - 0.26992*w**2
Rc = 8.31451e-2  #Constante universal dos gases - [L*Bar/K*mol]
b = 0.0778*Rc*Tc/Pc

#Input (Dados Experimentais)
Texp = [190, 220, 250, 280, 310, 340, 370, 400, 430, 460, 490, 520]  # K
Pexp = [1.5e-5, 5.20e-4, 6.82e-3, 4.78e-2, 0.2158, 0.7109, 1.8590, 4.0970, 7.9460, 14.032, 23.185, 36.673]  # bar

#Função para calcular os parâmetros dependentes de P
def calcular_parametros(P, T):
    Tr = T / Tc #Temperatura reduzida
    at = (1 + m * (1 - np.sqrt(Tr))) ** 2 #Parâmetro de atividade
    a = 0.45724 * (Rc**2 * Tc**2 / Pc) * at #Parâmetro de atração de Peng-Robinson
    A = a * P / (Rc**2 * T**2) #Constante Adimensional da EDE
    B = b * P / (Rc * T) #Constante Adiensional da EDE
    # Coeficientes Peng-Robinson
    C2 = -(1 - B)
    C3 = (A - 2 * B - 3 * B**2)
    C4 = (-A * B + B**2 + B**3)
    Q = (C2**2 - 3 * C3) / 9 #Correlação de coeficientes EDE Linear
    R = (2 * C2**3 - 9 * C2 * C3 + 27 * C4) / 54 #Correlação de coeficientes EDE Quadrática
    QR = Q**3 - R**2
    # Raízes
    if QR < 0:
        S = ((R**2 - Q**3)**0.5 + abs(R))**(1/3) #Discriminante da equação cúbica
        SGN = np.sign(R)
        X1 = -SGN * (S + (Q / S)) - C2/3
        X2 = X1
        X3 = X1
    else:
        theta = np.arccos(R / np.sqrt(Q**3)) #Método de Cardano
        X1 = -2 * np.sqrt(Q) * np.cos(theta / 3) - C2/3
        X2 = -2 * np.sqrt(Q) * np.cos((theta + 2 * np.pi) / 3) - C2/3
        X3 = -2 * np.sqrt(Q) * np.cos((theta + 4 * np.pi) / 3) - C2/3

    Zl = np.min([X1, X2, X3]) #Raiz fase líquida
    Zv = np.max([X1, X2, X3]) #Raiz fase vapor

    phil = np.exp((Zl - 1) - np.log(Zl - B) + (A / (2 * np.sqrt(2) * B)) *
                  np.log((Zl + (1 - np.sqrt(2)) * B) / (Zl + (1 + np.sqrt(2)) * B))) #Coeficiente de fugacidade fase líquida
    phiv = np.exp((Zv - 1) - np.log(Zv - B) + (A / (2 * np.sqrt(2) * B)) *
                  np.log((Zv + (1 - np.sqrt(2)) * B) / (Zv + (1 + np.sqrt(2)) * B))) #Coeficiente de fugacidade fase vapor
    return phil, phiv

#Função objetivo para ajustar P
def func_objetivo(P, i):
    T = Texp[i]
    phil_i, phiv_i = calcular_parametros(P, T)
    return np.abs(P*(phil_i/phiv_i) - P)

#Pressão Calculada a partir da função objetivo:
Pcalc = []
for i in range(len(Texp)):
    result = minimize(func_objetivo, Pexp[i], args=(i,), method='L-BFGS-B', bounds=[(Ppt, Pc)], tol=1e-20)
    Pcalc.append(result.x[0])

#Output
for i in range(len(Texp)):
    print(f"Iteração {i+1}: Pexp = {Pexp[i]:.3e} bar, Pcalc = {Pcalc[i]:.3e} bar")

plt.figure(figsize=(10, 6))
plt.plot(Texp, Pexp, 'D-', label='Pexp (Experimental)', markersize=8)
plt.plot(Texp, Pcalc, 'H-', label='Pcalc (Ajustado)', markersize=8)
plt.plot(Tpt, Ppt, 'X', label='Ponto triplo', markersize=8)
plt.plot(Tc, Pc, 'o', label='Ponto crítico', markersize=8)
plt.title('Diagrama PxT')
plt.xlabel('Temperatura (K)')
plt.ylabel('Pressão (bar)')
plt.legend()
plt.grid(True)
plt.show()