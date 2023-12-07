# Creado por Francisco Bolaños Becerra

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parámetros del modelo de Hodgkin-Huxley
C_m = 1.0
g_Na = 120.0
g_K = 36.0
g_L = 0.3
E_Na = 50.0
E_K = -77.0
E_L = -54.387
I_ext = 11

# Funciones de las tasas de cambio para las variables m, h, n y v
def alpha_m(v):
    return 0.1 * (v + 40.0) / (1.0 - np.exp(-(v + 40.0) / 10.0))

def beta_m(v):
    return 4.0 * np.exp(-(v + 65.0) / 18.0)

def alpha_h(v):
    return 0.07 * np.exp(-(v + 65.0) / 20.0)

def beta_h(v):
    return 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))

def alpha_n(v):
    return 0.01 * (v + 55.0) / (1.0 - np.exp(-(v + 55.0) / 10.0))

def beta_n(v):
    return 0.125 * np.exp(-(v + 65.0) / 80.0)

# Ecuaciones diferenciales del modelo de Hodgkin-Huxley completo
def hodgkin_huxley(y, t):
    v, m, h, n = y
    dvdt = (I_ext - g_Na * m**3 * h * (v - E_Na) - g_K * n**4 * (v - E_K) - g_L * (v - E_L)) / C_m
    dmdt = alpha_m(v) * (1 - m) - beta_m(v) * m
    dhdt = alpha_h(v) * (1 - h) - beta_h(v) * h
    dndt = alpha_n(v) * (1 - n) - beta_n(v) * n
    return [dvdt, dmdt, dhdt, dndt]

# Condiciones iniciales y tiempo de integración
y0 = [-65.0, 0.05, 0.23, 0.6]  # Valores iniciales para v, m, h, n
t = np.linspace(0, 100, 1000)

# Integración numérica de las ecuaciones diferenciales
sol = odeint(hodgkin_huxley, y0, t)
g = sol[:, 2]+ sol[:, 3]
y = 0.89-1.1*t

def hodgkin_huxley_2d(y, t, I):
    v, n = y
    # Ecuaciones del modelo de Hodgkin-Huxley 2D
    m_inf = (alpha_m(v))/(alpha_m(v)+beta_m(v))

    # Término de corriente de sodio
    dvdt = (1 / C_m) * (I - g_K * n**4 * (v - E_K)-g_Na* m_inf**3 *(0.89-1.1*n)*(v-E_Na) - g_L * (v - E_L))
    dndt = alpha_n(v) * (1 - n) - beta_n(v) * n
    return [dvdt, dndt]

# Condiciones iniciales
initial_conditions_2d = [-65, 0.6]  # Valores iniciales para v, n

# Resolver las ecuaciones diferenciales
solution_2d = odeint(hodgkin_huxley_2d, initial_conditions_2d, t, args=(I_ext,))

# Gráfica de las soluciones
plt.figure(figsize=(12, 8))
plt.plot(t, sol[:, 0], label='V')
plt.title('Simulación del modelo de Hodgkin-Huxley')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Potencial de membrana (mV)')
plt.legend()
plt.grid(True)
plt.show()

# Gráfica de h vs n
plt.figure(figsize=(8, 6))
plt.plot(sol[:, 3], sol[:, 2])
plt.title('Relación entre n y h en el modelo de Hodgkin-Huxley')
plt.plot(t, y, color='b')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('n')
plt.ylabel('h')
plt.grid(True)
plt.show()

#Grafica de n,h,m
plt.figure(figsize=(12, 8))

plt.plot(t, sol[:, 1], label='m')
plt.plot(t, sol[:, 2], label='h')
plt.plot(t, sol[:, 3], label='n')
plt.legend()
plt.title('Variables de Comportamiento')

plt.tight_layout()
plt.show()

#Grafica de n+h
plt.figure(figsize=(12, 8))

plt.plot(t, g, label='n+h')
plt.legend()
plt.title('n+h')
plt.ylim(0, 3)

plt.tight_layout()
plt.show()

#grafica comparación HH y HH reducido 

plt.figure(figsize=(12, 6))
plt.plot(t, solution_2d[:, 0],label='v reduccion')
plt.plot(t, sol[:, 0], label='V')
plt.ylim(-66, 150)
plt.title('comparación Potencial de Membrana (V)')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Potencial de Membrana (mV)')
plt.legend()
plt.tight_layout()
plt.show()

#Grafica comparación n y n reducida
plt.figure(figsize=(12, 6))
plt.plot(t, solution_2d[:, 1], label='n reducida')
plt.plot(t, sol[:, 3], label='n')
plt.legend()
plt.title('Comparación variable de Activación (n)')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Probabilidad')

plt.tight_layout()
plt.show()
