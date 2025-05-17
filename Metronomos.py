import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros físicos
N = 3  # número de metrónomos
m = 1.0  # masa del péndulo
L = 1.0  # longitud del péndulo
M = 5.0  # masa de la plataforma
g = 9.81  # gravedad

# Frecuencias naturales diferentes para cada metrónomo (omega_i = sqrt(g / L_i))
L_array = np.array([1.0, 0.95, 1.05])  
omega_array = np.sqrt(g / L_array)

# Constante de amortiguamiento y acoplamiento
b = 0.05 
k = 1.0   

# Ecuaciones diferenciales
def metronome_system(t, y):
    theta = y[0:N]
    omega = y[N:2*N]
    x = y[-2]
    v = y[-1]
    
    dtheta_dt = omega
    domega_dt = - (g / L_array) * np.sin(theta) - b * omega + (1 / L_array) * np.cos(theta) * (k * x)
    
    dx_dt = v
    dv_dt = -k * x - np.sum(m * L_array * domega_dt * np.cos(theta)) / M

    return np.concatenate([dtheta_dt, domega_dt, [dx_dt, dv_dt]])

# Condiciones iniciales: [theta_i, omega_i, x, v]
theta0 = [0.2, 0.4, -0.3] 
omega0 = [0.0, 0.0, 0.0]
x0 = 0.0
v0 = 0.0

y0 = np.concatenate([theta0, omega0, [x0, v0]])

# Tiempo de simulación
t_span = (0, 60)
t_eval = np.linspace(*t_span, 5000)

# Resolver el sistema
sol = solve_ivp(metronome_system, t_span, y0, t_eval=t_eval, method='RK45')

# Graficar los resultados
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(sol.t, sol.y[i], label=f'Metrónomo {i+1}')
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo (rad)')
plt.title('Sincronización de Metrónomos')
plt.legend()
plt.grid(True)
plt.show()
