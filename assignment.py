import numpy as np
import matplotlib.pyplot as plt

R1 = 50e3  # 50 K
C = 0.1e-6  # 0.1 u
Vcc = 5  # 5V

tau_charge = R1 * C

V_initial = 0
V_final = Vcc

t_charge_values = np.linspace(0, 5*tau_charge, 500)

V_charge_values = V_final + (V_initial - V_final) * np.exp(-t_charge_values/tau_charge)


time_to_final = 5 * tau_charge

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t_charge_values*1e3, V_charge_values)
plt.axhline(V_final, color='r', label="Final Voltage (5V)")
plt.axvline(time_to_final*1e3, color='g', label=f"5T = {time_to_final*1e3:.2f} ms")
plt.title("Voltage Increase over Time")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True)
plt.show()
