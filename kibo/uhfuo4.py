import numpy as np
import matplotlib.pyplot as plt

E = np.linspace(200, 10000, 100)
L = np.piecewise(E, 
    [E < 1800, (1800 <= E) & (E < 2000), E >= 2000],
    [lambda x: 2 + 0.01 * x,
     lambda x: 0.5 + 0.001 * x,
     lambda x: 10 + 0.02 * (x - 2000)]
)

T = np.exp(-200 / L)

plt.semilogx(E, T)
plt.xlabel("Photon Energy (eV)")
plt.ylabel("Transmission")
plt.title("X-ray Transmission Through 200 μm Silicon (with K-edge)")
plt.grid(True)
plt.show()
