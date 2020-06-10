# Bayes-Klassifikator fuer einzelnes Merkmal
#
# TU Darmstadt, Modul 18-ad-2100, Sommersemester 2020
# Machine Learning und Deep Learning in der Automatisierungstechnik
# Vorlesung 1: Konzepte des Machine Learning

import numpy as np
import matplotlib.pyplot as plt


#m1,s1,N1,m2,s2,N2 = (1,.5,50,3,.7,100)
#m1,s1,N1,m2,s2,N2 = (2,.4,50,4,1.,100)
m1,s1,N1,m2,s2,N2 = (2,.5,50,2.2,.7,100)

# --- Dichten und A-Posteriori-Wktn ---

x  = np.linspace(0, 5, 101)
f1 = np.exp(-.5 * (x-m1)**2 / s1**2) / (np.sqrt(2*np.pi) * s1);
f2 = np.exp(-.5 * (x-m2)**2 / s2**2) / (np.sqrt(2*np.pi) * s2);
p1 = N1*f1 / (N1*f1 + N2*f2);
p2 = N2*f2 / (N1*f1 + N2*f2);

# --- Ergebnisse plotten ---

plt.subplot(211)
plt.plot(x, f1, x, f2)
plt.ylabel('PDF')

plt.subplot(212)
plt.plot(x, p1, x, p2)
plt.xlabel('x')
plt.ylabel('A-Posteriori')

plt.show()
