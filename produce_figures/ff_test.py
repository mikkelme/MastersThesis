
import numpy as np
import matplotlib.pyplot as plt







N = 1e6
x = np.linspace(0, 1000, int(N))

a = 0.0135
b = 0.385
# y = np.sin(2*np.pi*f1*x)*np.sin(2*np.pi*f2*x)
y = np.sin(2*np.pi*a*x) * np.sin(2*np.pi*b*x)

# plt.plot(x, y)
# plt.show()

ran = np.arange(int(len(y)/2))
ff = np.fft.fft(y)/len(y)
ff = ff[ran]

freq = ran/x[-1]
A = abs(ff)

fzoom = freq[:len(freq)//200]
Azoom = A[:len(freq)//200]

sort = np.argsort(Azoom)

f1, f2 = fzoom[sort][-2:]
a = (f1+f2)/2
b = np.abs(f1-f2)/2

# Implement this in baseline_single_measurement
print(a,b)


plt.plot(fzoom, Azoom)
plt.show()