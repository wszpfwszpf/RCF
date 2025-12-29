import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

filepath = r'C:\Users\93084\Desktop\自己论文写作\2.降噪\RCF\data\LDV\Off_set1_trail5.txt' # The file path of the ldv data (.txt format)

with open(filepath, 'r', encoding='utf-8') as f:
    data = np.loadtxt(f, skiprows=5)
data_x = data[:,1]
data_t = data[:,0]
mask = (data_t >= 0.6) & (data_t <= 3.0)
t_sel = data_t[mask] - 0.6
x_sel = data_x[mask]
plt.figure(figsize=(5, 5))
plt.plot(t_sel, x_sel)
plt.title('LDV signal', fontsize=14, fontweight='bold')
plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
plt.ylabel('Velocity (m/s)', fontsize=14, fontweight='bold')
plt.show()
