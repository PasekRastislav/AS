import matplotlib.pyplot as plt
import numpy as np

# batch size (approx)
batch_size = 1000

# input times
times = [0.0017011165618896484, 0.0010199546813964844, 0.0009760856628417969, 0.0009999275207519531, 0.0010428428649902344, 0.000888824462890625, 0.0009698867797851562, 0.0009698867797851562, 0.0009059906005859375, 0.0009641647338867188]

# compute cumulative times
times_cumulative = np.cumsum(times)

# compute cumulative rows processed
rows_cumulative = [batch_size * (i+1) for i in range(len(times))]

plt.figure(figsize=(8,5))
plt.plot(times_cumulative, rows_cumulative, marker='o', color='orange')
plt.xlabel('Kumulatívny čas spracovania (s)')
plt.ylabel('Počet spracovaných riadkov')
plt.title('Hybrid: Priebeh spracovania v čase')
plt.grid(True)
plt.tight_layout()
plt.show()
