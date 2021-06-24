import matplotlib.pyplot as plt
import numpy as np

data = np.random.randint(0, 10, 1000)
counts = np.bincount(data)

# Switching to the OO-interface. You can do all of this with "plt" as well.
fig, ax = plt.subplots()
ax.bar(range(10), counts, width=1, align='center',edgecolor = 'black')
ax.set(xticks=range(10), xlim=[-1, 10])

plt.show()