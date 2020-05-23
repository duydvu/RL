import numpy as np
import matplotlib.pyplot as plt

from lib.utils import SMA


rewards = np.load(open('tmp/rewards.npy', 'rb'))
smoothed_rewards = SMA(rewards, 1000)

x = list(range(len(smoothed_rewards)))
y = smoothed_rewards

fig, ax = plt.subplots()
ax.plot(x, y)

ax.set(xlabel='episode', ylabel='reward',
       title='Reward over time')
ax.grid()

fig.savefig("test.png", dpi=300)
plt.show()
