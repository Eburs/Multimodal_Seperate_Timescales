from matplotlib import pyplot as plt
import numpy as np
import torch

data = torch.load(
    "./data/correlated_system/lorenz_bursting_1vec_correlated_train_reshaped.pt"
).numpy()
"""
N, T, dz = data.shape
bursting_neuron_data = data[:, :, :3]

example = bursting_neuron_data[0, :, :]

example_fft = np.fft.fft(example, axis=0)
plt.plot(example_fft[:, 0].real, label="X")
plt.show()
"""
bursting_neuron_data = data[:, :, :3]


N, T, dz = data.shape
n = int(np.ceil(np.sqrt(N)))

for i in range(5):
    fig = plt.figure(
        layout="compressed",
    )
    ax = fig.add_subplot(
        1,
        1,
        1,
        projection="3d",
    )
    ax.plot(
        data[i, :, 0],
        data[i, :, 1],
        data[i, :, 2],
        label="ground truth",
    )
    ax.set_title("Subject {}".format(i))
    ax.legend()

    plt.show()
