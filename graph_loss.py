import numpy as np
import matplotlib.pyplot as plt

losses = np.load('losses.npy')

def smooth_curve(losses: np.ndarray, factor=0.9):
    smoothed = [losses[0]]
    for point in losses[1:]:
        smoothed.append(factor * smoothed[-1] + (1-factor) * point)
    return smoothed

def plot_loss(losses: np.ndarray, label: str, color: str):
    epochs = list(range(1, len(losses)+1))
    losses = np.log10(losses)

    plt.scatter(epochs, losses, color=color, s=1, alpha=0.2)
    smoothed = smooth_curve(losses)
    plt.plot(epochs, smoothed, color=color, label=label)
    

plot_loss(losses, 'MSE Loss', 'orange')
plt.xlabel('Epochs')
plt.ylabel('Log MSE Loss')
plt.title('Training Loss')
plt.savefig('losses.png')
