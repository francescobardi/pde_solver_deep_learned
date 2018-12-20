import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_loss(losses):
    color_map = plt.get_cmap('cubehelix')
    colors = color_map(np.linspace(0.1, 1, 10))

    losses_fig = plt.figure()
    n_iter = np.arange(np.shape(losses)[0])
    plt.plot(n_iter[:], losses[:], color = colors[0], linewidth = 1, linestyle = "-", marker = "",  label='Loss')

    plt.legend(bbox_to_anchor=(0., -0.3), loc=3, borderaxespad=0.)
    plt.xlabel('n iteration', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Loss')
    plt.grid(True, which = "both", linewidth = 0.5,  linestyle = "--")


def plot_solution(gtt, output, N):
    Z_gtt = gtt.view(N, N).numpy()
    Z_output = output.detach().view(N, N).numpy()

    fig, axes = plt.subplots(nrows=1, ncols=2)

    fig.suptitle("Comparison")

    im_gtt = axes[0].imshow(Z_gtt)
    axes[0].set_title("Ground truth solution")

    im_output = axes[1].imshow(Z_output)
    axes[1].set_title("H method solution")

    fig.colorbar(im_gtt)
    fig.tight_layout()

    plt.show()