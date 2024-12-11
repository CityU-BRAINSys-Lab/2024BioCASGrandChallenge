import matplotlib.pyplot as plt
import matplotlib
from spiffyplots import MultiPanel
import seaborn as sns

import torch
import numpy as np
from scipy.signal import find_peaks

# prevent matplotlib from spamming the console
import logging

logging.getLogger("matplotlib").setLevel(logging.ERROR)

import stork


def plot_activity_snapshot(model, data, save_path=None):

    fig, ax = stork.plotting.plot_activity_CST(
        model,
        data=data,
        figsize=(10, 5),
        dpi=250,
        pos=(0, 0),
        off=(0.0, -0.05),
        turn_ro_axis_off=True,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=250)

    return fig, ax


def plot_training(
    results,
    nb_epochs,
    names=[
        "loss",
        "r2",
    ],
    save_path=None,
):
    fig, ax = plt.subplots(
        2,
        len(names),
        figsize=(7, 3),
        dpi=150,
        sharex=True,
        sharey="col",
        constrained_layout=True,
    )

    for i, n in enumerate(names):

        for j, s in enumerate(["x", "y"]):
            if "loss" in n:
                ax[0][i].plot(
                    results["train_{}".format(n)], color="black", label="train"
                )
                ax[0][i].plot(
                    results["val_{}".format(n)], color="black", alpha=0.5, label="valid"
                )

                ax[0][i].scatter(
                    nb_epochs,
                    results["test_{}".format(n)],
                    color="crimson",
                )
                ax[0][i].set_ylabel(n)
            else:
                ax[j][i].plot(
                    results["train_{}{}".format(n, s)], color="black", label="train"
                )
                ax[j][i].plot(
                    results["val_{}{}".format(n, s)],
                    color="black",
                    alpha=0.5,
                    label="valid",
                )

                ax[j][i].scatter(
                    nb_epochs,
                    results["test_{}{}".format(n, s)],
                    color="crimson",
                    label="test",
                )

                ax[j][i].set_ylabel("{} {}".format(n, s))
            
            if "r2" in n:
                ax[j][i].set_ylim(0, 1)

            ax[1][i].set_xlabel("Epochs")

    ax[0][-1].legend()
    ax[1][0].axis("off")
    ax[0][0].set_xlabel("Epochs")

    sns.despine()

    if save_path is not None:
        fig.savefig(save_path, dpi=250)
    return fig, ax


def plot_cumulative_mse(model, dataset, n_samples=5, save_path=None):

    fig, ax = plt.subplots(
        4, n_samples, figsize=(2 * n_samples, 3), dpi=250, sharex=True, sharey="row"
    )
    preds = model.predict(dataset)
    targets = dataset.labels

    for s in range(5):
        curr_preds = preds[s]
        curr_targets = targets[s]

        for i, idx in enumerate([0, 2]):
            ax[idx][s].hlines(0, len(curr_preds), 0, color="silver")
            ax[idx][s].plot(
                curr_targets[:, i], c="crimson", label="target", alpha=0.75, lw=1
            )
            ax[idx][s].plot(
                curr_preds[:, i], c="k", label="prediction", alpha=0.5, lw=1
            )

            # plot cumulative mse loss
            cs_se = np.cumsum((curr_targets[:, i] - curr_preds[:, i]) ** 2)
            ax[idx + 1][s].plot(cs_se / cs_se[-1], c="k")

            ax[0][s].set_title(f"$v_x$, cs_se = {cs_se[-1].item():.04f}")
            ax[2][s].set_title(f"$v_y$, cs_se = {cs_se[-1].item():.04f}")

            # Compute the absolute values of the segment
            abs_segment = torch.abs(curr_targets[:, i])

            # Find peaks in the absolute values

            peaks, _ = find_peaks(abs_segment.numpy())

            # threshold the peaks
            peaks = peaks[abs_segment.numpy()[peaks] > 0.25]

            # Plot the peaks
            ax[idx][s].vlines(peaks, -1.5, 1.5, color="silver", alpha=0.5)
            ax[idx + 1][s].vlines(peaks, 0, 1, color="silver", alpha=0.5)

            ax[idx][s].set_ylim(-1.5, 1.5)
            ax[idx + 1][s].set_ylim(0, 1)

        ax[0][0].set_ylabel(f"$v_x$")
        ax[1][0].set_ylabel("Cum.\nSE x")
        ax[2][0].set_ylabel(f"$v_y$")
        ax[3][0].set_ylabel("Cum.\nSE y")

    ax[0][-1].legend()

    sns.despine()
    if save_path is not None:
        fig.savefig(save_path, dpi=250)
    return fig, ax
