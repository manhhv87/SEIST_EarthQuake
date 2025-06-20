"""
This module provides visualization utilities for waveform predictions and targets.
It includes functions for plotting waveforms, predictions, targets, and phase picking results.

Functions:
    - vis_waves_preds_targets: Visualizes and saves waveforms, predictions, and targets.
    - vis_phase_picking: Visualizes and saves waveform phase picking results with predicted and true phases.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "mathtext.fontset": "stix",
        "font.sans-serif": ["Arial"],
        "axes.unicode_minus": False,
    }
)


def vis_waves_preds_targets(
    waveforms: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    sampling_rate=None,
    save_dir="./",
    format="png",
):
    """
    Visualizes and saves waveforms, predictions, and targets in a plot.

    This function creates a plot with subplots for each waveform, predicted waveform, and target waveform.
    The waveforms are plotted in a stacked manner and can be saved as an image to a specified directory.

    Args:
        waveforms (np.ndarray): The waveforms to be visualized, typically with shape (channels, length).
        preds (np.ndarray): The predicted waveforms, typically with shape (channels, length).
        targets (np.ndarray): The target waveforms, typically with shape (channels, length).
        sampling_rate (float, optional): The sampling rate of the waveforms, used to convert sample indices to time in seconds.
        save_dir (str, optional): Directory where the plot will be saved. Default is the current directory.
        format (str, optional): The format of the saved plot (e.g., 'png', 'jpg'). Default is 'png'.

    Returns:
        None
    """
    fig = plt.figure()
    num_row = waveforms.shape[0] + preds.shape[0] + targets.shape[0]
    for idx, wave in enumerate(waveforms):
        plt.subplot(num_row, 1, idx + 1)
        if sampling_rate is None:
            plt.plot(wave, "-", color="k", linewidth=0.15, alpha=0.8)
        else:
            x = [i / sampling_rate for i in range(len(wave))]
            plt.plot(x, wave, "-", color="k", linewidth=0.15, alpha=0.8)
        plt.text(
            0.001,
            0.95,
            f"Channel-{idx}",
            horizontalalignment="left",
            verticalalignment="top",
            transform=plt.gca().transAxes,
            fontsize="small",
            fontweight="normal",
        )
        plt.ylim(-1, 1)
        plt.yticks([])

    for idx, data in enumerate(preds):
        plt.subplot(num_row, 1, waveforms.shape[0] + idx + 1)
        if sampling_rate is None:
            plt.plot(data, "-", color="k", linewidth=0.15, alpha=0.8)
        else:
            x = [i / sampling_rate for i in range(len(data))]
            plt.plot(x, data, "-", color="k", linewidth=0.15, alpha=0.8)
        plt.text(
            0.001,
            0.95,
            f"Pred-{idx}",
            horizontalalignment="left",
            verticalalignment="top",
            transform=plt.gca().transAxes,
            fontsize="small",
            fontweight="normal",
        )
        plt.yticks([])

    for idx, data in enumerate(targets):
        plt.subplot(num_row, 1, waveforms.shape[0] + preds.shape[0] + idx + 1)
        if sampling_rate is None:
            plt.plot(data, "-", color="k", linewidth=0.15, alpha=0.8)
        else:
            x = [i / sampling_rate for i in range(len(data))]
            plt.plot(x, data, "-", color="k", linewidth=0.15, alpha=0.8)
        plt.text(
            0.001,
            0.95,
            f"Target-{idx}",
            horizontalalignment="left",
            verticalalignment="top",
            transform=plt.gca().transAxes,
            fontsize="small",
            fontweight="normal",
        )
        plt.yticks([])

    if sampling_rate is None:
        plt.xlabel("Sample points")
    else:
        plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.0)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(
        f'{os.path.join(save_dir,datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))}.{format}',
        dpi=400,
    )
    plt.close()


def vis_phase_picking(
    waveforms: np.ndarray,
    waveforms_labels: list,
    preds: np.ndarray,
    true_phase_idxs: list,
    true_phase_labels: list,
    pred_phase_labels: list,
    sampling_rate: int = None,
    save_name="",
    save_dir="./",
    formats=["png"],
):
    """
    Visualizes and saves waveform phase picking results, including predicted and true phase labels.

    This function generates a plot that displays waveforms along with true and predicted phase indices.
    The phase probabilities for each phase are also plotted in the last row.

    Args:
        waveforms (np.ndarray): The waveforms to be visualized, typically with shape (channels, length).
        waveforms_labels (list): List of labels for each waveform (e.g., channel names).
        preds (np.ndarray): The predicted phase probabilities, typically with shape (phases, length).
        true_phase_idxs (list): List of true phase indices for each phase.
        true_phase_labels (list): List of labels for true phases.
        pred_phase_labels (list): List of labels for predicted phases.
        sampling_rate (int, optional): The sampling rate of the waveforms. If None, samples are assumed.
        save_name (str, optional): A string appended to the filename of the saved plot.
        save_dir (str, optional): Directory where the plot will be saved. Default is the current directory.
        formats (list, optional): List of image formats to save the plot (e.g., 'png', 'jpg'). Default is ['png'].

    Returns:
        None
    """
    fig = plt.figure(figsize=(10 / 2.54, 10 / 2.54))
    if sampling_rate is None:
        x = list(range(len(waveforms[0])))
    else:
        x = [i / sampling_rate for i in range(len(waveforms[0]))]
    num_row = waveforms.shape[0] + 1
    tmp_min = np.min(waveforms)
    tmp_max = np.max(waveforms)
    number_map = {0: "(a)", 1: "(b)", 2: "(c)", 3: "(d)"}
    for idx, wave in enumerate(waveforms):
        plt.subplot(num_row, 1, idx + 1)

        plt.plot(
            x, wave, "-", color="k", linewidth=1, alpha=0.8, label=waveforms_labels[idx]
        )
        if idx == 0 and true_phase_idxs:
            plt.vlines(
                x=[true_phase_idxs[0]],
                ymin=tmp_min * 1.1,
                ymax=tmp_max * 1.1,
                colors=["C1"],
                linestyles="solid",
                label=true_phase_labels[0],
            )
            plt.vlines(
                x=[true_phase_idxs[1]],
                ymin=tmp_min * 1.1,
                ymax=tmp_max * 1.1,
                colors=["C5"],
                linestyles="solid",
                label=true_phase_labels[1],
            )
        plt.ylim(tmp_min * 1.2, tmp_max * 1.2)
        plt.ylabel("Amplitude")
        plt.yticks([])
        plt.xticks([])
        plt.text(
            0.05,
            0.78,
            number_map[idx],
            horizontalalignment="center",
            transform=plt.gca().transAxes,
            fontsize=8,
            fontweight="normal",
            bbox=None,
        )
        legend = plt.legend(loc="upper right", fontsize=8, ncol=1)
        legend.get_frame().set_linewidth(0.75)

    plt.subplot(num_row, 1, num_row)
    plt.text(
        0.05,
        0.78,
        number_map[3],
        horizontalalignment="center",
        transform=plt.gca().transAxes,
        fontsize=8,
        fontweight="normal",
        bbox=None,
    )
    plt.plot(x, preds[0], f"-.C0", linewidth=1, alpha=0.8, label=pred_phase_labels[0])
    plt.ylabel("Probability")
    plt.plot(x, preds[1], f"--C1", linewidth=1, alpha=0.8, label=pred_phase_labels[1])
    plt.ylabel("Probability")
    plt.plot(x, preds[2], f"--C5", linewidth=1, alpha=0.8, label=pred_phase_labels[2])
    plt.ylabel("Probability")

    if sampling_rate is None:
        plt.xlabel("Samples")
    else:
        plt.xlabel("Time (s)")

    legend = plt.legend(loc="upper right", fontsize=8, ncol=1)

    _width = 0.75
    legend.get_frame().set_linewidth(_width)

    ax = plt.gca()
    ax.spines["top"].set_linewidth(_width)
    ax.spines["right"].set_linewidth(_width)
    ax.spines["bottom"].set_linewidth(_width)
    ax.spines["left"].set_linewidth(_width)

    plt.tight_layout()
    plt.gcf().align_labels()
    plt.subplots_adjust(wspace=0.1, hspace=0.05)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not isinstance(formats, list):
        formats = [formats]

    for fmt in formats:
        plt.savefig(
            f'{os.path.join(save_dir,datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+save_name)}.{fmt}',
            dpi=400,
        )
    plt.close()
