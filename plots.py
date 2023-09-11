import torch
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Optional


def plot_tracking_error(save_dir: Path, errors: list):
    num_days = len(errors)
    fig = plt.figure()
    fig.add_subplot(111)
    plt.bar(list(range(359 - num_days, 359)), errors)
    plt.savefig(save_dir / Path('daily_tracking_errors.png'))
    plt.close()
    return


def plot_mean_tracking_error(save_dir: Path, errors: list):
    num_days = len(errors)
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(list(range(359 - num_days, 359)), errors)
    plt.savefig(save_dir / Path('cumulative_tracking_errors.png'))
    plt.close()
    return


def plot_epoch_losses(save_dir: Path, losses: list):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(losses)
    plt.savefig(save_dir / Path('epoch_loss.png'))
    plt.close()
    return


def plot_min_errors(save_dir: Path, min_errors: list):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(min_errors)
    plt.savefig(save_dir / Path('min_errors.png'))
    plt.close()
    return


