from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_correctness_bar(systems: List[str], correctness_vals: List[float]) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(systems, correctness_vals)
    plt.ylim(0, 10)
    plt.ylabel("Correctness (0–10)")
    plt.title("Correctness by System")
    plt.show()


def plot_hallucination_bar(systems: List[str], hallucination_vals: List[float]) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(systems, hallucination_vals)
    plt.ylim(0, 10)
    plt.ylabel("Hallucination (0–10, lower is better)")
    plt.title("Hallucination by System")
    plt.show()


def plot_relevance_bar(systems: List[str], relevance_vals: List[float]) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(systems, relevance_vals)
    plt.ylim(0, 10)
    plt.ylabel("Evidence Relevance (0–10)")
    plt.title("Evidence Relevance by System")
    plt.show()


def plot_radar_chart(
    baseline_vals: List[float],
    improved_vals: List[float],
    gpt_vals: List[float],
) -> None:
    """
    Radar chart comparing Baseline / Improved / GPT-only
    across correctness, hallucination, and evidence relevance.
    """

    metrics = ["Correctness", "Hallucination", "Evidence Relevance"]
    num_vars = len(metrics)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    def close(vals: List[float]) -> List[float]:
        return vals + vals[:1]

    baseline_plot = close(baseline_vals)
    improved_plot = close(improved_vals)
    gpt_plot = close(gpt_vals)

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    plt.xticks(angles[:-1], metrics)
    ax.set_rlabel_position(30)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"])
    plt.ylim(0, 10)

    ax.plot(angles, baseline_plot, linewidth=2)
    ax.fill(angles, baseline_plot, alpha=0.15)

    ax.plot(angles, improved_plot, linewidth=2)
    ax.fill(angles, improved_plot, alpha=0.15)

    ax.plot(angles, gpt_plot, linewidth=2)
    ax.fill(angles, gpt_plot, alpha=0.15)

    plt.legend(["Baseline", "Improved", "GPT-only"], loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.title("RAG System Comparison Radar Chart", fontsize=16)
    plt.show()
