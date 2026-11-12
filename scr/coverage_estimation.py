
import argparse

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def on_target_fraction(genome_size, target_size, method):

    target_size = np.asarray(target_size, dtype=np.float64)

    # if method == "WGS":
    #     return np.ones_like(target_size)
    
    if method == 'PCR':
        pcr_pot = np.where(target_size >= genome_size, 
                           1.0, 
                           0.92)
        return pcr_pot

    if method == 'Hybridization':
        # Based on the Michaelis-Menten equation and Mertes et al., 

        k_half = 1000.0
        hybrid_pot = np.where(target_size >= genome_size, 
                              1.0,
                              target_size / (target_size + k_half))
        return hybrid_pot

    else:
        raise ValueError(f"Unknown enrichment method: {method}")


def estimate_coverage(reaction_yield, target_size, on_target_fraction=1.0):
    """
    Estimate the coverage based on the fragment size.

    Parameters:
    target_size (int): The size of the target in kilobases (kb).
    reaction_yield (float): The yield in gigabases (Kb).

    Returns:
    float: The estimated coverage.
    """
    coverage = reaction_yield * on_target_fraction / target_size
    return coverage

def plot_coverage_vs_target_size(reaction_yield, genome_size, target_sizes, methods):

    if methods is None:
        raise ValueError("" \
        "Please provide a list of methods.")

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    # Continuous range from 1 kb to human genome size for smooth lines
    roi_continuous = np.geomspace(min(target_sizes), genome_size, 500)

    for method in methods:
        pot = on_target_fraction(roi_continuous, method=method)
        cov = estimate_coverage(reaction_yield, roi_continuous, on_target_fraction=pot)
        label_name = "PCR Enrichment (pot ≈ 92%)" if method == "pcr" else "Hybrid Capture (Dynamic pot)"
        ax.plot(roi_continuous, cov, lw=2.2, label=label_name)

    # Plot the specific discrete target sizes requested by the user
    target_sizes = np.array(target_sizes)
    pcr_user_pot = on_target_fraction(target_sizes, method="pcr")
    user_cov = estimate_coverage(reaction_yield, target_sizes, on_target_fraction=pcr_user_pot)
    ax.scatter(target_sizes, user_cov, color="red", zorder=5, s=40, label="Selected Targets (PCR)")

    # Benchmark point: Whole Genome Sequencing without enrichment (~32x)
    wgs_cov = estimate_coverage(reaction_yield, genome_size, on_target_fraction=1.0)
    ax.scatter([genome_size], [wgs_cov], color="black", s=70, zorder=6, label=f"WGS ({wgs_cov:.1f}x)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Target Size (kb)", fontsize=12)
    ax.set_ylabel("Estimated Coverage (x)", fontsize=12)
    ax.set_title(f"PacBio Revio Coverage vs Target Size (Yield: {reaction_yield:.0f} Kb)", fontsize=13)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig('images/coverage_vs_target_size.png', dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate PacBio Revio coverage across target sizes.")
    parser.add_argument("--reaction_yield", type=float, default=100.0, help="Yield in Kb (default: 100 for 1 Revio SMRT Cell)")
    parser.add_argument(
        "--target_sizes",
        nargs="+",
        type=int,
        default=[1, 5, 10, 15, 20, 50, 100, 200, 1024, 102400, 3145728],
        help="Example target sizes in kb"
    )
    args = parser.parse_args()

    plot_coverage_vs_target_size(args.reaction_yield, args.target_sizes)

