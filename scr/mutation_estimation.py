
import argparse
import scipy.stats as stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def estimate_phred(eps, n_passes):
    """
    Estimate the Phred quality score based on the subread 
    error probability and number of polymerase passes.

    Parameters:
    eps (float): The error probability (between 0 and 1).
    n_passes (int): The number of polymerase passes.

    Returns:
    float: The estimated Phred quality score.
    """
    return -10 * np.log10(3 * stats.binom.sf(n_passes//2, n_passes, eps / 3))


def phred_eps_dependency(min_eps, max_eps, n, n_passes):

    eps_values = np.linspace(min_eps, max_eps, n)
    p_err = 3 * stats.binom.sf(
        n_passes // 2,
        n_passes,
        eps_values / 3
    )
    phred_scores = -10 * np.log10(p_err)
    return eps_values, phred_scores

def plot_phred_eps_dependency(min_eps, max_eps, n, n_passes):
    eps_values, phred_scores = phred_eps_dependency(min_eps, max_eps, n, n_passes)

    sns.set_theme(style="darkgrid")
    
    ax = sns.relplot(x=eps_values, y=phred_scores, kind="line")
    ax.set(xlabel='Subread Error Probability (ε)', ylabel='Estimated Phred Quality Score', title='Phred Quality Score vs Subread Error Probability')
    plt.show()

if __name__ == "__main__":
    min_eps = 0.001
    max_eps = 0.2
    n = 100
    n_passes = 6

    parser = argparse.ArgumentParser(description='Estimate Phred quality scores based on subread error probability and number of polymerase passes.')
    parser.add_argument('--min_eps', type=float, default=min_eps, help='Minimum subread error probability (default: 0.001)')
    parser.add_argument('--max_eps', type=float, default=max_eps, help='Maximum subread error probability (default: 0.2)')
    parser.add_argument('--n', type=int, default=n, help='Number of points to evaluate (default: 100)')
    parser.add_argument('--n_passes', type=int, default=n_passes, help='Number of polymerase passes (default: 6)')

    args = parser.parse_args()

    plot_phred_eps_dependency(args.min_eps, args.max_eps, args.n, args.n_passes)
    