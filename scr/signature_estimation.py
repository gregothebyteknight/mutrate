
import numpy as np
import pandas as pd
import argparse


def expectation(
    mutation_vector: np.ndarray, 
    signature_matrix: pd.DataFrame, 
    signature_prop: np.ndarray
) -> tuple[np.float64, pd.DataFrame]:
    """
    Calculate the expected counts and the Q(pi|pi^t) function for a given set of mutations and signatures
    Input:
    mutation_matrix: A 1D numpy array of shape (96,) representing the observed mutation counts for each of the 96 trinucleotide contexts.
    signature_matrix: A 2D numpy array of shape (96, K) representing the mutation signatures, where K is the number of signatures
    signature_prop: A 1D numpy array of shape (K,) representing the proportions of
    each signature in the mixture model.
    Output:
    expected_counts_per_signature: A 2D numpy array of shape (96, K) representing the expected counts of mutations for each trinucleotide context and signature.
    q_function: A scalar representing the value of the Q(pi|pi^t) function for the given inputs.
    """


    # Marginal probability P(z) = sum_s P(z|s)P(s)
    marginal_prob = np.dot(signature_matrix, signature_prop)  # (n_mutation_types 96 x 1)
    marginal_prob = np.where(marginal_prob == 0, 1e-10, marginal_prob)  # Avoid division by zero

    # Posterior probability P(s|z) = P(z|s)P(s) / P(z)
    prob_type_per_signature = (signature_matrix * signature_prop).div(marginal_prob, axis=0) # gamma (n_mutation_types 96 x n_signatures_types 97)

    # Expected counts E[n_{z,s}] = n_z * P(s|z)
    expected_counts_per_signature = prob_type_per_signature.mul(mutation_vector, axis=0) # (n_mutation_types 96 x n_signatures_types 97)

    # q_function = sum_ks E[n_{z,s}] * log(P(s) * P(z|s))
    joint_prob = signature_prop * signature_matrix  # P(s) * P(z|s) (n_signatures_types 97 x n_mutation_types 96)
    q_function = np.float64(
        np.sum(
                np.array(expected_counts_per_signature * np.log(np.maximum(joint_prob, 1e-200)))
        )
    )

    return q_function, expected_counts_per_signature 

def maximization(
    expected_counts_per_signature: pd.DataFrame,
    dirichlet_beta: float
) -> np.ndarray:
    """
    Calculate the new signature proportions for a given set of expected counts per signature.
    Input:
    expected_counts_per_signature: A 2D numpy array of shape (96, K) representing the expected counts of mutations for each trinucleotide context and signature.
    Output:
    new_signature_prop: A 1D numpy array of shape (K,) representing the updated proportions of each signature in the mixture model.
    """

    
    total_expected_per_signature = np.array(np.sum(expected_counts_per_signature, axis=0))  # Sum over mutation types (result is a vector of length K)
    lagrange_lambda = np.sum(
        np.maximum(0.0, total_expected_per_signature - dirichlet_beta # sum and max over N axis (result is a vector 97, 1)
            ) # sum over K axis (result is a scalar)
        )
    
    if lagrange_lambda <= 0:
        return total_expected_per_signature / np.sum(total_expected_per_signature)
    
    new_signature_prop = np.maximum(0.0, total_expected_per_signature - dirichlet_beta) / lagrange_lambda
    return new_signature_prop


def em_algorithm(
    mutation_df: pd.DataFrame, 
    signature_matrix: pd.DataFrame, 
    signature_prop: np.ndarray | None = None, 
    max_iter: int = 1000, 
    tol: float = 1e-6,
    dirichlet_beta: float = 0.1
) -> tuple[np.ndarray, list[float], pd.DataFrame, pd.DataFrame]:
    """
    Perform the EM algorithm to estimate the proportions of mutation signatures in a given set of mutations.

    Input:
    * mutation_df: A pandas DataFrame containing the observed mutation counts for each trinucleotide context. 
    it should have columns "tri", "sub", and "count" representing the trinucleotide context, substitution type, and observed counts, respectively.
    * signature_matrix: A pandas DataFrame containing the mutation signatures.
    * signature_prop: A 1D numpy array of shape (K,) representing the initial proportions of each signature in the mixture model.
    * max_iter: An integer representing the maximum number of iterations to perform.
    * tol: A float representing the tolerance for convergence. The algorithm will stop if the change in log-likelihood is less than this value.
    * dirichlet_beta: A float representing the beta parameter for the Dirichlet prior.

    Output:
    * signature_prop: A 1D numpy array of shape (K,) representing the estimated proportions of each signature in the mixture model.
    * q_function_values: A list of floats representing the values of the Q(pi|pi^t) function at each iteration.
    * expected_counts_per_signature: A 2D numpy array of shape (96, K) representing the expected counts of mutations for each trinucleotide context and signature.
    * normalized_mutation_df: A pandas DataFrame containing the normalized mutation counts for each trinucleotide context.
    """

    # Mutation vector constructed from the mutation pd.DataFrame produced by sbs96_computation script
    mutation_df_copy = mutation_df.copy()
    mutation_df_copy["mut_type"] = mutation_df_copy["tri"].str[0] + "[" + mutation_df_copy["sub"] + "]" + mutation_df_copy["tri"].str[2]

    mutation_vector = np.array(mutation_df_copy["count"].values)
    if mutation_vector.sum() == 0:
        raise ValueError("Mutation vector sums to zero; cannot run EM on empty data.")

    signature_matrix = signature_matrix.loc[mutation_df_copy["mut_type"].values] # type: ignore # so mutation order would correspond


    # Initialization of signature proportions
    if signature_prop is None:
        signature_prop = np.ones(signature_matrix.shape[1]) / signature_matrix.shape[1]  # Uniform initialization if not provided

    q_function_values = []

    # Sanity check
    assert len(mutation_vector) == signature_matrix.shape[0]
    # EM algorithm iterations
    
    for i in range(max_iter):
        # E-step
        q_function, expected_counts_per_signature = expectation(mutation_vector, signature_matrix, signature_prop) # type: ignore
        q_function_values.append(q_function)
        
        # M-step
        new_signature_prop = maximization(expected_counts_per_signature, dirichlet_beta=dirichlet_beta)
        
        # Check for convergence
        if np.linalg.norm(new_signature_prop - signature_prop) < tol:
            signature_prop = new_signature_prop
            print(f"Converged after {i+1} iterations.")
            break
        
        signature_prop = new_signature_prop
    
    return signature_prop, q_function_values, expected_counts_per_signature, mutation_df_copy # type: ignore

def em_visualization(
        q_function_values: list[float]
    ) -> None:
    """
    Visualize the convergence of the EM algorithm by plotting the Q(pi|pi^t) function values over iterations.
    Input:
    q_function_values: A list of floats representing the values of the Q(pi|pi^t) function at each iteration.
    Output:
    None
    """


    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.figure(figsize=(10, 7))
    plt.plot(q_function_values, marker='o')
    plt.title('Convergence of EM Algorithm')
    plt.xlabel('Iteration')
    plt.ylabel('Q(pi|pi^t) Value')
    plt.grid()
    plt.show()

def signatures_visualization(
        expected_counts_per_signature: pd.DataFrame, 
        normalized_matrix: pd.DataFrame,
        n_top_signatures: int = 10,
        data_normalize: bool = False,
        save_image: bool = False
    ) -> None:


    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')
    sns.set_palette("coolwarm")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    total_counts_per_signature = expected_counts_per_signature.sum(axis=0)
    top_sbs = total_counts_per_signature.nlargest(n_top_signatures).index

    if data_normalize:
            total_mutations = normalized_matrix["count"].sum()
            normalized_matrix["count"] /= total_mutations
            expected_counts_per_signature /= total_mutations

    expected_counts_top_sbs = expected_counts_per_signature.loc[:, top_sbs]


#     sub_colors = {
#     "C>A": "#13DBF0",
#     "C>G": "#000000",
#     "C>T": "#E22642",
#     "T>A": "#999999",
#     "T>C": "#A2CD61",
#     "T>G": "#ECC4C5",
# }

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(normalized_matrix))
    bottom = np.zeros(len(normalized_matrix))

    for signature in expected_counts_top_sbs.columns:
        values = expected_counts_top_sbs[signature].to_numpy()
        ax.bar(x, 
               values, 
               bottom=bottom, 
               label=f"Signature {signature} ({values.sum():.2f})")
        bottom += values

    ax.plot(
        x,
        normalized_matrix["count"].to_numpy(),
        color='gray',
        alpha=0.5,
        label=f"Observed Mutations"
        
    )

    tick_positions = range(0, len(normalized_matrix), 4)
    ax.set_xticks(
        tick_positions,
        [normalized_matrix["mut_type"].iloc[i] for i in tick_positions]
    )

    # for i, tick_label in zip(tick_positions, ax.get_xticklabels()):
    #     tick_label.set_color(
    #         sub_colors[normalized_matrix["sub"].iloc[i]]
    # )

    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(title='Legend', bbox_to_anchor=(0.02, 1), 
              loc='upper left', title_fontsize=8, fontsize=7)
    ax.set_xlabel("Trinucleotide context", fontsize=8, labelpad=8)
    ax.set_ylabel("Mutation record", fontsize=8, labelpad=8)
    fig.tight_layout()

    ax.set_title('EM Derived Signature Contributions vs Observed Mutations',
                 fontsize=10, pad=20)
    if save_image:
        plt.savefig('images/signature_image.png', dpi=300, bbox_inches="tight")
    fig.show()


    if __name__ == "__main__":

        parser = argparse.ArgumentParser(description='Run EM algorithm for mutation signature estimation.')
        parser.add_argument('--mutation_df', type=str, required=True, help='Path to the mutation DataFrame CSV file.')
        parser.add_argument('--cosmic_path', type=str, required=True, help='Path to the COSMIC signature matrix CSV file.')
        parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations for the EM algorithm (default: 1000).')
        parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for convergence (default: 1e-6).')
        parser.add_argument('--dirichlet_beta', type=float, default=0.1, help='Dirichlet prior parameter (default: 0.1).')
        parser.add_argument('--is_visualize', type=bool, default=False, help='Whether to visualize the signature contribution')
        parser.add_argument('--n_top_signatures', type=int, default=10, help='Number of top signatures to visualize (default: 10).')
        parser.add_argument('--save_image', type=bool, default=False, help='Whether to save the visualization as an image (default: False).')

        args = parser.parse_args()

        mutation_df = pd.read_csv(args.mutation_df)
        _, q_function_values, expected_counts_per_signature, normalized_matrix = em_algorithm(
            mutation_df=mutation_df,
            cosmic_path=args.cosmic_path,
            max_iter=args.max_iter,
            tol=args.tol,
            dirichlet_beta=args.dirichlet_beta
        )

        
        if args.is_visualize:
            em_visualization(q_function_values)
            signatures_visualization(expected_counts_per_signature, normalized_matrix, n_top_signatures=args.n_top_signatures, save_image=args.save_image)