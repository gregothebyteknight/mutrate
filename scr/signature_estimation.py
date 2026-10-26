
import numpy as np
import pandas as pd


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
    prob_type_per_signature = (signature_matrix * signature_prop) / marginal_prob[:, None] # gamma (n_mutation_types 96 x n_signatures_types 97)

    # Expected counts E[n_{z,s}] = n_z * P(s|z)
    expected_counts_per_signature = mutation_vector[:, None] * prob_type_per_signature # (n_mutation_types 96 x n_signatures_types 97)

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
    cosmic_path: str, 
    signature_prop: np.ndarray | None = None, 
    max_iter: int = 1000, 
    tol: float = 1e-6,
    dirichlet_beta: float = 0.1
) -> tuple[np.ndarray, list[float], pd.DataFrame, pd.DataFrame]:
    """
    Perform the EM algorithm to estimate the proportions of mutation signatures in a given set of mutations.
    Input:
    mutation_df: A pandas DataFrame containing the observed mutation counts for each trinucleotide context. 
    it should have columns "tri", "sub", and "count" representing the trinucleotide context, substitution type, and observed counts, respectively.
    cosmic_path: A string representing the path to the COSMIC database file.
    signature_matrix: A 2D numpy array of shape (96, K) representing the mutation signatures, where K is the number of signatures
    signature_prop: A 1D numpy array of shape (K,) representing the initial proportions of each signature in the mixture model.
    max_iter: An integer representing the maximum number of iterations to perform.
    tol: A float representing the tolerance for convergence. The algorithm will stop if the change in log-likelihood is less than this value.
    
    Output:
    signature_prop: A 1D numpy array of shape (K,) representing the estimated proportions of each signature in the mixture model.
    q_function_values: A list of floats representing the values of the Q(pi|pi^t) function at each iteration.
    expected_counts_per_signature: A 2D numpy array of shape (96, K) representing the expected counts of mutations for each trinucleotide context and signature.
    normalized_mutation_df: A pandas DataFrame containing the normalized mutation counts for each trinucleotide context.
    """


    # Mutation vector constructed from the mutation pd.DataFrame produced by sbs96_computation script
    mutation_df_copy = mutation_df.copy()
    mutation_df_copy["mut_type"] = mutation_df_copy["tri"].str[0] + "[" + mutation_df_copy["sub"] + "]" + mutation_df_copy["tri"].str[2]

    mutation_vector = np.array(mutation_df_copy["count"].values)
    if mutation_vector.sum() == 0:
        raise ValueError("Mutation vector sums to zero; cannot run EM on empty data.")

    signature_matrix = pd.read_csv(cosmic_path, index_col=0, sep='\t')  # Load the signature matrix from a CSV file
    signature_matrix = signature_matrix.loc[mutation_df_copy["mut_type"].values] # so mutation order would correspond


    # Initialization of signature proportions
    if signature_prop is None:
        signature_prop = np.ones(signature_matrix.shape[1]) / signature_matrix.shape[1]  # Uniform initialization if not provided

    q_function_values = []

    # Sanity check
    assert len(mutation_vector) == signature_matrix.shape[0]
    # EM algorithm iterations
    
    for i in range(max_iter):
        # E-step
        q_function, expected_counts_per_signature = expectation(mutation_vector, signature_matrix, signature_prop)
        q_function_values.append(q_function)
        
        # M-step
        new_signature_prop = maximization(expected_counts_per_signature, dirichlet_beta=dirichlet_beta)
        
        # Check for convergence
        if np.linalg.norm(new_signature_prop - signature_prop) < tol:
            print(f"Converged after {i+1} iterations.")
            break
        
        signature_prop = new_signature_prop
    
    return signature_prop, q_function_values, expected_counts_per_signature, mutation_df_copy

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
        normalized_matrix: pd.DataFrame
    ) -> None:


    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')

    total_counts_per_signature = expected_counts_per_signature.sum(axis=0)
    top_sbs = total_counts_per_signature.nlargest(10).index

    expected_counts_top_sbs = expected_counts_per_signature.loc[:, top_sbs]

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.lineplot(data=expected_counts_top_sbs, ax=ax)
    ax.bar(normalized_matrix["mut_type"], normalized_matrix["count"], alpha=0.5, color='gray', label=f"Observed\nMutation\nCounts")

    ax.tick_params(axis="x", labelrotation=90, labelsize=5)
    ax.legend(title='Legend', bbox_to_anchor=(0.02, 1), loc='upper left')
    fig.tight_layout()

    ax.set_title('EM Derived Signature Contributions vs Observed Mutation Counts')
    plt.savefig('signature_image.png', dpi=300, bbox_inches="tight")
    fig.show()