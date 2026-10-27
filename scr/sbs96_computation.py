"""Helpers for COSMIC mutation counts and PacBio Phred quality estimates."""

import re

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sns.set_style("whitegrid")

_SAMPLE_ID_PATTERN = re.compile(
    r"^(?P<donor_biopsy_id>PD\d+\w)_(?P<tds>tds\d+)_(?P<assay>exome_2|exome|targeted)$"
)
_SUBSTITUTIONS = ("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
_BASES = ("A", "C", "G", "T")
_CHANNELS = tuple(
    (substitution, f"{left}{substitution[0]}{right}")
    for substitution in _SUBSTITUTIONS
    for left in _BASES
    for right in _BASES
)


def add_sample_metadata(mutations: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with donor, tds, and assay parsed from ``sampleID``."""
    if "sampleID" not in mutations.columns:
        raise ValueError("mutations must contain a 'sampleID' column")

    metadata = mutations["sampleID"].astype("string").str.extract(_SAMPLE_ID_PATTERN)
    invalid = metadata.isna().any(axis=1)
    if invalid.any():
        examples = mutations.loc[invalid, "sampleID"].head(5).tolist()
        raise ValueError(f"Could not parse sampleID values, for example: {examples}")

    result = mutations.copy()
    for column in metadata.columns:
        result[column] = metadata[column]
    return result


def tri_prop(seq_path: str, tri_types: list) -> np.array:
    """Count trinucleotide occurrences in a DNA sequence.

    Parameters:
    seq_path (str): The path to the FASTA file containing the DNA sequence.
    tri_types (list): A list of trinucleotide contexts to count.

    Returns:
    np.array: An array of proportions of each trinucleotide context in the sequence, ordered alphabetically.
    """
    from collections import Counter
    from pyfaidx import Fasta

    comp = str.maketrans("ACGT", "TGCA")
    sequence = Fasta(filename=seq_path)
    sequence = "".join(str(sequence[i]) for i in sorted(sequence.keys()))
    sequence = sequence.upper()
    print(f"Length of target sequence: {len(sequence)}")

    tri_count = Counter()
    tri_count.update({tri: 0 for tri in tri_types})

    for i in range(len(sequence) - 2):
        t = sequence[i:i+3]
        if "N" in t:
            continue
        if t[1] not in "TC":
            t = t.translate(comp)[::-1]
        tri_count[t] += 1
    tri_count = np.array([x[1] for x in sorted(tri_count.items())])
    tri_prop = tri_count / tri_count.sum()
    return tri_prop


def tri_correct(mut_matrix: pd.DataFrame, 
                genome_tri_prop_path: str,
                seq_path: str
                ) -> pd.DataFrame:
    """Correct mutation counts for trinucleotide context proportions in the genome. 
    Original matrix's counts are preserved, modified counts are returned.

    Parameters:
    mut_matrix (pd.DataFrame): A DataFrame containing mutation counts for each trinucleotide context.
    seq_path (str): The path to the FASTA file containing the DNA sequence.
    genome_tri_prop_path (str): The path to the CSV file containing the proportion of each trinucleotide in the genome.

    Returns:
    pd.DataFrame: A DataFrame with corrected mutation counts for each trinucleotide context.
    """
    genome_tri_prop = pd.read_csv(genome_tri_prop_path, index_col=0).iloc[:, 0]
    tri_types = sorted(list(genome_tri_prop.index))
    seq_tri_prop = tri_prop(seq_path, tri_types)
    weights = pd.Series(
        genome_tri_prop.to_numpy() / seq_tri_prop,
        index=genome_tri_prop.index.astype(str),
)
    matrix = mut_matrix.copy()

    matrix["count"] = (
        matrix["count"].astype(float)
        * matrix["tri"].map(weights)
    )

    return matrix


def select_mutations(
    mutations: pd.DataFrame,
    *,
    sample_id: str | None = None,
    donor_biopsy_id: str | None = None,
    assay: str | None = None,
    gene: str | None = None,
    variant_type: str | None = "snv",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Filter mutation rows by sample or donor/assay, gene, and variant type."""
    filters = {
        "sampleID": sample_id,
        "donor_biopsy_id": donor_biopsy_id,
        "assay": assay,
        "gene": gene,
        "type": variant_type,
    }
    required = [column for column, value in filters.items() if value is not None]
    missing = [column for column in required if column not in mutations.columns]
    if missing:
        raise ValueError(f"mutations is missing required columns: {missing}")
    if columns is not None:
        missing_output = [column for column in columns if column not in mutations.columns]
        if missing_output:
            raise ValueError(f"mutations is missing requested columns: {missing_output}")

    mask = pd.Series(True, index=mutations.index)
    for column, value in filters.items():
        if value is not None:
            mask &= mutations[column].eq(value)

    selected = mutations.loc[mask]
    if columns is not None:
        selected = selected.loc[:, columns]
    selected = selected.copy()
    selected.attrs["selection"] = {
        column: value for column, value in filters.items() if value is not None
    }
    return selected


def build_sbs96_matrix(
    mutations: pd.DataFrame,
    *,
    context_column: str = "tri",
) -> pd.DataFrame:
    """Count mutation records in the 96 C/T-centered trinucleotide channels.

    Each input row contributes one count. Contexts must be encoded as, for
    example, ``ATG>C``; the middle base must be C or T, as in the notebook's
    current channel mapping.
    """
    if context_column not in mutations.columns:
        raise ValueError(f"mutations must contain a {context_column!r} column")

    contexts = mutations[context_column].astype("string").str.extract(
        r"^([ACGT])([ACGT])([ACGT])>([ACGT])$"
    )
    invalid = contexts.isna().any(axis=1)
    if invalid.any():
        examples = mutations.loc[invalid, context_column].head(5).tolist()
        raise ValueError(f"Invalid trinucleotide mutation contexts, for example: {examples}")

    reference = contexts[1]
    alternate = contexts[3]
    unsupported = ~reference.isin(("C", "T")) | reference.eq(alternate)
    if unsupported.any():
        examples = mutations.loc[unsupported, context_column].head(5).tolist()
        raise ValueError(
            "SBS-96 counting requires a C/T reference base and a different "
            f"alternate base; unsupported contexts include: {examples}"
        )

    channel_positions = {channel: index for index, channel in enumerate(_CHANNELS)}
    counts = [0] * len(_CHANNELS)
    for left, middle, right, alternate_base in contexts.itertuples(
        index=False, name=None
    ):
        tri = f"{left}{middle}{right}"
        sub = f"{middle}>{alternate_base}"
        counts[channel_positions[(sub, tri)]] += 1

    result = pd.DataFrame(_CHANNELS, columns=["sub", "tri"])
    result["count"] = counts
    selection = mutations.attrs.get("selection", {})
    sample_label = selection.get("sampleID")
    if sample_label is None and selection.get("donor_biopsy_id") is not None:
        sample_label = selection["donor_biopsy_id"]
        if selection.get("assay") is not None:
            sample_label = f"{sample_label} ({selection['assay']})"
    if sample_label is None and "sampleID" in mutations.columns:
        sample_ids = mutations["sampleID"].dropna().unique()
        if len(sample_ids) == 1:
            sample_label = str(sample_ids[0])

    gene = selection.get("gene")
    if gene is None and "gene" in mutations.columns:
        genes = mutations["gene"].dropna().unique()
        if len(genes) == 1:
            gene = str(genes[0])

    result.attrs["sample_label"] = sample_label or "Selected data"
    result.attrs["gene"] = gene or "Unspecified"
    return result


def plot_sbs96_matrix(
    matrix: pd.DataFrame,
    *,
    sample_label: str | None = None,
    gene: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot SBS-96 counts with substitution colors, legend, and context ticks."""
    required = {"sub", "tri", "count"}
    missing = required.difference(matrix.columns)
    if missing:
        raise ValueError(f"matrix is missing required columns: {sorted(missing)}")

    indexed = matrix.set_index(["sub", "tri"])
    channels = pd.MultiIndex.from_tuples(_CHANNELS, names=["sub", "tri"])
    if len(indexed) != len(channels) or not indexed.index.is_unique:
        raise ValueError("matrix must contain exactly one row for each SBS-96 channel")
    ordered = indexed.reindex(channels)
    if ordered["count"].isna().any():
        raise ValueError("matrix does not contain all SBS-96 channels")

    if ax is None:
        _, ax = plt.subplots(figsize=(15, 7))

    palette = sns.color_palette(n_colors=len(_SUBSTITUTIONS))
    colors = dict(zip(_SUBSTITUTIONS, palette))
    ax.bar(
        range(len(_CHANNELS)),
        pd.to_numeric(ordered["count"]).to_numpy(),
        color=[colors[sub] for sub, _ in _CHANNELS],
    )
    ax.set_xticks(range(len(_CHANNELS)), [tri for _, tri in _CHANNELS])
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.set_xlabel("Trinucleotide context")
    ax.set_ylabel("Mutation record count")
    sample_label = sample_label or matrix.attrs.get("sample_label", "Selected data")
    gene = gene or matrix.attrs.get("gene", "Unspecified")
    ax.set_title(f"Mutation Count for Sample: {sample_label} and Gene: {gene}")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        handles=[
            Patch(facecolor=colors[substitution], label=substitution)
            for substitution in _SUBSTITUTIONS
        ],
        title="Substitution",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )
    ax.margins(x=0.01)
    if ax.figure.axes[0] is ax:
        ax.figure.subplots_adjust(right=0.84, bottom=0.24)

    plt.savefig('images/sbs96_matrix.png', dpi=300, bbox_inches="tight")
    return ax
