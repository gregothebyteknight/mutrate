# Mutrate

`mutrate` contains exploratory analyses of somatic mutations, SBS-96 mutation
spectra, mutational signatures, and genomic landscapes. The repository
includes Python helpers, R scripts, and Jupyter notebooks for exploring
thyroid Nanoseq data and blood-cell mutation landscapes.

## Analyses

### SBS-96 mutation spectra

[`scr/sbs96_computation.py`](scr/sbs96_computation.py) provides utilities to:

- filter mutation records by sample, donor, assay, gene, and variant type;
- count mutations in the 96 C/T-centered SBS trinucleotide contexts;
- correct counts using genome or target-sequence trinucleotide proportions;
- plot SBS-96 mutation spectra.

### Mutational signature estimation

[`scr/signature_estimation.py`](scr/signature_estimation.py) implements an
Expectation-Maximization algorithm for estimating signature contributions.
It accepts a 96-channel mutation count table and a COSMIC-style signature
matrix, and also includes plotting helpers.

### Genomic landscapes

[`scr/landscapes/sig_per_region.py`](scr/landscapes/sig_per_region.py) assigns
signatures at 1 Mb and 10 kb genomic scales. The landscape workflow combines
mutation calls, signature proportions, genomic features, and gene-expression
annotations.

[`scr/landscapes/expression_quartiles.rmd`](scr/landscapes/expression_quartiles.rmd)
classifies protein-coding gene expression into quartiles and annotates 10 kb
genomic regions as coding or intergenic.

## Notebooks

- [`play_with_thyroid.ipynb`](play_with_thyroid.ipynb) explores thyroid
  Nanoseq mutations, target-context correction, SBS-96 spectra, and signature
  estimation.
- [`play_with_landscapes.ipynb`](play_with_landscapes.ipynb) explores blood
  memory-cell mutation signatures across genomic regions and with genomic
  and expression features.

The notebooks are the primary exploratory entry points. Individual helper
functions are being separated into the `scr/` directory so analyses can be
rerun without repeating plotting or computation code.

## Repository layout

```text
scr/
  sbs96_computation.py       SBS-96 counting, correction, and plotting
  signature_estimation.py    EM-based signature estimation
  coverage_estimation.py     PacBio coverage calculations
  mutation_estimation.py     PacBio Phred estimates
  genome_tri32.r             hg19 trinucleotide frequencies
  landscapes/
    sig_per_region.py        Signature assignment across genomic regions
    expression_quartiles.rmd Gene-expression region annotation
data/                        Large input and intermediate files
images/                      Generated figures
logs/                        Analysis logs
```

Large data files are ignored by Git and are not reproduced automatically.

## Running

Open the notebooks from the repository root:

```bash
jupyter lab
```

The Python analyses use common scientific packages including `numpy`,
`pandas`, `matplotlib`, `seaborn`, `pyfaidx`, and `pygam`. The R scripts use
packages such as `Biostrings`, `BSgenome.Hsapiens.UCSC.hg19`,
`GenomicRanges`, `rtracklayer`, and `readr`.

Because this is an exploratory repository, dependency and environment files
are not yet included. Install only the packages required by the notebook or
script you intend to run.

## Data caveats

- SBS-96 analyses assume C/T-centered substitutions and the standard 96
  trinucleotide channel format.
- COSMIC matrices and mutation tables must use matching mutation-context
  labels.
- Regional signature estimates should be interpreted carefully where the
  number of mutations is small.
- Notebook outputs, intermediate tables, and generated figures may reflect
  work-in-progress analyses rather than a finalized pipeline.