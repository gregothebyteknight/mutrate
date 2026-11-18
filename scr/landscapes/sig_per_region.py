
import pandas as pd
import numpy as np

from scr.sbs96_computation import build_sbs96_matrix
from scr.signature_estimation import em_algorithm

def vcf_preselect(vcf: pd.DataFrame, 
                     metadata: pd.DataFrame, 
                     cell_type: list[str], 
                     save_to_csv: bool = False) -> pd.DataFrame:

    colony_list = list(metadata.loc[metadata["CellType"].isin(cell_type)].index)

    vcf = vcf[vcf["sample"].isin(colony_list)]
    vcf_b = vcf[
                vcf["ref"].str.len().eq(1) &
                vcf["mut"].str.len().eq(1)
                ]

    vcf_b["tri_modified"] = vcf_b["tri"].str[0:1] + "[" + vcf_b["tri"].str[1:2] + vcf_b["tri"].str[3:5] + "]" + vcf_b["tri"].str[2:3]

    if save_to_csv:
        vcf_b.to_csv("data/landscapes/SNV_Type_DF.vcf", sep="\t")
    return vcf_b


def sign_matrix_compose(cosmic_path: str,
                        additional_sign_object: pd.DataFrame | None,
                        selected_signs: list[str] = [], 
                        save_to_csv: bool = False) -> pd.DataFrame:

    cosmic_matrix = pd.read_csv(cosmic_path, sep="\t", index_col=0)
    if additional_sign_object is not None:
        cosmic_matrix_joined = cosmic_matrix.join(additional_sign_object, how="inner")
    else:
        cosmic_matrix_joined = cosmic_matrix
        
    if len(selected_signs) > 0:
        cosmic_matrix_joined = cosmic_matrix_joined[selected_signs]

    if save_to_csv:
        cosmic_matrix_joined.to_csv("data/landscapes/signatures/initial_cosmic_subset.csv")

    return cosmic_matrix_joined


def signatures_broad_region(vcf_df: pd.DataFrame, 
                      signature_matrix: pd.DataFrame,
                      selected_sbs: list[str] = [],
                      save_to_csv: bool = False
                      ) -> pd.DataFrame:

    signatures_per_mb = pd.DataFrame()
    signatures_per_mb[["mb_id", *selected_sbs]] = None


    cnt = 0
    regions = vcf_df["mb_id"].unique()
    n_regions = len(regions)

    for region in regions:
        matrix = build_sbs96_matrix(vcf_df[vcf_df["mb_id"] == region])

        signatures, _, _, _ = em_algorithm(matrix, 
                                       signature_matrix, 
                                       signature_prop=None,
                                       max_iter=100,
                                       tol=1e-5)

        
        signatures_per_mb.loc[len(signatures_per_mb)] = [region, *signatures]
        print(f"Progress: {cnt}/{n_regions} regions processed")
        cnt += 1

    if save_to_csv:
        signatures_per_mb.to_csv("data/landscapes/signatures/signatures_per_mb.csv", index=False)

    return signatures_per_mb



def signatures_narrow_region(vcf_df: pd.DataFrame, 
                             signature_matrix: pd.DataFrame,
                             signatures_per_mb: pd.DataFrame,
                             selected_sbs: list[str] = [],
                             save_to_csv: bool = False
                             ) -> pd.DataFrame:

    signatures_per_bp = pd.DataFrame(vcf_df["tri_modified"])
    signatures_per_bp[[*selected_sbs]] = None
    signatures_per_bp.set_index("tri_modified", inplace=True)

    def signature_per_bp(tri, mb_id):
        unnorm = signature_matrix.loc[tri] * signatures_per_mb.loc[mb_id]
        return unnorm / unnorm.sum()

    signatures_per_bp = vcf_df.apply(lambda row: signature_per_bp(row["tri_modified"], row["mb_id"]), axis=1)
    signatures_per_bp.index = vcf_df["kb_id"]

    signatures_per_kb = signatures_per_bp.groupby("kb_id").mean()
    signatures_per_kb = signatures_per_kb.map(lambda x: 1 if x > 0.5 else 0) # binarization of the signatures per 10kb
    signatures_per_kb["n_mutations"] = vcf_df.groupby("kb_id").size()

    if save_to_csv:
        signatures_per_kb.to_csv("data/landscapes/signatures/signatures_per_kb.csv", index=False)

    return signatures_per_kb # type: ignore


def signature_gam(genomic_features: pd.DataFrame,
                  signatures_per_kb: pd.DataFrame,
                  mask: np.ndarray):
    pass



