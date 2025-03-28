import pandas as pd
import numpy as np
from ..utils import read_gmt, GeneSymbolMapper, download_to_cache_ext


def benchmarking_taiga_paths(name='genetea-manuscript-bb10', version=7):
    return {
        "hallmark_collection":download_to_cache_ext(f'{name}.{version}/h.all.v2024.1.Hs.symbols'),
        "alphafold2_clusters":download_to_cache_ext(f'{name}.{version}/3-sapId_sapGO_repId_cluFlag_LCAtaxId'),
        "bioid_proximity":download_to_cache_ext(f'{name}.{version}/Supplementary table 4', "xlsx"),
        "rare_var_gwas":download_to_cache_ext(f'{name}.{version}/Supp Table 8 - Collapsing analysis top hits', "xlsx"),
        "perturbseq_clusters":download_to_cache_ext(f'{name}.{version}/NIHMS1812939-supplement-13', "xlsx"),
        "hgnc_table":"taiga"
    }

def get_benchmarking_sets(paths="taiga"):
    '''
    Get Hallmark and Experimentally derived queries.
    '''
    if paths == "taiga":
        print("loading from taiga")
        paths = benchmarking_taiga_paths()

    ###Hallmark Gene Sets ###
    hallmark_sets = read_gmt(paths["hallmark_collection"])["Genes"]

    ## AlphaFold2 Protein Clusters ##
    #https://afdb-cluster.steineggerlab.workers.dev/
    alphafold_foldseek = pd.read_csv(paths["alphafold2_clusters"], sep="\t", header=None)
    alphafold_foldseek.columns = ["Uniprot ID", "GO Terms", "Cluster", "Flag", "LDA"]
    #map to HGNC gene symbol, keep only clusters representing 2+ gene groups
    gsm = GeneSymbolMapper(paths["hgnc_table"])
    alphafold_foldseek = alphafold_foldseek.merge(gsm.hgnc_mapping, how="left", left_on="Uniprot ID", right_on="uniprot_ids")
    alphafold_clusters = alphafold_foldseek.dropna(subset=["symbol", "Cluster"]).groupby("Cluster").apply(
        lambda x: list(x["symbol"].sort_values()) if x["gene_group"].nunique() > 2 else []
    )

    ##BioID Proximity##
    #https://www.nature.com/articles/s41586-021-03592-2#MOESM2
    biotin = pd.read_excel(paths["bioid_proximity"], engine="openpyxl", sheet_name=2)
    biotin = biotin.set_index("bait")["recovered preys"].str.split(", ")

    ### Rare Variant Human Disease GWAS ###
    #https://www.nature.com/articles/s41586-021-03855-y#Fig2 
    rare_var_binary = pd.read_excel(paths["rare_var_gwas"], engine="openpyxl", sheet_name=0)
    rare_var_binary = rare_var_binary.groupby("root")["Gene"].apply(set).apply(list)
    rare_var_quant = pd.read_excel(paths["rare_var_gwas"], engine="openpyxl", sheet_name=1)
    rare_var_quant = rare_var_quant.groupby("Pheno")["Gene"].apply(set).apply(list)
    rare_var = pd.concat([rare_var_quant, rare_var_binary])


    ### PerturbSeq Clusters ###
    #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9380471/
    perturb_seq = pd.read_excel(paths["perturbseq_clusters"], engine='openpyxl', sheet_name=3)
    perturb_seq_pert_clusters = perturb_seq.set_index("best_description")["members"].str.split(",")

    perturb_seq2 = pd.read_excel(paths["perturbseq_clusters"], engine='openpyxl', sheet_name=1)
    perturb_seq2["manual_annotation"].fillna("unknown", inplace=True)
    mask = perturb_seq2["manual_annotation"].duplicated(keep=False)
    perturb_seq2.loc[mask, "manual_annotation"] += perturb_seq2.groupby("manual_annotation").cumcount().add(1).astype(str)
    perturb_seq_expr_clusters = perturb_seq2.dropna(subset=["members"]).set_index("manual_annotation")["members"].str.split(",")

    #combine, filter out short queries and duplicates
    queries = pd.concat({
        "Hallmark Collection":hallmark_sets,
        "BioID Interacting Proteins":biotin,
        "AlphaFold2 Protein Clusters":alphafold_clusters,
        "Perturb-seq Expression Modules":perturb_seq_expr_clusters,
        "Perturb-seq Perturbation Clusters":perturb_seq_pert_clusters,
        "Rare Variant GWAS":rare_var
    }).apply(lambda x: list(sorted(set(x))))
    return queries.loc[lambda x: (x.apply(len) > 2) & ~x.apply(" ".join).duplicated()]


def subsample_from_genesets(gene_sets, sizes=[10,25,50,100], n=1, random_seed=27):
    '''
    Subsample sets of specified sizes from exisiting gene sets.
    '''
    rng = np.random.default_rng(seed=random_seed)
    sampled = {}
    for s in sizes:
        sampled[s] = pd.Series(
            {k:rng.choice(v, (n, s)) for k, v in gene_sets.items()}
        ).explode()
    sampled = pd.concat(sampled).rename_axis(["size", "original"]).to_frame("genes").reset_index()
    sampled["genes"] = sampled["genes"].apply(list)
    sampled["gene_set"] = sampled["original"]+"_samp"+sampled.groupby("original").cumcount().astype(str)+"_size"+sampled["size"].astype(str)
    return sampled

def random_combo_subsample(sampled, max_size=500, n_combos=100, random_seed=27):
    '''
    Randomly combine pairs of subsamples.
    '''
    rng = np.random.default_rng(seed=random_seed)
    usable_sampled = sampled.loc[lambda x: x["size"] < max_size]
    combined_sampled = pd.concat([
        usable_sampled.sample(n_combos, random_state=rng.bit_generator).reset_index().add_suffix(" 1"), 
        usable_sampled.sample(n_combos, random_state=rng.bit_generator).reset_index().add_suffix(" 2")
    ], axis=1)
    combined_sampled["combined"] = combined_sampled.apply(lambda x:list(x["genes 1"])+list(x["genes 2"]), axis=1)
    return combined_sampled

def subsample_and_combo(gene_sets, subsample_kws={}, combo_kws={}):
    '''
    Get subsamples and combos of them.
    '''
    sampled = subsample_from_genesets(gene_sets, **subsample_kws)
    sampled_combos = random_combo_subsample(sampled, **combo_kws)
    return {
        "Random Sub-samples":sampled,
        "Random Combos":sampled_combos
    }