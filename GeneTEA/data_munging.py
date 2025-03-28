import pandas as pd
import numpy as np
from .utils import GeneSymbolMapper, download_to_cache_ext

def taiga_paths(name='genetea-manuscript-bb10', version=13):
    return {
        "ncbi_gene_info":download_to_cache_ext(f'{name}.{version}/Homo_sapiens_gene_info'),
        "ncbi_gene_summ":download_to_cache_ext(f'{name}.{version}/gene_summary'),
        "alliance_human":download_to_cache_ext(f'{name}.{version}/GENE-DESCRIPTION-TSV_HUMAN'),
        "uniprot_human":download_to_cache_ext(f'{name}.{version}/uniprotkb_human'),
        "civic":download_to_cache_ext(f'{name}.{version}/01-Feb-2025-FeatureSummaries'),
        "sources":download_to_cache_ext(f'{name}.{version}/sources'),
        "alliance_yeast":download_to_cache_ext(f'{name}.{version}/GENE-DESCRIPTION-TSV_SGD'),
        "uniprot_yeast":download_to_cache_ext(f'{name}.{version}/uniprotkb_yeast'),
        "pubchem_chebi":download_to_cache_ext(f'{name}.{version}/pubchem_record_des_ChEBI_2025-02-25'),
        "pubchem_ncit":download_to_cache_ext(f'{name}.{version}/pubchem_record_des_NCI%20Thesaurus%20(NCIt)_2025-02-25'),
        "medline_1g":download_to_cache_ext(f'{name}.{version}/medline_1-gram_2024'),
        "hgnc_table":"taiga"
    }

def clean_medline_ngrams(filepath):
    medline_ngrams = pd.read_csv(filepath, sep="|", error_bad_lines=False, header=None)
    medline_ngrams.columns = ["Doc Freq", "Word Freq", "Word"]
    return medline_ngrams

def get_seed_stopwords(paths="taiga"):
    if paths == "taiga":
        paths = taiga_paths()
    return clean_medline_ngrams(paths["medline_1g"]).nlargest(500, "Word Freq")["Word"].to_list()

def clean_ncbi_info(gene_info_path, gene_summary_path):
    '''
    Clean  NCBI gene info and gene summary, ie. from:
    https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz
    https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/gene_summary.gz
    '''
    #read in gene info and gene summaries
    ncbi_gene_info = pd.read_csv(gene_info_path, sep="\t").loc[
        lambda x:  (
            (x["type_of_gene"].isin(["protein-coding", "pseudo"])) & #throw out non-genes
            (x["Symbol_from_nomenclature_authority"] != "-") #ignore unmappable entries
        )
    ]
    ncbi_gene_summ = pd.read_csv(gene_summary_path, sep="\t").loc[lambda x: (
        (x["Source"] == "RefSeq") #keep only RefSeq summaries
    )]
    #combine into single dataframe
    all_ncbi = ncbi_gene_info.merge(ncbi_gene_summ, how="left", on=["#tax_id", "GeneID"])
    #split names/aliases out and separate by semicolon
    ncbi_names = all_ncbi.set_index("Symbol_from_nomenclature_authority")[
        ["description", "Full_name_from_nomenclature_authority", "Other_designations"]
    ].stack().str.split("|").explode().reset_index()
    ncbi_clean = ncbi_names.groupby("Symbol_from_nomenclature_authority")[0].apply(
        lambda x: "Also known as "+"; ".join(list(set(x.unique())))+"."
    ).to_frame("Names/Aliases")
    arm = "arm "+all_ncbi.set_index("Symbol_from_nomenclature_authority")["map_location"].str.extract("((?:\d+|X|Y|MT)(?:p|q))")[0]
    #add gene location info
    ncbi_clean["Gene Location"] = (
        "Located on chr"+all_ncbi.set_index("Symbol_from_nomenclature_authority")["chromosome"].astype(str).str.replace("|", "/")
        +", "+arm
        + " at "+all_ncbi.set_index("Symbol_from_nomenclature_authority")["map_location"].str.replace("\.", "_")
        +"."
    )
    #add RefSeq summary
    ncbi_clean["RefSeq"] = (
        all_ncbi.set_index("Symbol_from_nomenclature_authority")["Summary"]\
            .str.replace(r"\[provided by RefSeq.+\]", "", regex=True)
    )
    #confirm uniqueness
    assert ncbi_clean.index.is_unique
    return ncbi_clean

def clean_alliance(path):
    '''
    Get descriptions from Alliance GENE-DESCRIPTION-TSV_HUMAN.tsv.gz
    '''
    #load alliance
    alliance = pd.read_csv(path, sep="\t", skiprows=14, header=None)
    alliance.columns = ["HGNC", "Symbol", "Description"]
    #reformat
    alliance = alliance.loc[lambda x: (
        (x["Description"] != "No description available") & (x["Symbol"].notnull())
    )].set_index("Symbol")["Description"].to_frame("Alliance")
    assert alliance.index.is_unique
    return alliance


def clean_uniprot(path, columns=['Function [CC]', 'Miscellaneous [CC]','Subunit structure', 'Subcellular location [CC]', 'Domain [CC]', 'Developmental stage', 'Induction', 'Tissue specificity']):
    #load uniprot, dropping irrelevant columns and relying on primary symbol assigned
    uniprot = pd.read_csv(path, sep="\t").loc[lambda x: x["Gene Names (primary)"].notnull()]
    uniprot["symbol"] = uniprot["Gene Names (primary)"].str.split("; ")
    combined = uniprot.explode("symbol").loc[lambda x: x["symbol"].notnull()].set_index("symbol")[columns].dropna(how="all").apply(
        lambda x: " ".join(x.dropna()) if x.notnull().any() else None, axis=1
    ).dropna().to_frame("UniProt").reset_index()
    #combine genes with duplicate entries (multiple proteins)
    combined = combined.groupby("symbol")["UniProt"].apply(lambda x: " ".join(x.drop_duplicates()))
    assert combined.index.is_unique
    return combined.str.replace(r" \{[^}]*\}", "").str.replace("\.\.",".").dropna().to_frame("UniProt") #remove ECO/PubMed/UniProtKB tags

def clean_civic(path, hgnc_path):
    '''
    Get descriptions from a CIViC Release's FeatureSummaries.tsv
    '''
    #get table and set entrez as index
    civic = pd.read_csv(path, sep="\t").set_index("entrez_id").loc[lambda x: (x["feature_type"] == "Gene")]
    civic.index = civic.index.astype(int).astype(str)
    #map to symbol, dropping NAs
    gsm = GeneSymbolMapper(hgnc_path)
    civic["symbol"] = gsm.map_genes_from(civic.index, "entrez_id")
    civic = civic.loc[lambda x: x["symbol"].notnull()]
    assert civic.index.is_unique
    #drop missing descriptions
    return civic.set_index("symbol")["description"].dropna().to_frame("CIViC")


def get_gene_info(paths="taiga", organism="human"):
    '''
    Get a table of gene descriptions and sources.

    Parameters:
        paths (dict) - mapping from key to local filepath (taiga enables use of set of internal paths)
        organism (str) - indicates which set of files to prepare
    '''
    if paths == "taiga":
        paths = taiga_paths()
    if organism == "human":
        gene_info = pd.concat([
            clean_ncbi_info(paths["ncbi_gene_info"], paths["ncbi_gene_summ"]),
            clean_alliance(paths["alliance_human"]),
            clean_uniprot(paths["uniprot_human"]),
            clean_civic(paths["civic"], paths["hgnc_table"])
        ], axis=1)
    elif organism == "yeast":
        gene_info = pd.concat([
            clean_alliance(paths["alliance_yeast"]),
            clean_uniprot(paths["uniprot_yeast"]),
        ], axis=1)
    else:
        raise Exception("unknown organism")
    sources = pd.read_csv(paths["sources"])
    return gene_info, sources


import datetime
import requests

def fetch_pubchem_record_des(source):
    '''
    Get PubChem records.
    '''
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    def read_pubchem_record_description(data_json):
        parsed = [
            {
                'Compound':None if "LinkedRecords" not in a.keys() or "CID" not in a["LinkedRecords"].keys() else a["LinkedRecords"]["CID"], 
                'Source':a['SourceName'],
                'SourceID':a['SourceID'],
                'Description':a['Data'][0]['Value']['StringWithMarkup'][0]['String']
            } for a in data_json['Annotation']
        ]
        return pd.DataFrame(parsed)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/annotations/heading/JSON/?source={source}&heading_type=Compound&heading=Record%20Description"
    i, max_pages = 1, 1
    page_dict = {}
    while i < (max_pages+1):
        print(i)
        response = requests.get(url+f"&page={i}")
        try:
            if response.ok:
                data_json = response.json()['Annotations']
                page_dict[i] = read_pubchem_record_description(data_json)
                max_pages = data_json['TotalPages'] 
        except Exception as e:
            print(e)
            continue
        i+=1
    return pd.concat(page_dict).reset_index(drop=True).explode("Compound").to_csv(f"pubchem_record_des_{source}_{today}.csv", index=None)

def fetch_pubchem():
    fetch_pubchem_record_des("NCI%20Thesaurus%20(NCIt)")
    fetch_pubchem_record_des("ChEBI")

def clean_pubchem(path):
    records = pd.read_csv(path).loc[lambda x: x["Compound"].notnull()]
    assert records["Source"].nunique() == 1
    source_name = records["Source"].unique()[0]
    records["Compound"] = records["Compound"].astype(int).astype(str)
    description = records.groupby("Compound")["Description"].apply(lambda x: " ".join(x.dropna().drop_duplicates()))
    assert description.index.is_unique
    return description.to_frame(source_name).dropna()

def get_drug_info(paths="taiga"):
    '''
    Get a table of drug descriptions and sources.

    Parameters:
        paths (dict) - mapping from key to local filepath (taiga enables use of set of internal paths)
    '''

    if paths == "taiga":
        paths = taiga_paths()
    drug_info = pd.concat([
        clean_pubchem(paths["pubchem_chebi"]),
        clean_pubchem(paths["pubchem_ncit"])
    ], axis=1)
    sources = pd.DataFrame({
        'Name':['ChEBI', "NCI Thesaurus (NCIt)"], 
        'Link':'https://pubchem.ncbi.nlm.nih.gov/', 
        'Description':[
            'ChEBI description, downloaded from PubChem.',
            'NCI Thesaurus description, downloaded from PubChem.'
        ], 
        'DrugPage':None
    })
    return drug_info, sources