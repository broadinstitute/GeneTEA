import pandas as pd
import time
from ..utils import jaccard_index
import requests
import json
import concurrent.futures
from GeneTEA.utils import read_gmt

ENRICHR_LIBS = [
    'Achilles_fitness_decrease',
    'Achilles_fitness_increase',
    'Allen_Brain_Atlas_10x_scRNA_2021',
    'Allen_Brain_Atlas_down',
    'Allen_Brain_Atlas_up',
    'ARCHS4_Cell-lines',
    'ARCHS4_IDG_Coexp',
    'ARCHS4_Kinases_Coexp',
    'ARCHS4_TFs_Coexp',
    'ARCHS4_Tissues',
    'Azimuth_2023',
    'Azimuth_Cell_Types_2021',
    'BioCarta_2016',
    'BioPlanet_2019',
    'BioPlex_2017',
    'Cancer_Cell_Line_Encyclopedia',
    'CCLE_Proteomics_2020',
    'CellMarker_2024',
    'CellMarker_Augmented_2021',
    'ChEA_2022',
    'ClinVar_2019',
    'CORUM',
    'COVID-19_Related_Gene_Sets_2021',
    'dbGaP',
    'DepMap_CRISPR_GeneDependency_CellLines_2023',
    'DepMap_WG_CRISPR_Screens_Sanger_CellLines_2019',
    'Descartes_Cell_Types_and_Tissue_2021',
    'DGIdb_Drug_Targets_2024',
    'Diabetes_Perturbations_GEO_2022',
    'DisGeNET',
    'DrugMatrix',
    'DSigDB',
    'Elsevier_Pathway_Collection',
    'ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X',
    'ENCODE_Histone_Modifications_2015',
    'ENCODE_TF_ChIP-seq_2015',
    'Enrichr_Submissions_TF-Gene_Coocurrence',
    'Epigenomics_Roadmap_HM_ChIP-seq',
    'ESCAPE',
    'FANTOM6_lncRNA_KD_DEGs',
    'GeDiPNet_2023',
    'GeneSigDB',
    'Genome_Browser_PWMs',
    'GlyGen_Glycosylated_Proteins_2022',
    'GO_Biological_Process_2023',
    'GO_Cellular_Component_2023',
    'GO_Molecular_Function_2023',
    'GTEx_Aging_Signatures_2021',
    'GTEx_Tissue_Expression_Down',
    'GTEx_Tissue_Expression_Up',
    'GTEx_Tissues_V8_2023',
    'GWAS_Catalog_2023',
    'HDSigDB_Human_2021',
    'HDSigDB_Mouse_2021',
    'HMS_LINCS_KinomeScan',
    'HuBMAP_ASCT_plus_B_augmented_w_RNAseq_Coexpression',
    'HuBMAP_ASCTplusB_augmented_2022',
    'Human_Gene_Atlas',
    'Human_Phenotype_Ontology',
    'HumanCyc_2016',
    'huMAP',
    'IDG_Drug_Targets_2022',
    'Jensen_COMPARTMENTS',
    'Jensen_DISEASES',
    'Jensen_TISSUES',
    'KEA_2015',
    'KEGG_2021_Human',
    'Kinase_Perturbations_from_GEO_down',
    'Kinase_Perturbations_from_GEO_up',
    'KOMP2_Mouse_Phenotypes_2022',
    'LINCS_L1000_Chem_Pert_Consensus_Sigs',
    'LINCS_L1000_CRISPR_KO_Consensus_Sigs',
    'lncHUB_lncRNA_Co-Expression',
    'MAGMA_Drugs_and_Diseases',
    'MAGNET_2023',
    'Metabolomics_Workbench_Metabolites_2022',
    'MGI_Mammalian_Phenotype_Level_4_2024',
    'miRTarBase_2017',
    'MoTrPAC_2023',
    'Mouse_Gene_Atlas',
    'MSigDB_Computational',
    'MSigDB_Hallmark_2020',
    'MSigDB_Oncogenic_Signatures',
    'NCI-60_Cancer_Cell_Lines',
    'NCI-Nature_2016',
    'NURSA_Human_Endogenous_Complexome',
    'Old_CMAP_down',
    'Old_CMAP_up',
    'OMIM_Disease',
    'OMIM_Expanded',
    'Orphanet_Augmented_2021',
    'PanglaoDB_Augmented_2021',
    'Panther_2016',
    'PerturbAtlas',
    'PFOCR_Pathways_2023',
    'PhenGenI_Association_2021',
    'PheWeb_2019',
    'Phosphatase_Substrates_from_DEPOD',
    'PPI_Hub_Proteins',
    'Proteomics_Drug_Atlas_2023',
    'ProteomicsDB_2020',
    'Rare_Diseases_AutoRIF_ARCHS4_Predictions',
    'Rare_Diseases_AutoRIF_Gene_Lists',
    'Rare_Diseases_GeneRIF_ARCHS4_Predictions',
    'Rare_Diseases_GeneRIF_Gene_Lists',
    'Rummagene_kinases',
    'Rummagene_transcription_factors',
    'SILAC_Phosphoproteomics',
    'SubCell_BarCode',
    'SynGO_2022',
    'SynGO_2024',
    'Tabula_Muris',
    'Tabula_Sapiens',
    'TargetScan_microRNA',
    'TF-LOF_Expression_from_GEO',
    'TF_Perturbations_Followed_by_Expression',
    'TG_GATES_2020',
    'The_Kinase_Library_2024',
    'Tissue_Protein_Expression_from_Human_Proteome_Map',
    'Transcription_Factor_PPIs',
    'TRANSFAC_and_JASPAR_PWMs',
    'TRRUST_Transcription_Factors_2019',
    'UK_Biobank_GWAS_v1',
    'Virus-Host_PPI_P-HIPSTer_2020',
    'Virus_Perturbations_from_GEO_down',
    'Virus_Perturbations_from_GEO_up',
    'VirusMINT',
    'WikiPathways_2024_Human',
    'WikiPathways_2024_Mouse',
    'Reactome_Pathways_2024',
 ]

class EnrichrAPI():
    '''
    Class for calling the EnrichrAPI.
    '''
    def __init__(self, libraries=ENRICHR_LIBS, find_jaccard=False):
        self.LIB_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary"
        self.GENE_LIST_URL = 'https://maayanlab.cloud/Enrichr/addList'
        self.QUERY_URL = 'https://maayanlab.cloud/Enrichr/enrich'
        self.columns = [
            'Rank','Term name','P-value','Odds ratio',
            'Combined score','Overlapping genes','Adjusted p-value',
            'Old p-value','Old adjusted p-value'
        ]
        self.libraries = libraries
        self.find_jaccard = find_jaccard
        if self.find_jaccard:
            self.gene_sets = pd.DataFrame(
                {lib:self.get_library(lib) for lib in libraries}
            ).T.stack().dropna().rename_axis(["Library", "Term name"])
            self.gene_set_lens = self.gene_sets.apply(len).to_frame("n Matching Genes Overall")
    
    def get_library(self, gene_set_library, local_path=None):
        '''
        Get the gene sets in a library.
        '''
        if local_path:
            try:
                path = local_path+f"{gene_set_library}.txt"
                gmt = read_gmt(path)
                gmt["Library"] = gene_set_library
                return gmt.rename_axis("Set").drop(columns="Name")
            except Exception as e:
                print("Enrichr failed for", gene_set_library, "with Exception", e)
                return None
        url = self.LIB_URL + '?mode=text&libraryName=%s' % (gene_set_library)
        for attempt in range(3):
            try:
                response = requests.get(url)
                break
            except requests.exceptions.ChunkedEncodingError:
                print("attempt", attempt, "for", url)
                time.sleep(5)
        if not response.ok:
            print("Failed on", gene_set_library, response.status_code, response.reason, url)
            return None
        print("Succeeded on", gene_set_library)
        #read as if GMT with no name
        genesets_dict = {}
        for line in response.text.strip().split("\n"):
            entries = line.strip().split("\t")
            key = entries[0] 
            genesets_dict[key] = pd.Series({"Genes":entries[2:]})
        return pd.DataFrame(genesets_dict).T
    
    def get_libraries(self, max_workers=1, local_path=None):
        '''
        Get a set of libraries.
        '''
        if local_path:
            return [self.get_library(lib, local_path=local_path) for lib in self.libraries]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = pool.map(self.get_library, self.libraries)
        return list(results)
        
    def setup_genelist(self, genes):
        '''
        Add a gene list.
        '''
        payload = {
            'list': (None, '\n'.join(genes)),
            'description': (None, 'Gene list')
        }
        response = requests.post(self.GENE_LIST_URL, files=payload)
        response.raise_for_status()
        data = json.loads(response.text)
        return data['userListId']

    def query_library(self, user_list_id, gene_set_library):
        '''
        Query a library with a gene list.
        '''
        response = requests.get(
            self.QUERY_URL + '?userListId=%s&backgroundType=%s' % (user_list_id, gene_set_library)
         )
        response.raise_for_status()
        data = json.loads(response.text)
        data_df = pd.DataFrame(data[gene_set_library], columns=self.columns)
        data_df['Library'] = gene_set_library
        return data_df

    def query(self, gene_list, max_workers=50):
        '''
        Query all libraries with a gene list.
        '''
        user_list_id = self.setup_genelist(gene_list)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            query = lambda x: self.query_library(user_list_id, x)
            results = pool.map(query, self.libraries)
        result = pd.concat(list(results)).sort_values("Combined score")
        result['n Matching Genes in List'] = result['Overlapping genes'].apply(len)
        return result

    
class Competitors():
    '''
    Class that enables querying of competitors (g:GOSt and Enrichr)
    '''
    def __init__(self, names=["g:GOSt", "Enrichr"], find_jaccard=False):
        self.names = names
        from gprofiler import GProfiler
        self.gprofiler = GProfiler(return_dataframe=True)
        self.find_jaccard = find_jaccard
        self.enrichr_api = EnrichrAPI(find_jaccard=find_jaccard)
        self.name_map = {
            "g:GOSt":self.gprof,
            "Enrichr":self.enrichr
        }

    def gprof(self, gene_list):
        '''
        Query gProfiler's g:GOSt.
        '''
        try:
            result = self.gprofiler.profile(
                organism='hsapiens', query=gene_list, sources=[],  all_results=True,
            ).rename(columns={
                "name":"Term", "p_value":"FDR", 
                "term_size":"n Matching Genes Overall", "intersection_size":"n Matching Genes in List", 
            }).set_index('Term')
            #handle duplicated terms in g:GOSt by giving benefit of doubt and only keeping best (first) term
            result = result.loc[~result.index.duplicated(keep="first")].copy()
            if self.find_jaccard:
                result["Jaccard Index"] = jaccard_index(
                    result["n Matching Genes in List"], result["n Matching Genes Overall"], len(gene_list)
                )
            return result.reset_index()
        except Exception as e:
            print(f"g:GOSt errored on {gene_list}", e)
            return None
        
    def enrichr(self, gene_list):
        '''
        Query Enrichr.
        '''
        try:
            result = self.enrichr_api.query(gene_list).rename(columns={
                "Term name":"Term", "Adjusted p-value":"FDR"
            }).set_index('Term')
            result["Matching Genes in List"] = result["Overlapping genes"].apply(" ".join)
            result = result.loc[~result.index.duplicated(keep="first")].copy()
            return result.reset_index()
        except Exception as e:
            print(f"Enrichr errored on {gene_list}", e)
            return None
        
    def query(self, gene_list, sig=True, max_fdr=0.05, min_genes=2):
        '''
        Get all queries and filter.
        '''
        results = {k:self.name_map[k](gene_list) for k in self.names}
        if not sig: return results
        return {
            "g:GOSt":None if results["g:GOSt"] is None else results["g:GOSt"].loc[
                lambda x: x["significant"] & (x["n Matching Genes in List"] >= min_genes if min_genes is not None else True)
            ].sort_values("FDR"),
            "Enrichr":None if results["Enrichr"] is None else results["Enrichr"].loc[
                lambda x: ((x["FDR"] < max_fdr if max_fdr is not None else True) 
                            & (x["n Matching Genes in List"] >= min_genes if min_genes is not None else True))
            ].sort_values(["Rank", "Combined score"], ascending=[True, False])
        }
    
    def top_n(self, n=10, results=None, genes=None, max_fdr=0.05, min_genes=2):
        '''
        Get the top N for a query from all competitors.
        '''
        assert results is not None or genes is not None, "results or genes must be provided"
        assert not (results is not None and genes is not None), "only one of results or genes must be provided"
        if results is None:
            results = self.query(genes, max_fdr=max_fdr, min_genes=min_genes)
        return {
            "g:GOSt":None if results["g:GOSt"] is None else results["g:GOSt"].sort_values("FDR").head(n),
            "Enrichr":None if results["Enrichr"] is None else results["Enrichr"].sort_values(["Rank", "Combined score"], ascending=[True, False]).head(n)
        }



if __name__ == "__main__":
    enrichr_api = EnrichrAPI()
    enrichr_sets = enrichr_api.get_libraries(local_path="../enrichr_sets/")
    enrichr_sets = pd.concat(enrichr_sets).dropna()
    enrichr_sets["Genes"] = enrichr_sets["Genes"].apply("|".join)
    enrichr_sets.to_csv("../manuscript-resources/enrichr_sets_03_01_2025.csv")