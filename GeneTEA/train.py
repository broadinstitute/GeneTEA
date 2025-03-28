import glob
import os
from .data_munging import get_seed_stopwords, get_gene_info, get_drug_info
from . import GeneTEA, PharmaTEA, save, load
from .preprocessing import DescriptionCorpus, UMLSPhraseMatcher

def train_and_save_models(folder="trained_models"):
    '''
    Train GeneTEA, GeneTEA-yeast, and PharmaTEA.
    '''
    if not os.path.exists(folder):
        os.mkdir(folder)

    phraser = UMLSPhraseMatcher()
    drug_info, drug_sources = get_drug_info()
    gene_info, gene_sources = get_gene_info(organism="human")
    yeast_info, yeast_sources = get_gene_info(organism="yeast")
    seed_stopwords = get_seed_stopwords()

    teas = {
        "GeneTEA":(
            GeneTEA, "Gene", gene_info, gene_sources,
            dict(phraser=phraser)
        ),
        "GeneTEA-yeast":(
            GeneTEA, "Gene", yeast_info, yeast_sources, 
            dict(id_mapper=None, phraser=phraser)
        ),
        "PharmaTEA":(
            PharmaTEA, "Drug", drug_info, drug_sources, 
            dict(id_mapper=None, phraser=phraser)
        ),
    }
    USE_SYNS = None
    for k, (model, entity_type, texts, sources, kws) in teas.items():
        dc = DescriptionCorpus(texts, sources, entity_type=entity_type, synonyms=USE_SYNS, **kws)
        if "GeneTEA" in k:
            USE_SYNS, USE_STOPWORDS = dc.synonyms, dc.nominate_stopwords(seed_stopwords)
        dc.save_embedding(f"{folder}/{k}_embedding.csv")
        dc.del_embedding()
        tea = model(corpus=dc, custom_stopwords=USE_STOPWORDS)
        tea.fit()
        print(f"trained {k}")
        save(tea, f"{folder}/{k}.pkl")
    return

def load_models(folder="trained_models", name=None):
    '''
    Load saved model(s).
    '''
    teas = {}
    for file in glob.glob(os.path.join(folder, "*.pkl")):
        k = file.split("/")[-1].replace(".pkl", "")
        if name is not None:
            if name == k:
                return load(file)
            elif k in name:
                teas[k] = load(file)
        else:
            teas[k] = load(file)
    return teas

if __name__ == "__main__":
    train_and_save_models()