# Gene-Term Enrichment Analysis (GeneTEA)

GeneTEA is a tool that leverages ideas from natural language processing to parse gene descriptions into a de novo gene set database for overrepresentation analysis (ORA).
See bioRxiv preprint: "Natural language processing of gene descriptions for overrepresentation analysis with GeneTEA" (Boyle et al. 2025)

Trained models and other data supporting the manuscript can be found in the Figshare repo.


## Installation

To install, use the following commands in a python 3.9 environment:

```
git clone https://github.com/broadinstitute/GeneTEA-dev
cd GeneTEA
pip install -e .
pip install -r requirements.txt
pip install -r other-requirements.txt
```

Note: the requirements are split since those in other-requirements.txt are large and not necessary to run an exisiting GeneTEA model.


## Querying a GeneTEA model

A trained GeneTEA model can be loaded from a pickle file, such as those found in the Figshare.  
For a given query of genes, enriched terms can be obtained from the `get_enriched_terms` function.  The `result` table contains Term, p-value, FDR, etc.

```
from GeneTEA import GeneTEA, load
tea = load("GeneTEA.pkl")
result = tea.get_enriched_terms(
    ["CAD", "DHODH", "UMPS"], #query entities (genes)
    n=10, #number of terms to return, using None will return all significant terms
    max_fdr=0.05, #FDR threshold
)
```

## Training a GeneTEA model

The inputs and code required to train and save a GeneTEA model are as follows:
- `texts`: a pandas DataFrame of gene descriptions index by gene symbol where columns are named by source
- `sources`: a pandas DataFrame with at least columns the following columns
    - `Name`: name of source, ie. columns in `texts`
    - `Link`: html link to source (can be empty)
    - `Description`: description of source (can be empty)

```
from GeneTEA import GeneTEA, save
from GeneTEA.preprocessing import DescriptionCorpus
dc = DescriptionCorpus(
    texts, 
    sources, 
    id_mapper=None,
    map_id_to=None,
)
tea = GeneTEA(corpus=dc)
tea.fit()
save(tea, "GeneTEA.pkl")
```
See `GeneTEA/data_munging.py` and `GeneTEA/train.py` for how the manuscript model's were trained.



## Codebase organization

- The GeneTEA model is defined in the `GeneTEA/` folder
    - `eval/`: various benchmarking analyses and plotting functionality used to compare with competitor ORA models (g:GOSt and Enrichr) in the manuscript
    - `__init__.py`: contains the xTEA class and wrappers for GeneTEA and PharmaTEA
    - `data_munging.py`: functions for prepping the corpuses used in the paper
    - `preprocessing.py`: contains the UMLSPhraseMatcher and DescriptionCorpus classes
    - `synonyms.py`: contains the SynonymExtractor class
    - `train.py`: runs the model training
    - `utils.py`: contains helper functions and the GeneSymbolMapper class
- Notebooks for figure generation and other manuscript analysis can be found in `manuscript-figures/`

Note: in some places `download_to_cache_ext` functions are used, which refers to an Cancer Data Science internal version control client called `taigapy` that is not required to run GeneTEA.
These calls can be replaced with local paths corresponding to the desired file(s).  


# Reference
If you use GeneTEA in your research, please cite our manuscript.