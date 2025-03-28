import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle
import scipy.stats

rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
rcParams['savefig.dpi'] = 500
rcParams['savefig.transparent'] = True
rcParams['font.family'] = 'Arial'
rcParams['figure.dpi'] = 300
rcParams["figure.autolayout"] = True

rcParams['pdf.fonttype']=42
rcParams['ps.fonttype'] = 42

rcParams['axes.titlesize'] = 8
rcParams['axes.labelsize'] = 8
rcParams['font.size'] = 7
rcParams['legend.fontsize'] = 7

#https://coolors.co/5da899-7e2954-2e2585-dccd7d-94cbec-c26a77-9f4a96-337538-dddddd
PALETTE = ['#5da899', '#7e2954','#2e2585','#dccd7d','#94cbec','#c26a77','#9f4a96','#337538','#dddddd']
PALETTE_W_GROUPED = ['#5da899', "#95C6BC", '#7e2954','#2e2585','#dccd7d','#94cbec','#c26a77','#9f4a96','#337538','#dddddd']
ORDER = ["GeneTEA", "g:GOSt", "Enrichr"]
ORDER_W_GROUPED = ["GeneTEA", "GeneTEA-Grouped", "g:GOSt", "Enrichr"]
ORDER_W_RANDOM = ORDER+["Random"]
sns.set_palette(PALETTE)


def plot_hairball(coeffs, thresh=0.5, node_kws={}, edge_kws={}, ax=None):
    '''
    Create a hairball plot.
    '''
    #create a graph
    G=nx.from_pandas_edgelist(coeffs, 'Node1', 'Node2', 'Coeff')

    #remove edges below threshold
    to_remove = [(a,b) for a, b, attrs in G.edges(data=True) if attrs["Coeff"] < thresh]
    G.remove_edges_from(to_remove)
     
    # plot graph in circular layout
    pos = nx.circular_layout(G.nodes())
    nx.draw_networkx_nodes(
        G, node_size=5, pos=pos, ax=ax, **node_kws
    )
    nx.draw_networkx_edges(
        G, width=0.25, pos=pos, ax=ax, **edge_kws
    )

def scientific_format(x):
    '''
    Format float x scientifically.
    '''
    if x == 0:
        return "0"
    elif x > 1e-2:
        return "%.2f" % x
    return '%.2e' % x

def ttest_on_metric(scores, metric, anchor="GeneTEA", alternative="greater", groups=None, **kws):
    '''
    Run a t-test between models on metric with specific anchor and optional groups of comparisons.
    '''
    nans = scores[metric].isnull().sum()
    if nans > 0:
        print(f"dropping {nans} NAs")
        scores = scores.dropna(subset=[metric])
    pvals = {}
    if groups is None:
        grouped = [("all", scores)]
    else:
        grouped = scores.groupby(groups)
    for k, grouped_scores in grouped:
        inputs = grouped_scores.groupby("model")[metric].apply(list)
        group_pvals = {}
        for i in inputs.index:
            if i == anchor: continue
            res = scipy.stats.ttest_ind(inputs[anchor], inputs[i], alternative=alternative, **kws)
            group_pvals[i] = pd.Series({
                "statistic":res.statistic, "pvalue":res.pvalue, "df":res.df, 
                "n anchor":len(inputs[anchor]), "n other":len(inputs[i])
            })
        pvals[k] = pd.concat({anchor:pd.DataFrame(group_pvals)}, axis=1).T
    pvals = pd.concat(pvals)
    pvals["str"] = "p="+pvals["pvalue"].apply(scientific_format)
    if groups is None:
        pvals.reset_index(level=0, drop=True, inplace=True)
    return pvals

def annot_wrapper(ax, pairs, stats, bracket_lw=0.5, annot_format="star", **kws):
    '''
    Wrapper for statannotations.
    '''
    from statannotations.Annotator import Annotator
    annotator = Annotator(ax, pairs, verbose=False, **kws)
    annotator.line_width = bracket_lw
    if annot_format == "custom":
        annotator.set_custom_annotations(stats["str"]).annotate()
    else:
        annotator.configure(text_format=annot_format)
        annotator.set_pvalues(stats["pvalue"]).annotate()

def distplot(
        data, x, y, hue, s=3, dist_plot=sns.boxplot, strip=True,
        order=ORDER, hue_order=ORDER, palette=PALETTE, 
        dodge=False, dist_kws=dict(linewidth=1, notch=False, medianprops=dict(color='#dccd7d')),
        strip_kws=dict(linewidth=0.2, edgecolor="white", facecolor=None, alpha=0.85),
        stats=None, bracket_lw=0.5, annot_format="custom", ax=None
    ):
    '''
    Plot distribution as a box + strip plot.
    '''
    dist_plot(
        data=data, x=x, y=y, hue=hue, saturation=0.75,
        order=order, hue_order=hue_order, palette=palette,
        showfliers=False, dodge=dodge, ax=ax, **dist_kws
    )
    if strip:
        if hue is not None and hue_order is not None:
            hidden_hue = ["_"+h for h in hue_order]
            data2 = data.copy()
            data2[hue].replace({h:"_"+h for h in hue_order}, inplace=True)
            if order == hue_order: 
                order2 = hidden_hue
            else:
                order2 = order
        sns.stripplot(
            data=data2, x=x, y=y, hue=hue, s=s, dodge=dodge,
            order=order2, hue_order=hidden_hue, palette="dark:k", 
            ax=ax, **strip_kws
        )
    if stats is not None:
        if stats.index.nlevels == 2:
            pairs = [(i[0], i[1]) for i in stats.index]
        else:
            pairs = [((i[0], i[1]), (i[0], i[2])) for i in stats.index]
        ax = ax if ax is not None else plt.gca()
        annot_wrapper(
            ax, pairs, stats, bracket_lw=bracket_lw, annot_format=annot_format,
            data=data, x=x, y=y, order=order, hue="model", hue_order=hue_order
        )


def plot_examples(ex, tea, comps, path=None, n=10, order=ORDER, palette=PALETTE, gene_map={}, to_color={}, width_ratios=(3,1), figsize=(3.6,4), save_to=None):
    '''
    Plot an example queries top n terms for GeneTEA and competitors.
    '''
    if path is not None:
        saved_res = pd.read_csv(path, index_col=0)
        res = {k:saved_res.loc[k].reset_index() for k in saved_res.index.unique()}
    else:
        #get top terms
        res = comps.top_n(n=n, genes=ex)
        res["GeneTEA"] = tea.get_enriched_terms(ex, group_subterms=False, n=n, plot=False)
    #sort genes
    ordered_genes = pd.Series(range(len(ex)), index=sorted(ex))
    #setup colors and figure
    colors = dict(zip(order, palette))
    fig, axs = plt.subplots(3,2, figsize=figsize, sharex="col", sharey="row", width_ratios=width_ratios)
    for i, mod in enumerate(ORDER):
        curr_res = res[mod]
        if mod in gene_map.keys():
            curr_res["Matching Genes in List"] = gene_map[mod].loc[curr_res["Term"]].values
        #get term order, clip labels, and prep dfs for plot
        to_plot, term_to_entity = tea._prep_for_plot(curr_res, "Term", " ")
        ordered_terms = pd.Series(reversed(range(len(to_plot["Term"]))), index=to_plot["Term"]).sort_values()
        term_to_entity["OrderedGene"] = term_to_entity["Gene"].replace(ordered_genes)
        term_to_entity["OrderedTerm"] = term_to_entity["Term"].replace(ordered_terms)
        clipped_terms = pd.Series({t:t if len(t) < 40 else t[:37]+"..." for t in to_plot["Term"]}).loc[ordered_terms.index]
        to_plot["OrderedTerm"] = to_plot["Term"].replace(ordered_terms)
        #plot the scatter
        sns.scatterplot(
            data=term_to_entity.loc[lambda x: x["Count"] > 0], 
            x="OrderedGene", y="OrderedTerm", ax=axs[i, 0], color=colors[mod]
        )
        axs[i, 0].set_xticks(ticks=ordered_genes.values)
        axs[i, 0].set_xticklabels(rotation=90, labels=ordered_genes.index)
        axs[i, 0].set_yticks(ticks=ordered_terms.values)
        axs[i, 0].set_xlabel(None)
        axs[i, 0].set_ylabel(None)
        axs[i, 0].set_xlim(-0.5, len(ordered_genes)-0.5)
        axs[i, 0].set_ylim(-0.5, len(ordered_terms)-0.5)
        axs[i, 0].grid(axis="both", alpha=0.2)
        #plot the bars
        sns.barplot(
            data=to_plot, x="n Matching Genes in List", y="OrderedTerm", 
            ax=axs[i, 1], color=colors[mod], orient="h"
        )
        axs[i, 1].set_yticks(ticks=ordered_terms.values)
        axs[i, 1].set_ylabel(None)
        axs[i, 1].set_ylim(-0.5, len(ordered_terms)-0.5)
        axs[i, 1].set_xlabel(None if i != len(axs) -1 else "# Genes")

        #set and color yticks
        axs[i, 0].set_yticklabels(labels=clipped_terms.values)
        for _, (pattern, c)  in to_color.items():
            idxs = np.arange(len(clipped_terms))[clipped_terms.index.str.contains(pattern, case=False)]
            for idx in idxs:
                axs[i, 0].get_yticklabels()[idx].set_color(c)

    if len(to_color):
        col_labels = pd.Series({
            name:Rectangle((0, 0), 1, 1, fc=c, fill=True, edgecolor='none', linewidth=0)
            for name, (_, c) in to_color.items()
        })
        fig.legend(col_labels.values, col_labels.index, loc="lower left", title="Terms Related to")
    
    if save_to:
        pd.concat(res).to_csv(save_to)