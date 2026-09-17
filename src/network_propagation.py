"""
Phase 2: Network-propagated (pathway-level) features -- the NetBio-style
step that's the key differentiator vs. using raw gene expression alone.

Idea: instead of feeding thousands of individual gene expression values into
the model (which tends to overfit to cohort-specific batch signal and
generalizes poorly across cancer types), propagate each gene's expression
signal across a protein-protein interaction (PPI) network using a random-walk
with restart, then summarize propagated scores per pathway/gene-set. This
produces a smaller, more biologically robust feature set that travels better
across datasets and cancer types (this is the mechanism NetBio credits for
its cross-cancer-type generalization).

Requires a PPI network file (e.g. STRINGdb "protein.links" file, or
BioGRID). Download STRING for your organism at https://string-db.org and
filter to combined_score >= 700 (high confidence) before using here.
"""
import argparse
import numpy as np
import pandas as pd
import networkx as nx


def load_ppi_network(edge_list_path: str, score_col: str = "combined_score",
                      min_score: int = 700) -> nx.Graph:
    """
    Load a PPI edge list (protein1, protein2, combined_score) and build an
    undirected weighted graph, filtered to high-confidence edges.
    """
    edges = pd.read_csv(edge_list_path, sep=" ")
    edges = edges[edges[score_col] >= min_score]
    G = nx.from_pandas_edgelist(edges, source="protein1", target="protein2",
                                 edge_attr=score_col)
    print(f"[info] loaded PPI network: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges (score >= {min_score})")
    return G


def random_walk_with_restart(G: nx.Graph, seed_scores: dict, restart_prob: float = 0.7,
                              max_iter: int = 100, tol: float = 1e-6) -> dict:
    """
    Propagate `seed_scores` (e.g. per-gene expression z-scores for one
    sample) across the network. Returns a dict of propagated scores per node.

    p_{t+1} = (1 - restart_prob) * W @ p_t + restart_prob * p_0
    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    p0 = np.zeros(n)
    for gene, score in seed_scores.items():
        if gene in idx:
            p0[idx[gene]] = score
    if p0.sum() == 0:
        return {node: 0.0 for node in nodes}
    p0 = p0 / (np.abs(p0).sum())

    # Row-normalized adjacency (transition matrix)
    A = nx.to_numpy_array(G, nodelist=nodes, weight="combined_score")
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = A / row_sums

    p = p0.copy()
    for _ in range(max_iter):
        p_next = (1 - restart_prob) * (W.T @ p) + restart_prob * p0
        if np.linalg.norm(p_next - p, 1) < tol:
            p = p_next
            break
        p = p_next

    return {node: p[idx[node]] for node in nodes}


def propagate_pathway_scores(expression_df: pd.DataFrame, ppi_graph: nx.Graph,
                              gene_sets: dict, restart_prob: float = 0.7) -> pd.DataFrame:
    """
    For each sample, propagate its gene expression profile through the PPI
    network, then summarize propagated scores into pathway-level features by
    averaging over each gene set (e.g. Hallmark, KEGG, Reactome pathways).

    `gene_sets`: dict of {pathway_name: [gene1, gene2, ...]}
    """
    gene_cols = [c for c in expression_df.columns if c in ppi_graph.nodes()]
    if not gene_cols:
        raise ValueError(
            "None of the expression matrix's gene columns match PPI network "
            "node names -- check gene ID conventions (symbol vs Ensembl ID)."
        )

    results = []
    for _, row in expression_df.iterrows():
        seed_scores = {g: row[g] for g in gene_cols if pd.notna(row[g])}
        propagated = random_walk_with_restart(ppi_graph, seed_scores, restart_prob)

        sample_features = {"sample_id": row.get("sample_id", None)}
        for pathway, genes in gene_sets.items():
            vals = [propagated[g] for g in genes if g in propagated]
            sample_features[f"pathway_{pathway}"] = np.mean(vals) if vals else np.nan
        results.append(sample_features)

    return pd.DataFrame(results)


# A minimal starter gene-set dictionary. For real use, load MSigDB Hallmark
# or KEGG immune-related pathways instead (e.g. via `gseapy` or a downloaded
# .gmt file parsed with `parse_gmt` below).
DEFAULT_IMMUNE_PATHWAYS = {
    "antigen_presentation": ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2"],
    "t_cell_activation": ["CD3D", "CD3E", "CD28", "LCK", "ZAP70"],
    "interferon_response": ["IFNG", "STAT1", "IRF1", "CXCL9", "CXCL10"],
    "checkpoint_signaling": ["PDCD1", "CD274", "CTLA4", "LAG3", "HAVCR2"],
    "macrophage_polarization": ["CD68", "CD163", "MRC1", "NOS2", "TNF"],
}


def parse_gmt(gmt_path: str) -> dict:
    """Parse a standard MSigDB .gmt gene-set file into {name: [genes]}."""
    gene_sets = {}
    with open(gmt_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            name, _, genes = parts[0], parts[1], parts[2:]
            gene_sets[name] = genes
    return gene_sets


def main():
    parser = argparse.ArgumentParser(description="Compute network-propagated pathway features.")
    parser.add_argument("--expression", required=True)
    parser.add_argument("--ppi-network", required=True, help="STRINGdb-style edge list")
    parser.add_argument("--gmt", default=None, help="Optional .gmt gene-set file")
    parser.add_argument("--restart-prob", type=float, default=0.7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    expr = pd.read_csv(args.expression)
    G = load_ppi_network(args.ppi_network)
    gene_sets = parse_gmt(args.gmt) if args.gmt else DEFAULT_IMMUNE_PATHWAYS

    features = propagate_pathway_scores(expr, G, gene_sets, args.restart_prob)
    features.to_csv(args.output, index=False)
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
