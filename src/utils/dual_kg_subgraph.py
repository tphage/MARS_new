import itertools
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from GraphReasoning import find_best_fitting_node_list


@dataclass(frozen=True)
class KgMappingCaps:
    max_property_terms: int
    max_materials: int
    max_nodes_total: int
    max_pairs_evaluated: int
    max_shortest_path_len: int


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def map_terms_to_nodes_best_match(
    terms: Sequence[str],
    node_embeddings: Dict[str, np.ndarray],
    embedding_tokenizer: Any,
    embedding_model: Any,
    *,
    n_samples: int,
    similarity_threshold: float,
    max_terms: int,
) -> List[str]:
    """Map free-text terms to KG node IDs using embedding similarity (best match only)."""
    if not terms:
        return []
    mapped: List[str] = []
    for term in list(terms)[: max_terms]:
        if not term or not isinstance(term, str):
            continue
        matches = find_best_fitting_node_list(
            term,
            node_embeddings,
            embedding_tokenizer,
            embedding_model,
            N_samples=n_samples,
            similarity_threshold=similarity_threshold,
        )
        if matches:
            mapped.append(matches[0][0])
    # preserve order, de-dupe
    seen = set()
    out = []
    for n in mapped:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def build_connection_subgraph_shortest_paths(
    graph: nx.DiGraph,
    seed_nodes: Sequence[str],
    *,
    max_pairs_evaluated: int,
    max_shortest_path_len: int,
    max_nodes_total: int,
) -> nx.DiGraph:
    """Build a bounded connection subgraph by shortest paths among seed nodes.

    The input graph is converted to undirected so that shortest-path search
    can traverse edges in either direction.  The resulting subgraph is cast
    back to ``nx.DiGraph`` (the standard type expected by all downstream
    code), which creates two directed edges (A->B and B->A) for every
    undirected edge.  This is intentional: the merge and analysis layers
    treat edges symmetrically.
    """
    if graph is None or not seed_nodes:
        return nx.DiGraph()

    undirected = graph.to_undirected()
    seeds = [n for n in seed_nodes if n in undirected]
    if len(seeds) < 2:
        return nx.DiGraph()

    nodes_set = set(seeds)
    pairs = itertools.islice(itertools.combinations(seeds, 2), max_pairs_evaluated)
    for u, v in pairs:
        try:
            path = nx.shortest_path(undirected, u, v)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if len(path) > max_shortest_path_len:
            continue
        nodes_set.update(path)
        if len(nodes_set) >= max_nodes_total:
            break

    sub = undirected.subgraph(list(nodes_set)).copy()
    # Normalize to DiGraph for downstream code paths
    return nx.DiGraph(sub)


def merge_subgraphs_unify_by_embedding(
    matkg_subgraph: nx.DiGraph,
    patkg_subgraph: nx.DiGraph,
    matkg_node_embeddings: Dict[str, np.ndarray],
    patkg_node_embeddings: Dict[str, np.ndarray],
    *,
    similarity_threshold: float,
) -> Tuple[nx.DiGraph, Dict[str, str]]:
    """
    Merge two subgraphs by unifying patent-KG nodes onto material-KG nodes when
    their node embedding cosine similarity exceeds threshold.

    Returns merged graph + mapping patent_node_id -> merged_node_id.
    """
    merged = nx.DiGraph()

    # Start with all matkg nodes/edges
    for n, attrs in matkg_subgraph.nodes(data=True):
        merged.add_node(n, **dict(attrs), source_kgs=["material_properties"])
    for u, v, attrs in matkg_subgraph.edges(data=True):
        merged.add_edge(u, v, **dict(attrs), source_kgs=["material_properties"])

    mat_nodes = list(matkg_subgraph.nodes())
    pat_nodes = list(patkg_subgraph.nodes())

    mapping: Dict[str, str] = {}
    if mat_nodes and pat_nodes:
        mat_vecs = []
        mat_keep = []
        for n in mat_nodes:
            vec = matkg_node_embeddings.get(n)
            if vec is not None:
                mat_vecs.append(vec)
                mat_keep.append(n)
        pat_vecs = []
        pat_keep = []
        for n in pat_nodes:
            vec = patkg_node_embeddings.get(n)
            if vec is not None:
                pat_vecs.append(vec)
                pat_keep.append(n)

        if mat_vecs and pat_vecs:
            sims = _cosine_sim_matrix(np.vstack(pat_vecs), np.vstack(mat_vecs))
            # For each patent node, pick best mat node
            for i, pat_n in enumerate(pat_keep):
                j = int(np.argmax(sims[i]))
                best_sim = float(sims[i, j])
                if best_sim >= similarity_threshold:
                    mapping[pat_n] = mat_keep[j]

    # Add patents nodes/edges (mapped or unmapped)
    for n, attrs in patkg_subgraph.nodes(data=True):
        merged_n = mapping.get(n, n)
        if merged_n not in merged:
            merged.add_node(merged_n, **dict(attrs), source_kgs=["patents"])
        else:
            # Merge node attributes conservatively: keep existing keys, add patents-only keys under prefixed namespace
            for k, v in dict(attrs).items():
                if k not in merged.nodes[merged_n]:
                    merged.nodes[merged_n][k] = v
                else:
                    merged.nodes[merged_n][f"patents__{k}"] = v
            srcs = merged.nodes[merged_n].get("source_kgs", [])
            if "patents" not in srcs:
                merged.nodes[merged_n]["source_kgs"] = list(srcs) + ["patents"]

    for u, v, attrs in patkg_subgraph.edges(data=True):
        mu = mapping.get(u, u)
        mv = mapping.get(v, v)
        if merged.has_edge(mu, mv):
            # Preserve both attribute sets
            existing = dict(merged[mu][mv])
            for k, v_attr in dict(attrs).items():
                if k not in existing:
                    merged[mu][mv][k] = v_attr
                else:
                    merged[mu][mv][f"patents__{k}"] = v_attr
            srcs = merged[mu][mv].get("source_kgs", [])
            if "patents" not in srcs:
                merged[mu][mv]["source_kgs"] = list(srcs) + ["patents"]
        else:
            merged.add_edge(mu, mv, **dict(attrs), source_kgs=["patents"])

    return merged, mapping

