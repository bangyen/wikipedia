"""Link graph feature extraction for Wikipedia articles.

This module extracts graph-based features from Wikipedia article backlinks,
including PageRank, betweenness centrality, and orphan detection metrics.
These features capture the structural importance and connectivity patterns
of articles within the Wikipedia link graph.
"""

import math
from typing import Any, Dict, List

import networkx as nx


def linkgraph_features(article_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract link graph features from Wikipedia article data.

    Analyzes the link graph structure around an article to compute centrality
    metrics, connectivity patterns, and structural importance indicators.

    Args:
        article_data: Raw Wikipedia article JSON data containing backlinks,
                     internal links, and connectivity information

    Returns:
        Dictionary of normalized link graph features including:
        - pagerank_score: PageRank centrality score
        - betweenness_centrality: Betweenness centrality score
        - degree_centrality: Degree centrality score
        - closeness_centrality: Closeness centrality score
        - eigenvector_centrality: Eigenvector centrality score
        - clustering_coefficient: Local clustering coefficient
        - orphan_score: Measure of article isolation
        - hub_score: Hub-like connectivity score
        - authority_score: Authority-like connectivity score
        - connectivity_ratio: Ratio of actual to possible connections
        - structural_holes: Structural holes measure
        - core_periphery_score: Core-periphery position score
    """
    features = {}

    # Extract link data
    backlinks = _extract_backlinks(article_data)
    internal_links = _extract_internal_links(article_data)

    # Build local graph around the article
    graph = _build_local_graph(article_data, backlinks, internal_links)

    if graph.number_of_nodes() == 0:
        # Return zero values if no graph data
        return _get_zero_linkgraph_features()

    # Compute centrality measures
    features.update(_compute_centrality_metrics(graph, article_data.get("title", "")))

    # Compute structural metrics
    features.update(_compute_structural_metrics(graph, article_data.get("title", "")))

    # Compute connectivity patterns
    title = article_data.get("title", "")
    features.update(
        _compute_connectivity_metrics(graph, backlinks, internal_links, title)
    )

    # Compute orphan and isolation metrics
    features.update(_compute_orphan_metrics(graph, backlinks, internal_links))

    # Compute advanced graph metrics
    features.update(_compute_advanced_metrics(graph, article_data.get("title", "")))

    return features


def _build_local_graph(
    article_data: Dict[str, Any],
    backlinks: List[Dict[str, Any]],
    internal_links: List[Dict[str, Any]],
) -> nx.DiGraph:
    """Build a local directed graph around the article.

    Args:
        article_data: Article data containing title and metadata
        backlinks: List of pages linking to this article
        internal_links: List of pages this article links to

    Returns:
        NetworkX directed graph representing local link structure
    """
    graph: nx.DiGraph = nx.DiGraph()

    title = article_data.get("title", "")
    if not title:
        return graph

    # Add the central article
    graph.add_node(title)

    # Add backlinks (incoming edges)
    for backlink in backlinks:
        source_title = backlink.get("title", "")
        if source_title and source_title != title:
            graph.add_edge(source_title, title)

    # Add internal links (outgoing edges)
    for link in internal_links:
        target_title = link.get("title", "")
        if target_title and target_title != title:
            graph.add_edge(title, target_title)

    return graph


def _compute_centrality_metrics(graph: nx.DiGraph, title: str) -> Dict[str, float]:
    """Compute centrality metrics for the article.

    Args:
        graph: NetworkX directed graph
        title: Article title

    Returns:
        Dictionary of centrality metrics
    """
    features = {}

    if title not in graph.nodes():
        return _get_zero_centrality_features()

    try:
        # PageRank centrality
        if graph.number_of_nodes() > 1:
            try:
                pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100, tol=1e-06)
                pr_value = pagerank.get(title, 0.0)
                # Ensure we have a valid numeric value
                features["pagerank_score"] = (
                    float(pr_value) if pr_value is not None else 0.0
                )
            except (nx.PowerIterationFailedConvergence, ZeroDivisionError):
                # Fallback to uniform distribution if PageRank fails
                features["pagerank_score"] = 1.0 / graph.number_of_nodes()
        else:
            features["pagerank_score"] = 1.0 if graph.number_of_nodes() == 1 else 0.0

        # Degree centrality
        degree_centrality = nx.degree_centrality(graph)
        dc_value = degree_centrality.get(title, 0.0)
        features["degree_centrality"] = float(dc_value) if dc_value is not None else 0.0

        # Betweenness centrality (only for connected components)
        if graph.number_of_nodes() > 1:
            try:
                betweenness = nx.betweenness_centrality(
                    graph, k=min(100, graph.number_of_nodes())
                )
                features["betweenness_centrality"] = float(betweenness.get(title, 0.0))
            except nx.NetworkXError:
                features["betweenness_centrality"] = 0.0
        else:
            features["betweenness_centrality"] = 0.0

        # Closeness centrality (only for connected components)
        if graph.number_of_nodes() > 1:
            try:
                closeness = nx.closeness_centrality(graph)
                features["closeness_centrality"] = float(closeness.get(title, 0.0))
            except nx.NetworkXError:
                features["closeness_centrality"] = 0.0
        else:
            features["closeness_centrality"] = 0.0

        # Eigenvector centrality
        try:
            eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
            features["eigenvector_centrality"] = float(eigenvector.get(title, 0.0))
        except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
            features["eigenvector_centrality"] = 0.0

    except Exception:
        return _get_zero_centrality_features()

    return features


def _compute_structural_metrics(graph: nx.DiGraph, title: str) -> Dict[str, float]:
    """Compute structural metrics for the article.

    Args:
        graph: NetworkX directed graph
        title: Article title

    Returns:
        Dictionary of structural metrics
    """
    features = {}

    if title not in graph.nodes():
        return _get_zero_structural_features()

    try:
        # Local clustering coefficient
        clustering = nx.clustering(graph.to_undirected())
        features["clustering_coefficient"] = float(
            clustering.get(title, 0.0) if isinstance(clustering, dict) else 0.0
        )

        # Structural holes (effective size and efficiency)
        try:
            # Convert to undirected for structural holes calculation
            undirected_graph = graph.to_undirected()
            if title in undirected_graph.nodes():
                # Effective size: number of non-redundant contacts
                neighbors = list(undirected_graph.neighbors(title))
                if len(neighbors) > 1:
                    # Count non-redundant connections
                    non_redundant = 0
                    for neighbor in neighbors:
                        # Check if this neighbor is connected to other neighbors
                        other_neighbors = [n for n in neighbors if n != neighbor]
                        if not any(
                            undirected_graph.has_edge(neighbor, other)
                            for other in other_neighbors
                        ):
                            non_redundant += 1
                    features["structural_holes"] = float(non_redundant / len(neighbors))
                else:
                    features["structural_holes"] = 0.0
            else:
                features["structural_holes"] = 0.0
        except Exception:
            features["structural_holes"] = 0.0

        # Core-periphery score (simplified)
        try:
            # Use degree as a proxy for core-periphery position
            degree_dict = dict(graph.degree())  # type: ignore
            degree_value = degree_dict.get(title, 0)
            degree = int(degree_value) if isinstance(degree_value, (int, float)) else 0
            max_degree = max(degree_dict.values()) if graph.number_of_nodes() > 0 else 1
            features["core_periphery_score"] = (
                float(degree / max_degree) if max_degree > 0 else 0.0
            )
        except Exception:
            features["core_periphery_score"] = 0.0

    except Exception:
        return _get_zero_structural_features()

    return features


def _compute_connectivity_metrics(
    graph: nx.DiGraph,
    backlinks: List[Dict[str, Any]],
    internal_links: List[Dict[str, Any]],
    title: str,
) -> Dict[str, float]:
    """Compute connectivity pattern metrics.

    Args:
        graph: NetworkX directed graph
        backlinks: List of backlinks
        internal_links: List of internal links

    Returns:
        Dictionary of connectivity metrics
    """
    features = {}

    try:
        # Hub and authority scores (simplified HITS algorithm)
        if title in graph.nodes():
            node_in_degree = graph.in_degree(title)
            node_out_degree = graph.out_degree(title)
            total_degree = node_in_degree + node_out_degree

            if total_degree > 0:
                # Normalize by total degree for this node
                features["hub_score"] = float(node_out_degree / total_degree)
                features["authority_score"] = float(node_in_degree / total_degree)
            else:
                features["hub_score"] = 0.0
                features["authority_score"] = 0.0
        else:
            features["hub_score"] = 0.0
            features["authority_score"] = 0.0

        # Connectivity ratio (actual vs possible connections)
        n_nodes = graph.number_of_nodes()
        if n_nodes > 1:
            max_possible = n_nodes * (n_nodes - 1)  # Directed graph
            actual_edges = graph.number_of_edges()
            features["connectivity_ratio"] = float(actual_edges / max_possible)
        else:
            features["connectivity_ratio"] = 0.0

        # Link balance (incoming vs outgoing)
        if len(backlinks) > 0 or len(internal_links) > 0:
            total_links = len(backlinks) + len(internal_links)
            features["link_balance"] = float(
                1.0 - abs(len(backlinks) - len(internal_links)) / total_links
            )
        else:
            features["link_balance"] = 0.0

    except Exception:
        features["hub_score"] = 0.0
        features["authority_score"] = 0.0
        features["connectivity_ratio"] = 0.0
        features["link_balance"] = 0.0

    return features


def _compute_orphan_metrics(
    graph: nx.DiGraph,
    backlinks: List[Dict[str, Any]],
    internal_links: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute orphan and isolation metrics.

    Args:
        graph: NetworkX directed graph
        backlinks: List of backlinks
        internal_links: List of internal links

    Returns:
        Dictionary of orphan metrics
    """
    features = {}

    try:
        # Orphan score (inverse of connectivity)
        total_connections = len(backlinks) + len(internal_links)
        if total_connections > 0:
            # Normalize by log to handle scale differences
            features["orphan_score"] = float(
                1.0 / (1.0 + math.log(total_connections + 1))
            )
        else:
            features["orphan_score"] = 1.0  # Completely orphaned

        # Isolation score (based on graph connectivity)
        if graph.number_of_nodes() > 1:
            # Check if article is in largest connected component
            try:
                largest_cc = max(
                    nx.connected_components(graph.to_undirected()), key=len
                )
                features["isolation_score"] = float(1.0 / len(largest_cc))
            except Exception:
                features["isolation_score"] = 1.0
        else:
            features["isolation_score"] = 1.0

        # Dead-end score (only outgoing links, no incoming)
        if len(backlinks) == 0 and len(internal_links) > 0:
            features["dead_end_score"] = 1.0
        elif len(backlinks) > 0 and len(internal_links) == 0:
            features["dead_end_score"] = 0.5  # Only incoming links
        else:
            features["dead_end_score"] = 0.0

    except Exception:
        features["orphan_score"] = 1.0
        features["isolation_score"] = 1.0
        features["dead_end_score"] = 0.0

    return features


def _compute_advanced_metrics(graph: nx.DiGraph, title: str) -> Dict[str, float]:
    """Compute advanced graph metrics.

    Args:
        graph: NetworkX directed graph
        title: Article title

    Returns:
        Dictionary of advanced metrics
    """
    features = {}

    try:
        # Graph density
        n_nodes = graph.number_of_nodes()
        if n_nodes > 1:
            max_edges = n_nodes * (n_nodes - 1)  # Directed graph
            actual_edges = graph.number_of_edges()
            features["graph_density"] = float(actual_edges / max_edges)
        else:
            features["graph_density"] = 0.0

        # Assortativity (degree correlation)
        try:
            if graph.number_of_nodes() > 2:
                assortativity = nx.degree_assortativity_coefficient(
                    graph.to_undirected()
                )
                features["assortativity"] = float(assortativity)
            else:
                features["assortativity"] = 0.0
        except Exception:
            features["assortativity"] = 0.0

        # Small world coefficient (simplified)
        try:
            if graph.number_of_nodes() > 2:
                # Compare clustering to random graph
                clustering = nx.average_clustering(graph.to_undirected())
                # Random graph clustering approximation
                random_clustering = (
                    2 * graph.number_of_edges() / (n_nodes * (n_nodes - 1))
                )
                if random_clustering > 0:
                    features["small_world_coefficient"] = float(
                        clustering / random_clustering
                    )
                else:
                    features["small_world_coefficient"] = 0.0
            else:
                features["small_world_coefficient"] = 0.0
        except Exception:
            features["small_world_coefficient"] = 0.0

        # Log scaling for large values
        features["log_pagerank_score"] = math.log(
            max(features.get("pagerank_score", 0), 1e-10)
        )
        features["log_degree_centrality"] = math.log(
            max(features.get("degree_centrality", 0), 1e-10)
        )

    except Exception:
        features["graph_density"] = 0.0
        features["assortativity"] = 0.0
        features["small_world_coefficient"] = 0.0
        features["log_pagerank_score"] = 0.0
        features["log_degree_centrality"] = 0.0

    return features


def _get_zero_linkgraph_features() -> Dict[str, float]:
    """Return zero values for all link graph features."""
    return {
        "pagerank_score": 0.0,
        "betweenness_centrality": 0.0,
        "degree_centrality": 0.0,
        "closeness_centrality": 0.0,
        "eigenvector_centrality": 0.0,
        "clustering_coefficient": 0.0,
        "orphan_score": 1.0,
        "hub_score": 0.0,
        "authority_score": 0.0,
        "connectivity_ratio": 0.0,
        "structural_holes": 0.0,
        "core_periphery_score": 0.0,
        "isolation_score": 1.0,
        "dead_end_score": 0.0,
        "graph_density": 0.0,
        "assortativity": 0.0,
        "small_world_coefficient": 0.0,
        "log_pagerank_score": 0.0,
        "log_degree_centrality": 0.0,
        "link_balance": 0.0,
    }


def _get_zero_centrality_features() -> Dict[str, float]:
    """Return zero values for centrality features."""
    return {
        "pagerank_score": 0.0,
        "degree_centrality": 0.0,
        "betweenness_centrality": 0.0,
        "closeness_centrality": 0.0,
        "eigenvector_centrality": 0.0,
    }


def _get_zero_structural_features() -> Dict[str, float]:
    """Return zero values for structural features."""
    return {
        "clustering_coefficient": 0.0,
        "structural_holes": 0.0,
        "core_periphery_score": 0.0,
    }


# Helper functions for data extraction (reused from extractors.py)


def _extract_backlinks(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract backlinks from article data."""
    backlinks = []

    data = article_data.get("data", {})

    if "query" in data and "backlinks" in data["query"]:
        backlinks = data["query"]["backlinks"]

    return backlinks if isinstance(backlinks, list) else []


def _extract_internal_links(article_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract internal links from article data."""
    internal_links = []

    data = article_data.get("data", {})

    if "query" in data and "pages" in data["query"]:
        for page_id, page_data in data["query"]["pages"].items():
            if "links" in page_data:
                internal_links = page_data["links"]
                break

    return internal_links if isinstance(internal_links, list) else []
