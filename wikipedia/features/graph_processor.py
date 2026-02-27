"""Global graph processor for Wikipedia link analysis.

This module provides tools for building and analyzing large-scale Wikipedia
link graphs using DuckDB for data ingestion and NetworkX for graph metrics.
"""

from pathlib import Path
from typing import Dict, Optional

import duckdb
import networkx as nx
import pandas as pd


class GraphProcessor:
    """Processor for large-scale Wikipedia link graph analysis."""

    def __init__(self, db_path: str = "wiki_graph.duckdb") -> None:
        """Initialize the graph processor.

        Args:
            db_path: Path to the DuckDB database file
        """
        self.db_path = db_path
        self.con = duckdb.connect(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS links (
                from_id INTEGER,
                to_id INTEGER,
                from_title VARCHAR,
                to_title VARCHAR
            )
        """
        )

    def ingest_links(self, links_path: str) -> None:
        """Ingest link data from a Parquet or CSV file.

        Args:
            links_path: Path to the link data file
        """
        path = Path(links_path)
        if path.suffix == ".parquet":
            self.con.execute(
                f"INSERT INTO links SELECT * FROM read_parquet('{links_path}')"
            )
        elif path.suffix == ".csv":
            self.con.execute(
                f"INSERT INTO links SELECT * FROM read_csv_auto('{links_path}')"
            )
        else:
            raise ValueError(f"Unsupported link file format: {path.suffix}")

    def build_networkx_graph(self) -> nx.DiGraph:
        """Build a NetworkX DiGraph from the ingested links.

        Returns:
            NetworkX directed graph
        """
        print("Fetching links from database...")
        df = self.con.execute("SELECT from_title, to_title FROM links").df()

        print(f"Building graph with {len(df)} edges...")
        G = nx.from_pandas_edgelist(
            df, source="from_title", target="to_title", create_using=nx.DiGraph()
        )
        return G

    def compute_global_metrics(self, G: nx.DiGraph) -> Dict[str, Dict[str, float]]:
        """Compute global graph metrics for all nodes in the graph.

        Args:
            G: NetworkX directed graph

        Returns:
            Dictionary mapping article titles to their metrics
        """
        print("Computing PageRank...")
        pagerank = nx.pagerank(G, alpha=0.85)

        print("Computing Authority/Hub scores (HITS)...")
        # HITS can be slow on very large graphs, using a sample or simplified approach if needed
        hubs, authorities = nx.hits(G, max_iter=50)

        metrics = {}
        for node in G.nodes():
            metrics[node] = {
                "pagerank": pagerank.get(node, 0.0),
                "authority_score": authorities.get(node, 0.0),
                "hub_score": hubs.get(node, 0.0),
                "degree_centrality": G.in_degree(node) / len(G) if len(G) > 0 else 0.0,
            }

        return metrics

    def save_metrics_to_db(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """Save computed metrics back to the database for persistence."""
        df = pd.DataFrame.from_dict(metrics, orient="index").reset_index()
        df.columns = [
            "title",
            "pagerank",
            "authority_score",
            "hub_score",
            "degree_centrality",
        ]

        self.con.execute("CREATE TABLE IF NOT EXISTS node_metrics AS SELECT * FROM df")
        self.con.execute("INSERT OR REPLACE INTO node_metrics SELECT * FROM df")

    def get_article_metrics(self, title: str) -> Optional[Dict[str, float]]:
        """Retrieve pre-computed metrics for an article."""
        try:
            res = self.con.execute(
                "SELECT pagerank, authority_score, hub_score, degree_centrality FROM node_metrics WHERE title = ?",
                [title],
            ).fetchone()
            if res:
                return {
                    "pagerank": res[0],
                    "authority_score": res[1],
                    "hub_score": res[2],
                    "degree_centrality": res[3],
                }
        except Exception:
            return None
        return None
