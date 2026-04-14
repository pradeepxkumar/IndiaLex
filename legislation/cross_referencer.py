"""
IndiaLexABSA — Cross-Referencer
=================================
Builds a directed graph of inter-clause references using NetworkX.
Used for context window expansion during clause linking.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx
from loguru import logger


class CrossReferencer:
    """Builds and queries the clause cross-reference graph."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def build_from_clauses(self, clauses: list[dict]) -> None:
        """Build the reference graph from parsed clause data."""
        # Add all clause nodes
        for clause in clauses:
            self.graph.add_node(
                clause["clause_id"],
                title=clause.get("title", ""),
                level=clause.get("level", "section"),
                section_num=clause.get("section_num", 0),
            )

        # Add edges for cross-references
        for clause in clauses:
            for ref in clause.get("cross_refs", []):
                if ref in self.graph.nodes:
                    self.graph.add_edge(clause["clause_id"], ref, type="cross_ref")

        # Add parent-child edges
        for clause in clauses:
            if clause.get("parent_id") and clause["parent_id"] in self.graph.nodes:
                self.graph.add_edge(
                    clause["parent_id"],
                    clause["clause_id"],
                    type="parent_child",
                )

        logger.info(
            f"Cross-reference graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

    def get_related(self, clause_id: str, depth: int = 1) -> list[str]:
        """Return clause IDs related to the given clause within `depth` hops."""
        if clause_id not in self.graph:
            return []
        related = set()
        frontier = {clause_id}
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(self.graph.predecessors(node))
                next_frontier.update(self.graph.successors(node))
            next_frontier -= {clause_id}
            related.update(next_frontier)
            frontier = next_frontier
        return list(related)

    def get_most_connected(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Return the most cross-referenced clauses (by in-degree)."""
        degrees = [(n, self.graph.in_degree(n)) for n in self.graph.nodes]
        return sorted(degrees, key=lambda x: x[1], reverse=True)[:top_n]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)
        logger.info(f"Loaded cross-reference graph: {self.graph.number_of_nodes()} nodes")
