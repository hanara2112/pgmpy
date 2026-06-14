import networkx as nx
import numpy as np
import pandas as pd
import pytest

from pgmpy.base import DAG
from pgmpy.causal_discovery import ExpertInLoop, LLMPairwise
from pgmpy.causal_discovery._base import BaseCausalDiscovery


class FakePairwise(BaseCausalDiscovery):
    """Offline pairwise estimator that orients edges alphabetically."""

    def _fit(self, X: pd.DataFrame):
        u, v = list(X.columns)
        source, target = (u, v) if str(u) < str(v) else (v, u)
        self.causal_graph_ = DAG([(source, target)])
        return self


class RaisingPairwise(BaseCausalDiscovery):
    """Pairwise estimator that fails if ever fit (used for the cache test)."""

    def _fit(self, X: pd.DataFrame):
        raise AssertionError("pairwise_estimator should not be queried for cached edges")


@pytest.fixture
def chain_data():
    """Categorical data with associations A -> B -> C."""
    rng = np.random.default_rng(42)
    n = 500
    a = rng.integers(0, 2, n)
    b = a ^ (rng.random(n) < 0.1).astype(int)
    c = b ^ (rng.random(n) < 0.1).astype(int)
    return pd.DataFrame({"A": a, "B": b, "C": c}, dtype="category")


def test_pairwise_estimator(chain_data):
    """ExpertInLoop orients edges through a pairwise estimator."""
    est = ExpertInLoop(pairwise_estimator=FakePairwise(), effect_size_threshold=0.01, show_progress=False)
    est.fit(chain_data)

    assert nx.is_directed_acyclic_graph(est.causal_graph_)
    assert est.causal_graph_.number_of_edges() > 0
    for u, v in est.causal_graph_.edges():
        assert str(u) < str(v)


def test_llmpairwise_estimator(monkeypatch, chain_data):
    """LLMPairwise integrates through the same pairwise estimator path."""
    monkeypatch.setattr(LLMPairwise, "_query_llm", lambda self, messages: "1")

    est = ExpertInLoop(pairwise_estimator=LLMPairwise(), effect_size_threshold=0.01, show_progress=False)
    est.fit(chain_data)

    assert nx.is_directed_acyclic_graph(est.causal_graph_)
    assert est.causal_graph_.number_of_edges() > 0


def test_cached_edges_not_requeried(chain_data):
    """Cached orientations are reused without querying the estimator again."""
    cache = {("A", "B"), ("A", "C"), ("B", "C")}
    est = ExpertInLoop(pairwise_estimator=RaisingPairwise(), effect_size_threshold=0.01, show_progress=False)
    est.orientation_cache_ = set(cache)

    est.fit(chain_data)

    for edge in est.causal_graph_.edges():
        assert edge in cache
