import re
import unittest
from pathlib import Path

from graphrag_lab import BENCHMARKS, build_flat_index, build_graph, graph_rank, heuristic_triples, read_corpus


ROOT = Path(__file__).parent


class GraphRAGLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = read_corpus(ROOT / "dataset")
        cls.by_id = {doc.doc_id: doc for doc in cls.docs}

    def test_corpus_and_benchmarks_are_complete(self):
        self.assertEqual(len(self.docs), 70)
        self.assertEqual(len(BENCHMARKS), 20)
        self.assertEqual(len({item.benchmark_id for item in BENCHMARKS}), 20)
        for item in BENCHMARKS:
            self.assertIn(item.expected_doc, self.by_id)
            self.assertTrue(item.required_terms)

    def test_heuristic_triples_have_verbatim_provenance(self):
        doc = self.by_id["doc_6"]
        triples = heuristic_triples(doc)
        normalized_doc = re.sub(r"\s+", " ", doc.text).lower()
        self.assertTrue(triples)
        for triple in triples:
            if triple.predicate != "HAS_SOURCE":
                self.assertIn(re.sub(r"\s+", " ", triple.sentence).lower(), normalized_doc)

    def test_graph_query_returns_ranked_sources_and_evidence(self):
        docs = [self.by_id["doc_6"], self.by_id["doc_20"], self.by_id["doc_30"]]
        graph, triples, usage = build_graph(docs, "heuristic", "unused")
        ranked, matched, evidence = graph_rank("What did the Inflation Reduction Act accelerate?", docs, graph, build_flat_index(docs))
        self.assertEqual(usage.calls, 0)
        self.assertEqual(len(ranked), len(docs))
        self.assertTrue(matched)
        self.assertTrue(evidence)
        self.assertTrue(all(t.doc_id in {doc.doc_id for doc in docs} for t in triples))


if __name__ == "__main__":
    unittest.main(verbosity=2)
