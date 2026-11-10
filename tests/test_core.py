import unittest

import torch

from spna.losses import mixed_target, spna_objective, text_relation_target
from spna.metrics import retrieval_metrics
from spna.model import ModelConfig, SPNA


class SPNACoreTests(unittest.TestCase):
    def test_relations_and_mixed_target_are_valid_distributions(self):
        teacher = torch.randn(5, 12, requires_grad=True)
        relations = text_relation_target(teacher, 0.2)
        target = mixed_target(relations, 0.3)
        self.assertTrue(torch.allclose(relations.sum(-1), torch.ones(5), atol=1e-6))
        self.assertTrue(torch.allclose(relations.diag(), torch.zeros(5)))
        self.assertTrue(torch.allclose(target.sum(-1), torch.ones(5), atol=1e-6))
        self.assertTrue(torch.allclose(target.diag(), torch.full((5,), 0.7)))
        self.assertFalse(target.requires_grad)

    def test_uniform_and_shuffled_relations_exclude_diagonal(self):
        teacher = torch.randn(6, 8)
        uniform = text_relation_target(teacher, 0.2, mode="uniform")
        shuffled = text_relation_target(teacher, 0.2, mode="shuffled")
        self.assertTrue(torch.allclose(uniform.diag(), torch.zeros(6)))
        self.assertTrue(torch.allclose(shuffled.diag(), torch.zeros(6)))
        self.assertTrue(torch.allclose(shuffled.sum(-1), torch.ones(6), atol=1e-6))

    def test_prefix_is_normalized_after_truncation(self):
        vectors = torch.tensor([[3.0, 4.0, 100.0]])
        result = SPNA.prefix(vectors, 2)
        self.assertTrue(torch.allclose(result, torch.tensor([[0.6, 0.8]])))

    def test_objective_routes_gradients_through_student_logits(self):
        small = torch.randn(4, 4, requires_grad=True)
        large = torch.randn(4, 4, requires_grad=True)
        teacher = torch.randn(4, 7, requires_grad=True)
        relations = text_relation_target(teacher, 0.2)
        loss, _ = spna_objective(small, large, relations, eta=0.2)
        loss.backward()
        self.assertIsNotNone(small.grad)
        self.assertIsNotNone(large.grad)
        self.assertIsNone(teacher.grad)

    def test_full_bank_retrieval_metrics(self):
        vectors = torch.eye(4)
        result = retrieval_metrics(vectors, vectors, torch.arange(4), 4)
        self.assertEqual(result["r@1"], 1.0)
        self.assertEqual(result["mrr"], 1.0)
        self.assertEqual(result["n_candidates"], 4.0)


if __name__ == "__main__":
    unittest.main()

