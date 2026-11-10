import unittest

import torch

from spna.losses import hard_pair_loss, negative_relation_loss, spna_objective, text_relation_target
from spna.metrics import retrieval_metrics
from spna.model import ModelConfig, SPNA


class SPNACoreTests(unittest.TestCase):
    def test_relations_are_valid_off_diagonal_distributions(self):
        teacher = torch.randn(5, 12, requires_grad=True)
        relations = text_relation_target(teacher, 0.2)
        self.assertTrue(torch.allclose(relations.sum(-1), torch.ones(5), atol=1e-6))
        self.assertTrue(torch.allclose(relations.diag(), torch.zeros(5)))
        self.assertFalse(relations.requires_grad)

    def test_relation_kl_excludes_paired_candidates_and_backpropagates(self):
        logits = torch.randn(4, 4, requires_grad=True)
        relations = torch.full((4, 4), 1 / 3)
        relations.fill_diagonal_(0)
        relations[0] = torch.tensor([0.0, 0.7, 0.2, 0.1])
        loss = negative_relation_loss(logits, relations)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(logits.grad.diag().abs().sum()), 0.0)
        self.assertIsNone(relations.grad)

    def test_uniform_relation_kl_is_zero_for_uniform_negative_student(self):
        logits = torch.zeros(5, 5, requires_grad=True)
        relations = torch.full((5, 5), 0.25)
        relations.fill_diagonal_(0)
        loss = negative_relation_loss(logits, relations)
        self.assertTrue(torch.allclose(loss, torch.tensor(0.0), atol=1e-7))

    def test_uniform_and_shuffled_relations_exclude_diagonal(self):
        teacher = torch.randn(6, 8)
        uniform = text_relation_target(teacher, 0.2, mode="uniform")
        shuffled = text_relation_target(teacher, 0.2, mode="shuffled")
        self.assertTrue(torch.allclose(uniform.diag(), torch.zeros(6)))
        self.assertTrue(torch.allclose(uniform.sum(-1), torch.ones(6), atol=1e-6))
        self.assertTrue(torch.allclose(uniform[0, 1:], torch.full((5,), 0.2)))
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
        loss, components = spna_objective(small, large, relations, lambda_relation=0.1)
        expected = (0.5 * hard_pair_loss(small) + 0.5 * hard_pair_loss(large)
                    + 0.1 * negative_relation_loss(small, relations))
        loss.backward()
        self.assertTrue(torch.allclose(loss.detach(), expected.detach()))
        self.assertIsNotNone(small.grad)
        self.assertIsNotNone(large.grad)
        self.assertIsNone(teacher.grad)
        self.assertEqual(set(components), {"small", "large", "relation"})

    def test_full_bank_retrieval_metrics(self):
        vectors = torch.eye(4)
        result = retrieval_metrics(vectors, vectors, torch.arange(4), 4)
        self.assertEqual(result["r@1"], 1.0)
        self.assertEqual(result["mrr"], 1.0)
        self.assertEqual(result["n_candidates"], 4.0)


if __name__ == "__main__":
    unittest.main()

