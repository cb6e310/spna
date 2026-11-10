import unittest

import torch

from glim_spna_losses import nested_carg_objective


class GLIMSPNAIncrementTests(unittest.TestCase):
    def test_nested_carg_routes_gradients_to_students_only(self):
        eeg = torch.randn(5, 32, requires_grad=True)
        text = torch.randn(5, 32, requires_grad=True)
        teacher = torch.randn(5, 64, requires_grad=True)
        loss, parts = nested_carg_objective(
            eeg, text, teacher, small_dim=8, large_dim=24,
            eta=0.2, relation_temperature=0.1, logit_temperature=0.07,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(eeg.grad)
        self.assertIsNotNone(text.grad)
        self.assertIsNone(teacher.grad)
        self.assertEqual(set(parts), {"small", "large"})


if __name__ == "__main__":
    unittest.main()
