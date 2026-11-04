import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # the repository's default test install is intentionally ML-free
    torch = None
    F = None


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(torch is None, "install requirements-ml.txt for GradDiff prototype tests")
class GradDiffPrototypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.graddiff import causal_lm_nll, graddiff_objective
        from src.graddiff_train import (
            TinySyntheticCausalLM,
            assert_single_process,
            graddiff_training_step,
            run_synthetic_dry_run,
            select_device,
            synthetic_batches,
        )

        cls.causal_lm_nll = staticmethod(causal_lm_nll)
        cls.graddiff_objective = staticmethod(graddiff_objective)
        cls.TinySyntheticCausalLM = TinySyntheticCausalLM
        cls.assert_single_process = staticmethod(assert_single_process)
        cls.graddiff_training_step = staticmethod(graddiff_training_step)
        cls.run_synthetic_dry_run = staticmethod(run_synthetic_dry_run)
        cls.select_device = staticmethod(select_device)
        cls.synthetic_batches = staticmethod(synthetic_batches)

    def test_causal_nll_is_shifted_mean_over_valid_target_tokens(self):
        logits = torch.tensor(
            [[[0.2, 0.3, 1.1], [1.0, -0.2, 0.1], [0.1, 0.2, 0.3]],
             [[-0.1, 0.5, 0.2], [0.3, 1.2, -0.4], [0.6, -0.2, 0.1]]],
            dtype=torch.float32,
        )
        labels = torch.tensor([[-100, 2, 1], [-100, -100, 0]])
        actual = self.causal_lm_nll(logits, labels)
        expected_terms = torch.stack((
            F.cross_entropy(logits[0, 0].unsqueeze(0), labels[0, 1].view(1)),
            F.cross_entropy(logits[0, 1].unsqueeze(0), labels[0, 2].view(1)),
            F.cross_entropy(logits[1, 1].unsqueeze(0), labels[1, 2].view(1)),
        ))
        self.assertTrue(torch.allclose(actual, expected_terms.mean(), atol=1e-7, rtol=1e-7))
        self.assertFalse(torch.allclose(actual, expected_terms.sum()))

    def test_bfloat16_logits_are_upcast_before_causal_cross_entropy(self):
        logits = torch.tensor(
            [[[0.2, 0.3, 1.1], [1.0, -0.2, 0.1], [0.1, 0.2, 0.3]]],
            dtype=torch.bfloat16,
        )
        labels = torch.tensor([[-100, 2, 1]])
        actual = self.causal_lm_nll(logits, labels)
        expected = F.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, 3), labels[:, 1:].reshape(-1),
            ignore_index=-100, reduction="mean",
        )
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))

    def test_objective_uses_separate_mean_losses_and_official_coefficients(self):
        forget = torch.tensor(2.0, requires_grad=True)
        retain = torch.tensor(3.0, requires_grad=True)
        loss = self.graddiff_objective(forget, retain, gamma=0.5, alpha=2.0)
        self.assertEqual(loss.item(), 5.0)
        forget_grad, retain_grad = torch.autograd.grad(loss, (forget, retain))
        self.assertEqual(forget_grad.item(), -0.5)
        self.assertEqual(retain_grad.item(), 2.0)

    def test_matches_pinned_openunlearning_compute_loss_formula_and_gradients(self):
        # Source transcriptions of pinned Transformers 4.51.3 ForCausalLMLoss
        # and OpenUnlearning GradDiff.compute_loss at
        # a456aa277be14b836248e42b13b3befe47bb03f4.
        def pinned_transformers_causal_lm_loss(logits, labels):
            logits = logits.float()
            labels = F.pad(labels, (0, 1), value=-100)
            shift_labels = labels[..., 1:].contiguous()
            flat_logits = logits.view(-1, logits.shape[-1])
            flat_labels = shift_labels.view(-1).to(flat_logits.device)
            return F.cross_entropy(
                flat_logits, flat_labels, ignore_index=-100, reduction="mean"
            )

        def pinned_openunlearning_compute_loss(forget_outputs, retain_outputs, gamma, alpha):
            forget_loss = -forget_outputs.loss
            retain_loss = retain_outputs.loss
            return gamma * forget_loss + alpha * retain_loss

        forget_logits = torch.tensor(
            [[[0.4, -0.3, 0.7], [0.1, 0.5, -0.4], [0.3, -0.2, 0.8]],
             [[-0.1, 0.6, 0.2], [0.9, -0.5, 0.1], [0.2, 0.4, -0.3]]],
            requires_grad=True,
        )
        retain_logits = torch.tensor(
            [[[0.2, 0.8, -0.4], [-0.1, 0.5, 0.3], [0.6, -0.2, 0.1]],
             [[0.3, -0.6, 0.4], [0.7, 0.2, -0.5], [-0.4, 0.1, 0.8]]],
            requires_grad=True,
        )
        forget_labels = torch.tensor([[-100, 2, 1], [-100, 0, -100]])
        retain_labels = torch.tensor([[-100, 1, 0], [-100, 2, -100]])
        upstream_forget_nll = pinned_transformers_causal_lm_loss(forget_logits, forget_labels)
        upstream_retain_nll = pinned_transformers_causal_lm_loss(retain_logits, retain_labels)
        local_forget_nll = self.causal_lm_nll(forget_logits, forget_labels)
        local_retain_nll = self.causal_lm_nll(retain_logits, retain_labels)
        upstream_forget_outputs = SimpleNamespace(loss=upstream_forget_nll)
        upstream_retain_outputs = SimpleNamespace(loss=upstream_retain_nll)
        local_forget_outputs = SimpleNamespace(loss=local_forget_nll)
        local_retain_outputs = SimpleNamespace(loss=local_retain_nll)

        upstream_loss = pinned_openunlearning_compute_loss(
            upstream_forget_outputs, upstream_retain_outputs, gamma=0.7, alpha=1.3
        )
        local_loss = self.graddiff_objective(
            local_forget_outputs.loss, local_retain_outputs.loss, gamma=0.7, alpha=1.3
        )
        upstream_grads = torch.autograd.grad(
            upstream_loss, (forget_logits, retain_logits), retain_graph=True
        )
        local_grads = torch.autograd.grad(local_loss, (forget_logits, retain_logits))
        self.assertTrue(torch.allclose(upstream_loss, local_loss, atol=1e-7, rtol=1e-7))
        for upstream_grad, local_grad in zip(upstream_grads, local_grads):
            self.assertTrue(torch.allclose(upstream_grad, local_grad, atol=1e-7, rtol=1e-7))

    def test_extreme_finite_logits_keep_cross_entropy_and_gradients_finite(self):
        logits = torch.tensor(
            [[[10000.0, -10000.0], [10000.0, -10000.0],
              [-10000.0, 10000.0]]], requires_grad=True,
        )
        labels = torch.tensor([[-100, 1, 0]])
        nll = self.causal_lm_nll(logits, labels)
        nll.backward()
        self.assertTrue(torch.isfinite(nll))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_rejects_nonfinite_losses_logits_and_empty_shifted_labels(self):
        with self.assertRaises(FloatingPointError):
            self.causal_lm_nll(torch.tensor([[[float("nan"), 0.0], [0.0, 0.0]]]),
                                  torch.tensor([[-100, 1]]))
        with self.assertRaises(FloatingPointError):
            self.causal_lm_nll(torch.tensor([[[float("inf"), 0.0], [0.0, 0.0]]]),
                                  torch.tensor([[-100, 1]]))
        with self.assertRaises(ValueError):
            self.causal_lm_nll(torch.zeros(1, 2, 2), torch.tensor([[-100, -100]]))
        with self.assertRaises(FloatingPointError):
            self.graddiff_objective(torch.tensor(float("nan")), torch.tensor(1.0))
        with self.assertRaises(ValueError):
            self.graddiff_objective(torch.tensor(1.0), torch.tensor(1.0), -1.0, 1.0)

    def test_gradient_rows_are_ascent_for_forget_and_descent_for_retain(self):
        model = self.TinySyntheticCausalLM()
        forget_batch, retain_batch = self.synthetic_batches(torch.device("cpu"))
        forget_nll = self.causal_lm_nll(model(forget_batch["input_ids"]), forget_batch["labels"])
        retain_nll = self.causal_lm_nll(model(retain_batch["input_ids"]), retain_batch["labels"])
        forget_grad = torch.autograd.grad(forget_nll, model.task_logits, retain_graph=True)[0]
        retain_grad = torch.autograd.grad(retain_nll, model.task_logits, retain_graph=True)[0]
        combined_grad = torch.autograd.grad(
            self.graddiff_objective(forget_nll, retain_nll, gamma=1.5, alpha=0.25),
            model.task_logits,
        )[0]
        self.assertTrue(torch.allclose(combined_grad[0], -1.5 * forget_grad[0]))
        self.assertTrue(torch.allclose(combined_grad[1], 0.25 * retain_grad[1]))
        self.assertLess(torch.sum(combined_grad[0] * forget_grad[0]).item(), 0.0)
        self.assertGreater(torch.sum(combined_grad[1] * retain_grad[1]).item(), 0.0)
        self.assertTrue(torch.allclose(combined_grad[[0, 1]],
                                       torch.stack((-1.5 * forget_grad[0],
                                                    0.25 * retain_grad[1]))))

    def test_adamw_step_increases_forget_nll_and_preserves_retain_direction(self):
        model = self.TinySyntheticCausalLM()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
        forget, retain = self.synthetic_batches(torch.device("cpu"))
        record = self.graddiff_training_step(model, optimizer, forget, retain)
        self.assertGreater(record["forget_nll_after"], record["forget_nll_before"])
        self.assertLess(record["retain_nll_after"], record["retain_nll_before"])
        self.assertLess(record["objective_after"], record["objective_before"])
        self.assertGreater(record["gradient_norm"], 0.0)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        self.assertTrue(all(torch.isfinite(v).all() for state in optimizer.state.values()
                            for v in state.values() if isinstance(v, torch.Tensor)
                            and v.is_floating_point()))
        self.assertEqual(optimizer.state[model.task_logits]["step"].item(), 1.0)

    def test_device_and_process_guards_are_explicit(self):
        self.assertEqual(self.select_device("cpu").type, "cpu")
        chosen = self.select_device("auto")
        self.assertIn(chosen.type, {"cpu", "mps"})
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            self.assertEqual(self.select_device("mps").type, "mps")
        else:
            with self.assertRaises(RuntimeError):
                self.select_device("mps")
        self.assertEqual(self.assert_single_process({"WORLD_SIZE": "1"}), 1)
        with self.assertRaises(RuntimeError):
            self.assert_single_process({"WORLD_SIZE": "2"})
        with self.assertRaises(RuntimeError):
            self.assert_single_process({"WORLD_SIZE": "invalid"})

    def test_mps_dry_run_and_cpu_mps_update_agreement_when_available(self):
        if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
            self.skipTest("MPS is unavailable in this process")
        report = self.run_synthetic_dry_run(device="mps", steps=2)
        self.assertEqual(report["actual_device"], "mps")

        initial = {"task_logits": torch.tensor([[0.25, -0.5], [-0.2, 0.15]])}
        cpu_model = self.TinySyntheticCausalLM()
        mps_model = self.TinySyntheticCausalLM().to("mps")
        cpu_model.load_state_dict(initial)
        mps_model.load_state_dict(initial)
        cpu_forget, cpu_retain = self.synthetic_batches(torch.device("cpu"))
        mps_forget, mps_retain = self.synthetic_batches(torch.device("mps"))

        def losses_and_gradients(model, forget, retain):
            forget_nll = self.causal_lm_nll(model(forget["input_ids"]), forget["labels"])
            retain_nll = self.causal_lm_nll(model(retain["input_ids"]), retain["labels"])
            objective = self.graddiff_objective(forget_nll, retain_nll)
            objective.backward()
            return forget_nll.detach(), retain_nll.detach(), objective.detach(), model.task_logits.grad.detach()

        cpu_values = losses_and_gradients(cpu_model, cpu_forget, cpu_retain)
        mps_values = losses_and_gradients(mps_model, mps_forget, mps_retain)
        for cpu_value, mps_value in zip(cpu_values, mps_values):
            self.assertTrue(torch.allclose(cpu_value, mps_value.cpu(), atol=1e-5, rtol=1e-4))
        cpu_optimizer = torch.optim.AdamW(cpu_model.parameters(), lr=0.01, weight_decay=0.0)
        mps_optimizer = torch.optim.AdamW(mps_model.parameters(), lr=0.01, weight_decay=0.0)
        cpu_optimizer.step()
        mps_optimizer.step()
        self.assertTrue(torch.allclose(cpu_model.task_logits,
                                       mps_model.task_logits.cpu(), atol=1e-5, rtol=1e-4))

    def test_cli_runs_only_explicit_synthetic_dry_run(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.graddiff_train", "--dry-run", "--device", "cpu", "--steps", "2"],
            cwd=ROOT, check=True, capture_output=True, text=True,
            env={**os.environ, "WORLD_SIZE": "1"},
        )
        report = json.loads(result.stdout)
        self.assertEqual(report["mode"], "synthetic_dry_run")
        self.assertEqual(report["actual_device"], "cpu")
        self.assertFalse(report["checkpoint_loaded"])
        self.assertFalse(report["external_data_loaded"])
        self.assertTrue(report["assertions"]["forget_nll_increased_each_step"])

        missing_flag = subprocess.run(
            [sys.executable, "-m", "src.graddiff_train", "--device", "cpu"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertNotEqual(missing_flag.returncode, 0)
        self.assertIn("only --dry-run is supported", missing_flag.stderr)

        distributed = subprocess.run(
            [sys.executable, "-m", "src.graddiff_train", "--dry-run", "--device", "cpu"],
            cwd=ROOT, capture_output=True, text=True,
            env={**os.environ, "WORLD_SIZE": "2"},
        )
        self.assertNotEqual(distributed.returncode, 0)
        self.assertIn("supports only one process", distributed.stderr)


if __name__ == "__main__":
    unittest.main()
