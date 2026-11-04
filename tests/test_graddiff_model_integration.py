import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


@unittest.skipIf(torch is None, "install requirements-ml.txt for GradDiff integration tests")
class GradDiffModelIntegrationTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    class FakeTokenizer:
        chat_template = "fixture chat template"
        is_fast = True
        eos_token_id = 2
        pad_token_id = 2
        padding_side = "right"

        def __init__(self, *, break_prefix=False, pad_token_id=2, padding_side="right"):
            self.break_prefix = break_prefix
            self.pad_token_id = pad_token_id
            self.padding_side = padding_side
            self.calls = []

        def __len__(self):
            return 32

        @staticmethod
        def _text_ids(text):
            return [8 + sum(map(ord, word)) % 20 for word in text.strip().split()]

        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt, date_string
        ):
            self.calls.append((messages, tokenize, add_generation_prompt, date_string))
            ids = [1]
            role_ids = {"system": 3, "user": 4, "assistant": 5}
            for message in messages:
                ids.append(role_ids[message["role"]])
                ids.extend(self._text_ids(message["content"]))
                ids.append(self.eos_token_id)
            if add_generation_prompt:
                ids.append(role_ids["assistant"])
            elif self.break_prefix and messages[-1]["role"] == "assistant":
                ids[0] = 31
            return ids

    class TinyCausalLM(nn.Module):
        def __init__(self, *, corrupt_loss=False, nonfinite_logits=False):
            super().__init__()
            self.embedding = nn.Embedding(32, 8)
            self.lm_head = nn.Linear(8, 32)
            self.config = SimpleNamespace(
                model_type="llama",
                vocab_size=32,
                max_position_embeddings=64,
                eos_token_id=[2],
                tie_word_embeddings=False,
            )
            self.corrupt_loss = corrupt_loss
            self.nonfinite_logits = nonfinite_logits
            self.forward_count = 0
            self.inference_mode_seen = False

        def get_input_embeddings(self):
            return self.embedding

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids, attention_mask, labels, use_cache=False):
            from src.graddiff import causal_lm_nll

            self.forward_count += 1
            self.inference_mode_seen = self.inference_mode_seen or torch.is_inference_mode_enabled()
            logits = self.lm_head(self.embedding(input_ids))
            if self.nonfinite_logits:
                logits = logits * float("nan")
            loss = causal_lm_nll(logits, labels)
            if self.corrupt_loss:
                loss = loss + 1.0
            return SimpleNamespace(loss=loss, logits=logits)

    @classmethod
    def setUpClass(cls):
        from src.graddiff_integration import (
            choose_readonly_forget_retain_pair,
            collate_causal_examples,
            encode_tofu_qa,
            evaluate_readonly_pair,
            prepare_tofu_batch,
            resolve_smoke_device,
            validate_model_tokenizer,
        )
        from src.graddiff_model_smoke import memory_snapshot

        cls.choose_pair = staticmethod(choose_readonly_forget_retain_pair)
        cls.collate = staticmethod(collate_causal_examples)
        cls.encode = staticmethod(encode_tofu_qa)
        cls.evaluate = staticmethod(evaluate_readonly_pair)
        cls.prepare = staticmethod(prepare_tofu_batch)
        cls.resolve_device = staticmethod(resolve_smoke_device)
        cls.validate_compatibility = staticmethod(validate_model_tokenizer)
        cls.memory_snapshot = staticmethod(memory_snapshot)

    @staticmethod
    def record(question="Who wrote this?", answer="A short answer."):
        return {"question": question, "answer": answer}

    def test_chat_encoding_masks_prompt_and_supervises_answer_and_eot(self):
        tokenizer = self.FakeTokenizer()
        example = self.encode(tokenizer, "Who wrote this?", "A short answer.")
        self.assertGreater(example.assistant_target_start, 1)
        self.assertTrue(all(label == -100 for label in example.labels[:example.assistant_target_start]))
        self.assertEqual(
            list(example.labels[example.assistant_target_start:]),
            list(example.input_ids[example.assistant_target_start:]),
        )
        self.assertEqual(example.input_ids[-1], tokenizer.eos_token_id)
        self.assertEqual(len(tokenizer.calls), 2)
        self.assertTrue(tokenizer.calls[0][2])
        self.assertFalse(tokenizer.calls[1][2])
        self.assertEqual(tokenizer.calls[0][3], "10 Apr 2025")
        self.assertEqual([message["role"] for message in tokenizer.calls[1][0]],
                         ["system", "user", "assistant"])

    def test_realistic_singleton_batch_and_variable_length_right_padding(self):
        tokenizer = self.FakeTokenizer()
        singleton = self.prepare([self.record()], tokenizer)
        self.assertEqual(singleton["input_ids"].shape[0], 1)
        self.assertTrue(torch.all(singleton["attention_mask"] == 1))

        batch = self.prepare(
            [
                self.record("Short?", "Yes."),
                self.record("Please provide a considerably longer question here?", "A longer response."),
            ],
            tokenizer,
        )
        pad_count = int(batch["attention_mask"].eq(0).sum().item())
        self.assertGreater(pad_count, 0)
        self.assertTrue(torch.all(batch["input_ids"][batch["attention_mask"] == 0] == tokenizer.pad_token_id))
        self.assertTrue(torch.all(batch["labels"][batch["attention_mask"] == 0] == -100))
        self.assertTrue(torch.all(batch["labels"][:, :1] == -100))
        self.assertEqual(set(batch), {"input_ids", "attention_mask", "labels"})

    def test_readonly_pair_matches_model_and_local_mean_causal_losses(self):
        tokenizer = self.FakeTokenizer()
        forget = self.prepare([self.record("Forget question?", "Forget target.")], tokenizer)
        retain = self.prepare([self.record("Retain question?", "Retain target.")], tokenizer)
        model = self.TinyCausalLM()
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        report = self.evaluate(model, forget, retain)

        self.assertEqual(model.forward_count, 2)
        self.assertTrue(model.inference_mode_seen)
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.assertTrue(report["weight_values_unchanged"])
        self.assertFalse(report["backpropagation_performed"])
        self.assertFalse(report["optimizer_created"])
        self.assertFalse(report["checkpoint_write_performed"])
        for split in ("forget", "retain"):
            self.assertGreater(report[split]["target_tokens_after_shift"], 0)
            self.assertLess(report[split]["absolute_loss_difference"], 1e-6)

    def test_readonly_pair_rejects_model_loss_mismatch(self):
        tokenizer = self.FakeTokenizer()
        batch = self.prepare([self.record()], tokenizer)
        model = self.TinyCausalLM(corrupt_loss=True)
        with self.assertRaisesRegex(RuntimeError, "model loss differs"):
            self.evaluate(model, batch, batch)

    def test_readonly_pair_rejects_nonfinite_logits(self):
        tokenizer = self.FakeTokenizer()
        batch = self.prepare([self.record()], tokenizer)
        with self.assertRaises(FloatingPointError):
            self.evaluate(self.TinyCausalLM(nonfinite_logits=True), batch, batch)

    def test_readonly_pair_rejects_wrong_batch_shape_and_preexisting_gradients(self):
        tokenizer = self.FakeTokenizer()
        batch = self.prepare([self.record()], tokenizer)
        oversized = self.prepare([self.record(), self.record("Another?", "Answer.")], tokenizer)
        with self.assertRaisesRegex(ValueError, "batch size one"):
            self.evaluate(self.TinyCausalLM(), oversized, batch)
        model = self.TinyCausalLM()
        next(model.parameters()).grad = torch.zeros_like(next(model.parameters()))
        with self.assertRaisesRegex(RuntimeError, "existing gradients"):
            self.evaluate(model, batch, batch)

    def test_readonly_pair_rejects_padding_labels_and_left_padding(self):
        tokenizer = self.FakeTokenizer()
        batch = self.prepare([self.record()], tokenizer)
        padded_label_error = {key: value.clone() for key, value in batch.items()}
        padded_label_error["attention_mask"][0, -1] = 0
        with self.assertRaisesRegex(ValueError, "padded tokens must be ignored"):
            self.evaluate(self.TinyCausalLM(), padded_label_error, batch)

        left_padding_error = {key: value.clone() for key, value in batch.items()}
        left_padding_error["attention_mask"][0, 0] = 0
        left_padding_error["labels"][0, 0] = -100
        with self.assertRaisesRegex(ValueError, "right padding"):
            self.evaluate(self.TinyCausalLM(), left_padding_error, batch)

    def test_chat_encoding_rejects_invalid_text_prefix_mismatch_and_truncation(self):
        with self.assertRaisesRegex(ValueError, "question"):
            self.encode(self.FakeTokenizer(), "  ", "answer")
        with self.assertRaisesRegex(ValueError, "answer"):
            self.encode(self.FakeTokenizer(), "question", " ")
        with self.assertRaisesRegex(ValueError, "token prefix"):
            self.encode(self.FakeTokenizer(break_prefix=True), "question", "answer")
        with self.assertRaisesRegex(ValueError, "exceeding max_sequence_length"):
            self.encode(self.FakeTokenizer(), "A much longer question", "A long answer", max_sequence_length=4)

    def test_collation_rejects_empty_batches_missing_padding_and_bad_examples(self):
        tokenizer = self.FakeTokenizer()
        example = self.encode(tokenizer, "Question?", "Answer.")
        with self.assertRaisesRegex(ValueError, "empty batch"):
            self.collate([], tokenizer)
        with self.assertRaisesRegex(ValueError, "pad_token_id"):
            self.collate([example], self.FakeTokenizer(pad_token_id=None))
        with self.assertRaisesRegex(ValueError, "padding_side='right'"):
            self.collate([example], self.FakeTokenizer(padding_side="left"))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.collate([SimpleNamespace(input_ids=(1, 2), labels=(-100,), assistant_target_start=1)], tokenizer)

    def test_batch_preparation_rejects_empty_or_incomplete_records(self):
        tokenizer = self.FakeTokenizer()
        with self.assertRaisesRegex(ValueError, "empty TOFU batch"):
            self.prepare([], tokenizer)
        with self.assertRaisesRegex(ValueError, "missing: answer"):
            self.prepare([{"question": "Question?"}], tokenizer)

    def test_readonly_pair_selection_is_seeded_and_checks_empty_splits(self):
        forget = [self.record("F0", "A0"), self.record("F1", "A1")]
        retain = [self.record(f"R{i}", f"A{i}") for i in range(10)]
        first = self.choose_pair(forget, retain, seed=2026)
        second = self.choose_pair(forget, retain, seed=2026)
        self.assertEqual(first, second)
        self.assertEqual(first[0], forget[0])
        self.assertGreaterEqual(first[2]["retain_record_index"], 0)
        self.assertLess(first[2]["retain_record_index"], len(retain))
        with self.assertRaisesRegex(ValueError, "forget split is empty"):
            self.choose_pair([], retain)
        with self.assertRaisesRegex(ValueError, "retain split is empty"):
            self.choose_pair(forget, [])

    def test_model_tokenizer_compatibility_checks_vocab_context_and_tied_weights(self):
        tokenizer = self.FakeTokenizer()
        model = self.TinyCausalLM()
        report = self.validate_compatibility(model.config, tokenizer, max_sequence_length=32, model=model)
        self.assertEqual(report["vocab_size"], len(tokenizer))
        self.assertTrue(report["chat_template_present"])

        wrong_vocab = SimpleNamespace(**{**vars(model.config), "vocab_size": 31})
        with self.assertRaisesRegex(ValueError, "tokenizer length"):
            self.validate_compatibility(wrong_vocab, tokenizer, max_sequence_length=32)
        with self.assertRaisesRegex(ValueError, "exceeds model context"):
            self.validate_compatibility(model.config, tokenizer, max_sequence_length=65)
        wrong_model = SimpleNamespace(**{**vars(model.config), "model_type": "gpt2"})
        with self.assertRaisesRegex(ValueError, "expected a Llama"):
            self.validate_compatibility(wrong_model, tokenizer, max_sequence_length=32)

    def test_model_tokenizer_rejects_invalid_special_ids_and_embedding_size(self):
        model = self.TinyCausalLM()
        tokenizer = self.FakeTokenizer()
        no_pad = self.FakeTokenizer(pad_token_id=None)
        with self.assertRaisesRegex(ValueError, "pad_token_id"):
            self.validate_compatibility(model.config, no_pad, max_sequence_length=32)
        wrong_eos = self.FakeTokenizer()
        wrong_eos.eos_token_id = 7
        with self.assertRaisesRegex(ValueError, "eos_token_id"):
            self.validate_compatibility(model.config, wrong_eos, max_sequence_length=32)
        no_chat = self.FakeTokenizer()
        no_chat.chat_template = None
        with self.assertRaisesRegex(ValueError, "chat template"):
            self.validate_compatibility(model.config, no_chat, max_sequence_length=32)

        class WrongEmbeddingModel(self.TinyCausalLM):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(31, 8)

        with self.assertRaisesRegex(ValueError, "input embeddings"):
            wrong = WrongEmbeddingModel()
            self.validate_compatibility(wrong.config, tokenizer, max_sequence_length=32, model=wrong)

    def test_model_tokenizer_checks_configured_tied_embeddings(self):
        class TiedModel(self.TinyCausalLM):
            def __init__(self, *, untied=False):
                super().__init__()
                self.config.tie_word_embeddings = True
                if not untied:
                    self.lm_head.weight = self.embedding.weight

        tokenizer = self.FakeTokenizer()
        self.validate_compatibility(
            TiedModel().config, tokenizer, max_sequence_length=32, model=TiedModel()
        )
        with self.assertRaisesRegex(ValueError, "not tied"):
            untied = TiedModel(untied=True)
            self.validate_compatibility(
                untied.config, tokenizer, max_sequence_length=32, model=untied
            )

    def test_device_selection_prefers_mps_and_requires_explicit_cpu_fallback(self):
        class FakeMPS:
            def __init__(self, built, available):
                self._built = built
                self._available = available

            def is_built(self):
                return self._built

            def is_available(self):
                return self._available

        class FakeTorch:
            def __init__(self, mps):
                self.backends = SimpleNamespace(mps=mps)

            @staticmethod
            def device(name):
                return name

        self.assertEqual(self.resolve_device("auto", torch_module=FakeTorch(FakeMPS(True, True))), "mps")
        unavailable = FakeTorch(FakeMPS(True, False))
        with self.assertRaisesRegex(RuntimeError, "explicitly select"):
            self.resolve_device("auto", torch_module=unavailable)
        self.assertEqual(
            self.resolve_device("auto", allow_cpu_fallback=True, torch_module=unavailable), "cpu"
        )
        with self.assertRaisesRegex(RuntimeError, "refusing implicit CPU fallback"):
            self.resolve_device("mps", torch_module=unavailable)
        self.assertEqual(self.resolve_device("cpu", torch_module=unavailable), "cpu")

    def test_memory_snapshot_reports_host_and_process_memory_without_model_loading(self):
        from src.graddiff_model_smoke import memory_snapshot

        report = self.memory_snapshot(torch, torch.device("cpu"))
        self.assertGreater(report["total_physical_memory_bytes"], 0)
        self.assertGreater(report["process_peak_rss_bytes"], 0)
        self.assertIsNone(report["mps_tensor_allocated_bytes"])
        self.assertIsNone(report["mps_driver_allocated_bytes"])

    def test_readonly_smoke_cli_help_does_not_load_the_model(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.graddiff_model_smoke", "--help"],
            cwd=self.ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--device", result.stdout)
        self.assertIn("--allow-cpu-fallback", result.stdout)
        self.assertIn("--max-sequence-length", result.stdout)


if __name__ == "__main__":
    unittest.main()
