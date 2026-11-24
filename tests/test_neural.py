import importlib.util
import json
import math
from pathlib import Path
import pickle
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from author_match.neural import StarVerifier, _TrainingPosts, fine_tune_star


def training_posts():
    posts = []
    for author in range(3):
        for thread, text in enumerate(("never heard of this lol", "wait, who told you?", "no wayyyy")):
            posts.append(SimpleNamespace(
                id=f"p{author}-{thread}", text=f"{text} {author}", author=str(author),
                thread=str(thread), community="example",
            ))
    return posts


class NeuralConfigTests(unittest.TestCase):
    def test_sampler_respects_authors_and_threads(self):
        pool = _TrainingPosts(training_posts())
        sample = pool.sample(3, random.Random(31))
        self.assertEqual(sample, pool.sample(3, random.Random(31)))
        self.assertEqual(len({p.author for p in sample}), 3)
        for i in range(0, len(sample), 2):
            self.assertEqual(sample[i].author, sample[i + 1].author)
            self.assertNotEqual(sample[i].thread, sample[i + 1].thread)
        with self.assertRaises(ValueError):
            pool.sample(4, random.Random(31))

    def test_training_validates_data_without_importing_torch(self):
        with self.assertRaisesRegex(ValueError, "eligible authors"):
            fine_tune_star([], "/unused/nonexistent/checkpoint")

    def test_empty_split_needs_no_checkpoint(self):
        self.assertEqual(StarVerifier().scores([], []).size, 0)


@unittest.skipUnless(
    all(importlib.util.find_spec(dep) for dep in ("torch", "transformers", "tokenizers")),
    "install author-match[neural] to run the offline tiny RoBERTa integration test",
)
class OfflineNeuralTests(unittest.TestCase):
    def test_real_pooling_overflow_training_and_reload(self):
        import torch
        from tokenizers import ByteLevelBPETokenizer
        from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast

        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        with TemporaryDirectory() as root:
            base = Path(root) / "base"
            output = Path(root) / "trained"
            base.mkdir()
            bpe = ByteLevelBPETokenizer()
            bpe.train_from_iterator(
                ["hello, this is a post", "what??? lol", "never heard of this", "aha really"],
                vocab_size=280, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
            )
            bpe.save_model(str(base))
            tokenizer = RobertaTokenizerFast(
                vocab_file=str(base / "vocab.json"), merges_file=str(base / "merges.txt"), model_max_length=16,
            )
            tokenizer.save_pretrained(base)
            torch.manual_seed(77)
            model = RobertaModel(RobertaConfig(
                vocab_size=len(tokenizer), hidden_size=8, num_attention_heads=2,
                num_hidden_layers=1, intermediate_size=16, max_position_embeddings=18,
            )).eval()
            model.save_pretrained(base)
            encoder = StarVerifier(base, batch_size=2, local_files_only=True)

            with torch.inference_mode():
                expected = model(**tokenizer("hello", return_tensors="pt")).pooler_output
                expected = torch.nn.functional.normalize(expected, dim=1).numpy()
            np.testing.assert_allclose(encoder._encode(["hello"]), expected, atol=1e-6)

            long_text = "wait, what?? " * 5
            ids = tokenizer(long_text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
            accumulated = np.zeros(model.config.hidden_size, dtype=float)
            with torch.inference_mode():
                for start in range(0, len(ids), 14):
                    piece = ids[start:start + 14]
                    input_ids = torch.tensor([[tokenizer.bos_token_id, *piece, tokenizer.eos_token_id]])
                    vector = model(input_ids=input_ids).pooler_output
                    accumulated += len(piece) * torch.nn.functional.normalize(vector, dim=1).numpy()[0]
            accumulated /= np.linalg.norm(accumulated)
            np.testing.assert_allclose(encoder._encode([long_text])[0], accumulated, atol=1e-6)

            texts = [SimpleNamespace(id="a", text=long_text), SimpleNamespace(id="b", text="really?")]
            pairs = [SimpleNamespace(left="a", right="b"), SimpleNamespace(left="b", right="a")]
            forward, reverse = encoder.scores(texts, pairs)
            self.assertEqual(forward, reverse)
            self.assertAlmostEqual(forward, encoder.score_pair(long_text, "really?"), places=6)
            restored = pickle.loads(pickle.dumps(encoder))
            self.assertIsNone(restored._runtime)
            self.assertAlmostEqual(restored.score_pair(long_text, "really?"), forward, places=6)

            losses = fine_tune_star(
                training_posts(), output, checkpoint=base, steps=2, authors_per_batch=3,
                learning_rate=0.003, seed=5, local_files_only=True,
            )
            self.assertEqual(len(losses), 2)
            self.assertTrue(all(math.isfinite(loss) for loss in losses))
            self.assertEqual(json.loads((output / "author_match_training.json").read_text())["losses"], losses)
            updated = StarVerifier(output, local_files_only=True)
            self.assertTrue(math.isfinite(updated.score_pair("sample post", "another post")))
            self.assertFalse(torch.equal(
                model.pooler.dense.weight, updated._runtime[2].pooler.dense.weight,
            ))
            with self.assertRaisesRegex(ValueError, "new checkpoint"):
                fine_tune_star(training_posts(), output)
            changed = pickle.loads(pickle.dumps(encoder))
            (base / "config.json").write_text((base / "config.json").read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "changed after calibration"):
                changed.score_pair("sample", "text")
