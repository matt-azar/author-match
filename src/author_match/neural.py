"""Optional STAR encoder for individual posts, including a trainable contrastive path."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import tempfile

import numpy as np


STAR = "AIDA-UPM/star"
STAR_REVISION = "e2a08d0b3cf2dc78548e131045faec351ac46b33"


def _dependencies():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise ValueError("STAR requires the author-match[neural] dependencies.") from error
    return torch, AutoModel, AutoTokenizer


def _fingerprint(directory):
    """Bind a calibrated local model to the tokenizer and weights used for calibration."""
    digest = hashlib.sha256()
    files = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix in {".json", ".safetensors", ".bin", ".txt", ".model"}
    )
    for path in files:
        digest.update(path.name.encode() + b"\0")
        with path.open("rb") as stream:
            file_hash = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                file_hash.update(chunk)
            digest.update(file_hash.digest())
    return digest.hexdigest()


class StarVerifier:
    """Embed each individual post with STAR and return cosine similarity on [0, 1].

    The default checkpoint and bundled tokenizer use the same immutable revision.
    A local checkpoint must contain both weights and tokenizer. A serialized scorer
    contains only its configuration and the local checkpoint fingerprint.
    """

    method = "star"

    def __init__(self, checkpoint=None, *, device="cpu", batch_size=8, local_files_only=False):
        if batch_size < 1:
            raise ValueError("STAR batch_size must be positive.")
        self.checkpoint = None if checkpoint is None else str(Path(checkpoint).resolve())
        self.device = device
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self.checkpoint_sha256 = None
        self._runtime = None

    def __getstate__(self):
        return {**self.__dict__, "_runtime": None}

    def _load(self):
        if self._runtime is not None:
            return self._runtime

        torch, auto_model, auto_tokenizer = _dependencies()
        source = self.checkpoint or STAR
        options = {"local_files_only": self.local_files_only}
        if self.checkpoint is None:
            options["revision"] = STAR_REVISION
        else:
            directory = Path(source)
            if not directory.is_dir():
                raise ValueError(f"STAR checkpoint directory does not exist: {directory}")
            fingerprint = _fingerprint(directory)
            if self.checkpoint_sha256 is not None and self.checkpoint_sha256 != fingerprint:
                raise ValueError("Local STAR checkpoint changed after calibration.")
            self.checkpoint_sha256 = fingerprint
            options["local_files_only"] = True

        tokenizer = auto_tokenizer.from_pretrained(source, use_fast=True, **options)
        if not tokenizer.is_fast:
            raise ValueError("STAR requires a fast tokenizer for non-overlapping text windows.")
        model = auto_model.from_pretrained(source, **options)
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        self._runtime = torch, tokenizer, model
        return self._runtime

    def fit(self, posts, pairs):
        self._load()
        return self

    def _windows(self, texts):
        _, tokenizer, model = self._load()
        # RoBERTa positions start after the padding index. A 514-row table fits 512 tokens.
        model_limit = model.config.max_position_embeddings - model.config.pad_token_id - 1
        limit = min(512, tokenizer.model_max_length, model_limit)
        if limit <= tokenizer.num_special_tokens_to_add(pair=False):
            raise ValueError("STAR checkpoint has insufficient sequence length.")
        encoded = tokenizer(
            texts, max_length=limit, truncation=True, return_overflowing_tokens=True,
            return_attention_mask=False, padding=False,
        )
        return encoded["input_ids"], encoded["overflow_to_sample_mapping"]

    def _pool(self, windows):
        torch, tokenizer, model = self._load()
        batch = tokenizer.pad({"input_ids": windows}, padding=True, return_tensors="pt")
        outputs = model(**batch.to(model.device))
        if getattr(outputs, "pooler_output", None) is None:
            raise ValueError("STAR checkpoint must supply a trained pooler_output.")
        return torch.nn.functional.normalize(outputs.pooler_output.float(), dim=1)

    def _encode(self, texts):
        torch, tokenizer, model = self._load()
        embeddings = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
        specials = tokenizer.num_special_tokens_to_add(pair=False)
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                group = texts[start:start + self.batch_size]
                windows, owners = self._windows(group)
                vectors = np.zeros((len(group), model.config.hidden_size), dtype=np.float64)
                for i in range(0, len(windows), self.batch_size):
                    batch = windows[i:i + self.batch_size]
                    pooled = self._pool(batch).cpu().numpy()
                    for j, vector in enumerate(pooled):
                        token_count = max(1, len(batch[j]) - specials)
                        vectors[owners[i + j]] += token_count * vector
                norms = np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
                embeddings[start:start + len(group)] = vectors / norms
        return embeddings

    def scores(self, posts, pairs):
        if not pairs:
            return np.empty(0, dtype=float)
        text_by_id = {post.id: post.text for post in posts}
        ids = dict.fromkeys(key for pair in pairs for key in (pair.left, pair.right))
        index = {key: i for i, key in enumerate(ids)}
        vectors = self._encode([text_by_id[key] for key in ids])
        scores = np.empty(len(pairs), dtype=float)
        for start in range(0, len(pairs), 1024):
            block = pairs[start:start + 1024]
            left = [index[pair.left] for pair in block]
            right = [index[pair.right] for pair in block]
            cosines = np.einsum("ij,ij->i", vectors[left], vectors[right])
            scores[start:start + len(block)] = np.clip(0.5 + 0.5 * cosines, 0.0, 1.0)
        return scores

    def score_pair(self, clean_left, clean_right):
        vectors = self._encode([clean_left, clean_right])
        cosine = np.einsum("i,i->", vectors[0], vectors[1])
        return float(np.clip(0.5 + 0.5 * cosine, 0.0, 1.0))


class _TrainingPosts:
    """Each sampled author contributes posts from two different conversations."""

    def __init__(self, posts):
        by_author = {}
        for post in posts:
            if post.text.strip() and post.thread:
                threads = by_author.setdefault(post.author, {})
                threads.setdefault(post.thread, []).append(post)
        self.by_author = {key: value for key, value in by_author.items() if len(value) >= 2}
        self.authors = sorted(self.by_author)
        self.by_community = {}
        for author, threads in self.by_author.items():
            for thread_posts in threads.values():
                for post in thread_posts:
                    if post.community:
                        community = self.by_community.setdefault(post.community, {})
                        community.setdefault(author, []).append(post)
        self.community_authors = {key: sorted(values) for key, values in self.by_community.items()}

    def sample(self, size, rng):
        if size < 2 or len(self.authors) < size:
            raise ValueError(f"Need {max(2, size)} authors with posts in at least two distinct threads.")
        anchor = rng.choice(self.authors)
        first_threads = self.by_author[anchor]
        first = rng.choice(first_threads[rng.choice(list(first_threads))])
        peers = self.community_authors.get(first.community, [])
        candidates = rng.sample(peers, min(size, len(peers))) + rng.sample(self.authors, size)
        authors = list(dict.fromkeys([anchor, *candidates]))[:size]
        preferred = self.by_community.get(first.community, {})
        examples = []
        for author in authors:
            threads = self.by_author[author]
            if author == anchor:
                left = first
            elif author in preferred:
                left = rng.choice(preferred[author])
            else:
                left = rng.choice(threads[rng.choice(list(threads))])
            other_thread = rng.choice([thread for thread in threads if thread != left.thread])
            examples.extend([left, rng.choice(threads[other_thread])])
        return examples


def fine_tune_star(
    posts, output_dir, *, checkpoint=None, device="cpu", steps=100, authors_per_batch=4,
    learning_rate=1e-5, temperature=0.07, seed=42, local_files_only=False,
):
    """Train on the provided training split and publish a self-contained local checkpoint.

    Each row's sole positive is its adjacent paired row. Other examples from
    the batch become negatives, preferentially chosen from the same community.
    Uniformly sample one window per post for bounded training memory.
    """
    if steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Training needs positive steps and a finite positive learning rate.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    output = Path(output_dir)
    if output.exists():
        raise ValueError(f"Choose a new checkpoint directory: {output}")
    pool = _TrainingPosts(posts)
    if authors_per_batch < 2 or len(pool.authors) < authors_per_batch:
        raise ValueError("Each training batch needs at least two eligible authors and two threads per author.")
    rng = random.Random(seed)
    encoder = StarVerifier(checkpoint, device=device, local_files_only=local_files_only)
    torch, tokenizer, model = encoder._load()
    torch.manual_seed(seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []
    for _ in range(steps):
        examples = pool.sample(authors_per_batch, rng)
        windows, owners = encoder._windows([post.text for post in examples])
        by_post = [[] for _ in examples]
        for window, owner in zip(windows, owners):
            by_post[owner].append(window)
        optimizer.zero_grad(set_to_none=True)
        vectors = encoder._pool([rng.choice(choices) for choices in by_post])
        logits = (vectors @ vectors.T) / temperature
        diagonal = torch.eye(len(examples), dtype=torch.bool, device=model.device)
        logits = logits.masked_fill(diagonal, -torch.inf)
        positives = torch.arange(len(examples), device=model.device) ^ 1
        loss = torch.nn.functional.cross_entropy(logits, positives)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite STAR training loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp) / "checkpoint"
        model.save_pretrained(staged, safe_serialization=True)
        tokenizer.save_pretrained(staged)
        run = {
            "base": encoder.checkpoint or STAR,
            "base_revision": encoder.checkpoint_sha256 if checkpoint else STAR_REVISION,
            "steps": steps, "authors_per_batch": authors_per_batch, "seed": seed,
            "learning_rate": learning_rate, "temperature": temperature, "losses": losses,
        }
        (staged / "author_match_training.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf8")
        staged.rename(output)
    return losses
