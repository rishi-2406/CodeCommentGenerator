"""
CodeT5 Model — Week 9 ML
==========================
Fine-tunes Salesforce/codet5-small on a (function_signature → docstring)
corpus and generates comments via beam search.

Architecture:
  Pre-trained: codet5-small  (60M params, encoder-decoder T5)
  Task:        code summarisation  → fine-tuned on our corpus
  Inference:   beam search (num_beams=4)

Usage:
  model = CodeT5Model()
  model.fine_tune(corpus, epochs=3)
  comment = model.generate("def get_user(user_id: int) -> dict:")
  model.save("outputs/model/codet5")
  model = CodeT5Model.load("outputs/model/codet5")
"""
import os
import pathlib
import math
import random
import time
from typing import List, Optional, Tuple

try:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import Dataset as TorchDataset, DataLoader
    from transformers import (
        T5ForConditionalGeneration,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

from .corpus_builder import CorpusEntry


# ── Pre-trained model identifier ─────────────────────────────────────────────
MODEL_NAME = "Salesforce/codet5-small"
MAX_INPUT_LEN  = 128   # tokens
MAX_TARGET_LEN = 64    # tokens


# ── Torch Dataset wrapper ─────────────────────────────────────────────────────

class _CodeT5Dataset(TorchDataset if _TRANSFORMERS_AVAILABLE else object):
    def __init__(self, entries: List[CorpusEntry], tokenizer, max_in: int, max_out: int):
        self.entries    = entries
        self.tokenizer  = tokenizer
        self.max_in     = max_in
        self.max_out    = max_out

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        enc = self.tokenizer(
            entry.input_text,
            max_length=self.max_in,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        dec = self.tokenizer(
            text_target=entry.target_text,
            max_length=self.max_out,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = dec["input_ids"].squeeze()
        # Replace pad token id in labels with -100 (ignored in loss)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         labels,
        }


# ── Main model class ──────────────────────────────────────────────────────────

class CodeT5Model:
    """
    Wrapper around Salesforce/codet5-small for code comment generation.

    Provides fine-tuning, generation, save, and load methods.
    Falls back gracefully when transformers/torch are unavailable.
    """

    def __init__(self, model_name: str = MODEL_NAME, cache_dir: Optional[str] = None):
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required. Install with:\n"
                "pip3 install --break-system-packages transformers torch sentencepiece"
            )
        self._model_name = model_name
        self._cache_dir  = cache_dir
        self._model      = None
        self._tokenizer  = None
        self._device     = "cuda" if torch.cuda.is_available() else "cpu"
        self._fine_tuned = False
        self._train_loss_history: List[float] = []

    # ── Lazy loading ──────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        """Load the pre-trained model and tokenizer on first use."""
        if self._model is not None:
            return
        print(f"  [codet5] Loading {self._model_name} …")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            cache_dir=self._cache_dir,
            use_fast=False,
        )
        self._model = T5ForConditionalGeneration.from_pretrained(
            self._model_name,
            cache_dir=self._cache_dir,
        )
        self._model.to(self._device)
        print(f"  [codet5] Model loaded on {self._device}.")

    # ── Fine-tuning ───────────────────────────────────────────────────────────

    def fine_tune(
        self,
        corpus: List[CorpusEntry],
        epochs: int = 3,
        batch_size: int = 8,
        lr: float = 5e-4,
        warmup_ratio: float = 0.1,
        val_split: float = 0.1,
        seed: int = 42,
        verbose: bool = True,
    ) -> dict:
        """
        Fine-tune the model on the given corpus.

        Args:
            corpus:       List of CorpusEntry (input_text, target_text) pairs.
            epochs:       Number of training epochs.
            batch_size:   Mini-batch size (reduce to 4 if OOM).
            lr:           Peak learning rate.
            warmup_ratio: Fraction of steps used for LR warmup.
            val_split:    Fraction of corpus withheld for validation loss.
            seed:         Random seed for reproducibility.
            verbose:      Print per-epoch loss.

        Returns:
            dict with training metadata and loss history.
        """
        self._ensure_loaded()

        random.seed(seed)
        data = list(corpus)
        random.shuffle(data)

        split = max(1, int(len(data) * (1 - val_split)))
        train_data = data[:split]
        val_data   = data[split:]

        train_ds = _CodeT5Dataset(train_data, self._tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_dl = None
        if val_data:
            val_ds = _CodeT5Dataset(val_data, self._tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
            val_dl = DataLoader(val_ds, batch_size=batch_size)

        optimizer  = AdamW(self._model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_dl) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler  = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self._model.train()
        self._train_loss_history = []

        if verbose:
            print(f"  [codet5] Fine-tuning on {len(train_data)} samples "
                  f"(val={len(val_data)}) for {epochs} epoch(s) …")

        for epoch in range(epochs):
            epoch_loss = 0.0
            t0 = time.time()
            for batch in train_dl:
                batch = {k: v.to(self._device) for k, v in batch.items()}
                optimizer.zero_grad()
                out  = self._model(**batch)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(train_dl), 1)
            self._train_loss_history.append(avg_loss)

            val_loss = None
            if val_dl:
                self._model.eval()
                vl = 0.0
                with torch.no_grad():
                    for batch in val_dl:
                        batch = {k: v.to(self._device) for k, v in batch.items()}
                        vl += self._model(**batch).loss.item()
                val_loss = vl / max(len(val_dl), 1)
                self._model.train()

            elapsed = time.time() - t0
            if verbose:
                msg = (f"  [codet5] Epoch {epoch+1}/{epochs} — "
                       f"train_loss={avg_loss:.4f}")
                if val_loss is not None:
                    msg += f"  val_loss={val_loss:.4f}"
                msg += f"  ({elapsed:.1f}s)"
                print(msg)

        self._fine_tuned = True
        return {
            "model":          MODEL_NAME,
            "train_size":     len(train_data),
            "val_size":       len(val_data),
            "epochs":         epochs,
            "final_train_loss": self._train_loss_history[-1] if self._train_loss_history else None,
            "loss_history":   [round(l, 4) for l in self._train_loss_history],
            "device":         self._device,
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(self, func_signature: str, num_beams: int = 4,
                 max_length: int = 64) -> Tuple[str, float]:
        """
        Generate a docstring for the given function signature.

        Args:
            func_signature: e.g. "def get_user(user_id: int) -> dict:"
            num_beams:      Beam width for beam search.
            max_length:     Maximum output token length.

        Returns:
            (comment_text, confidence_score)
        """
        self._ensure_loaded()

        # Prepend the CodeT5 task prefix
        if not func_signature.startswith("Summarize Python:"):
            input_text = f"Summarize Python: {func_signature}"
        else:
            input_text = func_signature

        enc = self._tokenizer(
            input_text,
            max_length=MAX_INPUT_LEN,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        self._model.eval()
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                num_beams=num_beams,
                max_length=max_length,
                early_stopping=True,
                no_repeat_ngram_size=2,
                output_scores=True,
                return_dict_in_generate=True,
            )

        token_ids = out.sequences[0]
        text = self._tokenizer.decode(token_ids, skip_special_tokens=True).strip()

        # Compute approximate confidence from beam score
        if hasattr(out, "sequences_scores") and out.sequences_scores is not None:
            score = float(out.sequences_scores[0])
            confidence = float(min(1.0, math.exp(score)))
        else:
            confidence = 0.5 if self._fine_tuned else 0.3

        # Ensure output is formatted as a docstring
        if not text.startswith('"""'):
            text = f'"""{text}"""'

        return text, confidence

    def predict(self, func_name: str, feature_vector=None) -> Tuple[str, float]:
        """Adapter for ModelSelector API: predict(func_name, feature_vector)."""
        # Build a minimal signature from the function name
        sig = f"def {func_name}():"
        return self.generate(sig)

    def training_report(self) -> dict:
        return {
            "model":         MODEL_NAME,
            "fine_tuned":    self._fine_tuned,
            "device":        self._device,
            "loss_history":  [round(l, 4) for l in self._train_loss_history],
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save fine-tuned model + tokenizer to a local directory."""
        self._ensure_loaded()
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(directory)
        self._tokenizer.save_pretrained(directory)
        # Save small metadata
        import json
        meta = {
            "fine_tuned":   self._fine_tuned,
            "loss_history": self._train_loss_history,
            "model_name":   self._model_name,
        }
        with open(os.path.join(directory, "codet5_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [codet5] Model saved to {directory}/")

    @staticmethod
    def load(directory: str) -> "CodeT5Model":
        """Load a saved model from a local directory."""
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch are required.")
        import json
        meta_path = os.path.join(directory, "codet5_meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)

        instance = CodeT5Model.__new__(CodeT5Model)
        instance._model_name = meta.get("model_name", MODEL_NAME)
        instance._cache_dir  = None
        instance._device     = "cuda" if torch.cuda.is_available() else "cpu"
        instance._fine_tuned = meta.get("fine_tuned", False)
        instance._train_loss_history = meta.get("loss_history", [])

        instance._tokenizer = AutoTokenizer.from_pretrained(directory)
        instance._model     = T5ForConditionalGeneration.from_pretrained(directory)
        instance._model.to(instance._device)
        return instance
