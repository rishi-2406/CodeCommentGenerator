"""
AST Comment Model
=================
Fine-tunes ``google-t5/t5-small`` (60M params) to generate natural-language
docstrings from **structured AST feature text**.

Architecture
------------
  Base model : T5-small  (encoder-decoder)
  Task prefix: "Generate docstring: "
  Input      : structured AST feature text produced by ast_feature_formatter
               ─ NOT the source code or function name alone ─
  Output     : first sentence of a natural-language docstring

Training
--------
  Dataset  : CodeSearchNet Python  (via HuggingFace datasets)
  Input    : format_for_model(FunctionFeature, FunctionContext, raises)
  Target   : cleaned first-sentence docstring from the original function

Inference
---------
  Given a new function:
    1. Extract  FunctionFeature + FunctionContext + raises
    2. Call     format_for_model()
    3. Pass to  ASTCommentModel.generate(ff, fc, raises)
    4. Receive  (docstring_text, confidence)

Usage
-----
    model = ASTCommentModel()
    model.fine_tune(pairs, epochs=3)
    text, conf = model.generate(ff, fc, raises=["ValueError"])
    model.save("outputs/model/ast_model")
    model = ASTCommentModel.load("outputs/model/ast_model")
"""
import json
import math
import os
import pathlib
import random
import textwrap
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
    _TRANSFORMERS_OK = True
except ImportError:
    _TRANSFORMERS_OK = False

from .ast_feature_formatter import format_for_model

# Task prefix (T5-style)
TASK_PREFIX   = "Generate docstring: "
MODEL_NAME    = "google-t5/t5-small"
MAX_INPUT_LEN  = 256   # tokens for the structured AST feature text
MAX_TARGET_LEN = 64    # tokens for the generated docstring


# ── Torch Dataset ─────────────────────────────────────────────────────────────

class _ASTDataset(TorchDataset if _TRANSFORMERS_OK else object):
    """Maps ASTTrainPair list to tokenized tensors."""

    def __init__(self, pairs, tokenizer):
        self.pairs     = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        # T5 input = task_prefix + structured AST feature text
        input_text = TASK_PREFIX + pair.input_text

        enc = self.tokenizer(
            input_text,
            max_length=MAX_INPUT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        dec = self.tokenizer(
            text_target=pair.target_text,
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = dec["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         labels,
        }


# ── Main Model Class ──────────────────────────────────────────────────────────

class ASTCommentModel:
    """
    T5-small fine-tuned to generate docstrings from structured AST features.

    The model ONLY sees AST-derived information:
      - Control flow  (loops, conditionals, cyclomatic complexity)
      - Calls         (internal / external function calls)
      - Types         (parameter types, return type, data structures)
      - Decorators    (@property, @staticmethod, etc.)
      - Raises        (exception types from raise statements)
      - Metadata      (async, method, body size)

    It does NOT see raw source code or depend on the function name for meaning.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        if not _TRANSFORMERS_OK:
            raise ImportError(
                "transformers and torch are required:\n"
                "pip install transformers torch sentencepiece"
            )
        self._model_name      = model_name
        self._model           = None
        self._tokenizer       = None
        self._device          = "cuda" if torch.cuda.is_available() else "cpu"
        self._fine_tuned      = False
        self._loss_history: List[float] = []

    # ── Lazy loading ──────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._model is not None:
            return
        print(f"  [ast_model] Loading {self._model_name} …")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model     = T5ForConditionalGeneration.from_pretrained(self._model_name)
        self._model.to(self._device)
        print(f"  [ast_model] Model loaded on {self._device}.")

    # ── Fine-tuning ───────────────────────────────────────────────────────────

    def fine_tune(
        self,
        pairs,                       # List[ASTTrainPair]
        epochs: int = 4,
        batch_size: int = 16,
        lr: float = 3e-4,
        warmup_ratio: float = 0.1,
        val_split: float = 0.1,
        seed: int = 42,
        verbose: bool = True,
    ) -> dict:
        """
        Fine-tune T5 on (AST feature text, docstring) pairs.

        Args:
            pairs:       List of ASTTrainPair from ast_dataset_builder.
            epochs:      Training epochs (4 works well on GPU; reduce to 2 on CPU).
            batch_size:  Mini-batch size (reduce to 4 if OOM on CPU).
            lr:          Peak learning rate.
            warmup_ratio:Fraction of steps for LR warm-up.
            val_split:   Fraction withheld for validation loss reporting.
            seed:        Random seed.
            verbose:     Print per-epoch loss.

        Returns:
            Training metadata dict.
        """
        self._ensure_loaded()

        random.seed(seed)
        data = list(pairs)
        random.shuffle(data)
        split = max(1, int(len(data) * (1 - val_split)))
        train_data, val_data = data[:split], data[split:]

        train_ds = _ASTDataset(train_data, self._tokenizer)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=(self._device == "cuda"))

        val_dl = None
        if val_data:
            val_ds = _ASTDataset(val_data, self._tokenizer)
            val_dl = DataLoader(val_ds, batch_size=batch_size,
                                num_workers=2, pin_memory=(self._device == "cuda"))

        optimizer     = AdamW(self._model.parameters(), lr=lr, weight_decay=0.01)
        total_steps   = len(train_dl) * epochs
        warmup_steps  = int(total_steps * warmup_ratio)
        scheduler     = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self._model.train()
        self._loss_history = []

        if verbose:
            print(f"  [ast_model] Fine-tuning on {len(train_data)} pairs "
                  f"(val={len(val_data)}) for {epochs} epoch(s) on {self._device} …")

        for epoch in range(epochs):
            t0 = time.time()
            epoch_loss = 0.0
            for batch in train_dl:
                batch = {k: v.to(self._device) for k, v in batch.items()}
                optimizer.zero_grad()
                loss = self._model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            avg = epoch_loss / max(len(train_dl), 1)
            self._loss_history.append(avg)

            val_loss = None
            if val_dl:
                self._model.eval()
                vl = sum(
                    self._model(**{k: v.to(self._device) for k, v in b.items()}).loss.item()
                    for b in val_dl
                )
                val_loss = vl / max(len(val_dl), 1)
                self._model.train()

            elapsed = time.time() - t0
            if verbose:
                msg = (f"  [ast_model] Epoch {epoch+1}/{epochs} — "
                       f"train_loss={avg:.4f}")
                if val_loss is not None:
                    msg += f"  val_loss={val_loss:.4f}"
                msg += f"  ({elapsed:.1f}s)"
                print(msg)

        self._fine_tuned = True
        return {
            "model": self._model_name,
            "train_size": len(train_data),
            "val_size":   len(val_data),
            "epochs":     epochs,
            "final_loss": round(self._loss_history[-1], 4) if self._loss_history else None,
            "loss_history": [round(l, 4) for l in self._loss_history],
            "device": self._device,
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(
        self,
        ff,
        fc=None,
        raises: Optional[List[str]] = None,
        num_beams: int = 4,
        max_length: int = 64,
    ) -> Tuple[str, float]:
        """
        Generate a docstring from AST feature objects.

        Args:
            ff:        FunctionFeature (from ast_extractor).
            fc:        FunctionContext (from context_analyzer), optional.
            raises:    List of exception names raised in the body.
            num_beams: Beam search width.
            max_length:Max output tokens.

        Returns:
            (docstring_text, confidence_score)
        """
        self._ensure_loaded()

        # Build input: task prefix + formatted AST features
        feature_text = format_for_model(ff, fc, raises or [])
        input_text   = TASK_PREFIX + feature_text

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

        text = self._tokenizer.decode(
            out.sequences[0], skip_special_tokens=True
        ).strip()

        # Confidence from beam score
        if hasattr(out, "sequences_scores") and out.sequences_scores is not None:
            confidence = float(min(1.0, math.exp(float(out.sequences_scores[0]))))
        else:
            confidence = 0.5 if self._fine_tuned else 0.3

        # Wrap as docstring
        if not text.startswith('"""'):
            text = f'"""{text}"""'

        return text, confidence

    def generate_from_feature_text(
        self, feature_text: str, num_beams: int = 4, max_length: int = 64
    ) -> Tuple[str, float]:
        """
        Lower-level generate: accepts already-formatted AST feature text.
        Used by the evaluator and during corpus evaluation.
        """
        self._ensure_loaded()
        input_text = TASK_PREFIX + feature_text

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

        text = self._tokenizer.decode(
            out.sequences[0], skip_special_tokens=True
        ).strip()

        if hasattr(out, "sequences_scores") and out.sequences_scores is not None:
            confidence = float(min(1.0, math.exp(float(out.sequences_scores[0]))))
        else:
            confidence = 0.5 if self._fine_tuned else 0.3

        if not text.startswith('"""'):
            text = f'"""{text}"""'

        return text, confidence

    def training_report(self) -> dict:
        return {
            "model":        self._model_name,
            "fine_tuned":   self._fine_tuned,
            "device":       self._device,
            "loss_history": [round(l, 4) for l in self._loss_history],
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save model + tokenizer + metadata to a local directory."""
        self._ensure_loaded()
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(directory)
        self._tokenizer.save_pretrained(directory)
        meta = {
            "model_name":   self._model_name,
            "fine_tuned":   self._fine_tuned,
            "loss_history": self._loss_history,
        }
        with open(os.path.join(directory, "ast_model_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [ast_model] Saved to {directory}/")

    @staticmethod
    def load(directory: str) -> "ASTCommentModel":
        """Load a saved model from a local directory."""
        if not _TRANSFORMERS_OK:
            raise ImportError("transformers and torch are required.")
        meta_path = os.path.join(directory, "ast_model_meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)

        inst = ASTCommentModel.__new__(ASTCommentModel)
        inst._model_name    = meta.get("model_name", MODEL_NAME)
        inst._device        = "cuda" if torch.cuda.is_available() else "cpu"
        inst._fine_tuned    = meta.get("fine_tuned", False)
        inst._loss_history  = meta.get("loss_history", [])

        inst._tokenizer = AutoTokenizer.from_pretrained(directory)
        inst._model     = T5ForConditionalGeneration.from_pretrained(directory)
        inst._model.to(inst._device)
        return inst
