"""
Template Ranking Model — Week 9 ML
====================================
A lightweight offline Seq2Seq-style model based on TF-IDF cosine similarity.

Instead of a full neural network (which would require GPU + large downloads),
this model:
  1. Encodes the function name as a TF-IDF character n-gram vector.
  2. Encodes a curated bank of (name_pattern, comment_template) pairs.
  3. At inference time, ranks all templates by cosine similarity to the
     query function name and returns the top match.

This gives meaningful results deterministically with zero internet access
and runs in milliseconds, making it suitable for academic lab use.
"""
import pickle
import pathlib
import re
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Template bank ─────────────────────────────────────────────────────────────
# Each entry: (name_pattern, comment_template)
# The name_pattern is what TF-IDF is trained on; {name} in the comment
# is substituted at inference time.

_TEMPLATE_BANK: List[Tuple[str, str]] = [
    # getter/fetcher patterns
    ("get_user",            '"""Retrieves the user record by identifier."""'),
    ("get_data",            '"""Retrieves raw data from the data source."""'),
    ("fetch_record",        '"""Fetches the specified record from storage."""'),
    ("load_config",         '"""Loads configuration settings from file."""'),
    ("read_file",           '"""Reads and returns the contents of a file."""'),
    ("retrieve_item",       '"""Retrieves an item by its key."""'),

    # setter/updater patterns
    ("set_value",           '"""Sets the given value in the data store."""'),
    ("update_record",       '"""Updates an existing record with new field values."""'),
    ("save_data",           '"""Persists data to the backing storage."""'),
    ("write_output",        '"""Writes the output to the destination."""'),
    ("store_result",        '"""Stores the computed result for later retrieval."""'),

    # add/insert patterns
    ("add_item",            '"""Adds an item to the collection."""'),
    ("insert_row",          '"""Inserts a new row into the data structure."""'),
    ("append_entry",        '"""Appends an entry to the end of the sequence."""'),
    ("push_event",          '"""Pushes an event onto the queue."""'),

    # remove/delete patterns
    ("remove_item",         '"""Removes the specified item from the collection."""'),
    ("delete_record",       '"""Deletes a record from the data store."""'),
    ("clear_cache",         '"""Clears all cached entries."""'),

    # compute patterns
    ("calculate_area",      '"""Calculates and returns the area."""'),
    ("compute_hash",        '"""Computes a cryptographic hash of the input."""'),
    ("count_items",         '"""Counts and returns the number of matching items."""'),
    ("sum_values",          '"""Sums all values in the sequence."""'),

    # search/find patterns
    ("find_duplicates",     '"""Finds and returns all duplicate elements."""'),
    ("search_records",      '"""Searches records matching the given query."""'),
    ("lookup_entry",        '"""Looks up an entry by its key."""'),

    # validation/check patterns
    ("validate_email",      '"""Validates an email address format."""'),
    ("check_permissions",   '"""Checks whether the user has the required permissions."""'),
    ("is_valid",            '"""Returns True if the input is valid."""'),
    ("verify_signature",    '"""Verifies the cryptographic signature."""'),

    # parse/format patterns
    ("parse_config",        '"""Parses a configuration file and returns a dict."""'),
    ("format_output",       '"""Formats the output into the requested representation."""'),
    ("convert_units",       '"""Converts a value between measurement units."""'),
    ("encode_data",         '"""Encodes the payload for transmission."""'),
    ("decode_response",     '"""Decodes a raw response into structured data."""'),

    # process/run patterns
    ("process_batch",       '"""Processes items in configurable batch sizes."""'),
    ("run_pipeline",        '"""Runs the processing pipeline on the input data."""'),
    ("execute_query",       '"""Executes a database query and returns results."""'),
    ("handle_request",      '"""Handles an incoming request and produces a response."""'),

    # build/create patterns
    ("build_index",         '"""Builds a lookup index from the item list."""'),
    ("create_connection",   '"""Creates and returns a new connection."""'),
    ("initialize_logger",   '"""Initializes and returns a configured logger."""'),
    ("generate_report",     '"""Generates a formatted report from the data."""'),

    # send/receive patterns
    ("send_notification",   '"""Sends a notification to the target user."""'),
    ("close_connection",    '"""Closes the active connection and releases resources."""'),

    # test/assert patterns
    ("test_function",       '"""Tests the function for expected behaviour."""'),
    ("assert_valid",        '"""Asserts that the input conforms to the schema."""'),

    # generic fallbacks
    ("transform_record",    '"""Transforms a record using the provided field mapping."""'),
    ("merge_configs",       '"""Merges two configuration dicts, override wins."""'),
    ("split_text",          '"""Splits text by a delimiter and returns a list."""'),
    ("join_parts",          '"""Joins parts into a single string with a separator."""'),
    ("filter_results",      '"""Filters results below a threshold value."""'),
]


def _humanise(func_name: str) -> str:
    """Convert snake_case/CamelCase identifier to plain English."""
    parts = func_name.split("_")
    tokens = []
    for part in parts:
        sub = re.sub(r'([A-Z][a-z]+)', r' \1', part)
        sub = re.sub(r'([A-Z]+)([A-Z][a-z])', r' \1 \2', sub)
        tokens.extend(sub.strip().lower().split())
    return " ".join(t for t in tokens if t)


class TemplateRankingModel:
    """
    Offline template-ranking comment generator.

    Uses TF-IDF character n-gram cosine similarity to match an unseen
    function name to the closest entry in the template bank.
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
        )
        self._patterns: List[str] = [p for p, _ in _TEMPLATE_BANK]
        self._templates: List[str] = [t for _, t in _TEMPLATE_BANK]
        self._pattern_matrix: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self) -> "TemplateRankingModel":
        """Fit the TF-IDF vectorizer on the template bank patterns."""
        self._vectorizer.fit(self._patterns)
        self._pattern_matrix = self._vectorizer.transform(self._patterns).toarray()
        self._fitted = True
        return self

    def predict(self, func_name: str, feature_vector=None) -> Tuple[str, float]:
        """
        Return the best-matching template comment for the given function name.

        Args:
            func_name: Function identifier string.
            feature_vector: Unused (kept for API compatibility with TFIDFCommentModel).

        Returns:
            (comment_text, confidence_score)
        """
        if not self._fitted:
            self.fit()

        query_vec = self._vectorizer.transform([func_name]).toarray()
        sims = cosine_similarity(query_vec, self._pattern_matrix)[0]
        best_idx = int(np.argmax(sims))
        confidence = float(sims[best_idx])

        # Fill in the human-readable name
        tmpl = self._templates[best_idx]
        human_name = _humanise(func_name)

        # Replace the pattern's human name with the actual function's human name
        # The template already has a hard-coded description; we return it as-is
        # but replace a generic placeholder if present.
        comment = tmpl.replace("{name}", human_name)
        return comment, confidence

    def training_report(self) -> dict:
        return {
            "model": "TemplateRankingModel",
            "fitted": self._fitted,
            "template_bank_size": len(self._templates),
        }

    def save(self, path: str) -> None:
        """Serialize model to disk."""
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "TemplateRankingModel":
        """Load a previously saved model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)
