"""
Dataset Builder — Week 9 ML
============================
Builds a labelled dataset of (feature_vector, comment_text) pairs.

The labels are produced by the existing rule-based comment_generator so the
ML model learns to *replicate and improve upon* those comments using learned
patterns.  Additionally a built-in seed corpus of synthetic function stubs is
always included so the system works fully offline.
"""
import ast
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .feature_vectors import extract_feature_vector, FEATURE_DIM


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class DataPoint:
    func_name: str
    comment_text: str           # label — the rule-based comment
    feature_vector: np.ndarray  # shape (FEATURE_DIM,)


@dataclass
class Dataset:
    points: List[DataPoint] = field(default_factory=list)

    def __len__(self):
        return len(self.points)

    def X(self) -> np.ndarray:
        """Return feature matrix (N, FEATURE_DIM)."""
        if not self.points:
            return np.zeros((0, FEATURE_DIM), dtype=np.float32)
        return np.stack([p.feature_vector for p in self.points], axis=0)

    def y(self) -> List[str]:
        """Return list of comment strings (labels)."""
        return [p.comment_text for p in self.points]


# ── Seed corpus ───────────────────────────────────────────────────────────────
# A small built-in set of Python function stubs so the model can train
# without any external files.

_SEED_CORPUS = '''
def get_user(user_id: int) -> dict:
    result = {}
    return result

def set_password(user, new_password: str) -> None:
    pass

def calculate_area(width: float, height: float) -> float:
    return width * height

def parse_config(filepath: str) -> dict:
    data = {}
    return data

def validate_email(email: str) -> bool:
    if "@" in email:
        return True
    return False

def build_index(items: list) -> dict:
    index = {}
    for item in items:
        index[item] = True
    return index

def send_notification(user_id: int, message: str) -> None:
    pass

def compute_hash(data: bytes) -> str:
    result = ""
    return result

def load_dataset(filepath: str) -> list:
    rows = []
    return rows

def filter_results(results: list, threshold: float) -> list:
    filtered = []
    for r in results:
        if r > threshold:
            filtered.append(r)
    return filtered

def initialize_logger(name: str, level: str = "INFO") -> object:
    return object()

def process_batch(items: list, batch_size: int = 32) -> list:
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i+batch_size])
    return batches

def find_duplicates(sequence: list) -> list:
    seen = set()
    dupes = []
    for item in sequence:
        if item in seen:
            dupes.append(item)
        seen.add(item)
    return dupes

def generate_report(data: dict, title: str) -> str:
    lines = [title]
    for k, v in data.items():
        lines.append(f"{k}: {v}")
    return "\\n".join(lines)

def update_record(record_id: int, fields: dict) -> bool:
    if not fields:
        return False
    return True

def remove_duplicates(items: list) -> list:
    return list(set(items))

def search_records(query: str, records: list) -> list:
    return [r for r in records if query in str(r)]

def encode_data(payload: dict) -> bytes:
    return b""

def decode_response(raw: bytes) -> dict:
    return {}

def check_permissions(user, resource: str) -> bool:
    return True

def run_pipeline(stages: list, data) -> object:
    result = data
    for stage in stages:
        result = stage(result)
    return result

def create_connection(host: str, port: int) -> object:
    return object()

def close_connection(conn) -> None:
    pass

def format_output(data, fmt: str = "json") -> str:
    return str(data)

def merge_configs(base: dict, override: dict) -> dict:
    merged = copy.copy(base)
    merged.update(override)
    return merged

def split_text(text: str, delimiter: str = " ") -> list:
    return text.split(delimiter)

def join_parts(parts: list, separator: str = ", ") -> str:
    return separator.join(str(p) for p in parts)

def assert_valid(value, schema: dict) -> None:
    pass

def transform_record(record: dict, mapping: dict) -> dict:
    return {mapping.get(k, k): v for k, v in record.items()}

def count_occurrences(sequence: list, target) -> int:
    return sequence.count(target)

def fetch_metadata(resource_id: str) -> dict:
    return {}

def store_result(key: str, value, cache: dict) -> None:
    cache[key] = value
'''


def _build_seed_points() -> List[DataPoint]:
    """Parse the seed corpus and generate labelled data points."""
    # Late imports to avoid circular issues
    from ..parser_module import parse_code
    from ..ast_extractor import extract_features
    from ..context_analyzer import analyze_context
    from ..comment_generator import generate_comments

    try:
        tree = parse_code(_SEED_CORPUS)
    except SyntaxError:
        return []

    mf = extract_features(tree, _SEED_CORPUS, filepath="<seed>")
    cg = analyze_context(mf, tree, _SEED_CORPUS)
    comments = generate_comments(mf, cg)

    # Build lookup: func_name -> comment_text
    comment_map = {}
    for c in comments:
        if c.kind == "docstring" and c.node_type == "function":
            comment_map[c.target_name] = c.text

    # Build lookup for contexts
    fc_map = {fc.name: fc for fc in cg.function_contexts}

    points = []
    for ff in mf.functions:
        if ff.name not in comment_map:
            continue
        fc = fc_map.get(ff.name)
        vec = extract_feature_vector(ff, fc)
        points.append(DataPoint(
            func_name=ff.name,
            comment_text=comment_map[ff.name],
            feature_vector=vec,
        ))
    return points


def build_dataset(extra_source_files: Optional[List[str]] = None) -> Dataset:
    """
    Build a full Dataset.

    Args:
        extra_source_files: Optional list of .py file paths to include
                            in addition to the built-in seed corpus.

    Returns:
        Dataset with DataPoints ready for training.
    """
    from ..parser_module import read_file, parse_code
    from ..ast_extractor import extract_features
    from ..context_analyzer import analyze_context
    from ..comment_generator import generate_comments

    points = _build_seed_points()

    if extra_source_files:
        for fpath in extra_source_files:
            try:
                src = read_file(fpath)
                tree = parse_code(src)
                mf = extract_features(tree, src, filepath=fpath)
                cg = analyze_context(mf, tree, src)
                comments = generate_comments(mf, cg)

                comment_map = {}
                for c in comments:
                    if c.kind == "docstring" and c.node_type == "function":
                        comment_map[c.target_name] = c.text

                fc_map = {fc.name: fc for fc in cg.function_contexts}
                for ff in mf.functions:
                    if ff.name not in comment_map:
                        continue
                    fc = fc_map.get(ff.name)
                    vec = extract_feature_vector(ff, fc)
                    points.append(DataPoint(
                        func_name=ff.name,
                        comment_text=comment_map[ff.name],
                        feature_vector=vec,
                    ))
            except Exception:
                continue  # skip unreadable files

    return Dataset(points=points)
