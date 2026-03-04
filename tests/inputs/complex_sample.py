"""
Complex sample input for week7 pipeline testing.
Covers: typed functions, classes with methods, nested loops,
conditionals, recursion, decorators, and missing docstrings.
"""
from typing import List, Dict, Optional
import math


# ---------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------
MAX_RETRIES = 3
DEFAULT_THRESHOLD = 0.75


# ---------------------------------------------------------------
# Utility functions (no docstrings — generator should add them)
# ---------------------------------------------------------------

def calculate_discount(price: float, rate: float) -> float:
    if rate < 0 or rate > 1:
        raise ValueError("Rate must be between 0 and 1")
    return price * (1 - rate)


def find_max_element(numbers: List[float]) -> Optional[float]:
    if not numbers:
        return None
    max_val = numbers[0]
    for n in numbers:
        if n > max_val:
            max_val = n
    return max_val


def flatten_nested(nested: List) -> List:
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_nested(item))
        else:
            result.append(item)
    return result


def compute_statistics(data: List[float]) -> Dict[str, float]:
    if not data:
        return {}
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = math.sqrt(variance)
    min_val = min(data)
    max_val = max(data)
    return {
        "mean": mean,
        "variance": variance,
        "std_dev": std_dev,
        "min": min_val,
        "max": max_val,
        "count": n,
    }


def validate_email(email: str) -> bool:
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def generate_matrix(rows: int, cols: int, default: float = 0.0) -> List[List[float]]:
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(default)
        matrix.append(row)
    return matrix


def search_sorted(arr: List[int], target: int) -> int:
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


# ---------------------------------------------------------------
# Class with mixed methods (some documented, some not)
# ---------------------------------------------------------------

class DataProcessor:
    """Processes and transforms input datasets."""

    batch_size: int = 32
    threshold: float = DEFAULT_THRESHOLD

    def __init__(self, name: str, batch_size: int = 32):
        """Initialize the DataProcessor with a name and batch size."""
        self.name = name
        self.batch_size = batch_size
        self._cache: Dict = {}

    def load_data(self, filepath: str) -> List[Dict]:
        with open(filepath, 'r') as f:
            import json
            return json.load(f)

    def process_batch(self, batch: List[Dict]) -> List[Dict]:
        results = []
        for item in batch:
            if "value" not in item:
                continue
            value = item["value"]
            if value > self.threshold:
                processed = {"id": item.get("id"), "value": value, "flag": True}
            else:
                processed = {"id": item.get("id"), "value": value, "flag": False}
            results.append(processed)
        return results

    def get_cache(self, key: str) -> Optional[Dict]:
        return self._cache.get(key)

    def set_cache(self, key: str, value: Dict):
        self._cache[key] = value

    def clear_cache(self):
        self._cache.clear()

    def summarize(self) -> str:
        return f"DataProcessor(name={self.name}, batch_size={self.batch_size})"


# ---------------------------------------------------------------
# Standalone function with a docstring (generator should skip)
# ---------------------------------------------------------------

def format_output(value: float, precision: int = 2) -> str:
    """Format a float value to a fixed number of decimal places."""
    return f"{value:.{precision}f}"


# ---------------------------------------------------------------
# Async function example
# ---------------------------------------------------------------

async def fetch_remote_data(url: str, timeout: int = 30) -> Dict:
    import asyncio
    await asyncio.sleep(0)   # simulate I/O
    return {"url": url, "status": 200}
