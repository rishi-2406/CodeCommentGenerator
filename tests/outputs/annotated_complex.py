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

# Calculates discount.
# Cyclomatic complexity: 3 (moderate). Contains 0 loop(s) and 1 conditional(s).
def calculate_discount(price: float, rate: float) -> float:
    """
    Calculates discount.
    
    Args:
        price (float): price.
        rate (float): rate.
    
    Returns:
        float: The result of discount.
    """
    if rate < 0 or rate > 1:
        raise ValueError("Rate must be between 0 and 1")
    return price * (1 - rate)


# Finds max element.
# Cyclomatic complexity: 4 (moderate). Contains 1 loop(s) and 2 conditional(s).
def find_max_element(numbers: List[float]) -> Optional[float]:
    """
    Finds max element.
    
    Args:
        numbers (List[float]): numbers.
    
    Returns:
        Optional[float]: The result of max element.
    """
    if not numbers:
        return None
    max_val = numbers[0]
    for n in numbers:
        if n > max_val:
            max_val = n
    return max_val


# Handles flatten nested.
# Cyclomatic complexity: 3 (moderate). Contains 1 loop(s) and 1 conditional(s).
def flatten_nested(nested: List) -> List:
    """
    Handles flatten nested.
    
    Args:
        nested (List): nested.
    
    Returns:
        List: The result of flatten nested.
    """
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_nested(item))
        else:
            result.append(item)
    return result


# Computes statistics.
# Cyclomatic complexity: 3 (moderate). Contains 0 loop(s) and 1 conditional(s).
def compute_statistics(data: List[float]) -> Dict[str, float]:
    """
    Computes statistics.
    
    Args:
        data (List[float]): data.
    
    Returns:
        Dict[str, float]: The result of statistics.
    """
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
    """
    Validates email.
    
    Args:
        email (str): email.
    
    Returns:
        bool: The result of email.
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


# Generates matrix.
# Cyclomatic complexity: 3 (moderate). Contains 2 loop(s) and 0 conditional(s).
def generate_matrix(rows: int, cols: int, default: float = 0.0) -> List[List[float]]:
    """
    Generates matrix.
    
    Args:
        rows (int): rows.
        cols (int): cols.
        default (float): default, defaults to 0.0.
    
    Returns:
        List[List[float]]: The result of matrix.
    """
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(default)
        matrix.append(row)
    return matrix


# Searches sorted.
# Cyclomatic complexity: 4 (moderate). Contains 1 loop(s) and 2 conditional(s).
def search_sorted(arr: List[int], target: int) -> int:
    """
    Searches sorted.
    
    Args:
        arr (List[int]): arr.
        target (int): target.
    
    Returns:
        int: The result of sorted.
    """
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
        """
        Loads data.
        
        Args:
            filepath (str): filepath.
        
        Returns:
            List[Dict]: The result of data.
        """
        with open(filepath, 'r') as f:
            import json
            return json.load(f)

    # Processes batch.
    # Cyclomatic complexity: 4 (moderate). Contains 1 loop(s) and 2 conditional(s).
    def process_batch(self, batch: List[Dict]) -> List[Dict]:
        """
        Processes batch.
        
        Args:
            batch (List[Dict]): batch.
        
        Returns:
            List[Dict]: The result of batch.
        """
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
        """
        Retrieves cache.
        
        Args:
            key (str): key.
        
        Returns:
            Optional[Dict]: The result of cache.
        """
        return self._cache.get(key)

    def set_cache(self, key: str, value: Dict):
        """
        Sets cache.
        
        Args:
            key (str): key.
            value (Dict): value.
        """
        self._cache[key] = value

    def clear_cache(self):
        """
        Clears cache.
        """
        self._cache.clear()

    def summarize(self) -> str:
        """
        Handles summarize.
        
        Returns:
            str: The result of summarize.
        """
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
    """
    Fetches remote data.
    
    Args:
        url (str): url.
        timeout (int): timeout, defaults to 30.
    
    Returns:
        Dict: The result of remote data.
    """
    import asyncio
    await asyncio.sleep(0)   # simulate I/O
    return {"url": url, "status": 200}