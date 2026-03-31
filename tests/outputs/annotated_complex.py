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
# Body: 1 conditional(s).
# Cyclomatic complexity: 3 (moderate).
# May raise: ValueError.
def calculate_discount(price: float, rate: float) -> float:
    """
    Calculates discount.
    
    Applies 1 conditional branch to control flow.
    Has moderate control-flow complexity (cyclomatic complexity = 3).
    Calls external function(s): `ValueError`.
    May raise: ValueError.
    
    Args:
        price (float): price.
        rate (float): rate.
    
    Returns:
        float: The discount result.
    
    Raises:
        ValueError: If an error occurs during discount.
    """
    if rate < 0 or rate > 1:
        raise ValueError("Rate must be between 0 and 1")
    return price * (1 - rate)


# Finds max element.
# Body: 1 loop(s), 2 conditional(s).
# Cyclomatic complexity: 4 (moderate).
def find_max_element(numbers: List[float]) -> Optional[float]:
    """
    Finds max element.
    
    Iterates over a sequence using 1 loop.
    Applies 2 conditional branches to control flow.
    Has moderate control-flow complexity (cyclomatic complexity = 4).
    
    Args:
        numbers (List[float]): numbers.
    
    Returns:
        Optional[float]: The max element result.
    """
    if not numbers:
        return None
    max_val = numbers[0]
    for n in numbers:
        if n > max_val:
            max_val = n
    return max_val


# Handles flatten nested.
# Body: 1 loop(s), 1 conditional(s).
# Cyclomatic complexity: 3 (moderate).
def flatten_nested(nested: List) -> List:
    """
    Handles flatten nested.
    
    Iterates over a sequence using 1 loop.
    Applies 1 conditional branch to control flow.
    Has moderate control-flow complexity (cyclomatic complexity = 3).
    Calls external function(s): `result.extend`, `result.append`.
    Works with data structure(s): list.
    
    Args:
        nested (List): nested.
    
    Returns:
        List: The flatten nested result.
    """
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_nested(item))
        else:
            result.append(item)
    return result


# Computes statistics.
# Body: 1 conditional(s).
# Cyclomatic complexity: 3 (moderate).
def compute_statistics(data: List[float]) -> Dict[str, float]:
    """
    Computes statistics.
    
    Applies 1 conditional branch to control flow.
    Has moderate control-flow complexity (cyclomatic complexity = 3).
    Calls external function(s): `math.sqrt`.
    Works with data structure(s): len, math.sqrt, max, min.
    
    Args:
        data (List[float]): data.
    
    Returns:
        Dict[str, float]: The statistics result.
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
    
    Calls external function(s): `re.match`.
    
    Args:
        email (str): email.
    
    Returns:
        bool: The email result.
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


# Generates matrix.
# Body: 2 loop(s).
# Cyclomatic complexity: 3 (moderate).
def generate_matrix(rows: int, cols: int, default: float = 0.0) -> List[List[float]]:
    """
    Generates matrix.
    
    Iterates over sequences using 2 loops.
    Has moderate control-flow complexity (cyclomatic complexity = 3).
    Calls external function(s): `matrix.append`, `row.append`.
    Works with data structure(s): list.
    
    Args:
        rows (int): rows.
        cols (int): cols.
        default (float): default, defaults to ``0.0``.
    
    Returns:
        List[List[float]]: The matrix result.
    """
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(default)
        matrix.append(row)
    return matrix


# Searches sorted.
# Body: 1 loop(s), 2 conditional(s).
# Cyclomatic complexity: 4 (moderate).
def search_sorted(arr: List[int], target: int) -> int:
    """
    Searches sorted.
    
    Iterates over a sequence using 1 loop.
    Applies 2 conditional branches to control flow.
    Has moderate control-flow complexity (cyclomatic complexity = 4).
    
    Args:
        arr (List[int]): arr.
        target (int): target.
    
    Returns:
        int: The sorted result.
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
    """
    Represents a Data processor.
    
    Attributes:
        batch_size: batch size.
        threshold: threshold.
    
    Methods:
        load_data(): Loads data.
        process_batch(): Processes batch.
        get_cache(): Retrieves cache.
        set_cache(): Sets cache.
        clear_cache(): Clears cache.
        summarize(): Handles summarize.
    """
    

    batch_size: int = 32
    threshold: float = DEFAULT_THRESHOLD

    def __init__(self, name: str, batch_size: int = 32):
        """
        Initializes init.
        
        Args:
            name (str): name.
            batch_size (int): batch size, defaults to ``32``.
        """
        self.name = name
        self.batch_size = batch_size
        self._cache: Dict = {}

    def load_data(self, filepath: str) -> List[Dict]:
        """
        Loads data.
        
        Calls external function(s): `json.load`.
        
        Args:
            filepath (str): filepath.
        
        Returns:
            List[Dict]: The data result.
        """
        with open(filepath, 'r') as f:
            import json
            return json.load(f)

    # Processes batch.
    # Body: 1 loop(s), 2 conditional(s).
    # Cyclomatic complexity: 4 (moderate).
    def process_batch(self, batch: List[Dict]) -> List[Dict]:
        """
        Processes batch.
        
        Iterates over a sequence using 1 loop.
        Applies 2 conditional branches to control flow.
        Has moderate control-flow complexity (cyclomatic complexity = 4).
        Calls external function(s): `results.append`, `item.get`.
        Works with data structure(s): dict, list.
        
        Args:
            batch (List[Dict]): batch.
        
        Returns:
            List[Dict]: The batch result.
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
        
        Calls external function(s): `self._cache.get`.
        
        Args:
            key (str): key.
        
        Returns:
            Optional[Dict]: The cache result.
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
        
        Calls external function(s): `self._cache.clear`.
        """
        self._cache.clear()

    def summarize(self) -> str:
        """
        Handles summarize.
        
        Returns:
            str: The summarize result.
        """
        return f"DataProcessor(name={self.name}, batch_size={self.batch_size})"


# ---------------------------------------------------------------
# Standalone function with a docstring (generator should skip)
# ---------------------------------------------------------------

def format_output(value: float, precision: int = 2) -> str:
    """
    Formats output.
    
    Args:
        value (float): value.
        precision (int): precision, defaults to ``2``.
    
    Returns:
        str: The output result.
    """
    return f"{value:.{precision}f}"


# ---------------------------------------------------------------
# Async function example
# ---------------------------------------------------------------

async def fetch_remote_data(url: str, timeout: int = 30) -> Dict:
    """
    Asynchronously Fetches remote data.
    
    Calls external function(s): `asyncio.sleep`.
    
    Args:
        url (str): url.
        timeout (int): timeout, defaults to ``30``.
    
    Returns:
        Dict: The remote data result.
    """
    import asyncio
    await asyncio.sleep(0)  # simulate I/O
    return {"url": url, "status": 200}


# ---------------------------------------------------------------
# Advanced Algorithms (Testing block commenting)
# ---------------------------------------------------------------

# Handles quicksort.
# Body: 1 conditional(s).
# Cyclomatic complexity: 5 (moderate).
def quicksort(arr: List[int]) -> List[int]:
    """
    Handles quicksort.
    
    Applies 1 conditional branch to control flow.
    Has moderate control-flow complexity (cyclomatic complexity = 5).
    
    Args:
        arr (List[int]): arr.
    
    Returns:
        List[int]: The quicksort result.
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


# Handles dfs traverse.
# Body: 1 loop(s), 2 conditional(s).
# Cyclomatic complexity: 4 (moderate).
def dfs_traverse(graph: Dict[str, List[str]], start: str, visited: set = None) -> set:
    """
    Handles dfs traverse.
    
    Iterates over a sequence using 1 loop.
    Applies 2 conditional branches to control flow.
    Has moderate control-flow complexity (cyclomatic complexity = 4).
    Calls external function(s): `visited.add`, `graph.get`.
    Works with data structure(s): set.
    
    Args:
        graph (Dict[str, List[str]]): graph.
        start (str): start.
        visited (set): visited, defaults to ``None``.
    
    Returns:
        set: The dfs traverse result.
    """
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            dfs_traverse(graph, neighbor, visited)
    return visited


# Handles multiply matrices.
# Body: 3 loop(s).
# Cyclomatic complexity: 6 (complex).
def multiply_matrices(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """
    Handles multiply matrices.
    
    Iterates over sequences using 3 loops.
    Has high control-flow complexity (cyclomatic complexity = 6); consider
        splitting into smaller functions.
    
    Args:
        A (List[List[float]]): a.
        B (List[List[float]]): b.
    
    Returns:
        List[List[float]]: The multiply matrices result.
    """
    result = [[0.0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result


# Handles custom lru cache.
# Body: 2 conditional(s).
# Cyclomatic complexity: 3 (moderate).
def custom_lru_cache(capacity: int):
    """
    Handles custom lru cache.
    
    Applies 2 conditional branches to control flow.
    Has moderate control-flow complexity (cyclomatic complexity = 3).
    Calls external function(s): `func`, `order.append`, `order.remove`,
        `order.pop`.
    Works with data structure(s): dict, func, list, order.pop.
    
    Args:
        capacity (int): capacity.
    
    Returns:
        decorator or wrapper or result: The custom lru cache result.
    """
    def decorator(func):
        cache = {}
        order = []
        def wrapper(*args):
            
            if args in cache:
                order.remove(args)
                order.append(args)
                return cache[args]
            result = func(*args)
            
            if len(cache) >= capacity:
                oldest = order.pop(0)
                del cache[oldest]
            cache[args] = result
            order.append(args)
            return result
        return wrapper
    return decorator


# Handles knapsack 01.
# Body: 2 loop(s), 2 conditional(s).
# Cyclomatic complexity: 8 (complex).
def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Handles knapsack 01.
    
    Iterates over sequences using 2 loops.
    Applies 2 conditional branches to control flow.
    Has high control-flow complexity (cyclomatic complexity = 8); consider
        splitting into smaller functions.
    Works with data structure(s): len.
    
    Args:
        weights (List[int]): weights.
        values (List[int]): values.
        capacity (int): capacity.
    
    Returns:
        int: The knapsack 01 result.
    """
    n = len(values)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]


# Handles dijkstra shortest path.
# Body: 2 loop(s), 2 conditional(s).
# Cyclomatic complexity: 6 (complex).
def dijkstra_shortest_path(graph: Dict[str, Dict[str, int]], start: str) -> Dict[str, int]:
    """
    Handles dijkstra shortest path.
    
    Iterates over sequences using 2 loops.
    Applies 2 conditional branches to control flow.
    Has high control-flow complexity (cyclomatic complexity = 6); consider
        splitting into smaller functions.
    Calls external function(s): `heapq.heappop`, `graph[current_vertex].items`,
        `heapq.heappush`.
    Works with data structure(s): list.
    
    Args:
        graph (Dict[str, Dict[str, int]]): graph.
        start (str): start.
    
    Returns:
        Dict[str, int]: The dijkstra shortest path result.
    """
    import heapq
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        if current_distance > distances[current_vertex]:
            continue
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances


# Processes urls concurrently.
# Body: 1 loop(s).
# Cyclomatic complexity: 6 (complex).
def process_urls_concurrently(urls: List[str], max_workers: int = 4) -> List[Dict]:
    """
    Processes urls concurrently.
    
    Iterates over a sequence using 1 loop.
    Has high control-flow complexity (cyclomatic complexity = 6); consider
        splitting into smaller functions.
    Calls external function(s): `concurrent.futures.ThreadPoolExecutor`,
        `concurrent.futures.as_completed`, `urllib.request.Request`,
        `executor.submit`.
    Works with data structure(s): future.result, list, urllib.request.Request.
    
    Args:
        urls (List[str]): urls.
        max_workers (int): max workers, defaults to ``4``.
    
    Returns:
        List[Dict]: The urls concurrently result.
    """
    import concurrent.futures
    import urllib.request
    results = []
    
    def fetch_url(url: str):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                return {"url": url, "status": response.status, "length": len(response.read())}
        except Exception as e:
            return {"url": url, "error": str(e)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_url, url): url for url in urls}
        # Get results.
        for future in concurrent.futures.as_completed(future_to_url):
            data = future.result()
            results.append(data)
            
    return results


# ---------------------------------------------------------------
# Advanced Data Structures & Design Patterns
# ---------------------------------------------------------------

class TreeNode:
    """
    Represents a Tree node.
    """
    def __init__(self, value: int):
        """
        Initializes init.
        
        Args:
            value (int): value.
        """
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    """
    Represents a Binary search tree.
    
    Methods:
        insert(): Inserts.
        inorder_traversal(): Handles inorder traversal.
    """
    def __init__(self):
        """
        Initializes init.
        """
        self.root = None

    def insert(self, value: int):
        """
        Inserts insert.
        
        Applies 1 conditional branch to control flow.
        Calls external function(s): `TreeNode`, `self._insert_recursive`.
        
        Args:
            value (int): value.
        """
        if not self.root:
            self.root = TreeNode(value)
        else:
            self._insert_recursive(self.root, value)

    # Inserts recursive.
    # Body: 4 conditional(s).
    # Cyclomatic complexity: 5 (moderate).
    def _insert_recursive(self, node: TreeNode, value: int):
        """
        Inserts recursive.
        
        Applies 4 conditional branches to control flow.
        Has moderate control-flow complexity (cyclomatic complexity = 5).
        Calls external function(s): `TreeNode`, `self._insert_recursive`.
        
        Args:
            node (TreeNode): node.
            value (int): value.
        """
        if value < node.value:
            if node.left is None:
                node.left = TreeNode(value)
            else:
                self._insert_recursive(node.left, value)
        elif value > node.value:
            if node.right is None:
                node.right = TreeNode(value)
            else:
                self._insert_recursive(node.right, value)
                
    def inorder_traversal(self) -> List[int]:
        """
        Handles inorder traversal.
        
        Applies 1 conditional branch to control flow.
        Calls external function(s): `traverse`, `result.append`.
        Works with data structure(s): list.
        
        Returns:
            List[int]: The inorder traversal result.
        """
        result = []
        def traverse(node):
            if node:
                traverse(node.left)
                result.append(node.value)
                traverse(node.right)
        traverse(self.root)
        return result


class ThreadSafeSingleton:
    """
    Represents a Thread safe singleton.
    
    
        Internal attribute(s): _instance, _lock.
    """
    _instance = None
    _lock = __import__('threading').Lock()

    # Handles new.
    # Body: 1 conditional(s).
    # Cyclomatic complexity: 3 (moderate).
    def __new__(cls, *args, **kwargs):
        """
        Handles new.
        
        Applies 1 conditional branch to control flow.
        Has moderate control-flow complexity (cyclomatic complexity = 3).
        Calls external function(s): `super(ThreadSafeSingleton, cls).__new__`.
        """
        with cls._lock:
            if not cls._instance:
                cls._instance = super(ThreadSafeSingleton, cls).__new__(cls)
        return cls._instance


# Searches star.
# Body: 4 loop(s), 4 conditional(s).
# Cyclomatic complexity: 13 (very_complex).
def a_star_search(grid: List[List[int]], start: tuple, end: tuple) -> Optional[List[tuple]]:
    """
    Searches star.
    
    Iterates over sequences using 4 loops.
    Applies 4 conditional branches to control flow.
    Has very high control-flow complexity (cyclomatic complexity = 13);
        refactoring is strongly recommended.
    Calls external function(s): `heapq.heappush`, `heuristic`, `get_neighbors`,
        `heapq.heappop`.
    Works with data structure(s): dict, list.
    
    Args:
        grid (List[List[int]]): grid.
        start (tuple): start.
        end (tuple): end.
    
    Returns:
        Optional[List[tuple]]: The star result.
    """
    import heapq
    rows, cols = len(grid), len(grid[0])
    
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
    def get_neighbors(node):
        r, c = node
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if 0 <= r + dr < rows and 0 <= c + dc < cols and grid[r+dr][c+dc] == 0:
                neighbors.append((r+dr, c+dc))
        return neighbors

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor in get_neighbors(current):
            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None


# Processes data stream.
# Body: 1 loop(s), 2 conditional(s).
# Cyclomatic complexity: 4 (moderate).
async def process_data_stream(stream, db_session):
    """
    Asynchronously Processes data stream.
    
    Iterates over a sequence using 1 loop.
    Applies 2 conditional branches to control flow.
    Has moderate control-flow complexity (cyclomatic complexity = 4).
    Calls external function(s): `batch.append`, `batch.clear`,
        `db_session.execute_many`, `db_session.commit`.
    Works with data structure(s): list.
    
    Args:
        stream: stream.
        db_session: db session.
    
    Returns:
        bool: The data stream result.
    """
    batch = []
    async for record in stream:
        batch.append(record)
        if len(batch) >= 100:
            await db_session.execute_many("INSERT INTO records VALUES (:val)", batch)
            await db_session.commit()
            batch.clear()
            
    if batch:
        await db_session.execute_many("INSERT INTO records VALUES (:val)", batch)
        await db_session.commit()
    return True

# Handles retry with backoff.
# Body: 1 loop(s), 1 conditional(s).
# Cyclomatic complexity: 4 (moderate).
# May raise: e.
def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """
    Handles retry with backoff.
    
    Iterates over a sequence using 1 loop.
    Applies 1 conditional branch to control flow.
    Has moderate control-flow complexity (cyclomatic complexity = 4).
    Calls external function(s): `func`, `time.sleep`.
    May raise: e.
    
    Args:
        retries: retries, defaults to ``3``.
        backoff_in_seconds: backoff in seconds, defaults to ``1``.
    
    Returns:
        wrapper or inner or func: The retry with backoff result.
    
    Raises:
        e: If an error occurs during retry with backoff.
    """
    import time
    def wrapper(func):
        def inner(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise e
                    time.sleep((backoff_in_seconds * 2 ** x))
                    x += 1
        return inner
    return wrapper