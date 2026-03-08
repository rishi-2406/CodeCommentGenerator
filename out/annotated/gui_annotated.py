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
    """Calculate the discount rate for a purchase."""
    if rate < 0 or rate > 1:
        raise ValueError("Rate must be between 0 and 1")
    return price * (1 - rate)



def find_max_element(numbers: List[float]) -> Optional[float]:
    """Write a function to find the maximum element in a given list of numbers."""
    if not numbers:
        return None
    max_val = numbers[0]
    for n in numbers:
        if n > max_val:
            max_val = n
    return max_val



def flatten_nested(nested: List) -> List:
    """Flattens a nested nested list."""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_nested(item))
        else:
            result.append(item)
    return result



def compute_statistics(data: List[float]) -> Dict[str, float]:
    """Compute statistics from a dictionary of parameters."""
    # If the data is not a dictionary, return it.
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
    """Validates an email address format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))



def generate_matrix(rows: int, cols: int, default: float = 0.0) -> List[List[float]]:
    """Generate matrix of rows and columns."""
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(default)
        matrix.append(row)
    return matrix



def search_sorted(arr: List[int], target: int) -> int:
    """Searches records matching the given query."""
    low, high = 0, len(arr) - 1
    # while low <= target.
    while low <= high:
        mid = (low + high) // 2
        # Check if an array contains a target value.
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
        """Create a class for storing data in the database."""
        self.name = name
        self.batch_size = batch_size
        self._cache: Dict = {}


    def load_data(self, filepath: str) -> List[Dict]:
        """Loads the data from a file."""
        with open(filepath, 'r') as f:
            import json
            return json.load(f)


    def process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Processes items in configurable batch sizes."""
        results = []
        # iterate over multiple items.
        for item in batch:
            # Skip if the value is a value.
            if "value" not in item:
                continue
            value = item["value"]
            # Check if item is smaller than threshold.
            if value > self.threshold:
                processed = {"id": item.get("id"), "value": value, "flag": True}
            else:
                processed = {"id": item.get("id"), "value": value, "flag": False}
            results.append(processed)
        return results


    def get_cache(self, key: str) -> Optional[Dict]:
        """Get a cache entry."""
        return self._cache.get(key)


    def set_cache(self, key: str, value: Dict):
        """Set a cache value."""
        self._cache[key] = value


    def clear_cache(self):
        """Clears all cached entries."""
        self._cache.clear()


    def summarize(self) -> str:
        """Summarize the summary of all the test cases."""
        return f"DataProcessor(name={self.name}, batch_size={self.batch_size})"



# ---------------------------------------------------------------
# Standalone function with a docstring (generator should skip)
# ---------------------------------------------------------------


def format_output(value: float, precision: int = 2) -> str:
    """Formats the output into the requested representation."""
    return f"{value:.{precision}f}"



# ---------------------------------------------------------------
# Async function example
# ---------------------------------------------------------------


async def fetch_remote_data(url: str, timeout: int = 30) -> Dict:
    """Fetch remote data from a URL."""
    import asyncio
    await asyncio.sleep(0)  # simulate I/O
    return {"url": url, "status": 200}



# ---------------------------------------------------------------
# Advanced Algorithms (Testing block commenting)
# ---------------------------------------------------------------


def quicksort(arr: List[int]) -> List[int]:
    """Quick sort a list of integers using quicksort algorithm."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)



def dfs_traverse(graph: Dict[str, List[str]], start: str, visited: set = None) -> set:
    """Traverse a graph in depth first search."""
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            dfs_traverse(graph, neighbor, visited)
    return visited



def multiply_matrices(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Multiply matrices A and B using matrix multiplication."""
    result = [[0.0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result



def custom_lru_cache(capacity: int):
    """Custom lru cache decorator."""
    def decorator(func):
        cache = {}
        order = []
        def wrapper(*args):
            
            # Cache entry.
            if args in cache:
                order.remove(args)
                order.append(args)
                return cache[args]
            result = func(*args)
            
            # Clear the oldest item if available.
            if len(cache) >= capacity:
                oldest = order.pop(0)
                del cache[oldest]
            cache[args] = result
            order.append(args)
            return result
        return wrapper
    return decorator



def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """Knapsack01."""
    n = len(values)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    # Loop over all weights.
    for i in range(n + 1):
        # 尚未创建对象.
        for w in range(capacity + 1):
            # 尚未创建对象.
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]



def dijkstra_shortest_path(graph: Dict[str, Dict[str, int]], start: str) -> Dict[str, int]:
    """Returns a shortest path for the dijkstra_shortest_path."""
    import heapq
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]
    # current_distance = 0.
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        # skip if current_distance is smaller than current vertex.
        if current_distance > distances[current_vertex]:
            continue
        # calculate distance between current vertex.
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            # check if neighbor is nearby.
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances



def process_urls_concurrently(urls: List[str], max_workers: int = 4) -> List[Dict]:
    """Process urls concurrently."""
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
        """Sets the __init__ attribute of the class."""
        self.value = value
        self.left = None
        self.right = None


class BinarySearchTree:
    """
    Represents a Binary search tree.
    
    Methods:
        insert(): Inserts .
        inorder_traversal(): Handles inorder traversal.
    """
    def __init__(self):
        """Initialize attributes and methods from parent class."""
        self.root = None


    def insert(self, value: int):
        """Inserts a new row into the data structure."""
        if not self.root:
            self.root = TreeNode(value)
        else:
            self._insert_recursive(self.root, value)


    def _insert_recursive(self, node: TreeNode, value: int):
        """Insert a value in the tree recursively."""
        # Insert a recursive node if necessary.
        if value < node.value:
            # recursive.
            if node.left is None:
                node.left = TreeNode(value)
            else:
                self._insert_recursive(node.left, value)
        elif value > node.value:
            # Store the right node in recursion.
            if node.right is None:
                node.right = TreeNode(value)
            else:
                self._insert_recursive(node.right, value)
                
    def inorder_traversal(self) -> List[int]:
        """Traverses the graph in order of traversal."""
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
    
    """
    _instance = None
    _lock = __import__('threading').Lock()


    def __new__(cls, *args, **kwargs):
        """Creates a new instance of the class with the given attributes."""
        with cls._lock:
            if not cls._instance:
                cls._instance = super(ThreadSafeSingleton, cls).__new__(cls)
        return cls._instance



def a_star_search(grid: List[List[int]], start: tuple, end: tuple) -> Optional[List[tuple]]:
    """Astar search algorithm for a grid of grid elements."""
    import heapq
    rows, cols = len(grid), len(grid[0])
    
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
    def get_neighbors(node):
        r, c = node
        neighbors = []
        # compute neighbors.
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # add neighbors.
            if 0 <= r + dr < rows and 0 <= c + dc < cols and grid[r+dr][c+dc] == 0:
                neighbors.append((r+dr, c+dc))
        return neighbors


    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}


    # current = current.
    while open_set:
        current = heapq.heappop(open_set)[1]


        # current = current.
        if current == end:
            path = []
            # current = came_from.
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]


        # current = g_score[current].
        for neighbor in get_neighbors(current):
            tentative_g_score = g_score[current] + 1
            # If the neighbor score is less than the current g_score, then it's done.
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                # check if neighbor is in open_set.
                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None



async def process_data_stream(stream, db_session):
    """Processes items in configurable batch sizes."""
    batch = []
    async for record in stream:
        batch.append(record)
        # await db_session.execute_many(batch) == 100:.
        if len(batch) >= 100:
            await db_session.execute_many("INSERT INTO records VALUES (:val)", batch)
            await db_session.commit()
            batch.clear()
            
    # Persist each row into the session.
    if batch:
        await db_session.execute_many("INSERT INTO records VALUES (:val)", batch)
        await db_session.commit()
    return True


def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """Try to retry with a backoff in seconds."""
    import time
    def wrapper(func):
        def inner(*args, **kwargs):
            x = 0
            # they are not allowed to be retried.
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # if x == retries: raise an error.
                    if x == retries:
                        raise e
                    time.sleep((backoff_in_seconds * 2 ** x))
                    x += 1
        return inner
    return wrapper