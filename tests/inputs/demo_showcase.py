"""
Demo Showcase Input — exercises every feature of the Code Comment Generator.
Covers: call chains, all 12 security patterns, complexity levels,
classes, async, decorators, recursion, mutable defaults, and more.
"""
import hashlib
import pickle
import random
import subprocess
import yaml


# ---------------------------------------------------------------------------
# Simple functions (cyclomatic complexity 1-2)
# ---------------------------------------------------------------------------

def greet(name: str) -> str:
    return f"Hello, {name}"


def add(a: int, b: int) -> int:
    return a + b


def apply_discount(price: float, rate: float) -> float:
    if rate < 0 or rate > 1:
        raise ValueError("Rate must be between 0 and 1")
    return price * (1 - rate)


# ---------------------------------------------------------------------------
# Moderate complexity (CC 3-5) — forms a call chain
# ---------------------------------------------------------------------------

def sanitize_data(raw: str) -> str:
    cleaned = raw.strip().lower()
    for ch in "<>&'\"":
        cleaned = cleaned.replace(ch, "")
    return cleaned


def validate_input(data: str, min_len: int = 1) -> bool:
    clean = sanitize_data(data)
    if len(clean) < min_len:
        return False
    if any(c.isdigit() for c in clean):
        return True
    return len(clean) >= 3


def format_report(title: str, body: str) -> str:
    if not validate_input(title) or not validate_input(body, min_len=10):
        return "Invalid report data"
    sep = "=" * len(title)
    return f"{sep}\n{title}\n{sep}\n{body}\n"


# ---------------------------------------------------------------------------
# Complex (CC 6-10) — order processing chain
# ---------------------------------------------------------------------------

def calculate_total(items: list, tax_rate: float = 0.08) -> float:
    subtotal = 0.0
    for item in items:
        if "price" not in item:
            continue
        qty = item.get("quantity", 1)
        subtotal += item["price"] * qty
    subtotal = apply_discount(subtotal, 0.1)
    return subtotal * (1 + tax_rate)


def process_order(order: dict, config: dict) -> dict:
    if not order.get("items"):
        return {"status": "error", "message": "Empty order"}
    total = calculate_total(order["items"], config.get("tax_rate", 0.08))
    if total > config.get("max_amount", 10000):
        return {"status": "rejected", "reason": "Amount exceeds limit"}
    for item in order["items"]:
        if item.get("price", 0) < 0:
            return {"status": "error", "message": "Negative price"}
    return {"status": "approved", "total": total}


def inventory_report(orders: list, threshold: int = 5) -> dict:
    low_stock = []
    overstock = []
    for order in orders:
        result = process_order(order, {"tax_rate": 0.07, "max_amount": 5000})
        if result["status"] == "approved":
            for item in order.get("items", []):
                qty = item.get("quantity", 0)
                if qty < threshold:
                    low_stock.append(item)
                elif qty > threshold * 10:
                    overstock.append(item)
    return {"low_stock": low_stock, "overstock": overstock}


# ---------------------------------------------------------------------------
# Very complex (CC > 10)
# ---------------------------------------------------------------------------

def route_optimizer(deliveries: list, depot: tuple, max_stops: int = 8) -> list:
    best_route = []
    best_cost = float("inf")
    remaining = list(deliveries)
    current = depot
    for _ in range(min(len(deliveries), max_stops)):
        nearest = None
        nearest_dist = float("inf")
        for d in remaining:
            dist = abs(d[0] - current[0]) + abs(d[1] - current[1])
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = d
            elif dist == nearest_dist:
                if d[2] < nearest[2]:
                    nearest = d
        if nearest is None:
            break
        best_route.append(nearest)
        remaining.remove(nearest)
        current = (nearest[0], nearest[1])
        if len(best_route) >= max_stops:
            break
    if not best_route and deliveries:
        best_route.append(deliveries[0])
    return best_route


# ---------------------------------------------------------------------------
# Security Pattern Showcase — all 12 SEC patterns
# ---------------------------------------------------------------------------

# SEC001: eval / exec / compile
def run_user_expression(expr: str):
    eval(expr)


def execute_command(cmd: str):
    exec(cmd)


# SEC002: subprocess with shell=True
def shell_execute(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout


# SEC003: Hardcoded secret
def connect_to_api(endpoint: str) -> dict:
    api_key = "sk-abc123DEF456ghi789"
    password = "P@ssw0rd!2024"
    return {"endpoint": endpoint, "key": api_key}


# SEC004: Weak crypto (md5, sha1)
def compute_fingerprint(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# SEC005: SQL string concatenation
def search_users(name: str, role: str) -> str:
    query = "SELECT * FROM users WHERE name = '" + name + "' AND role = '" + role + "'"
    return query


# SEC006: Bare except
def safe_divide(a: float, b: float) -> float:
    try:
        return a / b
    except:
        return 0.0


# SEC007: Mutable default argument
def append_to_log(entry: str, log: list = []):
    log.append(entry)
    return log


# SEC008: assert in non-test code
def validate_config(config: dict):
    assert "host" in config, "Missing host"
    assert "port" in config, "Missing port"
    return True


# SEC009: pickle.load on untrusted data
def load_cached_data(filepath: str):
    with open(filepath, "rb") as f:
        return pickle.load(f)


# SEC010: yaml.load without SafeLoader
def parse_manifest(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.load(f)


# SEC011: Insecure random for security context
def generate_token(length: int = 16) -> str:
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choice(chars) for _ in range(length))


# SEC012: Hardcoded IP
def connect_to_server() -> str:
    host = "192.168.1.100"
    port = 8080
    return f"{host}:{port}"


# ---------------------------------------------------------------------------
# Class with methods and internal call chain
# ---------------------------------------------------------------------------

class DataPipeline:

    def __init__(self, name: str, batch_size: int = 64):
        self.name = name
        self.batch_size = batch_size
        self._buffer: list = []

    def ingest(self, raw: str) -> dict:
        clean = sanitize_data(raw)
        valid = validate_input(clean, min_len=2)
        return {"data": clean, "valid": valid}

    def process(self, records: list) -> list:
        results = []
        for rec in records:
            entry = self.ingest(rec.get("raw", ""))
            if entry["valid"]:
                results.append(entry)
        return results

    def flush(self) -> list:
        out = list(self._buffer)
        self._buffer.clear()
        return out

    def summarize(self) -> str:
        return f"Pipeline({self.name}, buffer={len(self._buffer)})"

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Async function
# ---------------------------------------------------------------------------

async def fetch_remote(url: str, timeout: int = 30) -> dict:
    import asyncio
    await asyncio.sleep(0)
    return {"url": url, "status": 200}


# ---------------------------------------------------------------------------
# Recursive function
# ---------------------------------------------------------------------------

def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# ---------------------------------------------------------------------------
# Decorated functions
# ---------------------------------------------------------------------------

def retry(max_attempts: int = 3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == max_attempts - 1:
                        raise
            return None
        return wrapper
    return decorator


@retry(max_attempts=5)
def unreliable_operation(data: str) -> bool:
    return validate_input(data)


# ---------------------------------------------------------------------------
# Already-documented function (generator should skip this)
# ---------------------------------------------------------------------------

def documented_helper(x: int) -> int:
    """Doubles the input value.

    Args:
        x: The integer to double.

    Returns:
        The doubled integer.
    """
    return x * 2


# ---------------------------------------------------------------------------
# Nested loops (deeply nested)
# ---------------------------------------------------------------------------

def build_3d_grid(x: int, y: int, z: int, default: float = 0.0) -> list:
    grid = []
    for i in range(x):
        layer = []
        for j in range(y):
            row = []
            for k in range(z):
                row.append(default)
            layer.append(row)
        grid.append(layer)
    return grid
