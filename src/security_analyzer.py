import ast
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .ast_extractor import ModuleFeatures, FunctionFeature
from .context_analyzer import FunctionContext, VariableInfo


@dataclass
class SecurityIssue:
    pattern_id: str
    severity: str
    function_name: str
    message: str
    lineno: int = 0
    remediation: str = ""


@dataclass
class SecurityReport:
    function_scores: Dict[str, float] = field(default_factory=dict)
    module_safe_pct: float = 100.0
    total_issues: int = 0
    by_severity: Dict[str, int] = field(default_factory=dict)
    issues: List[SecurityIssue] = field(default_factory=list)

    def add_issue(self, issue: SecurityIssue):
        self.issues.append(issue)
        self.total_issues += 1
        self.by_severity[issue.severity] = self.by_severity.get(issue.severity, 0) + 1

    def compute_function_score(self, func_name: str, issues: List[SecurityIssue]) -> float:
        weights = {"critical": 30, "high": 15, "medium": 5, "low": 1}
        penalty = sum(weights.get(i.severity, 0) for i in issues)
        return max(0.0, min(100.0, 100.0 - penalty))

    def compute_module_safe_pct(self, all_scores: Dict[str, float]) -> float:
        if not all_scores:
            return 100.0
        safe_count = sum(1 for s in all_scores.values() if s >= 80.0)
        return round(100.0 * safe_count / len(all_scores), 1)

    def to_dict(self) -> dict:
        return {
            "function_scores": self.function_scores,
            "module_safe_pct": self.module_safe_pct,
            "total_issues": self.total_issues,
            "by_severity": self.by_severity,
            "issues": [
                {
                    "pattern_id": i.pattern_id,
                    "severity": i.severity,
                    "function_name": i.function_name,
                    "message": i.message,
                    "lineno": i.lineno,
                    "remediation": i.remediation,
                }
                for i in self.issues
            ],
        }


_UNSAFE_BUILTINS = {"eval", "exec", "compile"}
_WEAK_CRYPTO = {"hashlib.md5", "hashlib.sha1", "md5", "sha1", "MD5", "SHA1",
                "Crypto.Hash.MD5", "Crypto.Hash.SHA"}
_SQL_CONCAT_PATTERNS = [
    re.compile(r'["\'].*\+.*["\']', re.IGNORECASE),
    re.compile(r'f["\'].*\{.*\}.*SELECT', re.IGNORECASE),
]
_DANGEROUS_DESERIALIZE = {"pickle.load", "pickle.loads", "cPickle.load",
                          "yaml.load", "yaml.unsafe_load"}
_INSECURE_RANDOM = {"random.randint", "random.random", "random.choice",
                    "random.randrange", "random.shuffle"}
_HARDCODED_SECRET_NAMES = {
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "access_key", "private_key", "auth_key", "credentials",
}
_MUTABLE_DEFAULTS = {"list", "dict", "set"}
_HARD_CODED_IP = re.compile(r'(?<!\w)(?:\d{1,3}\.){3}\d{1,3}(?!\w)')
_HARD_CODED_URL = re.compile(r'https?://[^\s"\']+')


def run_security_analysis(
    module_features: ModuleFeatures,
    context_graph,
    source_code: str = "",
) -> SecurityReport:
    """
    Comprehensive security analysis of Python source code.

    Scans AST features and context for security anti-patterns,
    producing a SecurityReport with per-function scores and module safety %.
    """
    report = SecurityReport()
    fc_map: Dict[str, FunctionContext] = {
        fc.name: fc for fc in context_graph.function_contexts
    }

    source_lines = source_code.splitlines() if source_code else []

    for ff in module_features.functions:
        fc = fc_map.get(ff.name)
        func_issues: List[SecurityIssue] = []

        for call in ff.calls_made:
            base_call = call.split("(")[0].split(".")[-1] if "." in call else call
            if base_call in _UNSAFE_BUILTINS or call in _UNSAFE_BUILTINS:
                func_issues.append(SecurityIssue(
                    pattern_id="SEC001",
                    severity="critical",
                    function_name=ff.name,
                    message=f"Uses dangerous builtin '{call}' — allows arbitrary code execution.",
                    lineno=ff.lineno,
                    remediation="Replace with ast.literal_eval() or a safe parser.",
                ))

            if "subprocess" in call:
                body = source_lines[ff.lineno - 1:ff.lineno + ff.body_lines]
                body_text = "\n".join(body)
                if "shell=True" in body_text:
                    func_issues.append(SecurityIssue(
                        pattern_id="SEC002",
                        severity="critical",
                        function_name=ff.name,
                        message="Potential shell injection via subprocess with shell=True.",
                        lineno=ff.lineno,
                        remediation="Use subprocess without shell=True; pass args as a list.",
                    ))

            for weak in _WEAK_CRYPTO:
                if weak in call:
                    func_issues.append(SecurityIssue(
                        pattern_id="SEC004",
                        severity="medium",
                        function_name=ff.name,
                        message=f"Uses weak cryptographic hash '{call}'.",
                        lineno=ff.lineno,
                        remediation="Use SHA-256 or stronger from hashlib.",
                    ))
                    break

            for danger in _DANGEROUS_DESERIALIZE:
                if danger in call:
                    sev = "high"
                    func_issues.append(SecurityIssue(
                        pattern_id="SEC009" if "pickle" in danger else "SEC010",
                        severity=sev,
                        function_name=ff.name,
                        message=f"Uses unsafe deserialization '{call}' — may execute arbitrary code.",
                        lineno=ff.lineno,
                        remediation="Use json.loads() or yaml.safe_load() instead.",
                    ))
                    break

            for rng in _INSECURE_RANDOM:
                if rng in call:
                    func_issues.append(SecurityIssue(
                        pattern_id="SEC011",
                        severity="medium",
                        function_name=ff.name,
                        message=f"Uses insecure random '{call}' — not suitable for cryptography.",
                        lineno=ff.lineno,
                        remediation="Use secrets module for security-sensitive randomness.",
                    ))
                    break

        if fc:
            for vi in fc.variables:
                low_name = vi.name.lower()
                if any(kw in low_name for kw in _HARDCODED_SECRET_NAMES):
                    if vi.inferred_type == "str":
                        func_issues.append(SecurityIssue(
                            pattern_id="SEC003",
                            severity="high",
                            function_name=ff.name,
                            message=f"Potential hardcoded secret in variable '{vi.name}'.",
                            lineno=vi.assigned_at[0] if vi.assigned_at else ff.lineno,
                            remediation="Load secrets from environment variables or a vault.",
                        ))

        if source_code:
            func_src = "\n".join(source_lines[ff.lineno - 1:ff.lineno + ff.body_lines])

            for pattern in _SQL_CONCAT_PATTERNS:
                if pattern.search(func_src) and ("SELECT" in func_src.upper() or "INSERT" in func_src.upper()):
                    func_issues.append(SecurityIssue(
                        pattern_id="SEC005",
                        severity="high",
                        function_name=ff.name,
                        message="Potential SQL injection via string concatenation.",
                        lineno=ff.lineno,
                        remediation="Use parameterized queries with placeholders.",
                    ))
                    break

            if re.search(r'\bexcept\s*:', func_src):
                func_issues.append(SecurityIssue(
                    pattern_id="SEC006",
                    severity="medium",
                    function_name=ff.name,
                    message="Bare 'except:' clause catches all exceptions including SystemExit and KeyboardInterrupt.",
                    lineno=ff.lineno,
                    remediation="Catch specific exceptions: except ValueError: or except Exception:",
                ))

            try:
                func_tree = ast.parse(func_src)
                for node in ast.walk(func_tree):
                    if isinstance(node, ast.FunctionDef):
                        for default in node.args.defaults:
                            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                                func_issues.append(SecurityIssue(
                                    pattern_id="SEC007",
                                    severity="low",
                                    function_name=ff.name,
                                    message="Mutable default argument — shared across all calls.",
                                    lineno=ff.lineno,
                                    remediation="Use None as default and initialize inside the function.",
                                ))
                                break
            except SyntaxError:
                pass

            if "assert" in func_src and not any(
                name in ("test_", "_test", "_tests") for name in [ff.name[:5]]
            ):
                if re.search(r'\bassert\b', func_src):
                    func_issues.append(SecurityIssue(
                        pattern_id="SEC008",
                        severity="low",
                        function_name=ff.name,
                        message="assert statement used in non-test code — disabled with -O flag.",
                        lineno=ff.lineno,
                        remediation="Use explicit if/raise for runtime checks.",
                    ))

            ips = _HARD_CODED_IP.findall(func_src)
            if ips:
                func_issues.append(SecurityIssue(
                    pattern_id="SEC012",
                    severity="low",
                    function_name=ff.name,
                    message=f"Hardcoded IP address(es) found: {', '.join(ips[:3])}.",
                    lineno=ff.lineno,
                    remediation="Use configuration files or environment variables for host addresses.",
                ))

        for issue in func_issues:
            report.add_issue(issue)

        score = report.compute_function_score(ff.name, func_issues)
        report.function_scores[ff.name] = score

    report.module_safe_pct = report.compute_module_safe_pct(report.function_scores)
    return report
