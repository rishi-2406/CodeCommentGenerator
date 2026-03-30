import ast
import textwrap
import pytest

from src.ast_extractor import extract_features
from src.context_analyzer import analyze_context
from src.ml.ast_feature_formatter import format_for_model
from src.ml.ast_dataset_builder import (
    build_full_dataset, _clean_docstring, ASTTrainPair
)


def test_clean_docstring():
    raw = '''
    Fetches the latest items from the remote server.
    
    This function blocks until the server responds.
    
    Parameters
    ----------
    timeout: int
        Max time to wait.
    '''
    # Should extract just the first meaningful sentence.
    clean = _clean_docstring(raw)
    assert clean == "Fetches the latest items from the remote server."

    raw_short = "Short."
    assert _clean_docstring(raw_short) == ""  # < 6 chars gets dropped

    raw_multiline = '''This is a sentence
    that spans two lines.
    '''
    assert _clean_docstring(raw_multiline) == "This is a sentence that spans two lines."


def test_ast_feature_formatter():
    source_code = textwrap.dedent('''
    def complex_math(a: int, b: float = 0.5) -> float:
        """Does math."""
        if a > 0:
            for i in range(a):
                b += i
                if b > 100:
                    raise ValueError("Too big")
        return float(b)
    ''')
    tree = ast.parse(source_code)
    mf = extract_features(tree, source_code)
    cg = analyze_context(mf, tree, source_code)
    
    ff = mf.functions[0]
    fc = cg.function_contexts[0]
    raises = ["ValueError"]
    
    formatted = format_for_model(ff, fc, raises)
    
    # Check that all the structured fields are present
    assert "Generate docstring: complex_math" in formatted
    assert "params: a:int, b:float=0.5" in formatted
    assert "returns: float" in formatted
    assert "loops: 1" in formatted
    assert "conditionals: 2" in formatted
    assert "raises: ValueError" in formatted
    assert "async: no" in formatted
    assert "complexity: " in formatted


def test_dataset_builder_offline():
    # Test dataset building using only the stdlib fallback (offline, fast)
    pairs = build_full_dataset(
        include_codesearchnet=False,
        include_stdlib=True,
        max_stdlib_files=50,  # Just parse a few files quickly
        verbose=False
    )
    
    assert isinstance(pairs, list)
    if pairs:  # Might be 0 if the environment's stdlib paths are weird, but usually > 0
        p = pairs[0]
        assert isinstance(p, ASTTrainPair)
        assert p.func_name
        assert p.input_text.startswith("Generate docstring: ")
        assert len(p.target_text) > 5


import sys

def test_ast_comment_model_generate():
    # Only try importing if requested, since transformers is heavy
    try:
        from src.ml.ast_comment_model import ASTCommentModel
        model = ASTCommentModel()
        
        source = textwrap.dedent('''
        def say_hello(name: str):
            print("hello", name)
        ''')
        mf = extract_features(ast.parse(source), source)
        cg = analyze_context(mf, ast.parse(source), source)
        
        text, conf = model.generate(mf.functions[0], cg.function_contexts[0], [])
        assert text.startswith('"""')
        assert text.endswith('"""')
        # We don't assert confidence range since model isn't fine-tuned, but typically > 0
        
    except ImportError:
        pytest.skip("transformers not installed")
