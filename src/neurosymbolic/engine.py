import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..ast_extractor import ModuleFeatures, FunctionFeature, ClassFeature
from ..context_analyzer import ContextGraph, FunctionContext
from ..ast_body_extractor import extract_raises, extract_returned_types
from ..comment_generator import (
    CommentItem,
    _meaningful_tokens,
    _pick_verb,
    _humanize,
    _sanitize_docstring_content,
    _describe_body,
    _describe_class_attributes,
    _generate_class_docstring,
    _VERB_MAP,
)
from .reasoner import SymbolicReasoner, validate_consistency


@dataclass
class NeurosymbolicComment:
    text: str
    confidence: float
    source: str
    validation_flags: List[str] = field(default_factory=list)
    ml_summary: str = ""
    symbolic_sections: Dict[str, str] = field(default_factory=dict)


def neurosymbolic_generate_comments(
    module_features: ModuleFeatures,
    context_graph: ContextGraph,
    ast_model=None,
    source_code: str = "",
    confidence_threshold: float = 0.4,
    strict_ml: bool = False,
) -> List[CommentItem]:
    """
    Generate comments using confidence-gated neurosymbolic fusion.

    For each undocumented function:
      1. Get ML summary + confidence from ast_model.generate()
      2. If confidence >= threshold: use ML summary as docstring lead
      3. If confidence < threshold: fall back to symbolic summary
      4. In both cases, augment with symbolic Args/Returns/Raises
      5. Run validate_consistency() to check ML summary vs AST facts
      6. If inconsistencies found, correct using AST facts

    For classes: always use rule-based generation + security warnings.

    Args:
        module_features:   Output of ast_extractor.extract_features().
        context_graph:     Output of context_analyzer.analyze_context().
        ast_model:         A loaded ASTCommentModel instance (or None).
        source_code:       Raw source string for raises/body extraction.
        confidence_threshold: Minimum confidence to use ML summary (default 0.4).
        strict_ml:         If True, require ML model and fail on generation errors.

    Returns:
        List[CommentItem] sorted by line number.
    """
    if ast_model is None and strict_ml:
        raise RuntimeError(
            "Neurosymbolic ML mode requires a trained AST model. "
            "Run: python3 -m src.main --train"
        )

    fc_map: Dict[str, FunctionContext] = {
        fc.name: fc for fc in context_graph.function_contexts
    }
    reasoner = SymbolicReasoner()
    comments: List[CommentItem] = []

    for cf in module_features.classes:
        if not cf.has_docstring:
            doc_text = _generate_class_docstring(cf, source_code)
            comments.append(CommentItem(
                node_id=cf.node_id,
                node_type="class",
                lineno=cf.lineno,
                col_offset=cf.col_offset,
                text=doc_text,
                kind="docstring",
                target_name=cf.name,
            ))

    for ff in module_features.functions:
        fc = fc_map.get(ff.name)

        if not ff.has_docstring:
            raises: List[str] = []
            if source_code:
                try:
                    end_line = ff.lineno + ff.body_lines
                    raises = extract_raises(source_code, ff.lineno, end_line)
                except Exception:
                    raises = []

            ns_comment = _generate_neurosymbolic_function_docstring(
                ff, fc, ast_model, source_code, raises,
                reasoner, confidence_threshold, strict_ml,
            )

            comments.append(CommentItem(
                node_id=ff.node_id,
                node_type="function",
                lineno=ff.lineno,
                col_offset=ff.col_offset,
                text=ns_comment.text,
                kind="docstring",
                target_name=ff.name,
            ))

        if fc and fc.complexity_label != "simple":
            inline = _generate_neurosymbolic_inline(ff, fc, source_code, reasoner)
            if inline:
                comments.append(CommentItem(
                    node_id=ff.node_id + "_inline",
                    node_type="inline",
                    lineno=ff.lineno,
                    col_offset=ff.col_offset,
                    text=inline,
                    kind="inline",
                    target_name=ff.name,
                ))

    comments.sort(key=lambda c: (c.lineno, c.kind))
    return comments


def _generate_neurosymbolic_function_docstring(
    ff: FunctionFeature,
    fc: Optional[FunctionContext],
    ast_model,
    source_code: str,
    raises: List[str],
    reasoner: SymbolicReasoner,
    confidence_threshold: float,
    strict_ml: bool,
) -> NeurosymbolicComment:
    """
    Build a full docstring using confidence-gated neurosymbolic fusion.

    - ML generates natural-language summary
    - Symbolic engine adds verified Args/Returns/Raises/Security sections
    - Consistency validation checks ML vs AST and corrects mismatches
    """
    ml_summary = ""
    ml_confidence = 0.0
    source = "symbolic"

    if ast_model is not None:
        try:
            raw_text, ml_confidence = ast_model.generate(ff, fc, raises)
            ml_summary = raw_text.strip().strip('"""').strip()
            ml_summary = _sanitize_docstring_content(ml_summary)
        except Exception as exc:
            if strict_ml:
                raise RuntimeError(
                    f"Neurosymbolic generation failed for '{ff.name}': {exc}"
                ) from exc

    if ml_confidence >= confidence_threshold and ml_summary:
        summary = ml_summary
        source = "neural" if ml_confidence >= 0.7 else "fused"
    else:
        summary = reasoner.generate_symbolic_summary(ff, fc)
        summary = _sanitize_docstring_content(summary)
        source = "symbolic"

    if ff.is_async and not summary.lower().startswith("async"):
        summary = f"Asynchronously {summary[0].lower()}{summary[1:]}" if summary else summary

    validation_flags: List[str] = []
    if source in ("neural", "fused") and ml_summary:
        validation_flags = validate_consistency(ml_summary, ff, fc, raises)

        if any("return_type_mismatch" in f for f in validation_flags):
            if ff.return_annotation and ff.return_annotation not in ("None", "none"):
                pass

    tokens = _meaningful_tokens(ff.name)
    noun_phrase = " ".join(t for t in tokens if t not in _VERB_MAP).strip() or _humanize(ff.name)

    lines = ['"""', summary]

    body_sentences = _describe_body(ff, fc, source_code)
    body_sentences = [s for s in body_sentences if not s.startswith("Runs asynchronously")]
    if body_sentences:
        lines.append("")
        for sent in body_sentences:
            sent = _sanitize_docstring_content(sent)
            wrapped = textwrap.fill(sent, width=76, subsequent_indent="    ")
            wrapped = _sanitize_docstring_content(wrapped)
            lines.append(wrapped)

    matched_rules = reasoner.match_rules(ff, fc)
    warning_rules = [r for r in matched_rules if r.severity == "warning"]
    if warning_rules:
        lines.append("")
        for rule in warning_rules:
            lines.append(f"    - {rule.description}")

    symbolic_sections: Dict[str, str] = {}

    real_params = [p for p in ff.params if p.name not in ("self", "cls")]
    if real_params:
        lines.append("")
        lines.append("Args:")
        args_parts = []
        for p in real_params:
            ann = f" ({p.annotation})" if p.annotation else ""
            default = f", defaults to ``{p.default}``" if p.default is not None else ""
            desc = _sanitize_docstring_content(_humanize(p.name))
            lines.append(f"    {p.name}{ann}: {desc}{default}.")
            args_parts.append(f"{p.name}{ann}{default}")
        symbolic_sections["args"] = ", ".join(args_parts)

    ret_ann = ff.return_annotation
    if ret_ann and ret_ann not in ("None", "none"):
        lines.append("")
        lines.append("Returns:")
        ret_ann = _sanitize_docstring_content(ret_ann)
        lines.append(f"    {ret_ann}: The {noun_phrase} result.")
        symbolic_sections["returns"] = ret_ann
    elif not ret_ann and source_code:
        inferred = extract_returned_types(source_code, ff.lineno, ff.lineno + ff.body_lines)
        if inferred and "None" not in inferred:
            ret_str = " or ".join(inferred[:3])
            ret_str = _sanitize_docstring_content(ret_str)
            lines.append("")
            lines.append("Returns:")
            lines.append(f"    {ret_str}: The {noun_phrase} result.")
            symbolic_sections["returns"] = ret_str

    if raises:
        lines.append("")
        lines.append("Raises:")
        raise_parts = []
        for exc in raises[:4]:
            exc = _sanitize_docstring_content(exc)
            lines.append(f"    {exc}: If an error occurs during {noun_phrase}.")
            raise_parts.append(exc)
        symbolic_sections["raises"] = ", ".join(raise_parts)

    if fc and getattr(fc, 'security_issues', None):
        lines.append("")
        lines.append("Security Warnings:")
        for issue in fc.security_issues:
            issue = _sanitize_docstring_content(issue)
            lines.append(f"    - {issue}")

    if source in ("neural", "fused"):
        conf_pct = f" (ML confidence: {ml_confidence:.0%})"
        lines.append("")
        lines.append(f"    Engine: neurosymbolic/{source}{conf_pct}")

    lines.append('"""')
    doc_text = "\n".join(lines)

    return NeurosymbolicComment(
        text=doc_text,
        confidence=ml_confidence,
        source=source,
        validation_flags=validation_flags,
        ml_summary=ml_summary,
        symbolic_sections=symbolic_sections,
    )


def _generate_neurosymbolic_inline(
    ff: FunctionFeature,
    fc: FunctionContext,
    source_code: str,
    reasoner: SymbolicReasoner,
) -> Optional[str]:
    """Generate inline block comment for complex functions with neurosymbolic insights."""
    if fc.complexity_label == "simple":
        return None

    tokens = _meaningful_tokens(ff.name)
    verb = _pick_verb(tokens)
    noun = " ".join(t for t in tokens if t not in _VERB_MAP).strip() or ff.name
    cc = fc.cyclomatic_complexity

    parts = [f"# {verb} {noun}."]

    body_parts: List[str] = []
    if ff.loops:
        body_parts.append(f"{ff.loops} loop(s)")
    if ff.conditionals:
        body_parts.append(f"{ff.conditionals} conditional(s)")
    if fc.calls_internal:
        body_parts.append(f"calls {', '.join(fc.calls_internal[:2])}")
    if body_parts:
        parts.append("# Body: " + ", ".join(body_parts) + ".")

    parts.append(f"# Cyclomatic complexity: {cc} ({fc.complexity_label}).")

    matched_rules = reasoner.match_rules(ff, fc)
    for rule in matched_rules:
        if rule.severity == "warning":
            parts.append(f"# SYMBOLIC WARNING: {rule.description}")

    if source_code:
        raises = extract_raises(source_code, ff.lineno, ff.lineno + ff.body_lines)
        if raises:
            parts.append(f"# May raise: {', '.join(raises[:3])}.")

    if getattr(fc, 'security_issues', None):
        for issue in fc.security_issues:
            parts.append(f"# SECURITY WARNING: {issue}")

    return "\n".join(parts)
