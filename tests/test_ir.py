"""
Week 8 IR Tests
===============
Tests for:
  - TestIRNodes      : IROpcode enum, IRInstruction, IRBlock, IRFunction, IRModule
  - TestIRBuilder    : build_ir() structure and instruction correctness
  - TestIRSerializer : serialize_ir() JSON output and pretty_print_ir() text output
"""
import json
import unittest

from src.parser_module import parse_code
from src.ast_extractor import extract_features
from src.context_analyzer import analyze_context
from src.ir.ir_nodes import (
    IROpcode, IRInstruction, IRBlock, IRFunction, IRModule
)
from src.ir.ir_builder import build_ir
from src.ir.ir_serializer import serialize_ir, pretty_print_ir


# ── Shared fixtures ──────────────────────────────────────────────────────────

SIMPLE_CODE = """\
def add(a: int, b: int) -> int:
    return a + b
"""

COMPLEX_CODE = """\
def binary_search(arr, target):
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
"""

CALL_CODE = """\
def process(data):
    result = sorted(data)
    result = list(result)
    return result
"""


def _build(code: str) -> IRModule:
    tree = parse_code(code)
    mf = extract_features(tree, source_code=code)
    cg = analyze_context(mf, tree, code)
    return build_ir(mf, cg)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IR Nodes Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIRNodes(unittest.TestCase):

    def test_opcode_enum_values(self):
        """All expected opcodes are present and have correct string values."""
        expected = [
            "ASSIGN", "LOAD", "STORE", "BINOP", "UNOP",
            "CALL", "RETURN", "BRANCH", "JUMP", "LABEL", "PHI", "NOP",
        ]
        names = [op.value for op in IROpcode]
        for e in expected:
            self.assertIn(e, names)

    def test_ir_instruction_defaults(self):
        instr = IRInstruction(op=IROpcode.NOP)
        self.assertIsNone(instr.result)
        self.assertEqual(instr.operands, [])
        self.assertEqual(instr.lineno, 0)
        self.assertEqual(instr.meta, {})

    def test_ir_instruction_repr(self):
        instr = IRInstruction(op=IROpcode.ASSIGN, result="t0",
                              operands=["x"], lineno=5)
        r = repr(instr)
        self.assertIn("t0", r)
        self.assertIn("ASSIGN", r)

    def test_ir_block_append_and_terminator(self):
        block = IRBlock(label="entry")
        block.append(IRInstruction(op=IROpcode.LOAD, result="t0", operands=["x"]))
        block.append(IRInstruction(op=IROpcode.RETURN, operands=["t0"]))
        self.assertEqual(len(block.instructions), 2)
        self.assertIsNotNone(block.terminator)
        self.assertEqual(block.terminator.op, IROpcode.RETURN)  # type: ignore

    def test_ir_block_no_terminator_when_empty(self):
        block = IRBlock(label="empty")
        self.assertIsNone(block.terminator)

    def test_ir_function_entry_block(self):
        f = IRFunction(name="foo")
        b = IRBlock(label="entry")
        f.blocks.append(b)
        self.assertIs(f.entry_block, b)

    def test_ir_function_get_block(self):
        f = IRFunction(name="foo")
        b1 = IRBlock(label="entry")
        b2 = IRBlock(label="exit")
        f.blocks += [b1, b2]
        self.assertIs(f.get_block("exit"), b2)
        self.assertIsNone(f.get_block("missing"))

    def test_ir_module_get_function(self):
        m = IRModule(source_file="test.py")
        f = IRFunction(name="bar")
        m.functions.append(f)
        self.assertIs(m.get_function("bar"), f)
        self.assertIsNone(m.get_function("baz"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IR Builder Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIRBuilder(unittest.TestCase):

    def test_module_has_function(self):
        ir = _build(SIMPLE_CODE)
        names = [f.name for f in ir.functions]
        self.assertIn("add", names)

    def test_function_params_preserved(self):
        ir = _build(SIMPLE_CODE)
        func = ir.get_function("add")
        self.assertIsNotNone(func)
        self.assertIn("a", func.params)  # type: ignore
        self.assertIn("b", func.params)  # type: ignore

    def test_entry_block_exists(self):
        ir = _build(SIMPLE_CODE)
        func = ir.get_function("add")
        self.assertIsNotNone(func.entry_block)  # type: ignore
        self.assertEqual(func.entry_block.label, "entry")  # type: ignore

    def test_exit_block_exists(self):
        ir = _build(SIMPLE_CODE)
        func = ir.get_function("add")
        exit_block = func.get_block("exit")  # type: ignore
        self.assertIsNotNone(exit_block)

    def test_exit_block_has_return_instr(self):
        ir = _build(SIMPLE_CODE)
        func = ir.get_function("add")
        exit_block = func.get_block("exit")  # type: ignore
        return_instrs = [i for i in exit_block.instructions  # type: ignore
                         if i.op == IROpcode.RETURN]
        self.assertGreater(len(return_instrs), 0)

    def test_complex_function_has_loop_blocks(self):
        ir = _build(COMPLEX_CODE)
        func = ir.get_function("binary_search")
        self.assertIsNotNone(func)
        block_labels = [b.label for b in func.blocks]  # type: ignore
        self.assertTrue(any("loop" in lbl for lbl in block_labels))

    def test_complex_function_has_branch_blocks(self):
        ir = _build(COMPLEX_CODE)
        func = ir.get_function("binary_search")
        block_labels = [b.label for b in func.blocks]  # type: ignore
        self.assertTrue(any("branch" in lbl for lbl in block_labels))

    def test_call_instructions_emitted(self):
        ir = _build(CALL_CODE)
        func = ir.get_function("process")
        self.assertIsNotNone(func)
        call_instrs = [i for i in func.all_instructions()  # type: ignore
                       if i.op == IROpcode.CALL]
        self.assertGreater(len(call_instrs), 0)

    def test_return_type_propagated(self):
        ir = _build(SIMPLE_CODE)
        func = ir.get_function("add")
        self.assertEqual(func.return_type, "int")  # type: ignore

    def test_source_lineno_set(self):
        ir = _build(SIMPLE_CODE)
        func = ir.get_function("add")
        self.assertGreater(func.source_lineno, 0)  # type: ignore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IR Serializer Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIRSerializer(unittest.TestCase):

    def setUp(self):
        self.ir = _build(SIMPLE_CODE)

    def test_serialize_ir_is_dict(self):
        d = serialize_ir(self.ir)
        self.assertIsInstance(d, dict)

    def test_serialize_ir_json_compatible(self):
        d = serialize_ir(self.ir)
        text = json.dumps(d)   # should not raise
        self.assertIsInstance(text, str)

    def test_serialize_ir_has_functions_key(self):
        d = serialize_ir(self.ir)
        self.assertIn("functions", d)
        self.assertIsInstance(d["functions"], list)

    def test_serialize_ir_function_has_blocks(self):
        d = serialize_ir(self.ir)
        func_dict = d["functions"][0]
        self.assertIn("blocks", func_dict)

    def test_serialize_ir_instruction_fields(self):
        d = serialize_ir(self.ir)
        func_dict = d["functions"][0]
        first_block = func_dict["blocks"][0]
        instr = first_block["instructions"][0]
        for key in ("op", "result", "operands", "lineno", "meta"):
            self.assertIn(key, instr)

    def test_pretty_print_ir_returns_string(self):
        text = pretty_print_ir(self.ir)
        self.assertIsInstance(text, str)

    def test_pretty_print_ir_non_empty(self):
        text = pretty_print_ir(self.ir)
        self.assertGreater(len(text), 0)

    def test_pretty_print_ir_contains_function_name(self):
        text = pretty_print_ir(self.ir)
        self.assertIn("add", text)

    def test_pretty_print_ir_contains_entry_label(self):
        text = pretty_print_ir(self.ir)
        self.assertIn("entry:", text)

    def test_pretty_print_ir_contains_return(self):
        text = pretty_print_ir(self.ir)
        self.assertIn("return", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
