"""
IR sub-package — Week 8
========================
Defines and constructs a language-agnostic Intermediate Representation
from the Week 7 AST features.
"""
from .ir_nodes import (
    IROpcode, IRInstruction, IRBlock, IRFunction, IRModule,
)
from .ir_builder import build_ir
from .ir_serializer import serialize_ir, pretty_print_ir

__all__ = [
    "IROpcode", "IRInstruction", "IRBlock", "IRFunction", "IRModule",
    "build_ir", "serialize_ir", "pretty_print_ir",
]
