# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Frontend vs make_ttir canonicalize on an scf.while with a dead after-arg.

`alive` is reassigned in the loop body (so the frontend puts it in inits) and
is read in the condition (before region), but the after-arg is never read and
the while result is unused. After `-canonicalize` (inside make_ttir), the
condition/after/results slot should disappear while inits/yield keep it.
"""

import os
import re

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions, make_ttir
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import ASTSource

pytestmark = pytest.mark.backend("none")

_WHILE_SIG = re.compile(
    r"scf\.while\b[^{]*:\s*\(([^)]*)\)\s*->\s*(\([^)]*\)|[^\s{]+)",
    re.MULTILINE,
)
_CONDITION = re.compile(r"scf\.condition\s*\([^)]*\)\s*([^:]*)\s*:")
_YIELD = re.compile(r"scf\.yield\b([^:]*)\s*:")
_DO_ARGS = re.compile(r"\}\s*do\s*\{\s*\^[^(]+\(([^)]*)\)", re.MULTILINE)


@triton.jit
def _while_dead_after_arg_kernel(n, out_ptr):
    i = 0
    alive = 1
    # `alive` is used in the condition (before) but only written in the body.
    # After the loop we only store `i`.
    while (i < n) & (alive != 0):
        i = i + 1
        alive = 0
    tl.store(out_ptr, i)


def _compile_ast_ttir(options):
    source = ASTSource(
        _while_dead_after_arg_kernel,
        {"n": "i32", "out_ptr": "*i32"},
        {},
    )
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    return ast_to_ttir(_while_dead_after_arg_kernel, source, context, options,
                       {}, {})


def _count_mlir_types(type_list):
    text = type_list.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    if not text:
        return 0
    return len([part for part in text.split(",") if part.strip()])


def _count_ssa_list(values):
    return len([part for part in values.split(",") if "%" in part])


def _parse_while_shape(mlir):
    sig = _WHILE_SIG.search(mlir)
    cond = _CONDITION.search(mlir)
    yield_op = _YIELD.search(mlir)
    do_args = _DO_ARGS.search(mlir)
    assert sig is not None, f"no scf.while signature in:\n{mlir}"
    assert cond is not None, f"no scf.condition in:\n{mlir}"
    assert yield_op is not None, f"no scf.yield in:\n{mlir}"
    assert do_args is not None, f"no scf.while after-args in:\n{mlir}"
    return {
        "num_inits": _count_mlir_types(sig.group(1)),
        "num_results": _count_mlir_types(sig.group(2)),
        "num_condition_args": _count_ssa_list(cond.group(1)),
        "num_yield_operands": _count_ssa_list(yield_op.group(1)),
        "num_after_args": _count_mlir_types(do_args.group(1)),
    }


def test_canonicalize_drops_unused_while_after_arg():
    options = NPUOptions(arch="Ascend910_95", enable_graph_optimize=False)

    frontend = str(_compile_ast_ttir(options))
    front = _parse_while_shape(frontend)
    assert front["num_inits"] == 2, frontend
    assert front["num_yield_operands"] == 2, frontend
    assert front["num_condition_args"] == 2, frontend
    assert front["num_after_args"] == 2, frontend
    assert front["num_results"] == 2, frontend

    after_canon = str(make_ttir(_compile_ast_ttir(options), {}, options))
    canon = _parse_while_shape(after_canon)
    assert canon["num_inits"] == 2, after_canon
    assert canon["num_yield_operands"] == 2, after_canon
    assert canon["num_condition_args"] == 1, after_canon
    assert canon["num_after_args"] == 1, after_canon
    assert canon["num_results"] == 1, after_canon
