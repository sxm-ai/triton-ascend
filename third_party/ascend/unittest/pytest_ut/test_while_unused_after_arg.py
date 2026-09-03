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

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _while_dead_after_arg_kernel(out_ptr, n):
    i = 0
    alive = 1
    # `alive` is read in the condition (before) but only written in the body,
    # and is not used after the loop. Frontend still carries it; canonicalize
    # may drop the after/result slot. If `alive` were dropped from before as
    # well, the loop would run `n` times instead of once.
    while (i < n) & (alive != 0):
        i = i + 1
        alive = 0
    tl.store(out_ptr, i)


def test_while_dead_after_arg():
    n = 4
    out = torch.zeros((1, ), dtype=torch.int32, device="npu")
    _while_dead_after_arg_kernel[(1, )](out, n)
    assert int(out[0].item()) == 1
