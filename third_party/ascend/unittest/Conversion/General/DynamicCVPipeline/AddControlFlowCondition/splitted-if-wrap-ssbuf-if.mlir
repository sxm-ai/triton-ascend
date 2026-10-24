// RUN: triton-opt --add-control-flow-condition %s --allow-unregistered-dialect | FileCheck %s

// Reduced from acc-ys.mlir (before this pass) and acc.mlir (after).
// One ssbuffer.if holds the tensor compute inside a single splitted_if, and
// the while_arg update sits outside that splitted_if. A scalar ssbuffer.if
// with no splitted_if stays unwrapped.
// SplittedIfWrapSsbufIf runs inside add-control-flow-condition. for's else
// reads cntArgs filled by the earlier passes in that pipeline.

// CHECK-LABEL: func.func @wrap_one_splitted_if
// The split condition is hoisted above the new if.
// CHECK: arith.cmpi sgt
// CHECK: scf.if
// Tensor compute from the splitted_if then stays in the ssbuffer.if then.
// CHECK: arith.mulf {{.*}} : tensor<4xf32>
// The while_arg update outside the splitted_if stays in the ssbuffer.if then.
// CHECK: arith.addi {{.*}} {ssbuffer.while_arg = 0 : i32}
// CHECK: } {ssbuffer.if = 2 : i32}
// The same update is cloned into the new if's else.
// CHECK: arith.addi
// CHECK: } {ssbuffer.splitted_if = 0 : i32}
// The scalar ssbuffer.if is not wrapped.
// CHECK: } {ssbuffer.if = 1 : i32}

module {
  func.func @wrap_one_splitted_if(%ub: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<4xf32>
    %one = arith.constant dense<1.000000e+00> : tensor<4xf32>
    %0:2 = scf.while (%arg0 = %c0_i32, %arg1 = %c0_i32) : (i32, i32) -> (i32, i32) {
      %1 = arith.cmpi slt, %arg0, %ub : i32
      scf.condition(%1) %arg0, %arg1 : i32, i32
    } do {
    ^bb0(%arg0: i32, %arg1: i32):
      %true = arith.constant true
      %2 = scf.if %true -> (i32) {
        %split = arith.cmpi sgt, %arg0, %c0_i32 : i32
        %inner = scf.if %split -> (i32) {
          %t = arith.mulf %zero, %one : tensor<4xf32>
          scf.yield %arg0 : i32
        } else {
          scf.yield %arg0 : i32
        } {ssbuffer.splitted_if = 0 : i32}
        %next = arith.addi %arg0, %c1_i32 {ssbuffer.while_arg = 0 : i32} : i32
        scf.yield %next : i32
      } else {
        scf.yield %arg0 : i32
      } {ssbuffer.if = 2 : i32}
      %3 = scf.if %true -> (i32) {
        %kept = arith.addi %arg1, %c1_i32 : i32
        scf.yield %kept : i32
      } else {
        scf.yield %arg1 : i32
      } {ssbuffer.if = 1 : i32}
      scf.yield %2, %3 : i32, i32
    } {ssbuffer.main_loop}
    return
  }
}
