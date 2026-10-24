// RUN: triton-opt --add-control-flow-condition %s --allow-unregistered-dialect | FileCheck %s --implicit-check-not=triton_ascend.dynamic_cv_pipeline.rc

// An ssbuffer.if has tensor compute and no splitted_if. checkCanWrap rejects
// the module before any rewrite.

// CHECK-LABEL: func.func @tensor_without_splitted_if
// CHECK-NOT: ssbuffer.splitted_if
// CHECK: arith.mulf {{.*}} : tensor<4xf32>
// CHECK: } {{.*}}ssbuffer.if = 2 : i32}
// CHECK-NOT: ssbuffer.splitted_if
module {
  func.func @tensor_without_splitted_if(%ub: i32) {
    %c0_i32 = arith.constant 0 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<4xf32>
    %one = arith.constant dense<1.000000e+00> : tensor<4xf32>
    %0 = scf.while (%arg0 = %c0_i32) : (i32) -> i32 {
      %1 = arith.cmpi slt, %arg0, %ub : i32
      scf.condition(%1) %arg0 : i32
    } do {
    ^bb0(%arg0: i32):
      %true = arith.constant true
      %2 = scf.if %true -> (i32) {
        %t = arith.mulf %zero, %one : tensor<4xf32>
        scf.yield %arg0 : i32
      } else {
        scf.yield %arg0 : i32
      } {ssbuffer.if = 2 : i32}
      scf.yield %2 : i32
    } {ssbuffer.main_loop}
    return
  }
}
