// RUN: triton-opt --add-control-flow-condition %s --allow-unregistered-dialect | FileCheck %s --implicit-check-not=triton_ascend.dynamic_cv_pipeline.rc

// One ssbuffer.if contains two splitted_ifs. checkCanWrap rejects the module
// before any rewrite.

// CHECK-LABEL: func.func @more_than_one_splitted_if
// CHECK: } {{.*}}ssbuffer.splitted_if = 0 : i32}
// CHECK: } {{.*}}ssbuffer.splitted_if = 0 : i32}
// CHECK: } {{.*}}ssbuffer.if = 2 : i32}
module {
  func.func @more_than_one_splitted_if(%ub: i32) {
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
        %3 = scf.if %true -> (i32) {
          %t = arith.mulf %zero, %one : tensor<4xf32>
          scf.yield %arg0 : i32
        } else {
          scf.yield %arg0 : i32
        } {ssbuffer.splitted_if = 0 : i32}
        %4 = scf.if %true -> (i32) {
          scf.yield %arg0 : i32
        } else {
          scf.yield %arg0 : i32
        } {ssbuffer.splitted_if = 0 : i32}
        scf.yield %3 : i32
      } else {
        scf.yield %arg0 : i32
      } {ssbuffer.if = 2 : i32}
      scf.yield %2 : i32
    } {ssbuffer.main_loop}
    return
  }
}
