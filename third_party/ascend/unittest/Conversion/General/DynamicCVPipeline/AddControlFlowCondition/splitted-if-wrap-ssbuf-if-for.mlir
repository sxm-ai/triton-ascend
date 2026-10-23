// RUN: triton-opt --add-control-flow-condition %s --allow-unregistered-dialect | FileCheck %s --implicit-check-not=triton_ascend.dynamic_cv_pipeline.rc

// for's else reads cntArgs. AddControlFlowConditionPass fills that map and
// then runs SplittedIfWrapSsbufIf, so this input is the pre-pipeline shape:
// one main scf.for, one block_id, and a splitted_if. The intraDeps pair keeps
// UpdateConditionInfo from bailing out before the wrap.

// CHECK-LABEL: func.func @wrap_for_splitted_if
// Tensor compute from the splitted_if then stays in the ssbuffer.if then.
// CHECK: arith.mulf {{.*}} : tensor<4xf32>
// CHECK: } {{.*}}ssbuffer.if = 1 : i32}
// The new if's else adds the block counter and the for step.
// CHECK: arith.addi
// CHECK: } {{.*}}ssbuffer.splitted_if = 0 : i32}

module {
  func.func @wrap_for_splitted_if() -> (i32, i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<4xf32>
    %one = arith.constant dense<1.000000e+00> : tensor<4xf32>
    %0:2 = scf.for %arg0 = %c0_i32 to %c4_i32 step %c1_i32
        iter_args(%arg1 = %c0_i32, %arg2 = %c0_i32) -> (i32, i32) : i32 {
      %produced = arith.addi %arg2, %c0_i32 {ssbuffer.block_id = 1 : i32, ssbuffer.intraDeps = [0 : i32, 1 : i32]} : i32
      %split = arith.cmpi sgt, %arg0, %c0_i32 {ssbuffer.block_id = 1 : i32} : i32
      %inner = scf.if %split -> (i32) {
        %t = arith.mulf %zero, %one : tensor<4xf32>
        %consumed = arith.addi %produced, %c0_i32 {ssbuffer.intraDeps = [0 : i32, 0 : i32]} : i32
        scf.yield %consumed : i32
      } else {
        scf.yield %arg1 : i32
      } {ssbuffer.block_id = 1 : i32, ssbuffer.splitted_if = 0 : i32}
      scf.yield %inner, %produced : i32, i32
    } {ssbuffer.main_loop = 0 : i32}
    return %0#0, %0#1 : i32, i32
  }
}
