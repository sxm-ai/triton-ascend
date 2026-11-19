module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.insertionOptimization, ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32, ssbuffer.load_store_buf_count = 1 : i32, ssbuffer.merge_small_block_first_run_done = true} {
  func.func @batched_gemm_3d_tma_kernel(%arg0: memref, %arg1: memref, %arg2: memref {tt.tensor_kind = 0 : i32}, %arg3: memref {tt.tensor_kind = 0 : i32}, %arg4: memref {tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 5 : i32} dense<[4, 8, 16, 16]> : tensor<4xi64>
    %cst_0 = arith.constant {ssbuffer.block_id = 5 : i32} dense<[128, 4, 16]> : tensor<3xi64>
    %cst_1 = arith.constant {ssbuffer.block_id = 4 : i32} 0.000000e+00 : f32
    %c63_i32 = arith.constant {ssbuffer.block_id = 4 : i32} 63 : i32
    %c127_i32 = arith.constant {ssbuffer.block_id = 4 : i32} 127 : i32
    %c64_i32 = arith.constant {ssbuffer.block_id = 4 : i32} 64 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 4 : i32} 128 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 4 : i32} 0 : i32
    %c-1_i32 = arith.constant {ssbuffer.block_id = 4 : i32} -1 : i32
    %c1_i32 = arith.constant {ssbuffer.block_id = 4 : i32} 1 : i32
    %c96_i32 = arith.constant {ssbuffer.block_id = 4 : i32} 96 : i32
    %c0_i64 = arith.constant {ssbuffer.block_id = 4 : i32} 0 : i64
    %c128 = arith.constant {ssbuffer.block_id = 4 : i32} 128 : index
    %c0 = arith.constant {ssbuffer.block_id = 4 : i32} 0 : index
    %c1 = arith.constant {ssbuffer.block_id = 4 : i32} 1 : index
    %c64 = arith.constant {ssbuffer.block_id = 4 : i32} 64 : index
    %cst_2 = arith.constant {ssbuffer.block_id = 4 : i32} 0.000000e+00 : f16
    %cst_3 = arith.constant {ssbuffer.block_id = 4 : i32} dense<[1, 128, 128]> : tensor<3xi64>
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<128x128xf32>
      %1 = linalg.fill {ssbuffer.block_id = 4 : i32} ins(%cst_1 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %2 = arith.addi %arg6, %c127_i32 {ssbuffer.block_id = 4 : i32} : i32
      %3 = arith.divsi %2, %c128_i32 {ssbuffer.block_id = 4 : i32} : i32
      %4 = arith.addi %arg7, %c127_i32 {ssbuffer.block_id = 4 : i32} : i32
      %5 = arith.divsi %4, %c128_i32 {ssbuffer.block_id = 4 : i32} : i32
      %6 = arith.addi %arg8, %c63_i32 {ssbuffer.block_id = 4 : i32} : i32
      %7 = arith.divsi %6, %c64_i32 {ssbuffer.block_id = 4 : i32} : i32
      %8 = arith.muli %3, %5 {ssbuffer.block_id = 4 : i32} : i32
      %9 = arith.extsi %arg5 {ssbuffer.block_id = 4 : i32} : i32 to i64
      %10 = arith.muli %arg5, %8 {ssbuffer.block_id = 4 : i32} : i32
      %11 = arith.divsi %10, %c96_i32 {ssbuffer.block_id = 4 : i32} : i32
      %12 = arith.remsi %10, %c96_i32 {ssbuffer.block_id = 4 : i32} : i32
      %13 = arith.cmpi slt, %arg12, %12 {ssbuffer.block_id = 4 : i32} : i32
      %14 = scf.if %13 -> (i32) {
        %25 = arith.addi %11, %c1_i32 {ssbuffer.block_id = 4 : i32} : i32
        scf.yield {ssbuffer.block_id = 4 : i32} %25 : i32
      } else {
        scf.yield {ssbuffer.block_id = 4 : i32} %11 : i32
      } {ssbuffer.block_id = 4 : i32}
      %15 = arith.subi %arg12, %c96_i32 {ssbuffer.block_id = 4 : i32} : i32
      %16 = arith.muli %arg8, %arg6 {ssbuffer.block_id = 4 : i32} : i32
      %17 = arith.extsi %16 {ssbuffer.block_id = 4 : i32} : i32 to i64
      %18 = arith.muli %arg7, %arg8 {ssbuffer.block_id = 4 : i32} : i32
      %19 = arith.extsi %18 {ssbuffer.block_id = 4 : i32} : i32 to i64
      %20 = arith.muli %arg6, %arg7 {ssbuffer.block_id = 4 : i32} : i32
      %21 = arith.extsi %20 {ssbuffer.block_id = 4 : i32} : i32 to i64
      %22 = arith.muli %7, %14 {ssbuffer.block_id = 4 : i32} : i32
      %23 = arith.subi %7, %c1_i32 {ssbuffer.block_id = 4 : i32} : i32
      %alloc = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_4 = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_5 = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_6 = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_7 = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_8 = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
      %24:18 = scf.for %arg15 = %c0_i32 to %22 step %c1_i32 iter_args(%arg16 = %15, %arg17 = %c-1_i32, %arg18 = %c0_i32, %arg19 = %c0_i32, %arg20 = %c0_i32, %arg21 = %c0_i32, %arg22 = %c0_i32, %arg23 = %c0_i32, %arg24 = %1, %arg25 = %15, %arg26 = %c0_i32, %arg27 = %c0_i32, %arg28 = %c0_i32, %arg29 = %c0_i32, %arg30 = %c-1_i32, %arg31 = %c-1_i32, %arg32 = %c0_i32, %arg33 = %c0_i32) -> (i32, i32, i32, i32, i32, i32, i32, i32, tensor<128x128xf32>, i32, i32, i32, i32, i32, i32, i32, i32, i32)  : i32 {
        %25 = arith.divsi %arg15, %c1_i32 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32} : i32
        %c2_i32 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32} 2 : i32
        %26 = arith.remsi %25, %c2_i32 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32} : i32
        %c0_i32_9 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32} 0 : i32
        %27 = arith.cmpi eq, %26, %c0_i32_9 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32} : i32
        %28 = arith.cmpi eq, %arg17, %23 {ssbuffer.block_id = 5 : i32} : i32
        %29 = arith.addi %arg17, %c1_i32 {ssbuffer.block_id = 5 : i32} : i32
        %30 = arith.select %28, %c0_i32, %29 {ssbuffer.block_id = 5 : i32, ssbuffer.dep_mark = [11 : i32]} : i32
        %31 = arith.cmpi eq, %30, %c0_i32 {ssbuffer.block_id = 5 : i32} : i32
        %32:7 = scf.if %31 -> (i32, i32, i32, i32, i32, i32, i32) {
          %245 = arith.addi %arg16, %c96_i32 {ssbuffer.block_id = 5 : i32} : i32
          %246 = arith.divsi %245, %8 {ssbuffer.block_id = 5 : i32} : i32
          %247 = arith.divsi %245, %5 {ssbuffer.block_id = 5 : i32} : i32
          %248 = arith.remsi %247, %3 {ssbuffer.block_id = 5 : i32} : i32
          %249 = arith.remsi %245, %5 {ssbuffer.block_id = 5 : i32} : i32
          %250 = arith.muli %248, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          %251 = arith.muli %249, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          scf.yield {ssbuffer.block_id = 5 : i32} %245, %248, %249, %246, %250, %251, %246 : i32, i32, i32, i32, i32, i32, i32
        } else {
          scf.yield {ssbuffer.block_id = 5 : i32} %arg16, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23 : i32, i32, i32, i32, i32, i32, i32
        } {ssbuffer.block_id = 5 : i32}
        %33 = arith.muli %30, %c64_i32 {ssbuffer.block_id = 5 : i32} : i32
        %34 = arith.extsi %32#6 {ssbuffer.block_id = 5 : i32, ssbuffer.dep_mark = [2 : i32]} : i32 to i64
        %35 = arith.muli %34, %17 {ssbuffer.block_id = 5 : i32} : i64
        %36 = arith.index_cast %35 {ssbuffer.block_id = 5 : i32} : i64 to index
        %37 = arith.index_cast %32#4 {ssbuffer.block_id = 5 : i32, ssbuffer.dep_mark = [3 : i32]} : i32 to index
        %38 = arith.index_cast %arg8 {ssbuffer.block_id = 5 : i32} : i32 to index
        %39 = arith.muli %37, %38 {ssbuffer.block_id = 5 : i32} : index
        %40 = arith.index_cast %33 {ssbuffer.block_id = 5 : i32} : i32 to index
        %41 = arith.addi %36, %39 {ssbuffer.block_id = 5 : i32} : index
        %42 = arith.addi %41, %40 {ssbuffer.block_id = 5 : i32} : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%42], sizes: [1, 128, 64], strides: [8192, %38, 1] {ssbuffer.block_id = 5 : i32} : memref to memref<1x128x64xf16, strided<[8192, ?, 1], offset: ?>>
        %43 = arith.cmpi sge, %34, %c0_i64 {ssbuffer.block_id = 5 : i32} : i64
        %44 = arith.cmpi slt, %34, %9 {ssbuffer.block_id = 5 : i32} : i64
        %45 = arith.andi %43, %44 {ssbuffer.block_id = 5 : i32} : i1
        %alloc_10 = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<1x128x64xf16>
        %46 = arith.index_castui %45 {ssbuffer.block_id = 5 : i32} : i1 to index
        %47 = arith.muli %46, %c128 {ssbuffer.block_id = 5 : i32} : index
        %48 = arith.addi %37, %c128 {ssbuffer.block_id = 5 : i32} : index
        %49 = arith.maxsi %37, %c0 {ssbuffer.block_id = 5 : i32} : index
        %50 = arith.minsi %48, %49 {ssbuffer.block_id = 5 : i32} : index
        %51 = arith.subi %50, %37 {ssbuffer.block_id = 5 : i32} : index
        %52 = arith.subi %48, %50 {ssbuffer.block_id = 5 : i32} : index
        %53 = arith.index_cast %arg6 {ssbuffer.block_id = 5 : i32} : i32 to index
        %54 = arith.maxsi %37, %53 {ssbuffer.block_id = 5 : i32} : index
        %55 = arith.minsi %48, %54 {ssbuffer.block_id = 5 : i32} : index
        %56 = arith.subi %55, %37 {ssbuffer.block_id = 5 : i32} : index
        %57 = arith.maxsi %51, %c0 {ssbuffer.block_id = 5 : i32} : index
        %58 = arith.addi %51, %52 {ssbuffer.block_id = 5 : i32} : index
        %59 = arith.minsi %58, %56 {ssbuffer.block_id = 5 : i32} : index
        %60 = arith.subi %59, %57 {ssbuffer.block_id = 5 : i32} : index
        %61 = arith.maxsi %60, %c0 {ssbuffer.block_id = 5 : i32} : index
        %62 = arith.minsi %46, %c1 {ssbuffer.block_id = 5 : i32} : index
        %63 = arith.maxsi %62, %c0 {ssbuffer.block_id = 5 : i32} : index
        %64 = arith.maxsi %57, %c0 {ssbuffer.block_id = 5 : i32} : index
        %65 = arith.addi %57, %61 {ssbuffer.block_id = 5 : i32} : index
        %66 = arith.minsi %47, %65 {ssbuffer.block_id = 5 : i32} : index
        %67 = arith.subi %66, %64 {ssbuffer.block_id = 5 : i32} : index
        %68 = arith.maxsi %67, %c0 {ssbuffer.block_id = 5 : i32} : index
        %69 = arith.addi %40, %c64 {ssbuffer.block_id = 5 : i32} : index
        %70 = arith.maxsi %40, %c0 {ssbuffer.block_id = 5 : i32} : index
        %71 = arith.minsi %69, %70 {ssbuffer.block_id = 5 : i32} : index
        %72 = arith.subi %71, %40 {ssbuffer.block_id = 5 : i32} : index
        %73 = arith.subi %69, %71 {ssbuffer.block_id = 5 : i32} : index
        %74 = arith.maxsi %40, %38 {ssbuffer.block_id = 5 : i32} : index
        %75 = arith.minsi %69, %74 {ssbuffer.block_id = 5 : i32} : index
        %76 = arith.subi %75, %40 {ssbuffer.block_id = 5 : i32} : index
        %77 = arith.maxsi %72, %c0 {ssbuffer.block_id = 5 : i32} : index
        %78 = arith.addi %72, %73 {ssbuffer.block_id = 5 : i32} : index
        %79 = arith.minsi %78, %76 {ssbuffer.block_id = 5 : i32} : index
        %80 = arith.subi %79, %77 {ssbuffer.block_id = 5 : i32} : index
        %81 = arith.maxsi %80, %c0 {ssbuffer.block_id = 5 : i32} : index
        %82 = arith.minsi %63, %c1 {ssbuffer.block_id = 5 : i32} : index
        %83 = arith.maxsi %82, %c0 {ssbuffer.block_id = 5 : i32, ssbuffer.dep_mark = [9 : i32]} : index
        %84 = arith.maxsi %64, %c0 {ssbuffer.block_id = 5 : i32, ssbuffer.dep_mark = [8 : i32]} : index
        %85 = arith.addi %64, %68 {ssbuffer.block_id = 5 : i32} : index
        %86 = arith.minsi %85, %c128 {ssbuffer.block_id = 5 : i32} : index
        %87 = arith.subi %86, %84 {ssbuffer.block_id = 5 : i32} : index
        %88 = arith.maxsi %87, %c0 {ssbuffer.block_id = 5 : i32, ssbuffer.dep_mark = [10 : i32]} : index
        %89 = arith.maxsi %77, %c0 {ssbuffer.block_id = 5 : i32} : index
        %90 = arith.addi %77, %81 {ssbuffer.block_id = 5 : i32} : index
        %91 = arith.minsi %90, %c64 {ssbuffer.block_id = 5 : i32} : index
        %92 = arith.subi %91, %89 {ssbuffer.block_id = 5 : i32} : index
        %93 = arith.maxsi %92, %c0 {ssbuffer.block_id = 5 : i32} : index
        %94 = arith.cmpi slt, %83, %c1 {ssbuffer.block_id = 5 : i32} : index
        %95 = arith.cmpi slt, %88, %c128 {ssbuffer.block_id = 5 : i32} : index
        %96 = arith.ori %94, %95 {ssbuffer.block_id = 5 : i32} : i1
        %97 = arith.cmpi slt, %93, %c64 {ssbuffer.block_id = 5 : i32} : index
        %98 = arith.ori %96, %97 {ssbuffer.block_id = 5 : i32} : i1
        scf.if %98 {
          linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst_2 : f16) outs(%alloc_10 : memref<1x128x64xf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 5 : i32}
        %subview = memref.subview %reinterpret_cast[0, %84, %89] [%83, %88, %93] [1, 1, 1] {ssbuffer.block_id = 5 : i32} : memref<1x128x64xf16, strided<[8192, ?, 1], offset: ?>> to memref>
        %subview_11 = memref.subview %alloc_10[0, %84, %89] [%83, %88, %93] [1, 1, 1] {ssbuffer.block_id = 5 : i32} : memref<1x128x64xf16> to memref>
        memref.copy %subview, %subview_11 {ssbuffer.block_id = 5 : i32} : memref> to memref>
        %99 = bufferization.to_tensor %alloc_10 restrict writable {ssbuffer.block_id = 5 : i32} : memref<1x128x64xf16>
        %100 = arith.muli %34, %19 {ssbuffer.block_id = 5 : i32} : i64
        %101 = arith.index_cast %100 {ssbuffer.block_id = 5 : i32} : i64 to index
        %102 = arith.index_cast %32#5 {ssbuffer.block_id = 5 : i32, ssbuffer.dep_mark = [5 : i32]} : i32 to index
        %103 = arith.muli %102, %38 {ssbuffer.block_id = 5 : i32} : index
        %104 = arith.addi %101, %103 {ssbuffer.block_id = 5 : i32} : index
        %105 = arith.addi %104, %40 {ssbuffer.block_id = 5 : i32} : index
        %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%105], sizes: [1, 128, 64], strides: [8192, %38, 1] {ssbuffer.block_id = 5 : i32} : memref to memref<1x128x64xf16, strided<[8192, ?, 1], offset: ?>>
        %alloc_13 = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<1x128x64xf16>
        %106 = arith.addi %102, %c128 {ssbuffer.block_id = 5 : i32} : index
        %107 = arith.maxsi %102, %c0 {ssbuffer.block_id = 5 : i32} : index
        %108 = arith.minsi %106, %107 {ssbuffer.block_id = 5 : i32} : index
        %109 = arith.subi %108, %102 {ssbuffer.block_id = 5 : i32} : index
        %110 = arith.subi %106, %108 {ssbuffer.block_id = 5 : i32} : index
        %111 = arith.index_cast %arg7 {ssbuffer.block_id = 5 : i32, ssbuffer.dep_mark = [4 : i32]} : i32 to index
        %112 = arith.maxsi %102, %111 {ssbuffer.block_id = 5 : i32} : index
        %113 = arith.minsi %106, %112 {ssbuffer.block_id = 5 : i32} : index
        %114 = arith.subi %113, %102 {ssbuffer.block_id = 5 : i32} : index
        %115 = arith.maxsi %109, %c0 {ssbuffer.block_id = 5 : i32} : index
        %116 = arith.addi %109, %110 {ssbuffer.block_id = 5 : i32} : index
        %117 = arith.minsi %116, %114 {ssbuffer.block_id = 5 : i32} : index
        %118 = arith.subi %117, %115 {ssbuffer.block_id = 5 : i32} : index
        %119 = arith.maxsi %118, %c0 {ssbuffer.block_id = 5 : i32} : index
        %120 = arith.maxsi %115, %c0 {ssbuffer.block_id = 5 : i32, ssbuffer.dep_mark = [7 : i32]} : index
        %121 = arith.addi %115, %119 {ssbuffer.block_id = 5 : i32, ssbuffer.dep_mark = [6 : i32]} : index
        %122 = arith.minsi %47, %121 {ssbuffer.block_id = 5 : i32} : index
        %123 = arith.subi %122, %120 {ssbuffer.block_id = 5 : i32} : index
        %124 = arith.maxsi %123, %c0 {ssbuffer.block_id = 5 : i32} : index
        %125 = arith.maxsi %120, %c0 {ssbuffer.block_id = 5 : i32} : index
        %126 = arith.addi %120, %124 {ssbuffer.block_id = 5 : i32} : index
        %127 = arith.minsi %126, %c128 {ssbuffer.block_id = 5 : i32} : index
        %128 = arith.subi %127, %125 {ssbuffer.block_id = 5 : i32} : index
        %129 = arith.maxsi %128, %c0 {ssbuffer.block_id = 5 : i32} : index
        %130 = arith.cmpi slt, %129, %c128 {ssbuffer.block_id = 5 : i32} : index
        %131 = arith.ori %94, %130 {ssbuffer.block_id = 5 : i32} : i1
        %132 = arith.ori %131, %97 {ssbuffer.block_id = 5 : i32} : i1
        scf.if %132 {
          linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst_2 : f16) outs(%alloc_13 : memref<1x128x64xf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 5 : i32}
        %subview_14 = memref.subview %reinterpret_cast_12[0, %125, %89] [%83, %129, %93] [1, 1, 1] {ssbuffer.block_id = 5 : i32} : memref<1x128x64xf16, strided<[8192, ?, 1], offset: ?>> to memref>
        %subview_15 = memref.subview %alloc_13[0, %125, %89] [%83, %129, %93] [1, 1, 1] {ssbuffer.block_id = 5 : i32} : memref<1x128x64xf16> to memref>
        memref.copy %subview_14, %subview_15 {ssbuffer.block_id = 5 : i32} : memref> to memref>
        %133 = bufferization.to_tensor %alloc_13 restrict writable {ssbuffer.block_id = 5 : i32} : memref<1x128x64xf16>
        %reshape = tensor.reshape %133(%cst_0) {ssbuffer.block_id = 5 : i32} : (tensor<1x128x64xf16>, tensor<3xi64>) -> tensor<128x4x16xf16>
        %134 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<4x128x16xf16>
        %transposed = linalg.transpose ins(%reshape : tensor<128x4x16xf16>) outs(%134 : tensor<4x128x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 5 : i32}
        %reshape_16 = tensor.reshape %transposed(%cst) {ssbuffer.block_id = 5 : i32} : (tensor<4x128x16xf16>, tensor<4xi64>) -> tensor<4x8x16x16xf16>
        %reshape_17 = tensor.reshape %99(%cst_0) {ssbuffer.block_id = 5 : i32} : (tensor<1x128x64xf16>, tensor<3xi64>) -> tensor<128x4x16xf16>
        %135 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<4x128x16xf16>
        %transposed_18 = linalg.transpose ins(%reshape_17 : tensor<128x4x16xf16>) outs(%135 : tensor<4x128x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 5 : i32}
        %reshape_19 = tensor.reshape %transposed_18(%cst) {ssbuffer.block_id = 5 : i32} : (tensor<4x128x16xf16>, tensor<4xi64>) -> tensor<4x8x16x16xf16>
        scf.if %27 {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
        } else {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 3
        } {ssbuffer.block_id = 5 : i32, ssbuffer.cross_buffer = 1 : i32}
        scf.if %27 {
          hivm.hir.copy ins(%reshape_16 : tensor<4x8x16x16xf16>) outs(%alloc : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32], ssbuffer.transfer_id = 0 : i32}
        } else {
          hivm.hir.copy ins(%reshape_16 : tensor<4x8x16x16xf16>) outs(%alloc_4 : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32], ssbuffer.transfer_id = 0 : i32}
        } {ssbuffer.block_id = 5 : i32, ssbuffer.cross_buffer = 1 : i32, ssbuffer.transfer_id = 0 : i32}
        scf.if %27 {
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
        } else {
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
        } {ssbuffer.block_id = 5 : i32, ssbuffer.cross_buffer = 1 : i32}
        scf.if %27 {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
        } else {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 4
        } {ssbuffer.block_id = 5 : i32, ssbuffer.cross_buffer = 1 : i32}
        scf.if %27 {
          hivm.hir.copy ins(%reshape_19 : tensor<4x8x16x16xf16>) outs(%alloc_5 : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32], ssbuffer.transfer_id = 1 : i32}
        } else {
          hivm.hir.copy ins(%reshape_19 : tensor<4x8x16x16xf16>) outs(%alloc_6 : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32], ssbuffer.transfer_id = 1 : i32}
        } {ssbuffer.block_id = 5 : i32, ssbuffer.cross_buffer = 1 : i32, ssbuffer.transfer_id = 1 : i32}
        scf.if %27 {
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        } else {
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 4
        } {ssbuffer.block_id = 5 : i32, ssbuffer.cross_buffer = 1 : i32}
        %136 = arith.cmpi eq, %arg31, %23 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 6 : i32, ssbuffer.clone = 5 : i32} : i32
        %137 = arith.addi %arg31, %c1_i32 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 6 : i32, ssbuffer.clone = 5 : i32} : i32
        %138 = arith.select %136, %c0_i32, %137 {ssbuffer.block_id = 6 : i32, ssbuffer.clone = 5 : i32, ssbuffer.dep_mark = [11 : i32]} : i32
        %139 = arith.cmpi eq, %138, %23 {ssbuffer.block_id = 6 : i32, ssbuffer.dep_mark = [1 : i32, 11 : i32]} : i32
        %140 = arith.cmpi eq, %arg31, %23 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 6 : i32} : i32
        %141 = arith.addi %arg31, %c1_i32 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 6 : i32} : i32
        %142 = arith.select %140, %c0_i32, %141 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 6 : i32, ssbuffer.dep_mark = [11 : i32]} : i32
        %143 = arith.divsi %arg15, %c1_i32 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32, ssbuffer.transfer_id = 0 : i32} : i32
        %c2_i32_20 = arith.constant {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32, ssbuffer.transfer_id = 0 : i32} 2 : i32
        %144 = arith.remsi %143, %c2_i32_20 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32, ssbuffer.transfer_id = 0 : i32} : i32
        %c0_i32_21 = arith.constant {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32, ssbuffer.transfer_id = 0 : i32} 0 : i32
        %145 = arith.cmpi eq, %144, %c0_i32_21 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32, ssbuffer.transfer_id = 0 : i32} : i32
        %146 = arith.cmpi eq, %arg30, %23 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : i32
        %147 = arith.addi %arg30, %c1_i32 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : i32
        %148 = arith.select %146, %c0_i32, %147 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32, ssbuffer.dep_mark = [11 : i32]} : i32
        %149 = arith.cmpi eq, %148, %c0_i32 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : i32
        %150:7 = scf.if %149 -> (i32, i32, i32, i32, i32, i32, i32) {
          %245 = arith.addi %arg25, %c96_i32 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 5 : i32} : i32
          %246 = arith.divsi %245, %8 {ssbuffer.block_id = 5 : i32} : i32
          %247 = arith.divsi %245, %5 {ssbuffer.block_id = 5 : i32} : i32
          %248 = arith.remsi %247, %3 {ssbuffer.block_id = 5 : i32} : i32
          %249 = arith.remsi %245, %5 {ssbuffer.block_id = 5 : i32} : i32
          %250 = arith.muli %248, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          %251 = arith.muli %249, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          scf.yield {ssbuffer.block_id = 5 : i32} %245, %248, %249, %246, %250, %251, %246 : i32, i32, i32, i32, i32, i32, i32
        } else {
          scf.yield {ssbuffer.arg = 5 : i32, ssbuffer.block_id = 5 : i32} %arg25, %arg27, %arg32, %arg28, %arg33, %arg29, %arg26 : i32, i32, i32, i32, i32, i32, i32
        } {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32}
        %151 = arith.extsi %150#6 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32, ssbuffer.dep_mark = [2 : i32]} : i32 to i64
        %152 = arith.index_cast %150#4 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32, ssbuffer.dep_mark = [3 : i32]} : i32 to index
        %153 = arith.cmpi sge, %151, %c0_i64 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : i64
        %154 = arith.cmpi slt, %151, %9 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : i64
        %155 = arith.andi %153, %154 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : i1
        %156 = arith.index_castui %155 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : i1 to index
        %157 = arith.muli %156, %c128 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %158 = arith.addi %152, %c128 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %159 = arith.maxsi %152, %c0 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %160 = arith.minsi %158, %159 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %161 = arith.subi %160, %152 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %162 = arith.subi %158, %160 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %163 = arith.index_cast %arg6 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : i32 to index
        %164 = arith.maxsi %152, %163 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %165 = arith.minsi %158, %164 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %166 = arith.subi %165, %152 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %167 = arith.maxsi %161, %c0 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %168 = arith.addi %161, %162 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %169 = arith.minsi %168, %166 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %170 = arith.subi %169, %167 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %171 = arith.maxsi %170, %c0 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %172 = arith.minsi %156, %c1 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %173 = arith.maxsi %172, %c0 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %174 = arith.maxsi %167, %c0 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %175 = arith.addi %167, %171 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %176 = arith.minsi %157, %175 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %177 = arith.subi %176, %174 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %178 = arith.maxsi %177, %c0 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %179 = arith.minsi %173, %c1 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %180 = arith.maxsi %179, %c0 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32, ssbuffer.dep_mark = [9 : i32]} : index
        %181 = arith.maxsi %174, %c0 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32, ssbuffer.dep_mark = [8 : i32]} : index
        %182 = arith.addi %174, %178 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %183 = arith.minsi %182, %c128 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %184 = arith.subi %183, %181 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %185 = arith.maxsi %184, %c0 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32, ssbuffer.dep_mark = [10 : i32]} : index
        %186 = arith.index_cast %150#5 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32, ssbuffer.dep_mark = [5 : i32]} : i32 to index
        %187 = arith.addi %186, %c128 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %188 = arith.maxsi %186, %c0 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %189 = arith.minsi %187, %188 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %190 = arith.subi %189, %186 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %191 = arith.subi %187, %189 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %192 = arith.index_cast %arg7 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32, ssbuffer.dep_mark = [4 : i32]} : i32 to index
        %193 = arith.maxsi %186, %192 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %194 = arith.minsi %187, %193 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %195 = arith.subi %194, %186 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %196 = arith.maxsi %190, %c0 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %197 = arith.addi %190, %191 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %198 = arith.minsi %197, %195 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %199 = arith.subi %198, %196 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %200 = arith.maxsi %199, %c0 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32} : index
        %201 = arith.maxsi %196, %c0 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32, ssbuffer.dep_mark = [7 : i32]} : index
        %202 = arith.addi %196, %200 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 5 : i32, ssbuffer.dep_mark = [6 : i32]} : index
        %203 = arith.cmpi eq, %148, %23 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 6 : i32, ssbuffer.dep_mark = [1 : i32, 11 : i32]} : i32
        scf.if %145 {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
        } else {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 5
        } {ssbuffer.block_id = 7 : i32, ssbuffer.cross_buffer = 1 : i32}
        %204 = scf.if %145 -> (tensor<128x128xf32>) {
          %memspacecast = memref.memory_space_cast %alloc_7 {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
          %245 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32>
          scf.yield %245 : tensor<128x128xf32>
        } else {
          %memspacecast = memref.memory_space_cast %alloc_8 {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
          %245 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32>
          scf.yield %245 : tensor<128x128xf32>
        } {ssbuffer.block_id = 7 : i32, ssbuffer.crossCoreDeps = [2 : i32, 0 : i32], ssbuffer.cross_buffer = 1 : i32, ssbuffer.transfer_id = 2 : i32}
        %205 = arith.addf %204, %arg24 {ssbuffer.add_from_matmul, ssbuffer.block_id = 7 : i32} : tensor<128x128xf32>
        %206 = arith.select %203, %1, %205 {ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = [1 : i32]} : tensor<128x128xf32>
        scf.if %203 {
          %245 = arith.truncf %205 {ssbuffer.block_id = 7 : i32} : tensor<128x128xf32> to tensor<128x128xf16>
          %reshape_22 = tensor.reshape %245(%cst_3) {ssbuffer.block_id = 7 : i32} : (tensor<128x128xf16>, tensor<3xi64>) -> tensor<1x128x128xf16>
          %246 = arith.muli %151, %21 {ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = [2 : i32]} : i64
          %247 = arith.index_cast %246 {ssbuffer.block_id = 7 : i32} : i64 to index
          %248 = arith.muli %152, %192 {ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = [3 : i32, 4 : i32]} : index
          %249 = arith.addi %247, %248 {ssbuffer.block_id = 7 : i32} : index
          %250 = arith.addi %249, %186 {ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = [5 : i32]} : index
          %reinterpret_cast_23 = memref.reinterpret_cast %arg4 to offset: [%250], sizes: [1, 128, 128], strides: [16384, %192, 1] {ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = [4 : i32]} : memref to memref<1x128x128xf16, strided<[16384, ?, 1], offset: ?>>
          %251 = arith.minsi %202, %c128 {ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = [6 : i32]} : index
          %252 = arith.subi %251, %201 {ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = [7 : i32]} : index
          %253 = arith.maxsi %252, %c0 {ssbuffer.block_id = 7 : i32} : index
          %extracted_slice = tensor.extract_slice %reshape_22[0, %181, %201] [%180, %185, %253] [1, 1, 1] {ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = [7 : i32, 8 : i32, 9 : i32, 10 : i32]} : tensor<1x128x128xf16> to tensor
          %subview_24 = memref.subview %reinterpret_cast_23[0, %181, %201] [%180, %185, %253] [1, 1, 1] {ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = [7 : i32, 8 : i32, 9 : i32, 10 : i32]} : memref<1x128x128xf16, strided<[16384, ?, 1], offset: ?>> to memref>
          bufferization.materialize_in_destination %extracted_slice in writable %subview_24 {ssbuffer.block_id = 7 : i32} : (tensor, memref>) -> ()
        } {ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = [1 : i32]}
        scf.if %145 {
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
        } else {
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
        } {ssbuffer.block_id = 7 : i32, ssbuffer.cross_buffer = 1 : i32}
        %207 = arith.cmpi eq, %arg30, %23 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 7 : i32} : i32
        %208 = arith.addi %arg30, %c1_i32 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 7 : i32} : i32
        %209 = arith.select %207, %c0_i32, %208 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = [11 : i32]} : i32
        %210 = arith.cmpi eq, %209, %c0_i32 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 7 : i32} : i32
        %211:7 = scf.if %210 -> (i32, i32, i32, i32, i32, i32, i32) {
          %245 = arith.addi %arg25, %c96_i32 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 5 : i32} : i32
          %246 = arith.divsi %245, %8 {ssbuffer.block_id = 5 : i32} : i32
          %247 = arith.divsi %245, %5 {ssbuffer.block_id = 5 : i32} : i32
          %248 = arith.remsi %247, %3 {ssbuffer.block_id = 5 : i32} : i32
          %249 = arith.remsi %245, %5 {ssbuffer.block_id = 5 : i32} : i32
          %250 = arith.muli %248, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          %251 = arith.muli %249, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          scf.yield {ssbuffer.block_id = 5 : i32} %245, %248, %249, %246, %250, %251, %246 : i32, i32, i32, i32, i32, i32, i32
        } else {
          scf.yield {ssbuffer.arg = 5 : i32, ssbuffer.block_id = 5 : i32} %arg25, %arg27, %arg32, %arg28, %arg33, %arg29, %arg26 : i32, i32, i32, i32, i32, i32, i32
        } {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 7 : i32}
        %212 = arith.cmpi eq, %arg30, %23 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 7 : i32} : i32
        %213 = arith.addi %arg30, %c1_i32 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 7 : i32} : i32
        %214 = arith.select %212, %c0_i32, %213 {ssbuffer.arg = 7 : i32, ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = [11 : i32]} : i32
        %215 = arith.cmpi eq, %214, %c0_i32 {ssbuffer.arg = 7 : i32, ssbuffer.block_id = 7 : i32} : i32
        %216:7 = scf.if %215 -> (i32, i32, i32, i32, i32, i32, i32) {
          %245 = arith.addi %arg16, %c96_i32 {ssbuffer.block_id = 5 : i32} : i32
          %246 = arith.divsi %245, %8 {ssbuffer.block_id = 5 : i32} : i32
          %247 = arith.divsi %245, %5 {ssbuffer.block_id = 5 : i32} : i32
          %248 = arith.remsi %247, %3 {ssbuffer.block_id = 5 : i32} : i32
          %249 = arith.remsi %245, %5 {ssbuffer.block_id = 5 : i32} : i32
          %250 = arith.muli %248, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          %251 = arith.muli %249, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          scf.yield {ssbuffer.block_id = 5 : i32} %245, %248, %249, %246, %250, %251, %246 : i32, i32, i32, i32, i32, i32, i32
        } else {
          scf.yield {ssbuffer.arg = 5 : i32, ssbuffer.block_id = 5 : i32} %arg16, %arg27, %arg32, %arg28, %arg33, %arg29, %arg26 : i32, i32, i32, i32, i32, i32, i32
        } {ssbuffer.arg = 7 : i32, ssbuffer.block_id = 7 : i32}
        %217 = arith.cmpi eq, %arg30, %23 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 7 : i32} : i32
        %218 = arith.addi %arg30, %c1_i32 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 7 : i32} : i32
        %219 = arith.select %217, %c0_i32, %218 {ssbuffer.arg = 2 : i32, ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = [11 : i32]} : i32
        %220 = arith.cmpi eq, %219, %c0_i32 {ssbuffer.arg = 2 : i32, ssbuffer.block_id = 7 : i32} : i32
        %221:7 = scf.if %220 -> (i32, i32, i32, i32, i32, i32, i32) {
          %245 = arith.addi %arg16, %c96_i32 {ssbuffer.block_id = 5 : i32} : i32
          %246 = arith.divsi %245, %8 {ssbuffer.block_id = 5 : i32} : i32
          %247 = arith.divsi %245, %5 {ssbuffer.block_id = 5 : i32} : i32
          %248 = arith.remsi %247, %3 {ssbuffer.block_id = 5 : i32} : i32
          %249 = arith.remsi %245, %5 {ssbuffer.block_id = 5 : i32} : i32
          %250 = arith.muli %248, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          %251 = arith.muli %249, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          scf.yield {ssbuffer.block_id = 5 : i32} %245, %248, %249, %246, %250, %251, %246 : i32, i32, i32, i32, i32, i32, i32
        } else {
          scf.yield {ssbuffer.arg = 5 : i32, ssbuffer.block_id = 5 : i32} %arg16, %arg27, %arg32, %arg28, %arg33, %arg29, %arg23 : i32, i32, i32, i32, i32, i32, i32
        } {ssbuffer.arg = 2 : i32, ssbuffer.block_id = 7 : i32}
        %222 = arith.cmpi eq, %arg30, %23 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 7 : i32} : i32
        %223 = arith.addi %arg30, %c1_i32 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 7 : i32} : i32
        %224 = arith.select %222, %c0_i32, %223 {ssbuffer.arg = 4 : i32, ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = [11 : i32]} : i32
        %225 = arith.cmpi eq, %224, %c0_i32 {ssbuffer.arg = 4 : i32, ssbuffer.block_id = 7 : i32} : i32
        %226:7 = scf.if %225 -> (i32, i32, i32, i32, i32, i32, i32) {
          %245 = arith.addi %arg16, %c96_i32 {ssbuffer.block_id = 5 : i32} : i32
          %246 = arith.divsi %245, %8 {ssbuffer.block_id = 5 : i32} : i32
          %247 = arith.divsi %245, %5 {ssbuffer.block_id = 5 : i32} : i32
          %248 = arith.remsi %247, %3 {ssbuffer.block_id = 5 : i32} : i32
          %249 = arith.remsi %245, %5 {ssbuffer.block_id = 5 : i32} : i32
          %250 = arith.muli %248, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          %251 = arith.muli %249, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          scf.yield {ssbuffer.block_id = 5 : i32} %245, %248, %249, %246, %250, %251, %246 : i32, i32, i32, i32, i32, i32, i32
        } else {
          scf.yield {ssbuffer.arg = 5 : i32, ssbuffer.block_id = 5 : i32} %arg16, %arg18, %arg32, %arg28, %arg33, %arg29, %arg23 : i32, i32, i32, i32, i32, i32, i32
        } {ssbuffer.arg = 4 : i32, ssbuffer.block_id = 7 : i32}
        %227 = arith.cmpi eq, %arg30, %23 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 7 : i32} : i32
        %228 = arith.addi %arg30, %c1_i32 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 7 : i32} : i32
        %229 = arith.select %227, %c0_i32, %228 {ssbuffer.arg = 6 : i32, ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = [11 : i32]} : i32
        %230 = arith.cmpi eq, %229, %c0_i32 {ssbuffer.arg = 6 : i32, ssbuffer.block_id = 7 : i32} : i32
        %231:7 = scf.if %230 -> (i32, i32, i32, i32, i32, i32, i32) {
          %245 = arith.addi %arg16, %c96_i32 {ssbuffer.block_id = 5 : i32} : i32
          %246 = arith.divsi %245, %8 {ssbuffer.block_id = 5 : i32} : i32
          %247 = arith.divsi %245, %5 {ssbuffer.block_id = 5 : i32} : i32
          %248 = arith.remsi %247, %3 {ssbuffer.block_id = 5 : i32} : i32
          %249 = arith.remsi %245, %5 {ssbuffer.block_id = 5 : i32} : i32
          %250 = arith.muli %248, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          %251 = arith.muli %249, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          scf.yield {ssbuffer.block_id = 5 : i32} %245, %248, %249, %246, %250, %251, %246 : i32, i32, i32, i32, i32, i32, i32
        } else {
          scf.yield {ssbuffer.arg = 5 : i32, ssbuffer.block_id = 5 : i32} %arg16, %arg18, %arg32, %arg20, %arg33, %arg29, %arg23 : i32, i32, i32, i32, i32, i32, i32
        } {ssbuffer.arg = 6 : i32, ssbuffer.block_id = 7 : i32}
        %232 = arith.cmpi eq, %arg30, %23 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 7 : i32} : i32
        %233 = arith.addi %arg30, %c1_i32 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 7 : i32} : i32
        %234 = arith.select %232, %c0_i32, %233 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = [11 : i32]} : i32
        %235 = arith.cmpi eq, %arg17, %23 {ssbuffer.arg = 3 : i32, ssbuffer.block_id = 7 : i32} : i32
        %236 = arith.addi %arg17, %c1_i32 {ssbuffer.arg = 3 : i32, ssbuffer.block_id = 7 : i32} : i32
        %237 = arith.select %235, %c0_i32, %236 {ssbuffer.arg = 3 : i32, ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = [11 : i32]} : i32
        %238 = arith.cmpi eq, %237, %c0_i32 {ssbuffer.arg = 3 : i32, ssbuffer.block_id = 7 : i32} : i32
        %239:7 = scf.if %238 -> (i32, i32, i32, i32, i32, i32, i32) {
          %245 = arith.addi %arg16, %c96_i32 {ssbuffer.block_id = 5 : i32} : i32
          %246 = arith.divsi %245, %8 {ssbuffer.block_id = 5 : i32} : i32
          %247 = arith.divsi %245, %5 {ssbuffer.block_id = 5 : i32} : i32
          %248 = arith.remsi %247, %3 {ssbuffer.block_id = 5 : i32} : i32
          %249 = arith.remsi %245, %5 {ssbuffer.block_id = 5 : i32} : i32
          %250 = arith.muli %248, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          %251 = arith.muli %249, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          scf.yield {ssbuffer.block_id = 5 : i32} %245, %248, %249, %246, %250, %251, %246 : i32, i32, i32, i32, i32, i32, i32
        } else {
          scf.yield {ssbuffer.arg = 5 : i32, ssbuffer.block_id = 5 : i32} %arg16, %arg18, %arg32, %arg20, %arg33, %arg22, %arg23 : i32, i32, i32, i32, i32, i32, i32
        } {ssbuffer.arg = 3 : i32, ssbuffer.block_id = 7 : i32}
        %240 = arith.cmpi eq, %arg17, %23 {ssbuffer.arg = 5 : i32, ssbuffer.block_id = 7 : i32} : i32
        %241 = arith.addi %arg17, %c1_i32 {ssbuffer.arg = 5 : i32, ssbuffer.block_id = 7 : i32} : i32
        %242 = arith.select %240, %c0_i32, %241 {ssbuffer.arg = 5 : i32, ssbuffer.block_id = 7 : i32, ssbuffer.dep_mark = [11 : i32]} : i32
        %243 = arith.cmpi eq, %242, %c0_i32 {ssbuffer.arg = 5 : i32, ssbuffer.block_id = 7 : i32} : i32
        %244:7 = scf.if %243 -> (i32, i32, i32, i32, i32, i32, i32) {
          %245 = arith.addi %arg16, %c96_i32 {ssbuffer.block_id = 5 : i32} : i32
          %246 = arith.divsi %245, %8 {ssbuffer.block_id = 5 : i32} : i32
          %247 = arith.divsi %245, %5 {ssbuffer.block_id = 5 : i32} : i32
          %248 = arith.remsi %247, %3 {ssbuffer.block_id = 5 : i32} : i32
          %249 = arith.remsi %245, %5 {ssbuffer.block_id = 5 : i32} : i32
          %250 = arith.muli %248, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          %251 = arith.muli %249, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          scf.yield {ssbuffer.block_id = 5 : i32} %245, %248, %249, %246, %250, %251, %246 : i32, i32, i32, i32, i32, i32, i32
        } else {
          scf.yield {ssbuffer.arg = 5 : i32, ssbuffer.block_id = 5 : i32} %arg16, %arg18, %arg19, %arg20, %arg33, %arg22, %arg23 : i32, i32, i32, i32, i32, i32, i32
        } {ssbuffer.arg = 5 : i32, ssbuffer.block_id = 7 : i32}
        scf.yield %32#0, %30, %32#1, %32#2, %32#3, %32#4, %32#5, %32#6, %206, %211#0, %216#6, %221#1, %226#3, %231#5, %234, %142, %239#2, %244#4 : i32, i32, i32, i32, i32, i32, i32, i32, tensor<128x128xf32>, i32, i32, i32, i32, i32, i32, i32, i32, i32
      } {ssbuffer.block_id = 8 : i32, ssbuffer.loop_depth = 1 : i32, ssbuffer.main_loop = 0 : i32}
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 4
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 3
      scope.return
    } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %0 = arith.addi %arg6, %c127_i32 {ssbuffer.block_id = 4 : i32} : i32
      %1 = arith.divsi %0, %c128_i32 {ssbuffer.block_id = 4 : i32} : i32
      %2 = arith.addi %arg7, %c127_i32 {ssbuffer.block_id = 4 : i32} : i32
      %3 = arith.divsi %2, %c128_i32 {ssbuffer.block_id = 4 : i32} : i32
      %4 = arith.addi %arg8, %c63_i32 {ssbuffer.block_id = 4 : i32} : i32
      %5 = arith.divsi %4, %c64_i32 {ssbuffer.block_id = 4 : i32} : i32
      %6 = arith.muli %1, %3 {ssbuffer.block_id = 4 : i32} : i32
      %7 = arith.muli %arg5, %6 {ssbuffer.block_id = 4 : i32} : i32
      %8 = arith.divsi %7, %c96_i32 {ssbuffer.block_id = 4 : i32} : i32
      %9 = arith.remsi %7, %c96_i32 {ssbuffer.block_id = 4 : i32} : i32
      %10 = arith.cmpi slt, %arg12, %9 {ssbuffer.block_id = 4 : i32} : i32
      %11 = scf.if %10 -> (i32) {
        %15 = arith.addi %8, %c1_i32 {ssbuffer.block_id = 4 : i32} : i32
        scf.yield {ssbuffer.block_id = 4 : i32} %15 : i32
      } else {
        scf.yield {ssbuffer.block_id = 4 : i32} %8 : i32
      } {ssbuffer.block_id = 4 : i32}
      %12 = arith.muli %5, %11 {ssbuffer.block_id = 4 : i32} : i32
      %13 = arith.subi %5, %c1_i32 {ssbuffer.block_id = 4 : i32} : i32
      %alloc = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_4 = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 3
      %alloc_5 = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_6 = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 4
      %alloc_7 = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %alloc_8 = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      %14 = scf.for %arg15 = %c0_i32 to %12 step %c1_i32 iter_args(%arg16 = %c-1_i32) -> (i32)  : i32 {
        %15 = arith.divsi %arg15, %c1_i32 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32} : i32
        %c2_i32 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32} 2 : i32
        %16 = arith.remsi %15, %c2_i32 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32} : i32
        %c0_i32_9 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32} 0 : i32
        %17 = arith.cmpi eq, %16, %c0_i32_9 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32} : i32
        %18 = arith.cmpi eq, %arg16, %13 {ssbuffer.block_id = 5 : i32} : i32
        %19 = arith.addi %arg16, %c1_i32 {ssbuffer.block_id = 5 : i32} : i32
        %20 = arith.select %18, %c0_i32, %19 {ssbuffer.block_id = 5 : i32} : i32
        %21 = arith.divsi %arg15, %c1_i32 {ssbuffer.block_id = 3 : i32, ssbuffer.clone = 5 : i32, ssbuffer.transfer_id = 0 : i32} : i32
        %c2_i32_10 = arith.constant {ssbuffer.block_id = 3 : i32, ssbuffer.clone = 5 : i32, ssbuffer.transfer_id = 0 : i32} 2 : i32
        %22 = arith.remsi %21, %c2_i32_10 {ssbuffer.block_id = 3 : i32, ssbuffer.clone = 5 : i32, ssbuffer.transfer_id = 0 : i32} : i32
        %c0_i32_11 = arith.constant {ssbuffer.block_id = 3 : i32, ssbuffer.clone = 5 : i32, ssbuffer.transfer_id = 0 : i32} 0 : i32
        %23 = arith.cmpi eq, %22, %c0_i32_11 {ssbuffer.block_id = 3 : i32, ssbuffer.clone = 5 : i32, ssbuffer.transfer_id = 0 : i32} : i32
        %24 = tensor.empty() {ssbuffer.block_id = 3 : i32} : tensor<64x128xf16>
        scf.if %23 {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
        } else {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
        } {ssbuffer.block_id = 3 : i32, ssbuffer.cross_buffer = 1 : i32}
        %25 = scf.if %23 -> (tensor<128x64xf16>) {
          %30 = hivm.hir.convert_layout %alloc output_shape [128, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 0 : i32} : (memref<4x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x64xf16, #hivm.address_space<cbuf>>
          %memspacecast = memref.memory_space_cast %30 {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x64xf16, #hivm.address_space<cbuf>> to memref<128x64xf16>
          %31 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x64xf16>
          scf.yield %31 : tensor<128x64xf16>
        } else {
          %30 = hivm.hir.convert_layout %alloc_4 output_shape [128, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 0 : i32} : (memref<4x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x64xf16, #hivm.address_space<cbuf>>
          %memspacecast = memref.memory_space_cast %30 {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x64xf16, #hivm.address_space<cbuf>> to memref<128x64xf16>
          %31 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x64xf16>
          scf.yield %31 : tensor<128x64xf16>
        } {ssbuffer.block_id = 3 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32], ssbuffer.cross_buffer = 1 : i32, ssbuffer.transfer_id = 0 : i32}
        %transposed = linalg.transpose ins(%25 : tensor<128x64xf16>) outs(%24 : tensor<64x128xf16>) permutation = [1, 0]  {ssbuffer.block_id = 3 : i32}
        %26 = tensor.empty() {ssbuffer.block_id = 3 : i32} : tensor<128x128xf32>
        %27 = linalg.fill {ssbuffer.block_id = 3 : i32} ins(%cst_1 : f32) outs(%26 : tensor<128x128xf32>) -> tensor<128x128xf32>
        scf.if %23 {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        } else {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 4
        } {ssbuffer.block_id = 3 : i32, ssbuffer.cross_buffer = 1 : i32}
        %28 = scf.if %23 -> (tensor<128x64xf16>) {
          %30 = hivm.hir.convert_layout %alloc_5 output_shape [128, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 1 : i32} : (memref<4x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x64xf16, #hivm.address_space<cbuf>>
          %memspacecast = memref.memory_space_cast %30 {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x64xf16, #hivm.address_space<cbuf>> to memref<128x64xf16>
          %31 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x64xf16>
          scf.yield %31 : tensor<128x64xf16>
        } else {
          %30 = hivm.hir.convert_layout %alloc_6 output_shape [128, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 1 : i32} : (memref<4x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x64xf16, #hivm.address_space<cbuf>>
          %memspacecast = memref.memory_space_cast %30 {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x64xf16, #hivm.address_space<cbuf>> to memref<128x64xf16>
          %31 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x64xf16>
          scf.yield %31 : tensor<128x64xf16>
        } {ssbuffer.block_id = 3 : i32, ssbuffer.crossCoreDeps = [1 : i32, 0 : i32], ssbuffer.cross_buffer = 1 : i32, ssbuffer.transfer_id = 1 : i32}
        %29 = linalg.matmul {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], input_precision = "ieee", ssbuffer.block_id = 3 : i32, ssbuffer.loop_carried_l0c} ins(%28, %transposed : tensor<128x64xf16>, tensor<64x128xf16>) outs(%27 : tensor<128x128xf32>) -> tensor<128x128xf32>
        scf.if %23 {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
        } else {
          hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
        } {ssbuffer.block_id = 3 : i32, ssbuffer.cross_buffer = 1 : i32}
        scf.if %23 {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 3 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32], ssbuffer.transfer_id = 2 : i32} ins(%29 : tensor<128x128xf32>) outs(%alloc_7 : memref<128x128xf32, #hivm.address_space<ub>>)
        } else {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 3 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32], ssbuffer.transfer_id = 2 : i32} ins(%29 : tensor<128x128xf32>) outs(%alloc_8 : memref<128x128xf32, #hivm.address_space<ub>>)
        } {ssbuffer.block_id = 3 : i32, ssbuffer.cross_buffer = 1 : i32, ssbuffer.transfer_id = 2 : i32}
        scf.if %23 {
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
        } else {
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 5
        } {ssbuffer.block_id = 3 : i32, ssbuffer.cross_buffer = 1 : i32}
        scf.if %23 {
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
        } else {
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 3
        } {ssbuffer.block_id = 3 : i32, ssbuffer.cross_buffer = 1 : i32}
        scf.if %23 {
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 1
        } else {
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 4
        } {ssbuffer.block_id = 3 : i32, ssbuffer.cross_buffer = 1 : i32}
        scf.yield %20 : i32
      } {ssbuffer.block_id = 8 : i32, ssbuffer.loop_depth = 1 : i32, ssbuffer.main_loop = 0 : i32}
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
      scope.return
    } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}