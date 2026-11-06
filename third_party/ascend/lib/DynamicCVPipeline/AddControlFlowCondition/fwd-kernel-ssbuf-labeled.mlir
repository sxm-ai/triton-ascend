// -----// IR Dump Before CreatePreload (create-preload) //----- //
#map = affine_map<(d0) -> (d0, 0)>
#map1 = affine_map<()[s0] -> (s0 + 1)>
#map2 = affine_map<()[s0, s1, s2, s3] -> (s0 * s1 + s2 + s3)>
#map3 = affine_map<()[s0, s1] -> (s0 - s1)>
#map4 = affine_map<()[s0, s1] -> ((s0 - s1) ceildiv 16)>
#map5 = affine_map<(d0) -> (d0 floordiv 16)>
#map6 = affine_map<(d0) -> (d0 mod 16)>
#map7 = affine_map<()[s0] -> (s0 * 16)>
#map8 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 768)>
#map9 = affine_map<()[s0, s1] -> (s0 - s1 + (s1 floordiv 768) * 768)>
#map10 = affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 768)>
#map11 = affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2)>
#map12 = affine_map<()[s0] -> (s0 * 16 + 16)>
#map13 = affine_map<()[s0, s1] -> (s0 - s1 * 16)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @fwd_kernel_mix_aiv_fused_5_outlined_merged_vf_0(%arg0: memref<16xf32, #hivm.address_space<ub>>, %arg1: memref<16xf32, #hivm.address_space<ub>>, %arg2: memref<16xf32, #hivm.address_space<ub>>, %arg3: memref<16xf32, #hivm.address_space<ub>>, %arg4: memref<16x64xf32, #hivm.address_space<ub>>, %arg5: memref<16x64xbf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [16] : vector<64xi1>
    %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<16xf32, #hivm.address_space<ub>>, vector<64xf32>
    %2 = vector.transfer_read %arg1[%c0], %cst, %0 {in_bounds = [true]} : memref<16xf32, #hivm.address_space<ub>>, vector<64xf32>
    %3 = math.log %2 : vector<64xf32>
    %4 = arith.addf %1, %3 : vector<64xf32>
    vector.transfer_write %4, %arg2[%c0], %0 {in_bounds = [true]} : vector<64xf32>, memref<16xf32, #hivm.address_space<ub>>
    scf.for %arg6 = %c0 to %c16 step %c1 {
      %subview = memref.subview %arg3[%arg6] [1] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg4[%arg6, 0] [1, 64] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg5[%arg6, 0] [1, 64] [1, 1] : memref<16x64xbf16, #hivm.address_space<ub>> to memref<1x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %5 = vector.transfer_read %subview_0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %6 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true, true], permutation_map = #map} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %7 = arith.divf %5, %6 : vector<1x64xf32>
      %8 = arith.truncf %7 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xbf16>
      vector.transfer_write %8, %subview_1[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xbf16>, memref<1x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @fwd_kernel_mix_aic(%arg0: memref {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref {tt.tensor_kind = 0 : i32}, %arg3: memref {tt.tensor_kind = 0 : i32}, %arg4: memref {tt.tensor_kind = 0 : i32}, %arg5: memref {tt.tensor_kind = 1 : i32}, %arg6: memref {tt.tensor_kind = 1 : i32}, %arg7: memref, %arg8: memref, %arg9: memref {tt.tensor_kind = 0 : i32}, %arg10: memref {tt.tensor_kind = 0 : i32}, %arg11: memref {tt.tensor_kind = 0 : i32}, %arg12: i32, %arg13: i32, %arg14: f32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false]> : vector<18xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c2_i64 = arith.constant 2 : i64
    %c40960_i64 = arith.constant 40960 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c26304_i64 = arith.constant 26304 : i64
    %c4224_i64 = arith.constant 4224 : i64
    %c28672_i64 = arith.constant 28672 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c22208_i64 = arith.constant 22208 : i64
    %c128_i64 = arith.constant 128 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %c0_i64 = arith.constant 0 : i64
    %c28_i32 = arith.constant 28 : i32
    %true = arith.constant true
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c1536_i32 = arith.constant 1536 : i32
    %c24_i32 = arith.constant 24 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i64 = arith.constant 64 : i64
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.muli %arg15, %arg16 : i32
    %1 = arith.muli %0, %arg17 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.extsi %arg12 : i32 to i64
    %5 = arith.muli %arg12, %c1536_i32 : i32
    %6 = arith.extsi %arg13 : i32 to i64
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 2
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 3
    scf.for %arg18 = %3 to %1 step %c28_i32  : i32 {
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %7 = arith.remsi %arg18, %arg15 : i32
      scf.for %arg19 = %7 to %5 step %c28_i32  : i32 {
        %8 = hivm.hir.pointer_cast(%c0_i64, %c24576_i64) : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
        annotation.mark %8 {hivm.multi_buffer = 2 : i32} : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
        %9 = arith.divsi %arg19, %c24_i32 : i32
        %10 = arith.remsi %arg19, %c24_i32 : i32
        %11 = arith.divsi %9, %arg12 : i32
        %12 = arith.remsi %9, %arg12 : i32
        %13 = arith.muli %12, %arg13 : i32
        %14 = arith.divsi %13, %arg12 : i32
        %15 = arith.index_cast %11 : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg10 to offset: [%15], sizes: [1], strides: [1] : memref to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %16 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %17 = affine.apply #map1()[%15]
        %reinterpret_cast_0 = memref.reinterpret_cast %arg10 to offset: [%17], sizes: [1], strides: [1] : memref to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %18 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %19 = arith.extsi %16 : i32 to i64
        %20 = arith.subi %18, %16 : i32
        %21 = arith.muli %10, %c32_i32 : i32
        %22 = arith.cmpi slt, %21, %20 : i32
        scf.if %22 {
          %reinterpret_cast_1 = memref.reinterpret_cast %arg11 to offset: [%15], sizes: [1], strides: [1] : memref to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %23 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %reinterpret_cast_2 = memref.reinterpret_cast %arg11 to offset: [%17], sizes: [1], strides: [1] : memref to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %24 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %25 = arith.extsi %23 : i32 to i64
          %26 = arith.subi %24, %23 : i32
          %27 = arith.extsi %21 : i32 to i64
          %28 = arith.extsi %26 : i32 to i64
          %29 = arith.cmpi slt, %27, %28 : i64
          scf.if %29 {
            %30 = arith.muli %19, %4 : i64
            %31 = arith.muli %30, %c64_i64 : i64
            %32 = arith.muli %12, %c64_i32 : i32
            %33 = arith.index_cast %31 : i64 to index
            %34 = arith.index_cast %32 : i32 to index
            %35 = arith.muli %arg12, %c64_i32 : i32
            %36 = arith.maxsi %21, %c0_i32 : i32
            %37 = arith.index_cast %36 : i32 to index
            %38 = arith.index_cast %35 : i32 to index
            %39 = affine.apply #map2()[%37, %38, %33, %34]
            %40 = arith.index_cast %20 : i32 to index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [32, 64], strides: [%38, 1] : memref to memref<32x64xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
            %41 = arith.muli %25, %6 : i64
            %42 = arith.muli %41, %c64_i64 : i64
            %43 = arith.muli %14, %c64_i32 : i32
            %44 = arith.index_cast %42 : i64 to index
            %45 = arith.index_cast %43 : i32 to index
            %46 = arith.muli %arg13, %c64_i32 : i32
            %47 = affine.apply #map3()[%40, %37]
            %48 = arith.maxsi %47, %c0 : index
            %49 = arith.minsi %48, %c32 : index
            %50 = arith.subi %c0_i32, %21 : i32
            %51 = arith.maxsi %50, %c0_i32 : i32
            %52 = arith.index_cast %51 : i32 to index
            %53 = arith.minsi %52, %49 : index
            %54 = affine.apply #map3()[%49, %53]
            %55 = arith.cmpi slt, %54, %c32 : index
            %subview = memref.subview %reinterpret_cast_3[0, 0] [%54, 64] [1, 1] : memref<32x64xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref, #hivm.address_space<gm>>
            %56 = affine.apply #map4()[%49, %53]
            %57 = affine.apply #map5(%53)
            %58 = affine.apply #map6(%53)
            %subview_4 = memref.subview %8[0, %57, %58, 0] [4, %56, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<4x?x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
            scf.if %55 {
              %collapse_shape = memref.collapse_shape %8 [[0, 1, 2, 3]] : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>> into memref<2048xbf16, #hivm.address_space<cbuf>>
              hivm.hir.vbrc ins(%cst : bf16) outs(%collapse_shape : memref<2048xbf16, #hivm.address_space<cbuf>>)
            } {hivm.unlikely_condition}
            hivm.hir.nd2nz {dst_continuous} ins(%subview : memref, #hivm.address_space<gm>>) outs(%subview_4 : memref<4x?x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>)
            %60 = hivm.hir.multi_buffer_counter -> i64
            %61 = arith.remui %60, %c2_i64 : i64
            %62 = arith.index_cast %61 : i64 to index
            %63 = arith.remsi %62, %c2 : index
            %64 = arith.addi %63, %c2 : index
            %65 = arith.index_cast %64 : index to i64
            %66 = arith.addi %63, %c1 : index
            %67 = arith.index_cast %66 : index to i64
            %68 = arith.addi %62, %c3 : index
            %69 = arith.index_cast %68 : index to i64
            %70 = hivm.hir.pointer_cast(%c8192_i64, %c24576_i64) : memref<4x2x16x16xf32, #hivm.address_space<cc>>
            annotation.mark %70 {hivm.multi_buffer = 2 : i32} : memref<4x2x16x16xf32, #hivm.address_space<cc>>
            %71 = hivm.hir.pointer_cast(%c16384_i64, %c40960_i64) : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
            annotation.mark %71 {hivm.multi_buffer = 2 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
            %72 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<4x2x16x16xf32, #hivm.address_space<cc>>
            annotation.mark %72 {hivm.multi_buffer = 2 : i32} : memref<4x2x16x16xf32, #hivm.address_space<cc>>
            %73 = hivm.hir.pointer_cast(%c8192_i64, %c32768_i64) : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
            annotation.mark %73 {hivm.multi_buffer = 2 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
            %74 = hivm.hir.pointer_cast(%c4224_i64, %c26304_i64) : memref<16x64xf32, #hivm.address_space<ub>>
            annotation.mark %74 {hivm.multi_buffer = 2 : i32, hivm.ssbuffer_Deps} : memref<16x64xf32, #hivm.address_space<ub>>
            %75 = hivm.hir.pointer_cast(%c4096_i64, %c28672_i64) : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
            annotation.mark %75 {hivm.multi_buffer = 2 : i32, hivm.ssbuffer_Deps} : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
            %76 = hivm.hir.pointer_cast(%c128_i64, %c22208_i64) : memref<16x64xf32, #hivm.address_space<ub>>
            annotation.mark %76 {hivm.multi_buffer = 2 : i32, hivm.ssbuffer_Deps} : memref<16x64xf32, #hivm.address_space<ub>>
            %59:2 = scf.for %arg20 = %21 to %26 step %c64_i32 iter_args(%arg21 = %21, %arg22 = %21) -> (i32, i32)  : i32 {
              %77 = scope.scope : () -> i32 {
                %79 = arith.index_cast %46 : i32 to index
                %80 = arith.index_cast %26 : i32 to index
                %81 = arith.maxsi %arg21, %c0_i32 : i32
                %82 = arith.index_cast %81 : i32 to index
                %83 = affine.apply #map2()[%82, %79, %44, %45]
                %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%83], sizes: [64, 64], strides: [%79, 1] : memref to memref<64x64xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
                %84 = affine.apply #map3()[%80, %82]
                %85 = arith.maxsi %84, %c0 : index
                %86 = arith.minsi %85, %c64 : index
                %87 = arith.subi %c0_i32, %arg21 : i32
                %88 = arith.maxsi %87, %c0_i32 : i32
                %89 = arith.index_cast %88 : i32 to index
                %90 = arith.minsi %89, %86 : index
                %91 = affine.apply #map3()[%86, %90]
                %92 = arith.cmpi slt, %91, %c64 : index
                %subview_6 = memref.subview %reinterpret_cast_5[0, 0] [%91, 64] [1, 1] : memref<64x64xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref, #hivm.address_space<gm>>
                %93 = affine.apply #map4()[%86, %90]
                %94 = affine.apply #map5(%90)
                %95 = affine.apply #map6(%90)
                %subview_7 = memref.subview %73[0, %94, %95, 0] [4, %93, 16, 16] [1, 1, 1, 1] : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>> to memref<4x?x16x16xbf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
                scf.if %92 {
                  %collapse_shape = memref.collapse_shape %73 [[0, 1, 2, 3]] : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>> into memref<4096xbf16, #hivm.address_space<cbuf>>
                  hivm.hir.vbrc ins(%cst : bf16) outs(%collapse_shape : memref<4096xbf16, #hivm.address_space<cbuf>>)
                } {hivm.unlikely_condition}
                hivm.hir.nd2nz {dst_continuous} ins(%subview_6 : memref, #hivm.address_space<gm>>) outs(%subview_7 : memref<4x?x16x16xbf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>)
                hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%8, %73, %true, %c32, %c64, %c64 : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>, memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%72 : memref<4x2x16x16xf32, #hivm.address_space<cc>>)
                annotation.mark %74 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<16x64xf32, #hivm.address_space<ub>>
                hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = %69
                hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%72 : memref<4x2x16x16xf32, #hivm.address_space<cc>>) outs(%74 : memref<16x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
                hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
                %96 = arith.addi %arg21, %c64_i32 : i32
                scope.return %96 : i32
              } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.ssbuffer,
                 hivm.ssbuffer_outs = [{dep_id = 1 : i32, multi_buffer = 2 : i32}], no_inline}
              scope.scope : () -> () {
                annotation.mark %75 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = -1 : index} : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
                scope.return
              } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.ssbuffer, no_inline}
              %78 = scope.scope : () -> i32 {
                %79 = arith.maxsi %arg22, %c0_i32 : i32
                %80 = arith.index_cast %79 : i32 to index
                %81 = arith.index_cast %46 : i32 to index
                %82 = affine.apply #map2()[%80, %81, %44, %45]
                %83 = arith.index_cast %26 : i32 to index
                %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%82], sizes: [64, 64], strides: [%81, 1] : memref to memref<64x64xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
                %84 = affine.apply #map3()[%83, %80]
                %85 = arith.maxsi %84, %c0 : index
                %86 = arith.minsi %85, %c64 : index
                %87 = arith.subi %c0_i32, %arg22 : i32
                %88 = arith.maxsi %87, %c0_i32 : i32
                %89 = arith.index_cast %88 : i32 to index
                %90 = arith.minsi %89, %86 : index
                %91 = affine.apply #map3()[%86, %90]
                %92 = arith.cmpi slt, %91, %c64 : index
                %subview_6 = memref.subview %reinterpret_cast_5[0, 0] [%91, 64] [1, 1] : memref<64x64xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref, #hivm.address_space<gm>>
                %93 = affine.apply #map4()[%86, %90]
                %94 = affine.apply #map5(%90)
                %95 = affine.apply #map6(%90)
                %subview_7 = memref.subview %71[0, %94, %95, 0] [4, %93, 16, 16] [1, 1, 1, 1] : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>> to memref<4x?x16x16xbf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
                scf.if %92 {
                  %collapse_shape = memref.collapse_shape %71 [[0, 1, 2, 3]] : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>> into memref<4096xbf16, #hivm.address_space<cbuf>>
                  hivm.hir.vbrc ins(%cst : bf16) outs(%collapse_shape : memref<4096xbf16, #hivm.address_space<cbuf>>)
                } {hivm.unlikely_condition}
                hivm.hir.nd2nz {dst_continuous} ins(%subview_6 : memref, #hivm.address_space<gm>>) outs(%subview_7 : memref<4x?x16x16xbf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>)
                hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
                hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%75, %71, %true, %c32, %c64, %c64 : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>, memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%70 : memref<4x2x16x16xf32, #hivm.address_space<cc>>)
                hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %65
                annotation.mark %76 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<16x64xf32, #hivm.address_space<ub>>
                hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = %67
                hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%70 : memref<4x2x16x16xf32, #hivm.address_space<cc>>) outs(%76 : memref<16x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
                hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
                %96 = arith.addi %arg22, %c64_i32 : i32
                scope.return %96 : i32
              } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.ssbuffer,
                 hivm.ssbuffer_ins = [{dep_id = 2 : i32, multi_buffer = 2 : i32}],
                 hivm.ssbuffer_outs = [{dep_id = 3 : i32, multi_buffer = 2 : i32}], no_inline}
              scf.yield %77, %78 : i32, i32
            } {fixpipe_for_mmad_result_already_inserted = true, hivm.cv_pipelined_loop}
          }
        }
      }
      hivm.hir.set_ctrl true at ctrl[60]
    } {autoblockify.subloop}
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
    return
  }
  func.func @fwd_kernel_mix_aiv_fused_0_outlined_vf_0(%arg0: memref<16x64xf32, #hivm.address_space<ub>>, %arg1: memref<16x64xi8, #hivm.address_space<ub>>, %arg2: f32, %arg3: memref<16x64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<-1.07374182E+9> : vector<1x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %cst_1 = arith.constant dense<0> : vector<1x64xi8>
    %c0_i8 = arith.constant 0 : i8
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %0 = vector.broadcast %arg2 : f32 to vector<1x64xf32>
    scf.for %arg4 = %c0 to %c16 step %c1 {
      %subview = memref.subview %arg0[%arg4, 0] [1, 64] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_3 = memref.subview %arg1[%arg4, 0] [1, 64] [1, 1] : memref<16x64xi8, #hivm.address_space<ub>> to memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_4 = memref.subview %arg3[%arg4, 0] [1, 64] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %1 = vector.transfer_read %subview[%c0, %c0], %cst_2 {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %2 = vector.transfer_read %subview_3[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xi8>
      %3 = arith.cmpi ne, %2, %cst_1 : vector<1x64xi8>
      %4 = arith.select %3, %cst_0, %cst : vector<1x64xi1>, vector<1x64xf32>
      %5 = arith.mulf %1, %0 : vector<1x64xf32>
      %6 = arith.addf %5, %4 : vector<1x64xf32>
      vector.transfer_write %6, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @fwd_kernel_mix_aiv_fused_0_outlined_vf_1(%arg0: memref<16x64xf32, #hivm.address_space<ub>>, %arg1: memref<16xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    scf.for %arg2 = %c0 to %c16 step %c1 {
      %subview = memref.subview %arg1[%arg2] [1] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg0[%arg2, 0] [1, 64] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %0 = vector.transfer_read %subview_0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %1 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1xf32>
      %2 = vector.multi_reduction <maximumf>, %0, %1 {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
      vector.transfer_write %2, %subview[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @fwd_kernel_mix_aiv_fused_1_outlined_vf_0(%arg0: memref<16xf32, #hivm.address_space<ub>>, %arg1: memref<16xf32, #hivm.address_space<ub>>, %arg2: memref<16xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [16] : vector<64xi1>
    %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<16xf32, #hivm.address_space<ub>>, vector<64xf32>
    %2 = vector.transfer_read %arg1[%c0], %cst, %0 {in_bounds = [true]} : memref<16xf32, #hivm.address_space<ub>>, vector<64xf32>
    %3 = arith.maximumf %1, %2 : vector<64xf32>
    vector.transfer_write %3, %arg2[%c0], %0 {in_bounds = [true]} : vector<64xf32>, memref<16xf32, #hivm.address_space<ub>>
    return
  }
  func.func @fwd_kernel_mix_aiv_fused_1_outlined_vf_1(%arg0: memref<16x64xf32, #hivm.address_space<ub>>, %arg1: memref<16xf32, #hivm.address_space<ub>>, %arg2: memref<16xf32, #hivm.address_space<ub>>, %arg3: memref<4x16x16xbf16, strided<[272, 16, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c16 step %c1 {
      %subview = memref.subview %arg2[%arg4] [1] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg0[%arg4, 0] [1, 64] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %arg1[%arg4] [1] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %0 = vector.transfer_read %subview_1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %1 = vector.transfer_read %subview_2[%c0], %cst {in_bounds = [true, true], permutation_map = #map} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %2 = arith.subf %0, %1 : vector<1x64xf32>
      %3 = math.exp %2 : vector<1x64xf32>
      %4 = arith.addf %3, %cst_0 {reductionOp} : vector<1x64xf32>
      %5 = arith.truncf %3 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xbf16>
      %subview_3 = memref.subview %arg3[0, %arg4, 0] [4, 1, 16] [1, 1, 1] : memref<4x16x16xbf16, strided<[272, 16, 1]>, #hivm.address_space<ub>> to memref<4x1x16xbf16, strided<[272, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %6 = vector.shape_cast %5 : vector<1x64xbf16> to vector<4x1x16xbf16>
      vector.transfer_write %6, %subview_3[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<4x1x16xbf16>, memref<4x1x16xbf16, strided<[272, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %7 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1xf32>
      %8 = vector.multi_reduction <add>, %4, %7 {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
      vector.transfer_write %8, %subview[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @fwd_kernel_mix_aiv_fused_2_outlined_vf_0(%arg0: memref<16xf32, #hivm.address_space<ub>>, %arg1: memref<16xf32, #hivm.address_space<ub>>, %arg2: memref<16xf32, #hivm.address_space<ub>>, %arg3: memref<16xf32, #hivm.address_space<ub>>, %arg4: memref<16xf32, #hivm.address_space<ub>>, %arg5: memref<16xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [16] : vector<64xi1>
    %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<16xf32, #hivm.address_space<ub>>, vector<64xf32>
    %2 = vector.transfer_read %arg1[%c0], %cst, %0 {in_bounds = [true]} : memref<16xf32, #hivm.address_space<ub>>, vector<64xf32>
    %3 = arith.subf %1, %2 : vector<64xf32>
    %4 = math.exp %3 : vector<64xf32>
    vector.transfer_write %4, %arg2[%c0], %0 {in_bounds = [true]} : vector<64xf32>, memref<16xf32, #hivm.address_space<ub>>
    %5 = vector.transfer_read %arg3[%c0], %cst, %0 {in_bounds = [true]} : memref<16xf32, #hivm.address_space<ub>>, vector<64xf32>
    %6 = vector.transfer_read %arg4[%c0], %cst, %0 {in_bounds = [true]} : memref<16xf32, #hivm.address_space<ub>>, vector<64xf32>
    %7 = arith.mulf %5, %4 : vector<64xf32>
    %8 = arith.addf %7, %6 : vector<64xf32>
    vector.transfer_write %8, %arg5[%c0], %0 {in_bounds = [true]} : vector<64xf32>, memref<16xf32, #hivm.address_space<ub>>
    return
  }
  func.func @fwd_kernel_mix_aiv_fused_3_outlined_vf_0(%arg0: memref<16xf32, #hivm.address_space<ub>>, %arg1: memref<16x64xf32, #hivm.address_space<ub>>, %arg2: memref<16x64xf32, #hivm.address_space<ub>>, %arg3: memref<16x64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c16 step %c1 {
      %subview = memref.subview %arg0[%arg4] [1] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg4, 0] [1, 64] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg2[%arg4, 0] [1, 64] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %arg3[%arg4, 0] [1, 64] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %0 = vector.transfer_read %subview_0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %1 = vector.transfer_read %subview_1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %2 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true, true], permutation_map = #map} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %3 = arith.mulf %1, %2 : vector<1x64xf32>
      %4 = arith.addf %0, %3 : vector<1x64xf32>
      vector.transfer_write %4, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @fwd_kernel_mix_aiv_outlined_vf_0(%arg0: memref<16x64xi8, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0> : vector<1x256xi8>
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1, 64] : vector<1x256xi1>
    scf.for %arg1 = %c0 to %c16 step %c1 {
      %subview = memref.subview %arg0[%arg1, 0] [1, 64] [1, 1] : memref<16x64xi8, #hivm.address_space<ub>> to memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      vector.transfer_write %cst, %subview[%c0, %c0], %0 {in_bounds = [true, true]} : vector<1x256xi8>, memref<1x64xi8, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @fwd_kernel_mix_aiv_outlined_vf_1(%arg0: memref<16x64xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    scf.for %arg1 = %c0 to %c16 step %c1 {
      %subview = memref.subview %arg0[%arg1, 0] [1, 64] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      vector.transfer_write %cst, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @fwd_kernel_mix_aiv_outlined_vf_2(%arg0: memref<16xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<-1.07374182E+9> : vector<64xf32>
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [16] : vector<64xi1>
    vector.transfer_write %cst, %arg0[%c0], %0 {in_bounds = [true]} : vector<64xf32>, memref<16xf32, #hivm.address_space<ub>>
    return
  }
  func.func @fwd_kernel_mix_aiv_outlined_vf_3(%arg0: memref<16xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [16] : vector<64xi1>
    vector.transfer_write %cst, %arg0[%c0], %0 {in_bounds = [true]} : vector<64xf32>, memref<16xf32, #hivm.address_space<ub>>
    return
  }
  func.func @fwd_kernel_mix_aiv(%arg0: memref {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref {tt.tensor_kind = 0 : i32}, %arg3: memref {tt.tensor_kind = 0 : i32}, %arg4: memref {tt.tensor_kind = 0 : i32}, %arg5: memref {tt.tensor_kind = 1 : i32}, %arg6: memref {tt.tensor_kind = 1 : i32}, %arg7: memref, %arg8: memref, %arg9: memref {tt.tensor_kind = 0 : i32}, %arg10: memref {tt.tensor_kind = 0 : i32}, %arg11: memref {tt.tensor_kind = 0 : i32}, %arg12: i32, %arg13: i32, %arg14: f32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false]> : vector<18xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c2_i64 = arith.constant 2 : i64
    %c31488_i64 = arith.constant 31488 : i64
    %c20032_i64 = arith.constant 20032 : i64
    %c31424_i64 = arith.constant 31424 : i64
    %c19968_i64 = arith.constant 19968 : i64
    %c22144_i64 = arith.constant 22144 : i64
    %c22080_i64 = arith.constant 22080 : i64
    %c64_i64 = arith.constant 64 : i64
    %c13568_i64 = arith.constant 13568 : i64
    %c13504_i64 = arith.constant 13504 : i64
    %c13440_i64 = arith.constant 13440 : i64
    %c9344_i64 = arith.constant 9344 : i64
    %c30400_i64 = arith.constant 30400 : i64
    %c8320_i64 = arith.constant 8320 : i64
    %c26304_i64 = arith.constant 26304 : i64
    %c4224_i64 = arith.constant 4224 : i64
    %c28672_i64 = arith.constant 28672 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c22208_i64 = arith.constant 22208 : i64
    %c128_i64 = arith.constant 128 : i64
    %c15744_i64 = arith.constant 15744 : i64
    %c15808_i64 = arith.constant 15808 : i64
    %c15872_i64 = arith.constant 15872 : i64
    %c0_i64 = arith.constant 0 : i64
    %c16 = arith.constant 16 : index
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i8 = arith.constant 0 : i8
    %c589824_i32 = arith.constant 589824 : i32
    %c24_i32 = arith.constant 24 : i32
    %c1536_i32 = arith.constant 1536 : i32
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c28_i32 = arith.constant 28 : i32
    %c0 = arith.constant 0 : index
    %0 = arith.muli %arg15, %arg16 : i32
    %1 = arith.muli %0, %arg17 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.extsi %arg12 : i32 to i64
    %5 = arith.muli %arg12, %c1536_i32 : i32
    %6 = hivm.hir.get_sub_block_idx -> i64
    %7 = arith.index_cast %6 : i64 to index
    %8 = affine.apply #map7()[%7]
    annotation.mark %1 {logical_block_num} : i32
    %9 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>
    call @fwd_kernel_mix_aiv_outlined_vf_3(%9) {hivm.vector_function, no_inline} : (memref<16xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
    scf.for %arg18 = %3 to %1 step %c28_i32  : i32 {
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %10 = arith.remsi %arg18, %arg15 : i32
      scf.for %arg19 = %10 to %5 step %c28_i32  : i32 {
        %11 = hivm.hir.pointer_cast(%c20032_i64, %c31488_i64) : memref<16x64xbf16, #hivm.address_space<ub>>
        annotation.mark %11 {hivm.multi_buffer = 2 : i32} : memref<16x64xbf16, #hivm.address_space<ub>>
        %12 = hivm.hir.pointer_cast(%c19968_i64, %c31424_i64) : memref<16xf32, #hivm.address_space<ub>>
        annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
        %13 = arith.divsi %arg19, %c24_i32 : i32
        %14 = arith.remsi %arg19, %c24_i32 : i32
        %15 = arith.divsi %13, %arg12 : i32
        %16 = arith.remsi %13, %arg12 : i32
        %17 = arith.index_cast %15 : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg10 to offset: [%17], sizes: [1], strides: [1] : memref to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %18 = affine.apply #map1()[%17]
        %19 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg10 to offset: [%18], sizes: [1], strides: [1] : memref to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %20 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %21 = arith.extsi %19 : i32 to i64
        %22 = arith.subi %20, %19 : i32
        %23 = arith.muli %14, %c32_i32 : i32
        %24 = arith.cmpi slt, %23, %22 : i32
        scf.if %24 {
          %reinterpret_cast_1 = memref.reinterpret_cast %arg11 to offset: [%17], sizes: [1], strides: [1] : memref to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %25 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %reinterpret_cast_2 = memref.reinterpret_cast %arg11 to offset: [%18], sizes: [1], strides: [1] : memref to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %26 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %27 = arith.subi %26, %25 : i32
          %28 = arith.extsi %23 : i32 to i64
          %29 = arith.extsi %27 : i32 to i64
          %30 = arith.cmpi slt, %28, %29 : i64
          scf.if %30 {
            %31 = arith.muli %21, %4 : i64
            %32 = arith.muli %31, %c64_i64 : i64
            %33 = arith.muli %16, %c64_i32 : i32
            %34 = arith.index_cast %32 : i64 to index
            %35 = arith.index_cast %33 : i32 to index
            %36 = arith.muli %arg12, %c64_i32 : i32
            %37 = arith.maxsi %23, %c0_i32 : i32
            %38 = arith.index_cast %37 : i32 to index
            %39 = arith.index_cast %36 : i32 to index
            %40 = affine.apply #map2()[%38, %39, %34, %35]
            %41 = arith.index_cast %22 : i32 to index
            %42 = affine.apply #map3()[%41, %38]
            %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%40], sizes: [32, 64], strides: [%39, 1] : memref to memref<32x64xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
            %43 = arith.index_cast %31 : i64 to index
            %44 = arith.index_cast %16 : i32 to index
            %45 = arith.index_cast %arg12 : i32 to index
            %46 = affine.apply #map2()[%38, %45, %43, %44]
            %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%46], sizes: [32], strides: [%45] : memref to memref<32xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %47 = arith.muli %15, %c589824_i32 : i32
            %48 = arith.index_cast %47 : i32 to index
            %49 = arith.maxsi %42, %c0 : index
            %50 = arith.minsi %49, %c32 : index
            %51 = arith.subi %c0_i32, %23 : i32
            %52 = arith.maxsi %51, %c0_i32 : i32
            %53 = arith.index_cast %52 : i32 to index
            %54 = arith.minsi %53, %50 : index
            %55 = affine.apply #map3()[%50, %54]
            %56 = hivm.hir.pointer_cast(%c15872_i64) : memref<16x64xf32, #hivm.address_space<ub>>
            func.call @fwd_kernel_mix_aiv_outlined_vf_1(%56) {hivm.vector_function, no_inline} : (memref<16x64xf32, #hivm.address_space<ub>>) -> ()
            %57 = hivm.hir.pointer_cast(%c15808_i64) : memref<16xf32, #hivm.address_space<ub>>
            func.call @fwd_kernel_mix_aiv_outlined_vf_2(%57) {hivm.vector_function, no_inline} : (memref<16xf32, #hivm.address_space<ub>>) -> ()
            %58 = hivm.hir.pointer_cast(%c15744_i64) : memref<16xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%9 : memref<16xf32, #hivm.address_space<ub>>) outs(%58 : memref<16xf32, #hivm.address_space<ub>>)
            %69 = hivm.hir.multi_buffer_counter -> i64
            %70 = arith.remui %69, %c2_i64 : i64
            %71 = arith.index_cast %70 : i64 to index
            %72 = arith.addi %71, %c1 : index
            %73 = arith.remsi %72, %c2 : index
            %74 = arith.addi %73, %c1 : index
            %75 = arith.index_cast %74 : index to i64
            %76 = arith.addi %71, %c1 : index
            %77 = arith.remsi %76, %c2 : index
            %78 = arith.addi %77, %c3 : index
            %79 = arith.index_cast %78 : index to i64
            %80 = arith.addi %77, %c2 : index
            %81 = arith.index_cast %80 : index to i64
            %82 = hivm.hir.pointer_cast(%c64_i64, %c22080_i64, %c22144_i64) : memref<16xf32, #hivm.address_space<ub>>
            annotation.mark %82 {hivm.multi_buffer = 3 : i32, hivm.ssbuffer_intraDeps，} : memref<16xf32, #hivm.address_space<ub>>
            %83 = hivm.hir.pointer_cast(%c8320_i64, %c30400_i64) : memref<16x64xi8, #hivm.address_space<ub>>
            annotation.mark %83 {hivm.multi_buffer = 2 : i32} : memref<16x64xi8, #hivm.address_space<ub>>
            %84 = hivm.hir.pointer_cast(%c4224_i64, %c26304_i64) : memref<16x64xf32, #hivm.address_space<ub>>
            annotation.mark %84 {hivm.multi_buffer = 2 : i32, hivm.ssbuffer_Deps} : memref<16x64xf32, #hivm.address_space<ub>>
            %85 = hivm.hir.pointer_cast(%c4096_i64, %c28672_i64) : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
            annotation.mark %85 {hivm.multi_buffer = 2 : i32, hivm.ssbuffer_Deps} : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
            %86 = hivm.hir.pointer_cast(%c128_i64, %c22208_i64) : memref<16x64xf32, #hivm.address_space<ub>>
            annotation.mark %86 {hivm.multi_buffer = 2 : i32, hivm.ssbuffer_Deps} : memref<16x64xf32, #hivm.address_space<ub>>
            %59:4 = scf.for %arg20 = %23 to %27 step %c64_i32 iter_args(%arg21 = %23, %arg22 = %56, %arg23 = %57, %arg24 = %58) -> (i32, memref<16x64xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>)  : i32 {
              annotation.mark %86 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<16x64xf32, #hivm.address_space<ub>>
              %subview_8 = memref.subview %85[0, %7, 0, 0] [4, 1, 16, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<4x1x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
              annotation.mark %84 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<16x64xf32, #hivm.address_space<ub>>
              %87:3 = scope.scope : () -> (memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>, i32) {
                %89 = arith.index_cast %27 : i32 to index
                %90 = arith.maxsi %arg21, %c0_i32 : i32
                %91 = arith.index_cast %90 : i32 to index
                %92 = affine.apply #map8()[%91, %48, %38]
                %93 = affine.apply #map9()[%89, %91]
                %94 = affine.apply #map10()[%41, %91, %38]
                %reinterpret_cast_9 = memref.reinterpret_cast %arg9 to offset: [%92], sizes: [32, 64], strides: [768, 1] : memref to memref<32x64xi8, strided<[768, 1], offset: ?>, #hivm.address_space<gm>>
                %95 = arith.maxsi %94, %c0 : index
                %96 = arith.minsi %95, %c32 : index
                %97 = arith.maxsi %93, %c0 : index
                %98 = arith.minsi %97, %c64 : index
                %99 = arith.minsi %53, %96 : index
                %100 = affine.apply #map3()[%96, %99]
                %101 = arith.subi %c0_i32, %arg21 : i32
                %102 = arith.maxsi %101, %c0_i32 : i32
                %103 = arith.index_cast %102 : i32 to index
                %104 = arith.minsi %103, %98 : index
                %105 = affine.apply #map3()[%98, %104]
                %106 = arith.cmpi slt, %100, %c32 : index
                %107 = arith.cmpi slt, %105, %c64 : index
                %108 = arith.ori %106, %107 : i1
                %109 = arith.minsi %8, %100 : index
                %110 = affine.apply #map11()[%109, %96, %99]
                %111 = arith.minsi %110, %c16 : index
                %subview_10 = memref.subview %reinterpret_cast_9[%8, 0] [%111, %105] [1, 1] : memref<32x64xi8, strided<[768, 1], offset: ?>, #hivm.address_space<gm>> to memref, #hivm.address_space<gm>>
                %112 = affine.apply #map12()[%7]
                %113 = arith.maxsi %99, %8 : index
                %114 = arith.minsi %96, %112 : index
                %115 = arith.maxsi %113, %114 : index
                %116 = affine.apply #map3()[%115, %113]
                %117 = affine.apply #map13()[%113, %7]
                %subview_11 = memref.subview %83[%117, %104] [%116, %105] [1, 1] : memref<16x64xi8, #hivm.address_space<ub>> to memref, #hivm.address_space<ub>>
                %118 = arith.remui %104, %c32 : index
                scf.if %108 {
                  func.call @fwd_kernel_mix_aiv_outlined_vf_0(%83) {hivm.vector_function, no_inline} : (memref<16x64xi8, #hivm.address_space<ub>>) -> ()
                } {hivm.unlikely_condition}
                hivm.hir.load ins(%subview_10 : memref, #hivm.address_space<gm>>) outs(%subview_11 : memref, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c0_i8 : i8 left_padding_num = %118 : index
                %119 = hivm.hir.pointer_cast(%c9344_i64) : memref<16x64xf32, #hivm.address_space<ub>>
                hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
                func.call @fwd_kernel_mix_aiv_fused_0_outlined_vf_0(%84, %83, %arg14, %119) {hivm.vector_function, no_inline} : (memref<16x64xf32, #hivm.address_space<ub>>, memref<16x64xi8, #hivm.address_space<ub>>, f32, memref<16x64xf32, #hivm.address_space<ub>>) -> ()
                hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = %79
                %120 = hivm.hir.pointer_cast(%c13440_i64) : memref<16xf32, #hivm.address_space<ub>>
                func.call @fwd_kernel_mix_aiv_fused_0_outlined_vf_1(%119, %120) {hivm.vector_function, no_inline} : (memref<16x64xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) -> ()
                %121 = hivm.hir.pointer_cast(%c13440_i64) : memref<16xf32, #hivm.address_space<ub>>
                func.call @fwd_kernel_mix_aiv_fused_1_outlined_vf_0(%arg23, %120, %121) {hivm.vector_function, no_inline} : (memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) -> ()
                %122 = hivm.hir.pointer_cast(%c13504_i64) : memref<16xf32, #hivm.address_space<ub>>
                %123 = hivm.hir.pointer_cast(%c13568_i64) : memref<4x17x16x1xbf16, #hivm.address_space<ub>>
                %subview_12 = memref.subview %123[0, 0, 0, 0] [4, 16, 16, 1] [1, 1, 1, 1] : memref<4x17x16x1xbf16, #hivm.address_space<ub>> to memref<4x16x16xbf16, strided<[272, 16, 1]>, #hivm.address_space<ub>>
                func.call @fwd_kernel_mix_aiv_fused_1_outlined_vf_1(%119, %121, %122, %subview_12) {hivm.vector_function, no_inline} : (memref<16x64xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>, memref<4x16x16xbf16, strided<[272, 16, 1]>, #hivm.address_space<ub>>) -> ()
                %expand_shape = memref.expand_shape %subview_12 [[0], [1, 2], [3]] output_shape [4, 1, 16, 16] : memref<4x16x16xbf16, strided<[272, 16, 1]>, #hivm.address_space<ub>> into memref<4x1x16x16xbf16, strided<[272, 256, 16, 1]>, #hivm.address_space<ub>>
                annotation.mark %85 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = 1 : index} : memref<4x2x16x16xbf16, #hivm.address_space<cbuf>>
                %collapse_shape = memref.collapse_shape %expand_shape [[0, 1], [2, 3]] : memref<4x1x16x16xbf16, strided<[272, 256, 16, 1]>, #hivm.address_space<ub>> into memref<4x256xbf16, strided<[272, 1]>, #hivm.address_space<ub>>
                %collapse_shape_13 = memref.collapse_shape %subview_8 [[0, 1], [2, 3]] : memref<4x1x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> into memref<4x256xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<cbuf>>
                hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %81
                hivm.hir.copy ins(%collapse_shape : memref<4x256xbf16, strided<[272, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_13 : memref<4x256xbf16, strided<[512, 1], offset: ?>, #hivm.address_space<cbuf>>) {"hivm.inserted-copy", tiled_op}
                hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
                %124 = hivm.hir.pointer_cast(%c15744_i64) : memref<16xf32, #hivm.address_space<ub>>
                func.call @fwd_kernel_mix_aiv_fused_2_outlined_vf_0(%arg23, %121, %82, %arg24, %122, %124) {hivm.vector_function, no_inline} : (memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) -> ()
                %125 = arith.addi %arg21, %c64_i32 : i32
                %126 = hivm.hir.pointer_cast(%c15808_i64) : memref<16xf32, #hivm.address_space<ub>>
                hivm.hir.copy ins(%121 : memref<16xf32, #hivm.address_space<ub>>) outs(%126 : memref<16xf32, #hivm.address_space<ub>>)
                scope.return %126, %124, %125 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>, i32
              } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.ssbuffer,
                 hivm.ssbuffer_ins = [{dep_id = 1 : i32, multi_buffer = 2 : i32}],
                 hivm.ssbuffer_outs = [{dep_id = 2 : i32, multi_buffer = 2 : i32},
                                       {dep_id = 4 : i32, multi_buffer = 3 : i32}], no_inline}
              %88 = scope.scope : () -> memref<16x64xf32, #hivm.address_space<ub>> {
                %89 = hivm.hir.pointer_cast(%c15872_i64) : memref<16x64xf32, #hivm.address_space<ub>>
                hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
                func.call @fwd_kernel_mix_aiv_fused_3_outlined_vf_0(%82, %86, %arg22, %89) {hivm.vector_function, no_inline} : (memref<16xf32, #hivm.address_space<ub>>, memref<16x64xf32, #hivm.address_space<ub>>, memref<16x64xf32, #hivm.address_space<ub>>, memref<16x64xf32, #hivm.address_space<ub>>) -> ()
                hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = %75
                scope.return %89 : memref<16x64xf32, #hivm.address_space<ub>>
              } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.ssbuffer,
                 hivm.ssbuffer_ins = [{dep_id = 3 : i32, multi_buffer = 2 : i32},
                                      {dep_id = 4 : i32, multi_buffer = 3 : i32}], no_inline}
              scf.yield %87#2, %88, %87#0, %87#1 : i32, memref<16x64xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>
            } {fixpipe_for_mmad_result_already_inserted = true, hivm.cv_pipelined_loop}
            func.call @fwd_kernel_mix_aiv_fused_5_outlined_merged_vf_0(%59#2, %59#3, %12, %59#3, %59#1, %11) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>, memref<16x64xf32, #hivm.address_space<ub>>, memref<16x64xbf16, #hivm.address_space<ub>>) -> ()
            %60 = affine.apply #map12()[%7]
            %61 = arith.maxsi %54, %8 : index
            %62 = arith.minsi %50, %60 : index
            %63 = arith.maxsi %61, %62 : index
            %64 = affine.apply #map3()[%63, %61]
            %65 = affine.apply #map13()[%61, %7]
            %subview = memref.subview %11[%65, 0] [%64, 64] [1, 1] : memref<16x64xbf16, #hivm.address_space<ub>> to memref, #hivm.address_space<ub>>
            %66 = arith.minsi %8, %55 : index
            %67 = affine.apply #map11()[%66, %50, %54]
            %68 = arith.minsi %67, %c16 : index
            %subview_5 = memref.subview %reinterpret_cast_3[%8, 0] [%68, 64] [1, 1] : memref<32x64xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref, #hivm.address_space<gm>>
            hivm.hir.store ins(%subview : memref, #hivm.address_space<ub>>) outs(%subview_5 : memref, #hivm.address_space<gm>>) {tiled_op}
            %subview_6 = memref.subview %12[%65] [%64] [1] : memref<16xf32, #hivm.address_space<ub>> to memref, #hivm.address_space<ub>>
            %subview_7 = memref.subview %reinterpret_cast_4[%8] [%68] [1] : memref<32xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref, #hivm.address_space<gm>>
            hivm.hir.store ins(%subview_6 : memref, #hivm.address_space<ub>>) outs(%subview_7 : memref, #hivm.address_space<gm>>) {tiled_op}
          }
        }
      }
      hivm.hir.set_ctrl true at ctrl[60]
    } {autoblockify.subloop}
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 2
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 3
    return
  }
}