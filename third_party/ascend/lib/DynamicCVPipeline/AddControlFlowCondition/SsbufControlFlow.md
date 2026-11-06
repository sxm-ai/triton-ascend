# ssbuf 控制流

模块里只要有一个带 `hivm.ssbuffer` 的 `scope.scope`，`hivm-create-preload`
就走这条路径，整个模块都不再走原来的 `preload_num` 软件流水；没有这种 scope
时这条路径完全不启动。

这里先把 scope 转成 if，再由 `processWhileCondArgs` 按 if 复制
`scf.while` 条件参数。后面才是加条件、改循环。

- [输入和输出](#输入和输出)
- [阶段](#阶段)
- [各函数的输入和输出](#各函数的输入和输出)
- [加条件](#加条件)
- [循环次数](#循环次数)
- [和 TA 的不同](#和-ta-的不同)

## 输入 IR

### 整体样例
 
`hivm-split-mix-kernel` 已经拆出两个 `func.func`：一个
`hivm.func_core_type<AIC>`，一个 `hivm.func_core_type<AIV>`。两边有带
`hivm.cv_pipelined_loop` 的主循环（`scf.for` 或 `scf.while`），循环体里是若干段
带 `hivm.ssbuffer` 的 `scope.scope`。

```mlir
func.func @fwd_kernel_mix_aic() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %buf_a = hivm.hir.pointer_cast(%c4224_i64) : memref<64x32xf32, #hivm.address_space<ub>>
  scf.for %i = %c0_i32 to %c64_i32 step %c64_i32
      iter_args(%n0 = %c0_i32) -> (i32) : i32 {
    %a = scope.scope : () -> i32 {
      hivm.hir.fixpipe ins(%l0c : tensor<2x4x16x16xf32>)
                       outs(%buf_a : memref<64x32xf32, #hivm.address_space<ub>>)
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
      %next = arith.addi %n0, %c64_i32 : i32
      scope.return %next : i32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.ssbuffer,
       hivm.ssbuffer_outs = [{dep_id = 1 : i32, multi_buffer = 2 : i32}], no_inline}
    scf.yield %a : i32
  } {hivm.cv_pipelined_loop}
  return
}

func.func @fwd_kernel_mix_aiv() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %buf_a = hivm.hir.pointer_cast(%c4224_i64) : memref<64x32xf32, #hivm.address_space<ub>>
  scf.for %i = %c0_i32 to %c64_i32 step %c64_i32
      iter_args(%n0 = %c0_i32) -> (i32) : i32 {
    %a = scope.scope : () -> i32 {
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
      hivm.hir.copy ins(%buf_a : memref<64x32xf32, #hivm.address_space<ub>>)
                    outs(%dst : memref<64x32xf32, #hivm.address_space<ub>>)
      %next = arith.addi %n0, %c64_i32 : i32
      scope.return %next : i32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.ssbuffer,
       hivm.ssbuffer_ins = [{dep_id = 1 : i32, multi_buffer = 2 : i32}], no_inline}
    scf.yield %a : i32
  } {hivm.cv_pipelined_loop}
  return
}
```

### 属性

| 属性 | 位置                      | 含义                                    |
| --- |-------------------------|---------------------------------------|
| `hivm.cv_pipelined_loop` | `scf.for` 或 `scf.while` | 这是一个主循环。   |
| `hivm.ssbuffer` | scope                   | 这段 scope 是一个 ssbuf block。最后删掉。 |
| `hivm.loop_core_type` | scope                   | `CUBE` 或 `VECTOR`。两端相同是核内，不同是核间。      |
| `hivm.ssbuffer_ins` | scope                   | 这段 scope 消费的依赖。最后删掉。                  |
| `hivm.ssbuffer_outs` | scope                   | 这段 scope 生产的依赖。最后删掉。                  |

### 标签

`ins` / `outs` 是 `{dep_id, multi_buffer}` 的数组。同一个 `dep_id` 只有一个
`outs` 和一个 `ins`：`outs` 是生产者，`ins` 是消费者。buffer数量取这个 `outs` 的 `multi_buffer`。没有核内边时核内 buffer 数就是 0，不再
读模块上的 `ssbuffer.intra_buf_count`。

`dep_id` 必须是整数。种类只看两端的 `hivm.loop_core_type`：相同是核内，不同是核间。跨轮传值的边也走这套，不再用 `dep_id = "arg1"`。`multi_buffer` 是这一对的 slot 数：

```mlir
hivm.ssbuffer_outs = [{dep_id = 2 : i32, multi_buffer = 2 : i32},
                      {dep_id = 4 : i32, multi_buffer = 3 : i32}]
```

| `dep_id` | 种类 | 怎么配对 | 条件变量                                                                  |
| --- | --- | --- |-----------------------------------------------------------------------|
| `2 : i32` | 核间或核内 | 这一个 id 只有一个 `ins` 和一个 `outs`。核类型相同是核内，不同是核间 | 上限是 `outs` 的 `multi_buffer`。核间两端不在同一个函数，计数放在 SSBuffer，两个 subblock 各一份（`组号 * 4` 和 `1024 + 组号 * 4`）。核内两端在同一个循环，用循环上追加的一个 i32 |
| `4 : i32` | 同上 | 同上 | 和 `dep_id = 2` 属于不同组的依赖                                               |

- 每个 `id` 必须同时有一个 `ins` 和一个 `outs`。`dep_id` 不是整数，或者一个`id`有多个 `ins` 或多个 `outs`，就失败。
- 生产者记成 `ProducerInfo {ifOp, bufferCount}`。`bufferCount` 来自 `outs` 的 `multi_buffer`。
- 核间边和核内边的两端落在同一个 if 上会在建 DAG 时报错。
- 没写进 `ins` / `outs` 的 if 之间就是没有依赖。

## 从 dump 识别依赖

下面是 `CreatePreload` 输入 dump 里的四段 scope。SplitMixKernel 会把另一侧的空壳留在函数里：AIC 里的 VECTOR scope，AIV 里的 CUBE scope。CreateIfOps 在转 if 之前把这种空壳删掉，不参与后面的依赖。

```
AIC  pipelined loop
  cube A   fixpipe -> %74 (multi_buffer = 2), set FIX/V flag 1,  yield i32
  cube B   wait MTE3/MTE1 flag 0, use %75, fixpipe -> %76,
           set FIX/V flag 0, yield i32

AIV  pipelined loop
  vector A wait FIX/V flag 1, use %84, copy -> %85 / %82,
           set MTE3/MTE1 flag 0, yield (memref, memref, i32)
  vector B wait FIX/V flag 0, use %86 / %82, yield memref
```

对应四条边，`id` 两边共用：

| id | outs | ins | core 类型 | 种类 | buffer |
| --- | --- | --- | --- | --- | --- |
| 1 | cube A（%74, set flag 1） | vector A（wait flag 1, %84） | CUBE → VECTOR | 核间 | 2 |
| 2 | vector A（%85, set flag 0） | cube B（wait flag 0, %75） | VECTOR → CUBE | 核间 | 2 |
| 3 | cube B（%76, set flag 0） | vector B（wait flag 0, %86） | CUBE → VECTOR | 核间 | 2 |
| 4 | vector A（%82） | vector B（%82） | VECTOR → VECTOR | 核内 | 3 |

写成标签：

```mlir
// cube A
hivm.ssbuffer_outs = [{dep_id = 1 : i32, multi_buffer = 2 : i32}]
// cube B
hivm.ssbuffer_ins  = [{dep_id = 2 : i32, multi_buffer = 2 : i32}]
hivm.ssbuffer_outs = [{dep_id = 3 : i32, multi_buffer = 2 : i32}]
// vector A
hivm.ssbuffer_ins  = [{dep_id = 1 : i32, multi_buffer = 2 : i32}]
hivm.ssbuffer_outs = [{dep_id = 2 : i32, multi_buffer = 2 : i32},
                      {dep_id = 4 : i32, multi_buffer = 3 : i32}]
// vector B
hivm.ssbuffer_ins  = [{dep_id = 3 : i32, multi_buffer = 2 : i32},
                      {dep_id = 4 : i32, multi_buffer = 3 : i32}]
```

## 输出 IR

每段 scope 变成一个条件真实的 `scf.if`，主循环多出若干迭代参数，循环上界被拉长，if根据依赖关系加上相应的条件

```mlir
func.func @fwd_kernel_mix_aic() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %zero = arith.constant 0 : i32
  %addr0 = arith.constant 0 : i64
  %ptr0 = hivm.hir.pointer_cast(%addr0) : memref<i32>
  memref.store %zero, %ptr0[] : memref<i32>
  %addr1 = arith.constant 1024 : i64
  %ptr1 = hivm.hir.pointer_cast(%addr1) : memref<i32>
  memref.store %zero, %ptr1[] : memref<i32>
  // ... 每个核间组两个 bank
  %newUb = arith.addi %c0_i32, %span : i32
  %out = scf.for %i = %c0_i32 to %newUb step %c64_i32
      iter_args(%n0 = %c0_i32, %cnt0 = %c0_i32) -> (i32, i32) : i32 {
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
    %l0 = memref.load %ptr0[] : memref<i32>
    annotation.mark %l0 {memref_ext.volatile} : i32
    %p0 = arith.cmpi slt, %l0, %c2_i32 : i32
    %l1 = memref.load %ptr1[] : memref<i32>
    annotation.mark %l1 {memref_ext.volatile} : i32
    %p1 = arith.cmpi slt, %l1, %c2_i32 : i32
    %buffers = arith.andi %p0, %p1 : i1
    %inRange = arith.cmpi slt, %cnt0, %c64_i32 : i32
    %cond = arith.andi %buffers, %inRange : i1
    %next:2 = scf.if %cond -> (i32, i32) {
      hivm.hir.fixpipe ...
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
      %u0 = arith.addi %l0, %c1_i32 : i32
      memref.store %u0, %ptr0[] : memref<i32>
      %n0.next = arith.addi %n0, %c64_i32 : i32
      %cnt0.next = arith.addi %cnt0, %c64_i32 : i32
      scf.yield %n0.next, %cnt0.next : i32, i32
    } else {
      scf.yield %n0, %cnt0 : i32, i32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, no_inline}
    hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
    scf.yield %next#0, %next#1 : i32, i32
  } {hivm.cv_pipelined_loop}
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
  return
}

func.func @fwd_kernel_mix_aiv() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
  %out = scf.for ... iter_args(%n0 = %c0_i32, %cnt0 = %c0_i32) -> (i32, i32) {
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
    // 消费者：计数 > 0
    %l = memref.load %ptr[] : memref<i32>
    %has = arith.cmpi sgt, %l, %c0_i32 : i32
    %inRange = arith.cmpi slt, %cnt0, %c64_i32 : i32
    %cond = arith.andi %has, %inRange : i1
    %next:2 = scf.if %cond -> (i32, i32) {
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
      ...
      scf.yield %n0.next, %cnt0.next : i32, i32
    } else {
      scf.yield %n0, %cnt0 : i32, i32
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, no_inline}
    scf.yield %next#0, %next#1 : i32, i32
  } {hivm.cv_pipelined_loop}
  return
}
```

输出里新出现的东西：

| 东西 | 说明                                                                                                                                                                          |
| --- |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `memref_ext.volatile` | 打在每次读 SSBuffer 的 `annotation.mark` 上。                                                                                                                                       |
| 多出来的迭代参数 | `scf.while` 先给每个 ssbuf if 复制一份条件原来读的 iter arg，这份更新链就是 while 的 block counter。（`scf.for` 再追加每个 if 一个 arg,初值取下界，if 用它做 `counter < ub`；），以及每条核内边一个初值 0 的 arg，核内 ssbuffer 条件用这个。 |
| if 多出来的结果 | 这个 if 改过的核内变量，加上它自己的 counter。then 给新值，else 原样转发。原来的 scope 结果排在最前面。                                                                                                          |
| `(S, S)` flag 15 | VECTOR 在循环前 set，CUBE 在循环后 wait；两侧都在体首 wait、体尾 set。cube 先跑。                                                                                                                  |
| 拉长的上界 | 见[循环次数](#循环次数)。                                                                                                                                                             |

scope 全部消失。

## 阶段

1. **CreateIfOps**。函数核类型和 scope 的 `hivm.loop_core_type` 对不上的空壳直接删掉：AIC 函数里的 VECTOR scope，AIV 函数里的 CUBE scope。这是 SplitMixKernel 留在另一侧的空壳，没有结果。结果还有人用就失败。其余每个 `hivm.ssbuffer` scope 变成条件恒为 `true` 的 `scf.if`。属性（`hivm.ssbuffer`、`ins` / `outs` 标签在内）搬到 if 上，scope 删掉。不再另写 `ssbuffer.if`。没有 block id，也不写 `hivm.matmul_limited_in_cube`。then / else yield 见 `createIfOps`。
2. **while 条件拷贝**。`processWhileCondArgs` 给每个 ssbuf if 一份
   `scf.while` 条件 iter arg 的私有拷贝，`whileBlockArgMap` 的内层键直接是
   这个 if。后面替换 if（`updateDAGAfterIfOpReplacement`）或重建 while
   （`retargetMainLoop`）时改这张表的键。
3. **InitDependentMap**。从 if 上读 `ins` / `outs`，按整数 `dep_id` 配对。一个 id 一组：一个消费者 if，生产者是 `ProducerInfo {ifOp, bufferCount}`。核类型相同进 `intraCoreBuffers`，不同进 `crossCoreBuffers`。然后 `computeProducerBufferCount`、`buildIfBlockDAG`、`detectCycle`、`collectFlowOptIfOpPairs`。DAG 从生产者 if 连到消费者 if。
4. **UpdateLoopOps**。`addBlockCountersAndInnerDepConds` 追加 block counter 和核内变量，
   `insertInterCorePipeS` 插入 `(S, S)`。
5. **UpdateConditionInfo**。`allocSSBuffer` 给核间组分配指针，`updateIfConds`
   把 `true` 换成真实条件，并在 then 里写回计数和变量。替换 if 时同步
   `whileBlockArgMap`。
6. **UpdateLoopIterTimes**。`ComputeMainLoopTimes`、重建更长的 `scf.for`、
   `replaceForOpCounterInIfOps`、`UpdateWhileLoopCondition`。
7. 清掉本路径专用的属性：`hivm.ssbuffer`、`hivm.ssbuffer_ins`、
   `hivm.ssbuffer_outs`。

## 各函数的输入和输出

下面按 `applySsbufControlFlow` 的调用顺序。每段只画这一步改掉的地方，
`....` 是上一步已经稳定、这一步不动的计算。例子是一条核间边：cube 生产
`dep_id = 1`，vector 消费，`multi_buffer = 2`。`scf.for` 原来是
`%lb` 到 `%ub`，step `%step`，一个 iter arg `%n0`。

`SsbufUtils.cpp` 里的 `migrateBody` 只在重建循环时把旧 block 里的算子搬过去，
不单独改变语义。

### `createIfOps`

函数核类型和 scope 对不上的空壳直接消失：AIC 里的 VECTOR，AIV 里的 CUBE。结果还有人用就失败。

```mlir
// 输入：AIC 函数里的 VECTOR，或 AIV 函数里的 CUBE
scope.scope { .... } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.ssbuffer}
scope.scope { .... } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.ssbuffer}
// 输出：这两段 scope 都没有了
```

其余 `hivm.ssbuffer` scope 变成条件恒真的 if。`hivm.ssbuffer` 和 `ins` / `outs` 跟过去，后面靠 if 上的 `hivm.ssbuffer` 认出它。有人用的结果才 yield：then 交 `scope.return` 里对应的操作数，else 交主循环 `scf.yield` 同一下标的 iter arg。没人用的结果丢掉；全部没人用时 if 没有 else。

```mlir
// 输入。yield 顺序可以和 scope 的结果列表不一样
scf.for ... iter_args(%n0 = ..., %n1 = ..., %n2 = ..., %n3 = ...) {
  %87:3 = scope.scope : () -> (memref<...>, memref<...>, i32) {
    ....
    scope.return %m0, %m1, %next : memref<...>, memref<...>, i32
  } {hivm.ssbuffer, hivm.ssbuffer_outs = [{dep_id = 1 : i32, multi_buffer = 2 : i32}], no_inline}
  %88 = ....
  scf.yield %87#2, %88, %87#0, %87#1 : i32, memref<...>, memref<...>, memref<...>
}

// 输出。then 按结果编号，else 按这个结果在主循环 yield 里的下标
%87:3 = scf.if %true -> (memref<...>, memref<...>, i32) {
  ....
  scf.yield %m0, %m1, %next : memref<...>, memref<...>, i32
} else {
  scf.yield %n2, %n3, %n0 : memref<...>, memref<...>, i32
  // %87#0 在 yield 第 2 位，else 交 %n2；%87#2 在第 0 位，else 交 %n0
} {hivm.ssbuffer, hivm.ssbuffer_outs = [{dep_id = 1 : i32, multi_buffer = 2 : i32}],
   no_inline}
```

### `processWhileCondArgs`

`scf.for` 原样通过。`scf.while` 给每个 ssbuf if 复制一份 before 条件读过的
iter arg。条件一个 iter arg 都没用时直接跳过：不改 IR，也不写
`whileBlockArgMap`。

```mlir
// 输入
scf.while (%arg = %init) : (i32) -> i32 {
  %cond = arith.cmpi slt, %arg, %bound : i32
  scf.condition(%cond) %arg : i32
} do {
^bb0(%arg : i32):
  %a = scf.if %true -> (i32) { .... scf.yield %next : i32 } else { scf.yield %arg : i32 }
  scf.yield %a : i32
}

// 输出：多一份 %arg，更新链克隆进这个 if
scf.while (%arg = %init, %arg.c = %init) : (i32, i32) -> (i32, i32) {
  %cond = arith.cmpi slt, %arg, %bound : i32
  scf.condition(%cond) %arg, %arg.c : i32, i32
} do {
^bb0(%arg : i32, %arg.c : i32):
  %a:2 = scf.if %true -> (i32, i32) {
    ....
    %arg.c.next = ....          // 克隆出来的 %arg 更新
    scf.yield %next, %arg.c.next : i32, i32
  } else {
    scf.yield %arg, %arg.c : i32, i32
  }
  scf.yield %a#0, %a#1 : i32, i32
}
// whileBlockArgMap[while][if][新下标] = 旧下标
```

### `initDependentMap`

IR 不变。读 if 上的 `ins` / `outs`，写出这几张表：

```text
dep_id = 1
  consumer = vector if
  producer = {cube if, bufferCount = 2}
  核类型不同 -> crossCoreBuffers
  核类型相同 -> intraCoreBuffers[该主循环]

crossCoreBufferCount = 2
intraCoreBufferCount = 0
DAG: cube if --核间--> vector if
flowOptIfOpPairs: 空
  // 要核间 buffer > 1 且核内 buffer > 2 才收集
  // 收集到的是「入度 0 的 if 跨 3 次核间边」落到的 if -> 起点 if
```

同一个 `dep_id` 同一侧出现第二个端点、缺一边、或 DAG 有环，这一步失败，
后面的函数不会跑。

### `updateLoopOps`

给主循环追加 iter arg，并插 `(S, S)` flag 15。if 的条件仍是 `%true`。

```mlir
// 输入
scf.for %i = %lb to %ub step %step iter_args(%n0 = ...) {
  %a = scf.if %true -> (i32) { .... } else { .... }
  scf.yield %a : i32
} {hivm.cv_pipelined_loop}

// 输出
// VECTOR 函数：循环前 sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
scf.for %i = %lb to %ub step %step
    iter_args(%n0 = ..., %cnt = %lb) -> (..., i32) {
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
  %a = scf.if %true -> (i32) { .... } else { .... }
  hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
  scf.yield %a, %cnt : i32, i32     // counter 这一步还不更新
} {hivm.cv_pipelined_loop}
// CUBE 函数：循环后 sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
```

`scf.for` 给每个 ssbuf if 追加一个 i32，初值是下界，if 用它做 `counter < 原上界`，then 里 `+ step`。`scf.while` 不追加这份 counter，它的 block counter 是前面复制的更新链 arg。每条核内边再追加一个初值 0 的 i32，for 和 while 都有，核内 ssbuffer 条件用这个。同一个主循环里既有 CUBE 又有 VECTOR，这一步失败。

### `updateIfConds`

`allocSSBuffer` 在 cube 和 vector 函数入口都把每个核间组的两个 bank 清零。
然后每个 if 的 `%true` 换成真实条件，then 多 yield 它改过的计数。

```mlir
// 新增，cube 和 vector 函数入口都有。组号 0：bank0 在 0，bank1 在 1024
%p0 = hivm.hir.pointer_cast(%c0) : memref<i32>
memref.store %c0_i32, %p0[] : memref<i32>
%p1 = hivm.hir.pointer_cast(%c1024) : memref<i32>
memref.store %c0_i32, %p1[] : memref<i32>

// cube if：生产者。输入条件是 %true，一个结果
// 输出
%b0 = arith.cmpi slt, %load0, %c2 : i32     // < multi_buffer，两个 bank 都比
%b1 = arith.cmpi slt, %load1, %c2 : i32
%banks = arith.andi %b0, %b1 : i1
%in = arith.cmpi slt, %cnt, %ub : i32       // block counter < 原上界
%cond = arith.andi %banks, %in : i1
%a:2 = scf.if %cond -> (i32, i32) {
  ....
  %u0 = arith.addi %load0, %c1 : i32
  memref.store %u0, %p0[] : memref<i32>
  %u1 = arith.addi %load1, %c1 : i32
  memref.store %u1, %p1[] : memref<i32>
  %next = arith.addi %n0, %step : i32
  %cnt.next = arith.addi %cnt, %step : i32
  scf.yield %next, %cnt.next : i32, i32
} else {
  scf.yield %n0, %cnt : i32, i32
}
scf.yield %a#0, %a#1 : i32, i32

// vector if：消费者。比的是 > 0，then 里 store load - 1
// 核内边不走指针：生产者 var < slot，消费者 var > 0，加减 1 后 yield
```

核间 buffer 大于 1 且核内 buffer 大于 2 时，第三次核间跳落到的 if 再 AND
一条 flowOpt，读的是起点 if 的 counter：

```mlir
// %cOpt 是 C++ 里算好的 min(核内 buffer - 1, 核间 buffer)，IR 里是常量
%cOpt = arith.constant 2 : i32
%offset = arith.muli %step, %cOpt : i32
%limit = arith.addi %lb, %offset : i32
%filled = arith.cmpi sge, %startCnt, %ub : i32
%ahead = arith.cmpi sge, %startCnt, %limit : i32
%flow = arith.ori %filled, %ahead : i1
```

`scf.while` 没有 counter `< ub`，改成用这个 if 自己那份 before 参数重算原条件。

### `updateLoopIterTimes`

`scf.for` 的上界被拉长。cube、vector 配对的两个循环用同一个更大的上界。
if 体里用到循环归纳变量的地方换成这个 if 自己的 block counter。

```mlir
// 输入：上界仍是 %ub，if 里可能直接用 %i
scf.for %i = %lb to %ub step %step iter_args(...) { .... }

// 输出：上界按「循环次数」那一节的式子拆开。无符号 / signless 用 ceildivui，有符号用 ceildivsi
%diff = arith.subi %ub, %lb : i32
%iters = arith.ceildivui %diff, %step : i32
%scaled = arith.muli %iters, %cReq : i32
%ceiled = arith.ceildivui %scaled, %cX : i32
%count = arith.addi %ceiled, %cIfCount : i32
%span = arith.muli %step, %count : i32
%newUb = arith.addi %lb, %span : i32
scf.for %i = %lb to %newUb step %step iter_args(...) {
  .... // if 内部原来用 %i 的地方换成这个 if 的 %cnt
}
```

`scf.while` 不上拉。before 条件变成每个 if 一份拷贝的 OR：还有一个 block
没结束，循环就继续。

```mlir
// 输入
%cond = arith.cmpi slt, %arg, %bound : i32
scf.condition(%cond) ...

// 输出：每个 if 用自己的拷贝重算一份，再 ori 起来交给 condition
%c0 = arith.cmpi slt, %arg, %bound : i32
%c1 = arith.cmpi slt, %arg.c, %bound : i32
%any = arith.ori %c0, %c1 : i1
scf.condition(%any) ...
```

### `clearSsbufAttrs`

前面都成功才跑。失败时这些属性还在。

```mlir
// 输入，if 上还带着
} {hivm.ssbuffer, hivm.ssbuffer_outs = [...], no_inline}

// 输出
} {no_inline}
```

删的是 `hivm.ssbuffer`、`hivm.ssbuffer_ins`、`hivm.ssbuffer_outs`。
`hivm.loop_core_type`、`no_inline`、`hivm.cv_pipelined_loop` 留下。

## 加条件

一个 if 的条件是下面几段 `arith.andi` 起来的结果。依赖已经是 if 对 if，
`getInputOutputValues` 直接比较 if，不再像 TA 那样走进 if 里找 Fixpipe/Copy。

### 核间

每个核间组对应 SSBuffer 上一个 4 字节计数，组号是它在 `crossCoreBuffers` 里
按 `id` 升序的下标。两个 bank：sub block 0 在 `组号 * 4`，sub block 1 在
`1024 + 组号 * 4`。cube 看两个 bank，vector 只用 `get_sub_block_idx` 算自己的。

- 消费者：`load > 0`，then 里 `store load - 1`。
- 生产者：`load < buffer 数`，then 里 `store load + 1`。

cube 和 vector 的函数入口都清零。

### 核内

用 `UpdateLoopOps` 追加的 i32 迭代参数，初值 0。两端核类型相同的边走这里，包括跨轮传值。

- 消费者：`var > 0`，then yield `var - 1`。
- 生产者：`var < buffer 数`，then yield `var + 1`。

同一个变量被前后两个 if 都改时，后一个读的是前一个 if 的结果。

### block counter

- `scf.for`：block counter 是 `UpdateLoopOps` 追加的那个 i32，初值是下界。条件是 `counter < 原上界`，then yield `counter + step`。
- `scf.while`：不追加 block counter。block counter 是更新链 arg。`processWhileCondArgs` 给每个 ssbuf if 复制一份 before 条件读过的 iter arg，记在 `whileBlockArgMap[while][if][新下标] = 旧下标`。这个 if 用自己那份拷贝把 before 条件再算一遍：这套参数走完，这个 if 就停。条件一个 iter arg 都没用时直接跳过，不写这张表。

### flowOpt

核间 buffer 数大于 1 且核内 buffer 数大于 2 时，从入度为 0 的 if 出发、跨三次
核间边到达的那个 if 会多一条：
`counter >= 原上界 || counter >= 下界 + step * min(核内 buffer - 1, 核间 buffer)`。

## 循环次数

下面是上界的含义，不是 IR。IR 里拆成 `arith.subi`、`arith.ceildivui`（有符号是 `arith.ceildivsi`）、`arith.muli`、`arith.addi`，见 `updateLoopIterTimes`。

```
newUb = lb + step * (ceil(ceil((ub - lb) / step) * requiredBuffers / x) + ifCount)
```

`requiredBuffers / x` 取所有边上最宽的比值，比较时交叉相乘：

- 核内边（两端同核，含跨轮传值）：`|消费者位置 - 生产者位置| + 1`，`x` 是 buffer 数。
- 核间边：两边都只数含 `sync_block_set` / `sync_block_wait` 的 if。距离是
  `消费者位置 - 生产者位置 + 1`，本侧先跑再减 1。消费者排在生产者前面时退回 1/1。

cube 侧第 i 个主循环和向量侧第 i 个主循环是一对，取同一个更大的次数。

`scf.while` 没有上界，不拉长。`UpdateWhileLoopCondition` 对每个 ssbuf if 用它的
私有拷贝克隆一份 before 条件，再 `arith.ori` 起来交给 `scf.condition`：还有一个
block 没结束，while 就继续。条件没用到 iter arg 时到不了这一步：
`buildWhileCounterCondition` 在 `whileBlockArgMap` 里找不到这个 while，pass 失败。

## 和 TA 的不同

对照的是提交 `70326674` 和 Triton Ascend `AddControlFlowCondition`。
TA 的源码在 `third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/`。
下面按阶段写到函数。同名或同职责的函数写行为差别；只在一边存在的函数单独列出。

### 入口和顺序

`applySsbufControlFlow`（`SsbufControlFlow.cpp`）由 `hivm-create-preload`
在看到 `hivm.ssbuffer` 时直接调用。状态是栈上的 `ControlFlowConditionInfo`，
失败是 `return failure()`，create-preload 再 `signalPassFailure`。成功后
`clearSsbufAttrs` 删掉 `hivm.ssbuffer`、`hivm.ssbuffer_ins`、
`hivm.ssbuffer_outs`。

TA 的 `AddControlFlowConditionPass::runOnOperation` 用 `PassManager` 挂子
pass，`ControlFlowConditionInfo` 用指针传下去。模块上已有 fallback 属性就直接
返回。先跑 `verifyControlFlowPrerequisites`：scope 带 `ssbuffer.skip` 则整段
不做。子 pass 失败时 `setFallbackAttr`，后面的子 pass 看到 fallback 自己返回，
不 `signalPassFailure`。父 pass 里没有清属性；清 `ssbuffer.*` 是外面的
`RemoveSsbufAttr`。

顺序：

| 顺序 | 这里 | TA |
| --- | --- | --- |
| 0 | 没有 | `CloneOpsPass` |
| 1 | 没有共享标量拆分 | `ProcessArgsPass`（while 条件拷贝也在这一步，CreateIf 之前） |
| 2 | `createIfOps` | `CreateIfOpsPass` |
| 2.5 | `processWhileCondArgs`（CreateIf 之后） | 已在 ProcessArgs 里做完 |
| 3 | `initDependentMap` | `InitDependentMapPass` |
| 4 | `updateLoopOps` | `UpdateLoopOpsPass` |
| 5 | `updateIfConds` | `UpdateConditionInfoPass` |
| 6 | `updateLoopIterTimes` | `UpdateLoopIterTimesPass` |
| 7 | `clearSsbufAttrs` | 没有。`SplittedIfWrapSsbufIfPass` 在拉长之后 |

我们整段不做的 pass：`CloneOps`、`SplittedIfWrapSsbufIf`。共享标量拆分也不做，
CV pipelining 在进这条路径之前把共用标量拆开并写好 `ins` / `outs`。

### `CloneOps`：整段没有

TA `CloneOps.cpp` 在建 if 之前，按 `ssbuffer.block_id` 把跨 block 共用的 op
克隆成每个 block 一份，再删掉不属于本核的副本。

只在 TA 里的函数：

- `cloneOpWithMapping`：按 SSA mapping 克隆一条 op。
- `cloneOpsInMainLoop`：主循环里按 block 克隆。
- `shouldEraseOpForCube`、`shouldEraseOpForVector`：副本落在错误核上就删。
- `cleanupClonedOpsInMainLoop`：删这些副本。
- `validateClonedSyncOpsErased`、`validateClonedOpsInVector`：删完再检查。
- `areBlockIdsConsecutive`、`validateBlockIdsConsecutive`：block id 必须连续。

我们的 scope 在进这条路径时已经是一段一块，没有按 block id 克隆这一步。

### `ProcessArgs` 和 `processWhileCondArgs`

TA `ProcessArgsPass::runOnOperation` 做两件事，都在 CreateIf 之前，映射的内层
键是 `ssbuffer.block_id`。

共享标量，我们没有这些函数：

- `processSharedIterArgs`、`processSharedIterArgsInLoop`
- `processSharedArgsInForOp`、`processSharedArgsInWhileOp`
- `prepareSharedArgsData`、`processSharedArgsIteration`
- `collectArgIndexToBlockIds`、`findSharedArgs`
- `buildCompInfoForSharedArgs`、`replaceIterArgsInBlock`
- `findYieldDefiningOp`、`collectChainOps`
- `updateIndependentCondsInWhileBlocks`
- `replaceMainLoopOpAndErase`、`getOpIterParams`

while 条件拷贝两边都有，职责相同，做法不同。

| | 这里 `processWhileCondArgs` | TA `processWhileIterArgsInWhileOp` |
| --- | --- | --- |
| 时机 | `createIfOps` 之后，键是 ssbuf if | CreateIf 之前，键是 block id |
| 空条件 | `condUsedArgs` 为空则 `continue`，不写 `whileBlockArgMap` | `collectConditionUsedIterArgIndices` 为空则直接成功，不写表 |
| 更新链克隆进哪 | `unwrapIfYield` 先从 while yield 穿过 if 结果，拿到 then yield 上的原更新。`collectChain` / `orderedChain` / `topologicalSort` 排好这条链。`cloneChainIntoIf` 把整条链克隆进这个 ssbuf if 的 then，插在 then yield 前面，链里读的是该 if 自己的新 after arg。定义 op 的每个结果都放进 mapping | `findLastOpWithBlockId` 找到该 block id 的最后一条 op，`cloneUpdateChainForWhileBlock`、`cloneWhileBlockChains` 把更新链克隆进那个 block。只把 `getResult(0)` 放进 mapping。消费者读到的是生产者的另一个结果时，克隆出来的条件仍指向生产者本轮的新值 |
| 克隆结果怎么交回 while | `appendIfResults` 把克隆出来的值加进 then yield，成为 if 的新结果；else yield 这份新 after arg，本轮不更新。while 在新下标上 yield 这个 if 结果 | 克隆链的结果直接成为该 block 交回 while 的值，不经过一层已经存在的 if |
| 嵌套循环 | `collectAfterSsbufIfs` 从这一个 while 往下走，while 外面的 if 不会进到这里。`enclosingLoop` 取最近的 `scf.for` / `scf.while`：if 再套在内层循环里时，最近的循环不是这个 while，这份 if 不复制条件 arg。`inNestedLoop` 用在收更新链时，链上落在内层循环里的 op 不克隆进 if | `getBlockIdsInOrder` 只扫这个 while 的 after body 的直接子 op，按 block id 收集。不单独判断内层 for/while |

后面 `buildWhileCounterCondition` 两边一样：表里没有这个 while 就失败。TA 失败
变成 fallback；我们是 `return failure()`。

### `CreateIfOps`

两边都把一块计算收成 `scf.if`：then 是原计算，else 是主循环里对应的 iter arg。

| | 这里 | TA `CreateIfOpsPass` |
| --- | --- | --- |
| 分组 | 一段 `hivm.ssbuffer` scope 就是一个 if。`convertScopeToIf` | `createIfInMainLoop` 按 `ssbuffer.block_id` 把循环体里的 op 分组 |
| 标记 | `copyDiscardableAttrs` 把 scope 上的属性抄到 if，其中包含 `hivm.ssbuffer`。不写 `ssbuffer.if` | if 上写整数 `ssbuffer.if`。CreateIf 还会标 `hivm.matmul_limited_in_cube`。我们不写这个属性，`NormalizeMatmul` 在 create-preload 之前 |
| yield | `computeYieldValues`、`findIterArgInMainLoop` 按 scope 的结果对 iter arg | `computeYieldValues`、`findIterArgInMainLoop` 扫描逃出 block 的值 |
| 搬 body | `moveOpsToThenBranch` 用 `migrateBody` | `moveOpsToThenBranch` 按 block 把 op 移进 then |
| 外部使用 | `replaceExternalIfOpUses` | `replaceExternalIfOpUses`、`isUsedOutsideRegion` |

只在我们这边：

- `isOtherCoreSkeleton`：AIC 函数里的 VECTOR scope，或 AIV 函数里的 CUBE scope。
- `eraseOtherCoreSkeleton`：这种空壳结果没有使用就删掉，有使用则失败。
- `nestedIn`：`walk` 是先把所有 ssbuf scope 收进列表，再删。
- `isOtherCoreSkeleton` 只有两对才为真：函数是 AIC 且 scope 是 VECTOR，或函数是 AIV 且 scope 是 CUBE。函数没有 `hivm.func_core_type`，或 scope 没有 `hivm.loop_core_type`，函数直接返回 false，scope 保留。

TA 没有删另一侧空壳。它的 `createIfOpForBlock` 按 block id 建 if，我们的同名
函数按 scope 的结果类型建 if。

### `InitDependentMap`

两边都读依赖、分成核间和核内、建 `ifBlockDAG`、查环、收 flowOpt 对。

| | 这里 | TA |
| --- | --- | --- |
| 输入 | `collectDepsById` 读 if 上的 `hivm.ssbuffer_ins` / `hivm.ssbuffer_outs`。字典键是 `dep_id`、`multi_buffer`。一个整数 id 一对一 | `buildCrossCoreProducerConsumerMapping`、`buildIntraCoreProducerConsumerMapping` 读 op 上的 `ssbuffer.crossCoreDeps` / `ssbuffer.intraDeps`，值是 `[group, role]`。一个 group 可以有多个生产者 |
| 落点 | 边的两端已经是 if。`getOrCreateEdge` 按 id 合成 `DependencyGroup` | `findIfOpContainingOp` 再从 op 找到包住它的 `ssbuffer.if` |
| 表 | `crossCoreBuffers`、`intraCoreBuffers`，元素是 consumer if + 一个 producer if | `crossCoreDependentMap`、`intraCoreDependentMap`，consumer op 到生产者 op 列表 |
| buffer 数 | `computeProducerBufferCount`：该侧 group 里生产者 `multi_buffer` 的最大值。没有核内边则 intra 保持 0 | 同名函数取生产者列表长度的最大值。核内表为空时读 `BufferCountManager` 的 IntraCore，模块属性 `ssbuffer.intra_buf_count`，缺省 3。不读 `ssbuffer.inter_core_buf_count` |
| 跨轮传值 | 和核内同一条路，`addGroupEdges` 按两端 `hivm.loop_core_type` 是否相同决定核间还是核内 | 核内只收同核标签。跨轮传值是后面 `analyzeTensorIterArgDependencies` 扫 iter arg，得到 `tensorIterArgDepsMap`，不是这条标签 |
| 环 | `dfsCycle`、`detectCycle`，发现环则失败 | `detectCycle`，失败则 `setFallbackAttr` |
| 要不要收 flowOpt | `crossCoreBufferCount > 1` 并且 `intraCoreBufferCount > 2` 才调用 `collectFlowOptIfOpPairs`。没有核内边时 intra 是 0，`0 > 2` 不成立，这个函数不跑 | 判断式相同。没有核内边时 intra 用 `ssbuffer.intra_buf_count`，属性不存在就是 3。`3 > 2` 成立，核内这一条过了。核间还要 `> 1`，核间只有 1 时仍然不跑 |
| 收的时候怎么数 | 从入度 0 的 if 出发，起点深度是 1。走一条核间边，深度加 1。走一条核内边，深度不变。深度正好是 3 的 if 和起点配成一对，写进 `flowOptIfOpPairs`。3 是写死的 | 同名函数，加法和目标深度与这里相同 |

只在 TA：`collectMainLoopById`、`isConsumerInMainLoop`、`sortedGroupIds`、
`printDependentMaps`、`printDAGInfo`。我们不按 `CVPipeline::kMainLoop` 的 id
收集主循环，也不打印这两张调试表。

`addIfBlockEdge`、`buildIfBlockDAG` 两边都有。我们的边是 if 到 if；TA 先从
op 找到 if 再加边。

### `UpdateLoopOps`

两边都给主循环追加迭代参数，并在主循环前后插 `(S, S)`，flag 是 15，set/wait
的 pipe 都是 `PIPE_S`。

| | 这里 | TA |
| --- | --- | --- |
| 追加什么 | `addBlockCountersAndInnerDepConds`。`scf.for` 的 block counter 个数等于 ssbuf if 个数，初值是下界。`scf.while` 的 block counter 个数是 0，不再追加一份 i32(0)。核内变量个数等于该循环的 intra group 数，初值 0。跨轮传值用的就是这份核内变量 | `deriveBlockCountersFromIfOps`、`addBlockCountersAndInnerDepConds`。for 和 while 都追加 block counter。另外 `analyzeTensorIterArgDependencies`、`recordMainLoopTensorIterArgs` 为每个 consumer 追加迭代依赖变量，初值 1 |
| 重建循环 | `buildMainLoopExtraInitArgs` 只填初值，真正重建在 `SsbufUtils.cpp` 的 `createMainLoopOpAndMigrateBody` | `computeMainLoopExtraArgs`、`buildMainLoopExtraInitArgs`、`getMainLoopBaseIdx`、`createMainLoopOpWithExtras`，再 `migrateWhileRegions` / `extendWhileCondition` / `extendWhileYield`，然后 `transferMainLoopInfoMaps` |
| `(S, S)` | `insertInterCorePipeS` 遍历主循环。`getMainLoopCoreType` 看这个循环里面 ssbuf if 的 `hivm.loop_core_type`。CUBE 和 VECTOR 都有就失败。只有一种时，`insertPipeSForMainLoopOp` 固定 cube 先跑：VECTOR 在循环前 set，CUBE 在循环后 wait，循环体内两边都是先 wait 再 set | `insertInterCorePipeS` 先找带 `hivm.tcore_type` 的 `scope.scope`，再找这个 scope 里面的主循环，用 scope 自己的核类型插。scope 是 CUBE 或 VECTOR 时，有 `kVectorFirst` 就 vector 先跑，没有就 cube 先跑。核类型不是这两类就什么都不插 |
| 插 sync 的帮手 | `insertSyncOpsInsideMainLoop`、`insertSyncOpsOutsideMainLoop`，set pipe 和 wait pipe 是同一个 `PIPE_S` | 同名两个函数，set pipe 和 wait pipe 分开传，实际都传 `PIPE_S`。插入失败会返回失败 |

只在 TA：`replaceMainLoopOpUsesAndErase`、`recordMainLoopBlockCountersAndConds`。
我们没有 tensor iter arg 这一类变量，所以没有 `analyzeTensorIterArgDependencies`
和 `recordMainLoopTensorIterArgs`。

### `updateIfConds` 和 `UpdateConditionInfoPass`

两边都把 if 的恒真条件换成核间、核内、flowOpt 的与，并让 yield 把更新后的
计数带出循环。

核间条件相同：消费者 `> 0` 再减，生产者 `< bufferCount` 再加。SSBuffer 一个
计数 4 字节，vector 的第二份从 1024 起。cube 读两个 bank，vector 用
`get_sub_block_idx` 只读自己的 bank。

| | 这里                                                                                                                                               | TA |
| --- |--------------------------------------------------------------------------------------------------------------------------------------------------| --- |
| 分配 | `allocSSBuffer`：有核间组时，每个带 ssbuf if 的函数在入口把两个 bank 都 store 0。cube 函数把指针留在 `ssbufferPtrs`。普通函数没有 ssbuf if，不写                                       | `allocSSBuffer`：在第一个 scope 前分配并清零一次。两个核当时还在同一个模块里，所以只清一次 |
| 指针 | `createPointerCastOp`、`computeVectorSSBufferMemref`、`getSSBufferMemrefs`、`createSsbufLoad`                                                       | `createPointerCastOp` 在 `SSBufferManager.h`。`computeVectorSSBufferMemrefs`、`getSSBufferMemref`、`createSsbufLoads` |
| 核间 | `getInputOutputValues` 直接拿 if。`addCrossCoreConditions`、`updateCrossCoreControlVars`、`setCrossCoreCondition`                                      | 同名四个函数。先 `buildIdxToVarMap`、`buildBufferDependencyMappings`，op 再经 `findIfOpContainingOp`。`getVarValue` 按迭代参数下标取值 |
| 核内 | `setSlotCondition`、`setIntraCoreCondition`。初值 0，比较式和核间一样。跨轮传值走这里                                                                                 | `collectIntraCoreInputConditions`、`collectIntraCoreOutputConditions`、`setIntraCoreCondition`。比较式相同 |
| 迭代依赖 | 按照核内依赖处理                                                                                                                                         | `buildTensorIterArgIfOpVarMap`、`collectTensorIterArgInputConditions`、`collectTensorIterArgOutputConditions`。消费者 `var == 1` 再减，生产者对每个消费者 `var == 0` 再加，初值 1，不看 `multi_buffer` |
| flowOpt | `setFlowOptCondition` 只对 `scf.for`。条件是起点 if 的 block counter：`counter >= 原上界` 或 `counter >= 下界 + step * min(intra - 1, cross)`，再和原条件与             | 同名函数，式子相同。intra 可能是 `BufferCountManager` 的 3，所以 `intra - 1` 在缺省配置下是 2 |
| while 的块条件 | `cloneConditionDefChain`、`buildWhileCounterCondition`。表里没有 while 则失败                                                                             | 同名两个函数，另有 `getLoopRegionIterArgs`、`getLoopRegionIterArg`。缺失同样失败，但是 fallback |
| 换 if | `combineConditions` 与起来。`populateNewThenBlock`、`populateNewElseBlock`。`retargetIfOp`、`retargetGroups`、`updateDAGAfterIfOpReplacement` 把表的键改到新 if | `combineConditions`、`createNewIfOpWithBlocks`、`buildNewIfResultTypes`、`collectYieldOperands`、`populateNewThenBlock`、`populateNewElseBlock`、`updateDAGAfterIfOpReplacement`、`updateControlVarToLatestValue`。另有 `collectSSBufferIfOps`、`validateBlockCounters`、`getIfBlockId` |
| yield | `updateLoopYield` 一个函数覆盖 for 和 while                                                                                                             | `updateForOpYield`、`updateWhileOpYield`，由 `updateLoopYield` 分发 |

只在 TA、只为调试：`logConditionGroupIndices`、`logOutputGroupValues`。
`findTcbGroupId` 在这个 cpp 里还有一份，用来对 tightly coupled buffer 分组；
我们没有 TCB 分组。

### `UpdateLoopIterTimes`

上界公式两边相同：

```
newUb = lb + step * (ceil(ceil((ub - lb) / step) * requiredBuffers / x) + ifCount)
```

`requiredBuffers / x` 取最宽的一条边，交叉相乘比较。`scf.while` 都不拉上界。

| | 这里 | TA `UpdateLoopIterTimesPass` |
| --- | --- | --- |
| 配对 | `getMainLoopIdToLoopOpMap` 按模块里的出现顺序编号。cube 第 i 个和 vector 第 i 个是一对，核类型用 `getMainLoopCoreType` | `GetMainLoopIdToLoopOpMap` 读 `CVPipeline::kMainLoop` 上的 id。`getOtherScopeMainloop`、`findOtherSideMainloopAndIfOps` 按这个 id 找对侧 |
| 收集 if | `collectIfOps`。核间只用 `syncBlockFilter`，只数带 `sync_block_set` / `sync_block_wait` 的 if。`defaultIfOpFilter` 用于核内 | 同名过滤函数，规则相同 |
| 核内距离 | `calculateIntraDepsFactor`：`\|consumerIdx - producerIdx\| + 1`。消费者排在生产者前面也算，当成跨轮传值，不失败。`x` 是 `multi_buffer` | `calculateIntraDepsFactor`：要求 `ConsumerIdx > producerIdx`，否则失败。`x` 是生产者 op 个数 |
| 核间距离 | `calculateCrossDepsFactor`：`consumerIdx - producerIdx + 1`，本侧 `isRunFirst` 再减 1。消费者下标小于生产者下标时退回 `{1, 1}` | 同名函数，这条规则相同。生产者下标用 `getProducerIfOpIndex`，一组里多个生产者取一个代表 |
| 迭代依赖 | 没有。跨轮传值已经在核内因子里 | `calculateIterDepsFactor`：先消费再生产，`producerIdx - consumerIdx + 1`，参与同一轮 max |
| 合成 | `calculateFactor` 把核内、核间的比值取更宽的一对。对侧主循环或过滤后的 if 找不到则 `{-1, -1}`，本阶段失败 | 同名函数。还会并进 `calculateIterDepsFactor`。没有依赖时 `{1, 1}` |
| 改上界 | `computeNewLoopUpperBound`、`cloneForOpWithNewUpperBound`、`extendForOpIterationCount`、`updateCntArgsAfterClone`。一对循环用 `collectForOpsAndUpdateMax` 取两边更大的 `ifCount` 和更宽的比值，`UpdateForLoopIteration` 写到两侧 | 同名函数都是 pass 方法。`UpdateForLoopIteration`、`collectForOpsAndUpdateMax`、`updateMainLoopMaps` 做同一件事 |
| 归纳变量 | `replaceForOpCounterInIfOps`：if 里面用到的 for 归纳变量换成该 if 的 block counter | 同名函数，同一件事 |
| while | `UpdateWhileLoopCondition`：每个 ssbuf if 用私有拷贝克隆 before 条件，`arith.ori` 后交给 `scf.condition`。while 不在 `whileBlockArgMap` 或某个 if 没有条目则失败 | 同名函数。while 缺失或该 block 的映射为空返回 -1，再变成 fallback |

`getConsumerIfOpIndex` 两边都有。TA 另有 `findIfOpIndexInList`，因为 if 的编号
表是 `DenseMap<Operation *, int>`；我们的 `getIfOpIndex` 在 `SsbufUtils.cpp`，
直接在 if 数组里找 1-based 下标。

### `SsbufUtils` 和 TA 的 `Utils`

`SsbufUtils.cpp` 是给打标签的 pass 和这条改写一起用的。TA 的对应头是
`AddControlFlowCondition/Utils.h`，另有 `Common/Utils.h` 里的 `isMainLoopOp`。

两边都有、职责相同的：

- `migrateBody`：把旧 block 的非终结 op 搬到新 block，参数按位置替换。
- 主循环重建。我们是 `createMainLoopOpAndMigrateBody`，内部 `createForOpAndMigrateBody`、`createWhileOpAndMigrateBody`，搬完 body 就 `retargetMainLoop`。TA 拆成 `createNewForOpWithExtras`、`createNewWhileOpWithExtras`、`createMainLoopOpWithExtras`、`migrateWhileBodies`、`buildNewYieldOp`、`buildNewWhileCondition`、`replaceOpResultUses`，调用方自己迁表。

只在我们这边：

- 标签：`isSsbufScope`、`isSsbufIfOp`。判断的是 `hivm.ssbuffer`，不是 `ssbuffer.if`。
- 主循环：`isMainLoopOp` 看 `hivm.cv_pipelined_loop`。TA `CVPipeline::isMainLoopOp` 看 `ssbuffer.main_loop`。
- `getParentMainLoop`、`getMainLoopBody`、`getNumIterArgs`、`getLoopRegionIterArg`、`getIfCoreType`、`copyDiscardableAttrs`。
- `getMainLoopCoreType`：一个循环里的 ssbuf if 核类型不一致则返回空，并把 `mixed` 设上。TA 不需要这个函数，`(S, S)` 按 scope 的 `hivm.tcore_type` 插。
- `retargetMapKey`、`retargetMainLoop`、`retargetSsbufBlock`。
- `getExistingTerminator`、`eraseTerminator`、`ensureYield`。TA 用 `Block::getTerminator`，ForOp 带 iter arg 时新建 body 没有终结符，我们这两支是为了避开那个断言。
- `getIfOpIndex`。

只在 TA 的 `Utils.h`，因为它们靠 block id：

- `collectAllNestedOps`、`collectOpsByBlockId`、`getBlockIdsInOrder`
- `countUniqueIfBlockIds`、`getLoopDirectChildBlockId`
- `topologicalSort` 的两个重载。我们只在 `ProcessWhileCondArgs.cpp` 里有一份，只排序 while 条件的更新链。
- `findTcbGroupId`、`getScopeType`、`isIfOpWithOnlySyncOps`、`dumpWhileBlockArgMap`

### 状态结构

TA `ControlFlowConditionInfo` 的键很多是 op 和 block id：
`crossCoreDependentMap`、`intraCoreDependentMap`、`tensorIterArgDepsMap`、
`tensorIterArgIndicesMap`、`blockCounterNums`。`whileBlockArgMap` 是
while 到 block id 再到参数下标。

我们的表以 if 为主：`crossCoreBuffers`、`intraCoreBuffers` 是
`DependencyGroup`（一个 consumer if、一个 producer if、`bufferCount`）。
另有 `ifOpsByMainLoop`、`ssbufferPtrs`、`vectorSSBufferPtrs`、
`controlVarToLatestValue`。没有 tensor iter arg 的两张表，也没有
`blockCounterNums`。`whileBlockArgMap` 的内层键是 if。拉长用的
`IterationTimesInfo`（`ifCount`、`requiredBuffers`、`x`）只活在
`updateLoopIterTimes` 的局部 map 里，不放进 `ControlFlowConditionInfo`。
