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

下面是 `CreatePreload` 输入 dump 里的四段 scope。AIC 循环里那一段
空的 VECTOR scope 只打了 `annotation.mark`。SplitMixKernel 会留下这个空壳，
CreateIfOps 在转 if 之前把它删掉，不参与后面的依赖。

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

1. **CreateIfOps**。`hivm.func_core_type<AIC>` 函数里、`hivm.loop_core_type`
   为 VECTOR 的 `hivm.ssbuffer` scope 直接删掉。这是 SplitMixKernel 留在 AIC
   里的空壳，没有结果。其余每个 `hivm.ssbuffer` scope 变成条件恒为 `true` 的
   `scf.if`。属性（`ins` / `outs` 标签在内）搬到 if 上，再加单位属性
   `ssbuffer.if`，scope 删掉。没有 block id，也不写
   `hivm.matmul_limited_in_cube`。then / else yield 见 `createIfOps`。AIV 函数里的 VECTOR
   scope 照常转 if。
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
   `hivm.ssbuffer_outs`、`ssbuffer.if`。

## 各函数的输入和输出

下面按 `applySsbufControlFlow` 的调用顺序。每段只画这一步改掉的地方，
`....` 是上一步已经稳定、这一步不动的计算。例子是一条核间边：cube 生产
`dep_id = 1`，vector 消费，`multi_buffer = 2`。`scf.for` 原来是
`%lb` 到 `%ub`，step `%step`，一个 iter arg `%n0`。

`Utils.cpp` 里的 `migrateBody` 只在重建循环时把旧 block 里的算子搬过去，
不单独改变语义。

### `createIfOps`

AIC 函数里、`loop_core_type` 为 VECTOR 的空壳直接消失。结果还有人用就失败。

```mlir
// 输入：AIC 函数里
scope.scope { .... } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.ssbuffer}
// 输出：这段 scope 没有了
```

其余 `hivm.ssbuffer` scope 变成条件恒真的 if。标签跟过去，多一个 `ssbuffer.if`。有人用的结果才 yield：then 交 `scope.return` 里对应的操作数，else 交主循环 `scf.yield` 同一下标的 iter arg。没人用的结果丢掉；全部没人用时 if 没有 else。

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
   ssbuffer.if, no_inline}
```

### `processWhileCondArgs`

`scf.for` 原样通过。`scf.while` 给每个 ssbuf if 复制一份 before 条件读过的
iter arg。条件一个 iter arg 都没用时，IR 不动，只在 `whileBlockArgMap` 里
给每个 if 记一条空映射。

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

每个 ssbuf if 追加一个 i32。`scf.for` 初值是下界，if 用它做 `counter < 原上界`，then 里 `+ step`。`scf.while` 这份初值是 0，if 条件不读它；while 的 block counter 是前面复制的更新链 arg。每条核内边再追加一个初值 0 的 i32，核内 ssbuffer 条件用这个。同一个主循环里既有 CUBE 又有 VECTOR，这一步失败。

### `updateIfConds`

`allocSSBuffer` 只在 cube 函数入口把每个核间组的两个 bank 清零。然后每个 if
的 `%true` 换成真实条件，then 多 yield 它改过的计数。

```mlir
// 新增，cube 函数入口。组号 0：bank0 在 0，bank1 在 1024
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
} {hivm.ssbuffer, hivm.ssbuffer_outs = [...], ssbuffer.if, no_inline}

// 输出
} {no_inline}
```

删的是 `hivm.ssbuffer`、`hivm.ssbuffer_ins`、`hivm.ssbuffer_outs`、`ssbuffer.if`。
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

只在 cube 侧函数入口清零。

### 核内

用 `UpdateLoopOps` 追加的 i32 迭代参数，初值 0。两端核类型相同的边走这里，包括跨轮传值。

- 消费者：`var > 0`，then yield `var - 1`。
- 生产者：`var < buffer 数`，then yield `var + 1`。

同一个变量被前后两个 if 都改时，后一个读的是前一个 if 的结果。

### block counter

- `scf.for`：block counter 是 `UpdateLoopOps` 追加的那个 i32，初值是下界。条件是 `counter < 原上界`，then yield `counter + step`。
- `scf.while`：block counter 是更新链 arg。`processWhileCondArgs` 给每个 ssbuf if 复制一份 before 条件读过的 iter arg，记在 `whileBlockArgMap[while][if][新下标] = 旧下标`。这个 if 用自己那份拷贝把 before 条件再算一遍：这套参数走完，这个 if 就停。后面那个初值 0 的 i32 不参与这个比较。条件一个 iter arg 都没用时 IR 不动，仍给每个 if 记一条空映射。

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
block 没结束，while 就继续。条件没用到 iter arg 时，`scf.condition` 保持原样。

## 和 TA 的不同

对应的是 Triton Ascend `AddControlFlowCondition` 里的子 pass。下面按 pass 说现在的差别。

| 阶段 | 这里                                                                                                                                                                            | TA |
| --- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --- |
| 入口 | `applySsbufControlFlow`，由 `hivm-create-preload` 在看到 `hivm.ssbuffer` 时调用。不是一串子 pass。                                                                                           | 一个父 pass，里面 `PassManager` 挂子 pass，用指针传 `ControlFlowConditionInfo`。 |
| 标量拆分 | 不做。CV pipelining 把共用的标量 iter arg 拆开，并写好 `ins` / `outs`。                                                                                                                       | `ProcessArgs` 在 CreateIf 之前拆多个 scope 共用的标量。 |
| CreateIfOps | 一段 `hivm.ssbuffer` scope 就是一个 if。AIC 函数里 `hivm.loop_core_type` 为 VECTOR 的 scope 先删掉，不转 if。`ssbuffer.if` 是单位属性，没有 block id。then yield scope 用到的结果，else yield 主循环里对应的 iter arg。 | 循环体按 `ssbuffer.block_id` 分组。`ssbuffer.if` 带整数编号。yield 靠扫描跨组逃逸的值。没有 AIC 里删 VECTOR scope 这一步。 |
| while 条件拷贝 | `processWhileCondArgs` 在 CreateIf 之后。`whileBlockArgMap` 的内层键是 if。条件没用到 iter arg 时仍给每个 if 插一条空映射。                                                                              | 在 CreateIf 之前做，映射按 block id。空集合直接跳过。 |
| InitDependentMap | 从 if 上的 `hivm.ssbuffer_ins` / `outs` 读。一个整数 `dep_id` 只有一个 `ins` 和一个 `outs`，上限是该 `outs` 的 `multi_buffer`。核类型分核间 / 核内。跨轮传值和核内走同一套。`getInputOutputValues` 直接比 if。                | 从单个 op 上的 `ssbuffer.crossCoreDeps` / `ssbuffer.intraDeps` 读 `[group, role]`。同一个 group 的多个生产者共用一个计数，上限是生产者个数。再 `findIfOpContainingOp`。迭代依赖是 `analyzeTensorIterArgDependencies` 扫 iter arg 得到的 `tensorIterArgDepsMap`，不是标签。 |
| UpdateLoopOps | 追加 block counter 和核内变量。核内初值是 0，跨轮传值也用这个变量。`(S, S)` flag 15，cube 先跑：VECTOR 在循环前 set，CUBE 在循环后 wait。同一个主循环里混有 CUBE 和 VECTOR 直接失败。                                               | 三类：block counter、核内变量、迭代依赖变量。迭代依赖变量初值是 1，个数按每个 consumer 计。`(S, S)` 的先后由 vector-first 决定。 |
| UpdateConditionInfo | 核间走 SSBuffer：消费者 `> 0` 再减，生产者 `< buffer` 再加。核内也是这套，初值 0，跨轮传值同样走核内。只在 cube 函数入口把两份 SSBuffer 清零。                                                                                | 核间、核内的比较式相同。迭代依赖另算：消费者 `var == 1` 再减，生产者每个消费者的 `var == 0` 再加，初值 1，不看 `multi_buffer`。flowOpt 用核内 buffer 数，核内边为空时还会读 `BufferCountManager` 的 IntraCore 计数，缺省 3。指针在第一个 scope 前分配，因为两个核还在同一个模块里。 |
| UpdateLoopIterTimes | 核间边和核内边都拉长 `scf.for`。核内距离是 `\|消费者 - 生产者\| + 1`，消费者排在前面也不失败。跨轮传值算在核内距离里。主循环按模块里的出现顺序配对。                                                                                        | 核内要求消费者在生产者后面，否则失败。迭代依赖另算 `calculateIterDepsFactor`：先消费再生产，`producerIdx - consumerIdx`，并参与拉长。主循环按 `CVPipeline::kMainLoop` 上的 id 配对。 |
| 收尾 | 成功后删掉 `hivm.ssbuffer`、`hivm.ssbuffer_ins`、`hivm.ssbuffer_outs`、`ssbuffer.if`。失败时保留。                                                                                           | 没有这一步。 |
| 不做的 pass | `CloneOps`、`SplittedIfWrapSsbufIf` 不在这里。不写 `hivm.matmul_limited_in_cube`（`NormalizeMatmul` 在 create-preload 之前）。                                                              | 这两个子 pass 都在父 pass 里。CreateIf 会标 matmul-limited。 |
