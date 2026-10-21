# PP 层迁移开发计划：shadow compute、异步状态同步与候选 CUDA Graph 预热

日期：2026-10-19。用途：分阶段设计与验收计划。本文主体仍描述完整目标；下方 implementation checkpoint 记录当前工作区已经落地的原型范围。

## Implementation checkpoint

The current worktree has implemented and tested the following local prototype:

- S0 measurement and CUDA Graph instrumentation.
- S1 fixed `(layer, device)` KV/recurrent state banks and independent complete metadata views.
- S2 bounded active/candidate full-placement runtimes with independent schedulers and handle-swap promotion.
- S3a asynchronous backend copy tickets represented by a pending migration, destination completion events, cancellation, and stale frontier/generation rejection.
- A correctness-first S5 shadow mode that runs the complete candidate model for each ubatch and transactionally restores all state outside the moved interval.
- S6 candidate warmup that runs the real final-address TG graph and restores candidate state tensors and metadata before promotion.
- S4 owned entry-hidden journal and interval-only token replay for Qwen3Next and Qwen3.5-MoE. Implementation details and acceptance status are tracked in `pp-replay-s4.md`.

Current limits are important:

- Async begin synchronizes the active checkpoint, enqueues full allocated tensors, and overlaps those copies with candidate scheduler reserve. It has no event-query API and no incremental valid-row transfer.
- Shadow mode is a semantic reference path, not the final efficient replay engine. It recomputes the full candidate graph and copies shared state through host snapshots. It proves view and promotion correctness but does not yet exploit a PP bubble.
- Warmup is transactional but uses full host snapshots. It is not yet integrated with a tool-window budget or automatic migration controller.
- The S4 journal currently captures synchronously to host memory, and explicit replay pauses the foreground. There is no lag controller, transfer chunking, RPC performance validation, or production server integration.
- A caller must explicitly invoke candidate warmup or enable shadow before commit to avoid a cold first candidate graph. Scheduler reserve is already outside commit in either case.

The implementation is therefore a working semantic-exact infrastructure prototype, not completion of every S0-S8 optimization in this plan.

## 1. 目标、结论与阅读顺序

目标是在预先限定的相邻 pipeline backend 之间移动层边界，将权重准备、KV/递归状态同步、图构建、内存规划和 CUDA Graph 捕获尽量移出迁移提交的关键路径。前台仍然执行同一个模型，不压缩有效上下文、不近似更新状态、不丢弃历史 token。

推荐路线不是在 shadow compute 和异步复制之间二选一，而是：

1. 用异步复制建立可靠的检查点和低风险基线。
2. 用同一套区间重放引擎实现缺口追赶和持续 shadow compute。
3. 用固定状态存储和候选 TG runtime 避免提交时重新 reserve。
4. 单独预热真正用于切换后推理的 CUDA Graph，不能用 shadow 图已预热来替代。
5. 工具执行窗口优先承担较重任务；推理期间只执行预算内的小任务。

“无感”应定义为条件成立时的热切换路径，而不是任何时刻都强制零停顿：当目标状态、候选图或 native graph 未准备好时，继续原位置推理，推迟迁移；显式选择阻塞模式时才走同步回退。

本文中的 PP 若无特别说明指 pipeline parallel。源码 `sched_reserve()` 中的 `pp` 指 prompt processing，二者不是同一概念。TG 指 token generation。

阅读建议：先看第 2 到 5 节理解现状和正确性约束，再按第 6 节的 S0 到 S8 实施。第 7 到 10 节是跨阶段验收与交接要求。

## 2. 基于当前仓库的起点

核对基线为 HEAD `3cfc792db` 加工作区已有修改。开发会话必须重新记录完整 HEAD、分支和工作区差异，不能只检出该提交就认为得到本文描述的状态。

当前工作区已有 `llama-context`、KV cache、recurrent memory、memory 接口和 `tests/test-llama-archs.cpp` 的修改，以及阶段一测量文档。它们不是本计划生成的改动，不得覆盖或清理。

| 能力 | 当前代码事实 | 对后续开发的含义 |
| --- | --- | --- |
| 权重冗余与层放置 | 存在 `layer_on_device()`、预置副本及 `active_layers` | 复用权重基础设施，首版不做按需权重传输 |
| 候选验证 | `validate_candidate_backends()` 枚举候选，临时切换状态和位置，reserve 后恢复 | 目前保存的是通过验证的布局，不是可立即执行的常驻 runtime |
| 同步迁移 | `migrate_layers_sync()` 同步、准备并复制状态、swap、更新 placement epoch、调用 `sched_reserve()` | 保留为正确性基准及显式阻塞回退 |
| 状态搬运 | `prepare_layer_move()` 为目的设备准备存储并复制完整状态张量 | 尚不是固定备用 bank，也不是仅复制有效 KV 增量 |
| reserve | 重建 scheduler 和 graph result，执行 prompt/TG/prompt 的 reserve | 不是只更新一个 layer ownership 标志 |
| 常规 decode | `process_ubatch()` 在允许复用时设置新输入，否则重新 build/alloc | 正常推理也不是每 token 必定重建逻辑图 |
| 激活获取 | 有 `t_layer_inp`、层输入提取接口和本轮输出 buffer | 不是持久激活日志，也不是任意区间重放引擎 |
| CUDA Graph | 根据实际 split graph 的节点身份、UID 和节点属性判断复用/更新 | 缓存一个 `ggml_cgraph *` 不足以保持 native graph 可用 |
| RPC | 客户端识别最近 graph UID；服务端每设备保存最近一张图；没有 `cpy_tensor_async` 实现 | 多 runtime 与后台重放会打断当前复用，不能直接套用本地热切换结论 |

主要代码入口：

- `src/llama-context.cpp/.h`：候选验证、同步迁移、`sched_reserve()`、`process_ubatch()`、层输入提取。
- `src/llama-graph.cpp/.h`：图参数、`allow_reuse()`、图输入与结果对象、实际层副本选择。
- `src/llama-memory.h`、`src/llama-memory-hybrid.cpp/.h`：混合状态及状态迁移抽象。
- `src/llama-kv-cache.cpp/.h`：KV 张量、cell/slot 元数据、层状态复制。
- `src/llama-memory-recurrent.cpp/.h`：递归状态、卷积历史和相关游标。
- `src/models/qwen3next.cpp`、`src/models/qwen35moe.cpp`：逐层计算与混合注意力路径。
- `ggml/src/ggml-backend.cpp`：切图、scheduler 分配、事件与异步复制回退。
- `ggml/src/ggml-cuda/ggml-cuda.cu`：native graph 检测、预热、捕获和执行。
- `ggml/src/ggml-rpc/ggml-rpc.cpp`：RPC 队列、图序列化、服务端图缓存。
- `tests/test-llama-archs.cpp`：复用现有 PP fixture、同步迁移和失败注入测试。

已有测量见 `docs/development/pp-sync-migration-phase1.md`。小型 Qwen35MoE fixture 的一次 mixed 迁移中，复制约 0.172 ms，reserve 约 3.439 ms。这只说明该配置下 reserve 值得优化，不代表真实模型和边缘网络也是同一瓶颈。已有 layer callback 计时会同步并改变执行重叠，不能当作无扰动的 GPU kernel 时间。

此前 `docs/superpowers/plans/2026-10-17-pp-state-reconstruction-migration.md` 可作为背景。涉及本主题时，以本文的依赖顺序、预热隔离和验收条件为准；不要将旧文中的工具/skill 名称当作开发环境已具备的能力。

## 3. 首版范围与非目标

首版限定单序列、append-only、普通 host sampling、相邻 backend 的连续边界层迁移。沿用当前实验路径支持的 Qwen3Next/Qwen35MoE、layer split、预加载权重副本等限制。

第一阶段先在本地双 CUDA GPU 验证；RPC 是独立阶段。首版不同时支持 speculative decoding、多序列连续批处理、任意 KV 删除/移动、context shift、运行中 adapter 更换或任意拓扑重排。遇到这些操作必须明确拒绝，或先取消后台任务并进入已实现的失效处理路径，不能带着旧状态继续热切换。

候选布局必须是显式、有限的白名单。当前从初始布局生成的候选不等于任意多次迁移后都允许继续向外扩张。首版建议一个初始布局加一到两个邻接候选；是否允许 A/B/C 之间的转换，逐条登记并验证。

不用第二个公共 `llama_context` 来偷偷承载 shadow。当前实验放置路径有 model/context guard，应在受控的同一上下文内引入私有执行视图，或明确设计必要的内部抽象，不绕过所有权保护。

“semantic exact”在这里表示执行相同模型语义和完整依赖，不做有损状态近似；它不自动等于浮点逐位一致，也不保证任何采样配置下 token 永远逐个相同。跨设备、kernel 和 chunk 策略的浮点差异可能改变抽样或近似并列的 argmax。需要同时报告固定输入下的 logits/状态误差、确定性生成回归以及真实任务质量；如果要求严格相同 token 序列，还需更强的数值与采样约束。

## 4. 全程不能破坏的正确性约束

### 4.1 统一使用排他 frontier

定义 `F` 为“已经完整处理的 token 数/下一待处理位置”，状态包含 `[0, F)`，下一步处理位置 `F`。检查点 `P` 是处理完 `P - 1` 后的状态，补齐区间为 `[P, F)`。不要混用“最后完成的位置”和“下一位置”。

例：前台已经处理位置 0 到 99，则 `F = 100`。目标仅完成到 `P = 80`，需要重放 80 到 99，不能直接从位置 100 接管。若重放期间前台推进到 105，目标必须继续追到 105，或等下一个安全窗口；不能将旧的 `F = 100` 当作仍然可提交。

针对每次迁移记录：请求 ID、sequence epoch、源 placement/version、候选布局、层区间、目标设备和源/目标 frontier。异步完成消息必须携带这些标识，取消后的旧消息不得推进新请求状态。

### 4.2 状态就绪不等于图就绪

| 就绪项 | 精确定义 |
| --- | --- |
| 权重就绪 | 该布局所有需要的权重副本已经驻留且版本一致 |
| 状态就绪 | 所有迁移层在目的端都具有同一当前 frontier 的完整有效状态与一致元数据 |
| 逻辑图就绪 | 实际下一 TG batch 的逻辑图及输入对象仍然存活、签名匹配 |
| scheduler 就绪 | 对应切分、复制依赖、输入槽和计算 buffer 已分配且未失效 |
| native 就绪 | 会执行的各 CUDA split/输入轮换槽具有有效 native graph，或被明确标记为不支持 |
| 可提交 | 上述条件满足，且 source/target 的相关 GPU 工作完成，没有后台写入会越过提交点 |

必须分开表示这些状态。不得用一个 `ready = true` 掩盖“只 reserve 过”或“只有某个 shadow 图 capture 过”。

### 4.3 固定 bank 必须包含一致的元数据视图

一个状态 bank 不只是 K/V 或 R/S/P 张量的指针集合，还包括解释其内容所需的 cell/slot、位置、序列映射、head/tail、有效长度和模型相关历史。

当前混合 memory 的若干元数据是组件级共享的，不是天然的逐层独立副本。shadow 已到 80、前台已到 100 时，不能共用会被 `apply()` 修改的同一套游标。需要将物理存储和执行元数据视图分开，或采用经过审计的等价设计。未完成这个隔离之前，不开展异步 shadow。

只读权重可以共享；不同执行任务的可变输入、scratch、输出、记忆游标和写入区域不能未经证明就共享。常驻 bank 的 tensor 对象、底层地址、view 与 generation 必须在引用它的 runtime 生命周期内保持有效。

### 4.4 KV 重建依赖真实的层输入

已有历史 KV 本身不能产生下一 segment 的 KV。对于层 `l`，还需要区间 `[P, F)` 中进入该层的 hidden input、正确的绝对位置/RoPE、权重和相应模型操作。

重放连续层 `[a, b)` 时，保存每个 token 进入 `a` 的 pre-norm residual hidden 即可作为入口，但区间内部必须正常计算注意力、残差、FFN/MoE 等，才能得到下一层输入。线性层必须恢复全部递归/卷积历史，而不只是某个递归矩阵。

若末端 full-attention 层只需要生成 K/V、无需向下游输出 hidden，可另做 KV-only 投影优化；这不是任意混合区间都可使用的捷径，放在完整 replay 验证之后。

### 4.5 迁移方向不改变层的因果顺序

层归属可以在允许的两个方向移动，执行仍须遵守原层顺序。

例如将后端 B 的第一层移动到上游 A，A 可能已有该层所需的边界 hidden，但必须在 buffer 被复用前保存它。将 A 的末端层移到下游 B，B 原来拿到的可能只是这些层处理后的 hidden，需要额外发送更早的入口 hidden。两种方向的激活通信成本通常不同，必须分别测量。

### 4.6 预热也是执行，执行可能写状态

不能在真实序列活跃时用“假 token”直接运行共享其状态的候选全图。一次预热可能写入未迁移层的 KV/递归状态、推进 memory 游标或覆盖输出。

在另一套 scratch 地址上捕获的图，不能未经验证就当作最终 bank 地址上的图。复制 native graph 或修改指针不是首版方案。

## 5. 推荐的最小架构

下列名称用于解释职责，最终命名服从仓库模式，不要求照搬一套新框架。

| 对象 | 拥有什么 | 不应拥有什么 |
| --- | --- | --- |
| 状态 bank | 某层在某设备的持久张量、版本、有效 frontier，以及与执行视图绑定的状态元数据 | 不隐式拥有全模型第二份权重 |
| 候选 TG runtime | 固定布局、完整图结果/输入对象、scheduler、计算 buffer、状态绑定与签名 | 不只是一个裸 `ggml_cgraph *` |
| replay runtime | 连续层区间的计算图、独立输入/scratch、目标状态视图 | 不推进前台采样器，不调用全模型普通 decode 冒充区间重放 |
| checkpoint | frontier、不可变状态快照/被 pin 的有效前缀、完成依赖 | 不引用仍会覆盖的源 recurrent 张量而声称快照不可变 |
| activation journal | 有所有权的 hidden segment、位置、dtype/layout、版本、完成依赖 | 不引用本轮稍后会被覆盖的输出 buffer |
| migration controller | 请求状态机、任务预算、frontier 与就绪检查、提交和取消 | 不在后台线程任意修改主上下文全局放置 |

物理状态副本优先按 `(layer, device)` 复用，不给每个候选布局机械地复制全模型状态。未移动层可以绑定到相同的权威物理 bank，但不同 runtime 的可变元数据/输入对象必须隔离且在执行前正确准备。若共享元数据无法安全拆分，应明确设计替代方案和显存代价，不把它隐藏在“swap 指针”里。

建议将以下版本分开：

- sequence epoch：清空/替换序列后，旧 journal、checkpoint 和完成事件全部失效。
- placement generation：用于判定迁移请求是否仍属于当前源布局。
- runtime signature：实际图结构、地址、配置和输入形状是否可复用。
- bank generation：物理对象被重建或回收时，使引用它的 runtime 失效。

runtime signature 至少审计：层布局和副本版本、图类型、token/sequence/output 形状、输入 token/embedding 模式、KV 的有效形状/stride/view offset、状态绑定及地址 generation、FA/fused ops 配置、adapter、backend/scheduler 模式和输入轮换槽。具体字段以实际 `can_reuse()` 与 CUDA 属性检查为依据，不用一个手写的不完整 hash 替代现有检查。

仅仅 ownership epoch 变化不应强制废弃物理签名仍有效的候选图，但不能直接删除当前 epoch 校验。要先建立新的等价性判定和失效机制。尤其不能保留旧 UID 同时悄悄修改 tensor/view，让 CUDA 的 UID 快路径跳过属性检查。

候选 TG runtime 的最小范围建议是“一个候选全模型布局及其实际 backend splits”，先复用现有全图 builder。首版不把所有模型改造成每层独立编译器；单层 shadow 图也不被当作未来多层 TG split 的替代品。

推荐控制流程：

```text
REQUESTED -> BOOTSTRAP_COPY -> CATCH_UP -> READY_FOR(F) -> SEALED -> COMMITTED
                  |               |           |
                  +---------------+-----------+-> CANCELLED / FAILED

READY_FOR(F) 在前台推进后可以重新变成 CATCH_UP。
SEALED 阶段禁止目标 bank 接收新后台写入，提交前重新核对当前 frontier。
```

## 6. 分阶段开发与验收

| 阶段 | 主要成果 | 依赖 | 阶段结束时能做到什么 |
| --- | --- | --- | --- |
| S0 | 基线、计时与失效清单 | 当前代码 | 知道延迟和流量分别花在哪里 |
| S1 | 固定状态 bank 和隔离的元数据视图 | S0 | 同步复制后安全切换 bank，地址生命周期可靠 |
| S2 | 有界候选 TG runtime 缓存 | S1 | 匹配形状时切换不重新 build/reserve；native 预热尚不保证 |
| S3 | checkpoint 与异步 KV 增量/状态同步 | S1，接入提交依赖 S2 | 工具窗口提前同步，满足条件后提交 |
| S4 | 有所有权的激活日志和 exact 区间 replay | S1，测量复用 S0 | 用旧状态加 hidden 补齐缺失 segment |
| S5 | 预算驱动的 shadow/catch-up 调度 | S2、S3、S4 | 组合复制和重放，推理期有界执行、空闲期加速 |
| S6 | 实际候选 TG 的安全 CUDA Graph 预热 | S2，集成依赖 S5 | 满足签名时首个迁移后 decode 不再冷捕获 |
| S7 | RPC/边缘链路适配 | 本地 S3 到 S6 通过 | 远端同样有正确的完成语义和可量化的通信 |
| S8 | 真实模型评测与策略收敛 | S0 到 S7 | 给出可靠的通信/停顿/开销权衡与论文结果 |

### S0. 补齐测量，不先改迁移语义

任务：

1. 记录编译选项、CUDA Graph 是否启用、模型/权重与 KV dtype、FA/fused 模式、设备、实际 KV capacity/padding、batch/ubatch、context 长度和已有 dirty diff。
2. 在现有统计基础上分开记录逻辑 build、backend split/分配、reserve 总时长、输入准备、GPU 等待、复制提交/完成、native warmup/capture/update 和提交耗时。
3. `sched_reserve()` 正常有 prompt/TG/prompt 三次 reserve；实验放置路径的 `graph_reserve()` 还可能构建校验图，fused probe 或失败重试还会增加次数。统计实际次数，不把 `reserve_us` 全部解释成一次 CPU build。
4. 记录同步路径搬运的物理 tensor 字节，并单列有效 KV rows 的逻辑字节；后者才是未来增量复制比较口径。
5. 建立事件时间线：迁移请求、任务入队、GPU 开始/结束、ready frontier、提交开始/结束、下一 token 完成。后台异步阶段的耗时有重叠，不能直接相加得到关键路径。
6. 列出会使候选失效的操作；首版对不支持操作明确拒绝。补足现有失败注入与前台连续生成对照。

代码落点：`llama-context` 统计、现有 CUDA/backend 诊断入口、`tests/test-llama-archs.cpp`。GPU 事件/trace 模式独立开关；普通性能数据不能使用会在每层同步的 probe。

验收：同步迁移原有用例继续通过；同一输入可复现实验；日志能区分 build/reserve/capture；不新增默认同步点或改变默认推理行为。只做文档/测量时不声称已实现任何重叠。

### S1. 固定状态 bank：先解决地址和元数据，再谈缓存图

任务：

1. 将“分配目的存储”“填充内容”“绑定为活动状态”“回收”分离。保留 `prepare_layer_move()` 的同步语义作为封装或参考路径，避免一次性推翻所有调用。
2. 初始化时为有限候选层预留目的 bank，权重沿用已有副本；记录 tensor/view/data 地址、buffer 所属 backend、capacity 和版本。
3. 审计 KV cell/slot、recurrent head/tail、hybrid context `apply()`，设计独立的执行元数据视图。用一次“目标滞后但源继续前进”的测试证明两者不会互相推进。
4. 对 full-attention bank 定义有效 token rows 和可读取前缀；对 recurrent bank 保存 R/S 及该模型需要的 P/卷积历史、索引等完整集合。以实际 `layer_state_tensors()` 和 forward 读取为准。
5. 实现受控 bind/promotion；未迁移层仍保持原权威 bank。提交后旧 bank 标记为旧 frontier 的 standby，而不是永久同步副本。
6. 通过引用/任务 pin 保持仍被 graph/event 使用的 bank 存活。回收必须在最后一次 GPU 使用完成之后，不允许 allocator 复用地址产生悬挂 graph。

元数据切换也必须计时。如果实现需要在提交时复制随 context 长度增长的大型 cell 表，就不能称为仅切换指针；应尽量提前准备一致的目的视图，并以受控绑定发布。未迁移层在候选 runtime 中的元数据同样必须对应当前 frontier，不能停留在候选创建时的旧值。

此阶段先同步复制，再沿用 reserve 完成切换；不追求低延迟，只建立新的正确性基线。

代码落点：memory 接口、KV/recurrent/hybrid 实现，context 的迁移事务。尽量扩展现有类型，不增加通用任务系统。

验收：full-only、recurrent-only、mixed、双向往返和失败恢复通过；前台持续推进不改变 standby 元数据；连续切换不重复分配相同 bank；低显存时初始化明确失败或减少候选，不在生成中突然无界分配。

### S2. 候选 TG runtime：把 build/reserve 移出提交路径

任务：

1. 将目前 context 中绑定活动布局的 graph result、输入对象、scheduler 和相关 buffer 收束成可拥有、可存活的 runtime。必须审计 graph params 中对 batch、memory context、callback、`active_layers` 和参数对象的引用生命周期。
2. 为每个允许布局准备完整 TG runtime，绑定 S1 的最终物理 bank。候选构图不应临时改写正在执行的全局 `active_layers`；初始化串行临时切换若被保留，也必须有完整事务和无并发前提。
3. 初始化可 reserve 最坏容量，但要区分“容量已预留”与“实际 TG 图已 alloc 并保持”。不要把 sizing graph 直接当作下一步实际执行图。
4. 首版只缓存一个有限的 TG batch/shape 类别，例如单 token、单 sequence、固定 output 方式。KV 长度/掩码形状增长会带来新类别，显式记录 hit/miss。
5. hot activation 必须选择匹配下一 ubatch 和 memory view 的 runtime，并使用正确的新位置、mask、slot 输入。审计 `process_ubatch()` 先执行 `mctx->apply()` 的顺序，避免检查时还是旧形状，提交后才发现不匹配。
6. 修改复用判断时分别维护语义版本与物理签名，保留现有输入检查。保存 graph 时要保存它的全部依赖，不能只保存指针。
7. candidate scheduler 不得被 reserve/reset 再复用其旧 graph；缓存淘汰和普通 decode 重建必须执行明确失效。若 native 缓存没有对应回收机制，需要一并审计，不能只限制 host runtime 数。
8. 保留初始 runtime，确保 A/B/A 在同一有效形状下可返回。不同 scheduler 可以引用同一 backend，但首版只由一个驱动线程按顺序提交。

候选的下一步输入预备应使用私有执行视图，不能为了检查 shape 提前 `apply()` 前台 memory 并推进其状态。同一 backend 的多个 scheduler 通常仍共享 stream/内部缓存，并不自动产生 GPU 并行；若后续引入独立 backend handle/stream，需要另外审计共享权重 tensor 的 backend 附加信息、CUDA 缓存及线程安全。

此阶段可以有首个 decode 的 CUDA 冷启动，日志必须如实标记。尚不要求 CUDA Graph 热命中。

验收：命中候选的提交路径中没有全模型 `sched_reserve()`、`model.build_graph()` 和大块计算 buffer 分配；匹配形状的下一 decode 也不重新 build/alloc；故意改变 KV shape 时正确拒绝热切换或转入明确冷路径；A/B/A 数值、地址和资源稳定性通过。

### S3. checkpoint 与异步同步：先实现最可靠的数据通路

建议分两小步：S3a 在显式 tool-idle 窗口复制；S3b 在真实生成期间进行受控分块复制。不要把从后台线程调用同步函数视作 S3b 已完成。

任务：

1. 在安全 token 边界创建 frontier 为 `P` 的 checkpoint。确保源写入完成，之后才读取其内容。
2. full-attention 在 append-only 且不会 wrap/shift/reuse 的前提下可 pin 已完成前缀；如该前提不成立则复制到不可变 snapshot 或拒绝。需要同时处理 cell map、V 转置布局、量化 block 对齐和 padding，不能假定第 `p` 个 token 就是内存中连续的一段 K/V。
3. recurrent 状态在下一 token 会覆盖，必须先在有序依赖下复制到不可变 snapshot，再允许后续源写入。snapshot 开销计入预算，不能让网络线程直接读取持续变化的 R/S/P。
4. 建立有界 transfer ticket：包含请求版本、层/bank、区间、source completion、destination completion、字节数和 staging 所有权。只有目标写入真正完成才推进 frontier。
5. KV 先同步 `[0, P)`，再发送缺口 `[P, Q)`。覆盖已有行不得与目标 replay 读写竞态。recurrent 同步是某个 frontier 的完整状态快照，不是按 token 行追加；可与 KV 的增量一起组成同一 frontier 的检查点。
6. CPU staging 使用有限 buffer pool；DMA 未完成不能复用。分块大小受延迟和链路带宽约束，不一次入队巨量低优先级任务阻塞前台。
7. 审计各 backend 的真正异步能力。当前通用 async copy 可走同步回退；没有已验证非阻塞完成查询时，不虚构 event query 接口。可由独立 completion worker 等待后投递完成消息，但主线程仍需非阻塞，且底层 backend 并发访问必须受支持。
8. 提交时匹配所有层的 frontier。部分层先完成不能提前变成 active；取消后只能回收已完成任务的资源，不能把正在 DMA 的 buffer 立刻释放。

建议先提供 capability 状态：异步已验证、仅 idle 同步可用、不支持。支持能力不足时继续源推理，不自动退化成前台长阻塞。

验收：强制延迟/重排完成消息、取消请求、断开目标、旧 epoch 消息到达等情况下均不会提交损坏状态；同步 copy 与异步 copy 的逻辑有效状态一致；能证明数据传输与前台 GPU 活动是否实际重叠；未重叠也须正确标记，不作性能承诺。

### S4. 激活日志与 exact replay：为 KV 和线性状态共用一条重建路径

任务：

1. 为可迁移区间入口 `[a, b)` 捕获 `a` 层 pre-norm residual hidden。沿用模型已有 `t_layer_inp` 的语义，但新增有所有权、带完成依赖的持久 segment。
2. journal 条目记录 sequence epoch、模型/adapter 配置版本、入口层、token 范围、绝对位置、dtype/shape/stride、存储设备、ready event 和 pin 计数。首版保留原生精度，不做有损激活量化。
3. 有界保留，按最早未完成 replay 需求 pin。空间不足时取消/推迟对应候选或重新 checkpoint，不阻塞所有生成，也不静默丢弃仍需要的 segment。
4. 从两个目标模型的现有 layer loop 中提取最小可复用构建逻辑，增加私有区间执行入口。避免复制一份完整 model forward 后独立维护。
5. replay 输入是 checkpoint/bank、入口 hidden、位置和目标状态视图；输出是更新后的层状态与必要的区间末端 hidden。无需 sampling、lm_head 或前台 logits 输出。
6. 第一版逐 token 重放，沿用正常 TG 算法和 kernel 配置，先排除线性注意力 chunk 路径与 AR 路径的差异。之后再单独实现 chunk replay 并建立误差/收益基线。
7. 每个 token 在区间内按原顺序执行；完整恢复 full-attention 前缀和 recurrent 历史。相邻多层可以只传一个入口 hidden，但计算开销包含区间内 attention/FFN/MoE。
8. 首版任务完成后再统一发布新区间 frontier；若需要逐层细粒度进度，必须证明恢复/取消不会留下错误的可见版本。

代码落点：graph result/input、模型构图 helper、memory 私有执行视图、context journal 与 replay 控制。先实现窄范围，别将整个模型框架改成通用分布式执行器。

验收：从 `P` 恢复并重放 `[P, F)` 后，与不中断运行到 `F` 的有效 KV、recurrent 状态及后续 logits 对照。覆盖 `P = 0`、空区间、单 token、多 token、full/recurrent/mixed、两种迁移方向及缺失 hidden 的拒绝路径。证明前台 memory、采样状态和输出未被 replay 推进或覆盖。

### S5. shadow compute 与追赶调度：由同一 replay 引擎实现

两种模式共用 S4，不实现两套模型计算：

- 持续 shadow：每个新 hidden 到达后尽早更新 standby，适合明确高概率迁移的少数边界层。
- checkpoint + catch-up：正常时少量维护/记录，在工具窗口同步大前缀，接近迁移时重放短缺口，作为默认策略。

任务：

1. 增加显式应用信号，例如工具执行开始/结束或允许后台预算；名称由实现决定。先由测试驱动接入，不通过识别输出文本猜测 tool use。生产 server 接入前阅读其开发约束，另行界定范围。
2. 一个主控制线程拥有 placement 和状态机。后台 I/O 可报告完成，不能自行 swap。首版每次只推进一个受控后台任务，不并发运行多个共享 backend 的 replay scheduler。
3. 以小 replay segment 或小复制块为调度单位，先用单 token replay 确保可解释性。当前任务完成后才允许新任务入队，避免前台恢复时已经积压大量不可取消 GPU 工作。
4. 对每个 backend 分别评估可用窗口。单序列自回归依赖上一 token 结果，不保证存在大 PP 气泡；其它设备工作时的空闲也不意味着本设备的带宽完全空闲。使用事件时间线验证，不凭 `pipeline_parallel = true` 推断。
5. 维护 lag `D = F_active - F_shadow`、最近追赶速度、前台 token 到达速度、预计复制/重放成本、journal 剩余空间以及候选有效期。
6. 选择完整且可验证的追赶方案：COPY 所有缺失有效状态；或从一致 checkpoint REPLAY 完整区间。按层混合策略只有在依赖和 frontier 已明确后再加入，不只重建某个线性终态却遗漏其后 full-attention 所需 hidden。
7. 前台有延迟预算时，后台任务服从 admission control。不能承诺 GPU kernel 入队后可被立即抢占；低优先级 stream 也不保证没有显存带宽争用。
8. 工具窗口结束即停止提交新后台任务；已在执行的任务按正确完成/取消协议处理。若剩余工作超预算，迁移保持 pending，前台沿源布局继续。

若按墙钟计的有效 replay 速率为 `mu`、前台增长为 `lambda`，持续生成时必须 `mu > lambda` 才能长期追上；近似追赶时间为 `D / (mu - lambda)`。实际还受 checkpoint、journal 通信和任务启动成本影响。`mu <= lambda` 时应等待工具窗口、改用复制或放弃该次迁移，而不是无限追赶。

提交协议：

1. 停在已完成 token 的安全边界，读取当前 `F`。
2. 对目标置 SEALED，阻止新后台写入；确认此前目标写和源读写依赖完成。
3. 再次验证请求版本、全部迁移层 frontier、下一步实际 runtime 签名及所选热路径要求。
4. 原子发布新的活动 runtime/布局/状态绑定，再允许位置 `F` 的 decode。
5. 按完成依赖退休旧执行引用；旧 bank 可保留为 standby，但它不会自动随新布局推进。

提交前失败可取消并保持源布局。提交后新布局已经执行了 token，则回切也必须恢复旧 bank 到新的 frontier；不能把切换前快照直接恢复成当前状态。

验收：长时间推理、随机 tool-idle、频繁取消和 A/B/A 通过；后台任务不干扰前台 logits；追不上时有明确状态和降级策略；给出无迁移时仅启用后台维护的 token latency/吞吐/显存开销，不能只展示成功切换瞬间。

### S6. CUDA Graph 预热：预热未来真正要执行的图

任务分为状态观察、无副作用预热、失效管理三部分。

状态观察：

1. 为每个实际候选 TG split 记录 CUDA Graph 是否被禁用、是否 warmup complete、是否已有 executable、发生过 capture/update 的次数和最近签名。
2. 当前 CUDA graph key 是 split 的首节点指针，UID 有快速复用判断。不同的单层 shadow 图和多层候选 TG 图通常不是同一 key/拓扑，必须分别观察。
3. pipeline scheduler 可能有多个输入 copy/轮换槽；预热一次不一定覆盖它们。枚举实际会轮换执行的 split 变体，分别验收，不硬编码“执行两次必然全热”。

安全预热的默认路径：

1. 在真实 prefill 开始之前，使用最终 bank、最终 runtime 和最终计算 buffer 地址进行初始化预热。此时没有用户序列在这些状态上运行。
2. 构造合法测试输入和元数据，按实际 backend 条件运行到 native ready。需要覆盖未来使用的形状类别；若无法安全构造某类有效输入，不将其标记为已预热。
3. 预热结束后在原地址恢复初始状态、cell/slot/游标和输出元数据，验证所有状态与预热前等价。恢复不得触发 bank 重分配或 scheduler reset。
4. 同时验证恢复后的真实 `set_inputs()`、位置和 memory view 没有改变图结构/地址；如改变，则预热结果失效，不能继续宣称热命中。

运行中新增形状的策略：

- 默认允许候选保持冷状态，提前准备逻辑图，在允许的后续窗口完成预热；严格热切换模式不强制立刻提交。
- 若需要在 tool-idle 预热全候选图，先停止所有可能写共享状态的工作，对该图会写入的全部状态和元数据做事务性快照，运行并等待完成，再原址恢复、验证。范围包括未迁移但被全图执行的层，不只是 standby 层。
- 该事务也必须保护输出、位置、输入槽、调度器轮换状态等必要依赖；不推进采样器。普通公共 state restore 可能使当前 hybrid memory 失去 append-only 属性，不能直接滥用，需设计内部恢复语义并测试。
- 若完整快照代价过高，推迟热切换或明确选择冷路径。完整私有 state/runtime 副本是可讨论的高显存替代，不作为默认实现；不同地址的 scratch capture 不能直接转移为正式 capture。

失效管理：

1. 监控 KV 长度相关 padding/shape、mask、view offset、输入 batch、FA/fused 配置、物理 bank generation、scheduler 重建及候选淘汰。
2. 有界缓存当前和近期可能使用的 shape 类别；不预建所有 token pos 的所有候选组合。哪些变化只是输入数据、哪些改变节点属性，由实际图决定。
3. 若采用更大固定形状加 mask 以提高命中，单独评估额外算力、未使用 KV 读取安全和语义，不作为免费优化。
4. CUDA Graph 不支持的 split 明确记为 unsupported，可运行正常 kernel，但不能计入“完全热 native 切换”成功率。

验收：热命中提交及第一步 decode 无新增 build/reserve/native capture/update；日志能指出冷 miss 原因；预热前后真实状态与后续 logits 一致；跨 KV shape 边界、输入轮换、A/B/A、淘汰后重建均正确。所有“无 capture”结论用 backend 计数或 trace 验证，不能仅看总耗时下降。

### S7. RPC 与边缘链路：单独解决队列和图身份

本阶段开始前确认当前构建确实启用 RPC，本地 CUDA 成功不等于 RPC 已测试。

任务：

1. 给传输增加实际线缆/协议口径的字节统计，包括 hidden journal、KV、recurrent checkpoint、图序列化、重传及 staging。区分设备内部复制字节、逻辑 payload 与网络字节。
2. 当前 RPC dispatcher 单队列/FIFO，`cpy_tensor_async` 为空。先验证 host 异步提交与远端实际执行的区别，用小块和有界队列降低队头阻塞。后台线程不能消除同连接队列串行。
3. 当前服务端每设备仅保存最近一图，shadow/replay/TG 交替会覆盖最近图；客户端虽可重新序列化保证路径执行，但不具备多 runtime 热缓存。先记录开销，再加入最小有界远端 graph handle/cache 设计。
4. 若增加协议，必须有能力协商、graph/session ID、buffer generation、显式失效与释放、容量上限、断线清理。旧服务端不支持时降级为正确但不保证热切换的模式。
5. 远端完成确认必须表示实际数据或计算已完成，不能以发送成功代替。跨连接若引入独立数据通道，需要额外证明顺序与取消语义，不能沿用单队列隐含顺序。
6. 测试延迟、限速、连接中断、候选被淘汰、远端重启和失败请求；不能继续使用已无效的远端地址/handle。

验收：本地验收矩阵在 RPC 的受支持能力上重跑；独立报告不支持项；提供网络流量和端到端 token 时间线。若边缘设备不支持 CUDA Graph，仍可评测 S1 到 S5，不能强制将 native graph 作为所有 backend 的前提。

### S8. 真实模型评测与策略收敛

至少比较以下消融，保持模型、输入 token、总生成长度、设备和候选布局一致：

| 方案 | 要回答的问题 |
| --- | --- |
| 无迁移、无后台维护 | 原始性能与质量基线 |
| 现有同步完整迁移 | 迁移代价基线 |
| 固定 bank + 候选 runtime | 单独去掉 reserve 能省多少 |
| 加异步 checkpoint/KV 增量 | 只移动传输时机能降低多少停顿 |
| 加 replay/shadow | 相比复制，通信减少多少、额外算力多少 |
| 加候选 CUDA Graph 预热 | 首步冷启动和尾延迟改善多少 |

评测要求：

1. 小 fixture 保留做错误定位，再用真实目标模型和实际 dtype/context/网络。两种方向、不同层区间、不同迁移概率都测。
2. 固定 token 输入比较每步 logits/状态，避免前一步抽样分叉使后续数值不可比。另做固定 seed/greedy 生成与真实任务质量测试，明确容差与数据集。
3. 记录用户可见相邻 token 间隔、迁移额外停顿、请求到提交时间、首步/稳态 token latency、吞吐、成功热切换率、lag、后台 GPU 时间、显存/主存峰值和初始化时间。
4. 独立统计整个请求生命周期与切换瞬间的通信。初始化权重副本、checkpoint、hidden、状态增量、被取消的工作和图协议成本都不能消失，只能分栏并给出摊销假设。
5. 同时报 p50/p95 和样本数、预热规则、失败/取消数量；尾延迟需要足够重复样本，不能用几次运行推断稳定 p99。
6. 在预算与准确性达标前，不引入复杂自适应控制器。先用可解释阈值，再通过实验决定 COPY/REPLAY 切换条件和 shadow 层数。

HYPIC 风格的线性终态重建作为后续独立扩展：先核实论文具体假设及目标模型映射，再与逐 token exact replay 比较。如果混合区间下游 full-attention 仍需要每 token hidden，只有终态并不足以完成该区间重建。它不应阻塞 S0 到 S8 的基础实现。

## 7. 通信、内存和时延的预算方法

令缺口 token 数为 `D`，有效完整 KV 的每 token 字节数为 `B_kv`，入口 hidden 每 token 字节数为 `B_h`，则可先作下列粗估，再以实测修正：

```text
COPY payload   ~= D * sum(B_kv for moved full-attention layers)
                  + recurrent_snapshot_bytes_at_target_frontier

REPLAY payload ~= missing_checkpoint_bytes + D * B_h

REPLAY compute = moved interval's actual forward cost over D tokens
```

这里不是承诺 replay 一定省通信。GQA/MQA 的 K/V 维度可能较小，hidden 反而更大；KV 量化和 hidden 精度也会改变结论。若目标本来拥有入口 hidden，则该项跨链路流量可以为零，但保存/读取成本仍存在。checkpoint 已在目的端时不能再次计入本次缺口，却必须在请求总流量中计入其建立成本。

候选额外显存分账：权重副本、持久状态 bank、checkpoint staging、各 runtime 计算 buffer、journal、CUDA executable 及 backend 缓存。限制候选数、shape 数、journal token 数和 in-flight ticket 数；显存不足时优先取消低价值候选，而不是破坏主推理。

热路径仍包含必要的事件确认、版本检查、输入更新和极少量主机切换。目标是移除大状态传输与重建，不承诺数学上的零耗时。具体停顿阈值通过 S0 和目标设备 SLA 确定，本文不预设未经测量的亚毫秒保证。

## 8. 通用测试清单与阶段门槛

复用 `tests/test-llama-archs.cpp` 的现有 fixture 和 PP 入口；未经另行同意，不新增 `tests/*` 文件。新选项只在实现后进入文档，不将计划中的参数写成已经可执行的命令。

当前已有的基础命令如下；路径和可用 GPU 应由开发会话先确认：

```sh
cmake --build build-pp-cuda --target test-llama-archs -j 8
CUDA_VISIBLE_DEVICES=0,1 build-pp-cuda/bin/test-llama-archs --pp-replicas -s 1234
CUDA_VISIBLE_DEVICES=0,1 build-pp-cuda/bin/test-llama-archs --pp-model /absolute/path/to/model.gguf -s 1234
```

每阶段至少覆盖其涉及的项目：

- full/recurrent/mixed；空缺口、单 token、长缺口；初始位置和边界位置。
- KV 实际 shape/padding 改变前后；若该配置按 256 对齐，再测试 255/256/257 等边界，不无条件硬编码。
- 双向迁移、A/B/A、重复 request、候选白名单外请求。
- 前台继续推进、目标滞后、journal 溢出、任务取消、旧事件迟到。
- 分配失败、copy 失败、replay 失败、提交前失败与已经执行新 token 后的回切。
- CUDA Graph 开/关、不支持节点、输入轮换槽、native cache miss/淘汰。
- 请求结束/上下文析构时仍有未完成任务；所有 buffer/event/runtime 均安全释放。
- 默认功能关闭时的普通推理回归；禁止依赖实验路径才能正确运行普通 decode。

阶段报告统一包含：已实现范围、关键不变量、触及文件、可执行测试命令、通过/失败/跳过原因、性能/字节数据、已知限制、下一阶段风险。跳过硬件测试不等于通过。

在 S1 之前提交一份简短设计说明供用户确认，重点解释元数据隔离与所有权；S2 的 runtime 生命周期、S6 的预热恢复事务、S7 的协议变更若需要超出本文的大改，也必须先说明设计与代价。不要一次提交整个新子系统供用户被动接受。

## 9. 不要采用的捷径

1. 仅删掉 `placement_epoch` 检查，让旧图“复用”。
2. 仅预分配权重/KV，却宣称候选逻辑图和 CUDA Graph 已就绪。
3. 用层数或 token pos 当唯一图缓存 key。
4. 用一个备用层图的 warmup 代替未来完整 backend split 的 warmup。
5. 直接在真实状态上跑假 token，随后只恢复迁移层。
6. 将 scratch capture 的地址替换成真实 bank，或保留旧 UID 绕过 CUDA 检查。
7. 用旧 prefix KV 推测新 hidden；用线性最终状态代替整个混合区间的逐 token 输出。
8. 在后台线程调用同步复制，就宣称复制与前台 GPU 已重叠。
9. 目标只追到过去位置便提交，或提交后直接恢复切换前快照回滚。
10. 只统计切换瞬间字节，忽略提前同步和 hidden 通信，再宣称总通信接近零。

## 10. 给开发会话的交接文本

以下内容可直接作为另一个会话的起始请求：

> 请基于当前工作区执行 `docs/development/pp-shadow-async-runtime-plan.md`。先完整阅读仓库 AGENTS.md、相关实现及已有工作区差异，不覆盖已有修改。该任务是私有研究分支的开发，不提交 commit、不 push、不创建 PR。
>
> 先完成 S0，给出当前基线与设计核对；然后说明 S1 的状态 bank/元数据隔离方案供我确认，再按阶段推进。每阶段保持可独立验证，不一次性重构整个推理框架。固定状态地址先于候选图缓存；区间 replay 与候选 TG runtime 分开；预热必须保护真实状态。
>
> 首版限定单序列、append-only、相邻 backend、有限候选布局和本地 CUDA。复用现有测试文件；无法验证的硬件能力明确标为未验证。任何热切换都必须检查当前 frontier、版本、后台写入完成和真实下一步图签名。条件不满足时继续源推理，不悄悄进入阻塞全量迁移。
>
> 每阶段报告实现范围、关键不变量、文件变化、测试命令、结果与未解决问题。把可复现数据写入阶段记录，不将小 fixture 的数值/性能外推到真实模型。RPC 和真实模型评测按后续阶段单独完成。
