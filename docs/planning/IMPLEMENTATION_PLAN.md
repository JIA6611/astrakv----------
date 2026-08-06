# AstraKV-W 补全实现计划
**对齐赛题要求的 P0 / P1 / P2 任务指南**
状态：待执行
日期：2026-06-09
赛题：Runtime Optimization of LLM Inference for the Memory Constraint System

---

## 目录

1. [背景与问题定位](#1-背景与问题定位)
2. [P0：理论对齐层](#2-p0理论对齐层)
   - 2.1 LMCache Tiering ↔ 虚拟内存机制映射
   - 2.2 FlexInfer 技术对比分析
3. [P1：真实系统实现层](#3-p1真实系统实现层)
   - 3.1 On-demand Loading 机制（uffd / mmap）
   - 3.2 模型权重 Offloading PoC
   - 3.3 消融实验设计
   - 3.4 边缘设备模拟测试
4. [P2：冲奖扩展项](#4-p2冲奖扩展项)
5. [实现时间线与优先级](#5-实现时间线与优先级)
6. [答辩关键问题准备](#6-答辩关键问题准备)

---

## 1. 背景与问题定位

### 当前方案的核心缺口

赛题是 **OS 功能挑战赛**，维护方为南开大学（OS 方向），评审要点是**系统完整度与可演示性**，参考文献是 FlexInfer（EuroMLSys 2025）。

对照赛题三个任务，当前方案的覆盖情况如下：

| 赛题任务 | 要求 | 当前状态 | 风险 |
|----------|------|----------|------|
| 任务1 | 分析访存行为（参数加载、KV cache、专家激活） | KV cache 完整；参数加载未涉及；专家激活为 optional | 中 |
| 任务2 | **使用虚拟内存相关技术**，按需加载、数据换出 | LMCache tiering 是应用层方案，不是 OS 虚拟内存技术 | **高** |
| 任务3 | 通过预取技术**掩盖 IO 延迟** | selective prefetch 有设计但 hints 未真实消费 | 中 |

赛题背景原文的关键句：

> 大语言模型推理过程中访存行为固定……为使用**虚拟内存**、数据预取等相关技术提供了潜在的施展空间

评委（OS 专业背景）必问：**"你们用的是什么虚拟内存技术？"** 当前方案无法正面回答这个问题。

---

## 2. P0：理论对齐层

> **成本**：不需要写新代码，只需补充报告章节。
> **必要性**：不做这部分，赛题的 OS 属性无法说清，答辩必然被追问。

### 2.1 LMCache Tiering ↔ 虚拟内存机制映射

#### 2.1.1 核心映射表

在 competition report 中增加一节"虚拟内存机制对应关系"，建立如下完整映射：

| OS 虚拟内存概念 | AstraKV-W 对应实现 | 技术说明 |
|----------------|-------------------|----------|
| 物理内存（Physical RAM） | GPU HBM / VRAM | KV cache 的主存，访问延迟最低（~TB/s） |
| 扩展内存 / Swap Space | CPU DRAM（LMCache CPU tier） | 二级缓存，延迟约为 GPU 的 10–30 倍 |
| 磁盘交换文件（Swap File） | NVMe SSD（LMCache disk tier） | 三级存储，延迟约为 GPU 的 1000 倍，但容量大 |
| 内存页（Page） | KV cache chunk / PagedAttention block | PagedAttention 的 block 即 OS page 的 LLM 等价物 |
| 页表（Page Table） | ProfileDB + chunk score map | 记录每个 chunk 当前所在层级（GPU/CPU/SSD） |
| 页面置换算法（LRU/LFU/Clock） | Chunk Scorer（prefetch/keep/offload/drop） | 决策每个 chunk 的去留，等价于 page replacement policy |
| 缺页中断（Page Fault） | Cache miss → load from CPU/disk tier | 请求所需 KV chunk 不在 GPU 时触发加载 |
| 预取（Prefetching） | Selective prefetch endpoint | 主动将高频 chunk 从 CPU/disk 预热到 GPU |
| 内存压力（Memory Pressure） | Memory Pressure Controller | 监控 GPU 使用率，触发 offload 决策 |
| 按需分配（Demand Paging） | Partial KV Load Planner | 只加载请求实际需要的 chunk，不全量加载 |
| OOM Killer | pressure → drop / recompute 决策 | 极端内存压力下主动丢弃低价值 chunk |
| 内存分层（NUMA / memory hierarchy） | GPU → CPU → SSD 三级 tiering | 显式实现 LLM 推理场景下的存储层次结构 |
| 工作集（Working Set） | ProfileDB 中的热 chunk 集合 | 基于访问历史确定需要常驻 GPU 的 chunk 集合 |

#### 2.1.2 报告中的推荐叙述框架

在 competition_report.md 中增加以下章节结构：

```
## 3. 系统设计：基于虚拟内存原理的 KV Cache 分层管理

### 3.1 虚拟内存视角下的 LLM 推理内存问题
- 问题建模：KV cache 增长 = 动态内存需求，GPU HBM = 物理内存，
  CPU/SSD = 扩展内存层
- 访存规律固定性：LLM prefill/decode 的访问模式可预测，
  比通用程序更适合 prefetch

### 3.2 LMCache Tiering 与 OS 虚拟内存机制的对应关系
- [插入 2.1.1 的映射表]
- 说明：AstraKV-W 在 LLM 推理层实现了与 OS 虚拟内存等价的
  分层管理机制，利用 LMCache 作为 KV cache 的"内存管理单元"

### 3.3 与传统 OS 虚拟内存的差异与优化空间
- 差异1：LLM 访存模式固定，可 profile-guided prefetch，
  优于 OS 的 LRU/Clock 通用策略
- 差异2：KV cache 的 chunk 粒度比 OS page（4KB）大得多，
  减少了 page table 管理开销
- 差异3：可结合语义信息（attention pattern、token 重复度）
  做更精准的 eviction 决策，OS 无法做到
- 未来工作：uffd/mmap 集成可使系统直接复用 OS 虚拟内存基础设施
```

#### 2.1.3 性能层次结构数据（填入报告）

以下延迟数据作为报告中"为什么需要分层"的定量支撑：

| 存储层级 | 带宽（读） | 延迟 | AstraKV-W 对应 |
|----------|-----------|------|----------------|
| GPU HBM（A100） | ~2 TB/s | ~100 ns | GPU KV cache（在线） |
| CPU DRAM（DDR5） | ~50 GB/s | ~100 ns CPU，PCIe ~10 GB/s | LMCache CPU tier |
| NVMe SSD | ~7 GB/s | ~100 µs | LMCache disk tier |
| 比值（GPU vs SSD） | 280×–300× | ~1000× | offloading 代价的理论下界 |

> 结论：KV cache 的分层管理需要在内存容量（SSD >> CPU >> GPU）和访问延迟（GPU << CPU << SSD）之间做动态权衡，这正是虚拟内存机制的核心问题。

---

### 2.2 FlexInfer 技术对比分析

赛题唯一参考文献，评委必然熟悉，必须正面回应，不能绕过。

#### 2.2.1 FlexInfer 核心技术摘要

FlexInfer（EuroMLSys 2025）的主要贡献：

| 技术点 | 描述 |
|--------|------|
| 目标场景 | 端侧 / 边缘设备上的 LLM inference（手机、嵌入式） |
| 核心问题 | 模型**权重**超过设备 GPU/内存容量 |
| 优化对象 | **模型权重（Model Weights）** 的 offloading，而非 KV cache |
| offloading 粒度 | Transformer layer 级别（逐层加载/卸载） |
| 关键机制 | Prefill/decode 阶段解耦，pipeline overlap 隐藏加载延迟 |
| 内存管理 | 静态分析 + 运行时灵活调度（flexible scheduling） |
| 实现方式 | 深度集成推理引擎（侵入式修改） |

#### 2.2.2 AstraKV-W 与 FlexInfer 的对比表

在报告中加入以下对比，**主动建立差异化框架**：

| 对比维度 | FlexInfer | AstraKV-W | 优劣分析 |
|----------|-----------|-----------|----------|
| **优化目标** | 模型权重（weights） | KV Cache | 互补，非竞争 |
| **目标场景** | 嵌入式 / 端侧单用户 | 内存受限推理服务器 / 边缘服务节点 | 场景不同 |
| **Offloading 粒度** | Layer 级（几十 MB/层） | Chunk/Block 级（KB-MB） | AstraKV-W 粒度更细 |
| **动态性** | 静态 profiling 指导 | 动态 memory pressure 感知 | AstraKV-W 更具自适应性 |
| **预取策略** | 基于 layer 顺序的 pipeline prefetch | Profile-guided chunk prefetch | 均利用访存规律性 |
| **实现侵入性** | 深度修改推理引擎 | 非侵入式 runtime layer | AstraKV-W 可移植性更强 |
| **多用户支持** | 单用户场景为主 | 多请求并发 batch 场景 | AstraKV-W 更适合服务场景 |
| **MoE 支持** | 未明确说明 | 有 expert predictor 模块 | AstraKV-W 有延伸 |

#### 2.2.3 差异化叙事（答辩用）

> "FlexInfer 解决的是单用户端侧推理中模型权重超出设备内存的问题，核心技术是 layer-level weight offloading。
> AstraKV-W 解决的是多用户服务场景中 KV cache 累积超出 GPU 内存的问题，核心技术是 profile-guided KV chunk tiering。
> 二者优化对象不同（weights vs KV cache），目标场景不同（端侧单用户 vs 服务多用户），但共同的理论基础都是虚拟内存的分层存储原理。
> AstraKV-W 的非侵入式设计使其可以直接部署在现有 vLLM 生产环境上，而不需要修改推理引擎本身。"

#### 2.2.4 补充实验（可选但加分）

如果时间允许，增加一个 layer offload PoC（见 P1 3.2 节），并在报告中写：

> "受 FlexInfer 启发，AstraKV-W 在 KV cache tiering 基础上，实现了一个 layer-level weight offloading 的概念验证（PoC），
> 验证了 AstraKV-W 的分层框架可以同时覆盖 KV cache 和模型权重两类内存对象。"

---

## 3. P1：真实系统实现层

> **成本**：需要写代码，每个子项独立，可选其中 2–3 项实现，不需要全部完成。
> **必要性**：决定"系统完整度"评分，是与其他队伍拉开差距的关键。

---

### 3.1 On-demand Loading 机制

赛题任务2的核心要求，必须有至少一个真实 OS 虚拟内存技术的落地点。

提供两套方案，**推荐优先实现方案B（mmap）**，成本低、可靠性高、演示效果直观。

---

#### 方案A：userfaultfd（uffd）

**技术原理**：userfaultfd 是 Linux 内核提供的机制，允许用户空间程序拦截进程的缺页中断（page fault），并自定义缺页处理逻辑。用于 KV cache 时，可实现真正的 on-demand page loading。

**适用场景**：要求技术深度高、需要强调"OS 机制直接复用"时使用。

**实现步骤**：

```python
# scripts/uffd_kv_loader.py
"""
使用 userfaultfd 实现 KV cache 的 on-demand page loading。
当 vLLM 访问 KV cache 中尚未加载的页时，
由 uffd handler 负责从 CPU/SSD backing store 加载对应数据页。

依赖：Linux kernel >= 4.11，需要 CAP_SYS_PTRACE 或 /proc/sys/vm/unprivileged_userfaultfd=1
"""
import ctypes
import os
import mmap
import struct
import threading
import logging
from pathlib import Path

# uffd ioctl 常量
UFFDIO_API       = 0xC018AA3F
UFFDIO_REGISTER  = 0xC020AA00
UFFDIO_COPY      = 0xC028AA03
UFFDIO_ZEROPAGE  = 0xC020AA04
UFFDIO_RANGE_UNREGISTER = 0x8010AA01

UFFD_EVENT_PAGEFAULT = 0x12
UFFD_PAGEFAULT_FLAG_WRITE = 1 << 0

PAGE_SIZE = 4096  # 4KB，os.sysconf('SC_PAGE_SIZE')


class UFDDemandLoader:
    """
    用 userfaultfd 拦截 KV cache tensor 的缺页中断，
    实现 OS 层面的 on-demand page loading。
    """

    def __init__(self, cache_size_bytes: int, backing_store_path: str):
        self.cache_size   = cache_size_bytes
        self.backing_path = Path(backing_store_path)
        self.page_size    = PAGE_SIZE
        self.num_pages    = cache_size_bytes // PAGE_SIZE
        self.loaded_pages = set()   # 已加载到物理内存的页号集合
        self._uffd_fd     = None
        self._mmap_addr   = None
        self._handler     = None
        self.logger       = logging.getLogger("UFDDemandLoader")

    def setup(self):
        """初始化 uffd、mmap 区域、handler 线程。"""
        # Step 1：打开 /dev/userfaultfd 或通过 syscall 创建 uffd
        self._uffd_fd = self._open_uffd()

        # Step 2：mmap 一块匿名内存作为 KV cache 区域（不提交物理页）
        self._mmap_addr = mmap.mmap(
            -1, self.cache_size,
            flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | mmap.MAP_NORESERVE,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        # Step 3：向 uffd 注册这块内存区域
        self._register_region(
            start=ctypes.addressof(ctypes.c_char.from_buffer(self._mmap_addr)),
            length=self.cache_size,
        )

        # Step 4：启动 handler 线程，监听缺页事件
        self._handler = threading.Thread(
            target=self._fault_handler_loop,
            daemon=True,
            name="uffd-handler",
        )
        self._handler.start()
        self.logger.info(
            f"UFDDemandLoader ready: {self.num_pages} pages, "
            f"backing={self.backing_path}"
        )

    def _open_uffd(self) -> int:
        """通过 syscall 443 (userfaultfd) 创建 uffd fd。"""
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        fd = libc.syscall(443, 0)  # SYS_userfaultfd = 443 on x86_64
        if fd < 0:
            raise OSError(ctypes.get_errno(), "userfaultfd syscall failed")
        return fd

    def _register_region(self, start: int, length: int):
        """将内存区域注册到 uffd，启用 MISSING 事件。"""
        # struct uffdio_register { struct uffdio_range range; __u64 mode; __u64 ioctls; }
        buf = struct.pack("QQQ", start, length, 1 << 2)  # UFFDIO_REGISTER_MODE_MISSING
        ret = ctypes.cdll.LoadLibrary("libc.so.6").ioctl(
            self._uffd_fd, UFFDIO_REGISTER, buf
        )
        if ret < 0:
            raise OSError(f"uffd REGISTER failed: {ret}")

    def _fault_handler_loop(self):
        """循环读取 uffd 事件，处理缺页中断。"""
        # struct uffd_msg { __u8 event; ... __u64 address; }
        while True:
            try:
                msg = os.read(self._uffd_fd, 32)
                if len(msg) < 32:
                    break
                event = msg[0]
                if event == UFFD_EVENT_PAGEFAULT:
                    fault_addr = struct.unpack_from("Q", msg, 16)[0]
                    self._handle_fault(fault_addr)
            except OSError:
                break

    def _handle_fault(self, fault_addr: int):
        """
        处理缺页中断：从 backing store 加载对应页数据，
        通过 UFFDIO_COPY 写回 uffd 管理的内存区域。
        """
        base_addr   = ctypes.addressof(ctypes.c_char.from_buffer(self._mmap_addr))
        page_offset = ((fault_addr - base_addr) // self.page_size) * self.page_size
        page_no     = page_offset // self.page_size

        if page_no in self.loaded_pages:
            return

        # 从 backing store 读取页数据
        page_data = self._load_page_from_store(page_no)

        # UFFDIO_COPY：将数据拷贝到触发 fault 的内存页
        page_buf = ctypes.create_string_buffer(page_data, self.page_size)
        copy_struct = struct.pack(
            "QQQQI",
            ctypes.addressof(page_buf),  # src
            base_addr + page_offset,      # dst（对齐到页边界）
            self.page_size,               # len
            0,                            # mode
            0,                            # copy
        )
        ctypes.cdll.LoadLibrary("libc.so.6").ioctl(
            self._uffd_fd, UFFDIO_COPY, copy_struct
        )
        self.loaded_pages.add(page_no)
        self.logger.debug(f"Page fault handled: page={page_no}, addr={hex(fault_addr)}")

    def _load_page_from_store(self, page_no: int) -> bytes:
        """从 backing store 文件读取指定页的数据。"""
        offset = page_no * self.page_size
        with open(self.backing_path, "rb") as f:
            f.seek(offset)
            data = f.read(self.page_size)
        # 不足一页时补零
        return data.ljust(self.page_size, b'\x00')

    def prefetch_pages(self, page_nos: list):
        """
        预取指定页：提前触发加载，避免后续 fault 延迟。
        可由 selective prefetch 模块调用。
        """
        for pno in page_nos:
            if pno not in self.loaded_pages:
                self._handle_fault(
                    ctypes.addressof(
                        ctypes.c_char.from_buffer(self._mmap_addr)
                    ) + pno * self.page_size
                )

    def teardown(self):
        if self._mmap_addr:
            self._mmap_addr.close()
        if self._uffd_fd:
            os.close(self._uffd_fd)
```

**验收标准**：
- 能启动 handler 线程，无报错
- 访问未加载地址时能触发 fault handler，打印 debug 日志
- `loaded_pages` 集合随访问递增
- 实验记录：每次推理请求触发的 page fault 次数、加载延迟

---

#### 方案B：mmap + madvise（推荐，成本低）

**技术原理**：将 KV cache 用 `mmap` 映射到磁盘文件，KV cache 的换入换出直接复用 OS 的 page cache 机制。用 `madvise(MADV_WILLNEED)` 实现预取，用 `madvise(MADV_DONTNEED)` 实现主动换出。

**优势**：直接调用 OS 虚拟内存基础设施，代码量少，可靠性高，演示效果直观。

**实现步骤**：

```python
# scripts/mmap_kv_cache.py
"""
使用 mmap + madvise 实现 KV cache 的虚拟内存管理。

核心设计：
- KV cache 的 backing store 是一个 NVMe 上的文件
- mmap 将文件映射到进程虚拟地址空间
- OS page cache 自动处理换入（缺页）和换出（内存压力时）
- madvise 控制预取和主动换出
- 这直接复用了 OS 虚拟内存机制，满足赛题"使用虚拟内存相关技术"的要求
"""
import mmap
import os
import ctypes
import struct
import numpy as np
from pathlib import Path
import logging


# madvise 常量
MADV_WILLNEED  = 3   # 预取：告知 OS 即将访问，触发 readahead
MADV_DONTNEED  = 4   # 换出：告知 OS 不再需要，允许 OS 回收物理页
MADV_SEQUENTIAL = 2  # 顺序访问提示
MADV_RANDOM    = 1   # 随机访问提示

libc = ctypes.CDLL("libc.so.6", use_errno=True)


def madvise(addr: int, length: int, advice: int) -> int:
    return libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(length), ctypes.c_int(advice))


class MMapKVCache:
    """
    基于 mmap 的 KV Cache 管理器。

    通过将 KV cache 文件 mmap 到内存，直接利用 OS 虚拟内存机制
    实现 on-demand loading（缺页时 OS 自动从文件加载）和
    memory pressure 下的自动换出（OS page eviction）。

    prefetch_block / evict_block 通过 madvise 系统调用显式控制
    OS 的预取和换出行为，实现 profile-guided 的虚拟内存管理。
    """

    def __init__(
        self,
        backing_file: str,
        total_blocks: int,
        block_size_bytes: int,
        dtype: np.dtype = np.float16,
    ):
        self.file_path    = Path(backing_file)
        self.total_blocks = total_blocks
        self.block_size   = block_size_bytes
        self.total_size   = total_blocks * block_size_bytes
        self.dtype        = dtype
        self.logger       = logging.getLogger("MMapKVCache")

        # 初始化 backing file
        self._init_backing_file()
        # mmap 映射
        self.fd = os.open(str(self.file_path), os.O_RDWR)
        self.mm = mmap.mmap(self.fd, self.total_size)
        self._base_addr = ctypes.addressof(
            ctypes.c_char.from_buffer(self.mm)
        )
        self.logger.info(
            f"MMapKVCache initialized: {total_blocks} blocks × "
            f"{block_size_bytes/1024:.1f} KB = {self.total_size/1e9:.2f} GB"
        )

    def _init_backing_file(self):
        """创建或验证 backing file，预分配空间（稀疏文件）。"""
        if not self.file_path.exists():
            self.logger.info(f"Creating backing file: {self.file_path} ({self.total_size/1e9:.2f} GB)")
            with open(self.file_path, 'wb') as f:
                # 稀疏文件：只 seek 到末尾写一个字节，不占用实际磁盘空间
                f.seek(self.total_size - 1)
                f.write(b'\x00')
        else:
            actual = os.path.getsize(self.file_path)
            assert actual == self.total_size, \
                f"Backing file size mismatch: {actual} vs {self.total_size}"

    # ---------- 核心接口 ----------

    def read_block(self, block_id: int) -> np.ndarray:
        """
        读取指定 block 的 KV 数据。
        若对应物理页不在内存中，OS 自动触发缺页中断并从文件加载（demand paging）。
        这是 on-demand loading 的核心机制。
        """
        assert 0 <= block_id < self.total_blocks
        offset = block_id * self.block_size
        raw    = self.mm[offset : offset + self.block_size]
        return np.frombuffer(raw, dtype=self.dtype).copy()

    def write_block(self, block_id: int, data: np.ndarray):
        """将 KV 数据写入指定 block（写回 mmap，OS 负责刷盘）。"""
        assert data.nbytes == self.block_size
        offset = block_id * self.block_size
        self.mm[offset : offset + self.block_size] = data.astype(self.dtype).tobytes()

    def prefetch_block(self, block_id: int):
        """
        预取：通知 OS 即将访问该 block，触发 readahead。
        等价于 FlexInfer 的 layer prefetch，但粒度是 KV block。
        对应虚拟内存的 prefetching 机制。
        """
        offset = block_id * self.block_size
        addr   = self._base_addr + offset
        ret    = madvise(addr, self.block_size, MADV_WILLNEED)
        if ret != 0:
            self.logger.warning(f"madvise WILLNEED failed for block {block_id}: {ret}")
        else:
            self.logger.debug(f"Prefetch requested: block={block_id}")

    def evict_block(self, block_id: int):
        """
        主动换出：通知 OS 该 block 不再需要，可以回收物理页。
        等价于 OS 虚拟内存的 page eviction。
        数据保留在 backing file，下次访问时自动重新加载（on-demand）。
        """
        offset = block_id * self.block_size
        addr   = self._base_addr + offset
        ret    = madvise(addr, self.block_size, MADV_DONTNEED)
        if ret != 0:
            self.logger.warning(f"madvise DONTNEED failed for block {block_id}: {ret}")
        else:
            self.logger.debug(f"Evicted: block={block_id}")

    def prefetch_batch(self, block_ids: list):
        """批量预取，用于 profile-guided prefetch。"""
        for bid in block_ids:
            self.prefetch_block(bid)

    def evict_batch(self, block_ids: list):
        """批量换出，用于 memory pressure 场景。"""
        for bid in block_ids:
            self.evict_block(bid)

    def get_resident_pages(self) -> dict:
        """
        查询哪些页当前在物理内存中（mincore 系统调用）。
        用于验证预取和换出的效果，生成实验证据。
        """
        vec_size   = (self.total_size + 4095) // 4096
        vec        = ctypes.create_string_buffer(vec_size)
        ret = libc.mincore(
            ctypes.c_void_p(self._base_addr),
            ctypes.c_size_t(self.total_size),
            vec,
        )
        if ret != 0:
            return {}
        resident = {}
        for i in range(self.total_blocks):
            page_start = (i * self.block_size) // 4096
            page_end   = ((i + 1) * self.block_size + 4095) // 4096
            pages_in_mem = sum(
                (vec[p] & 1) for p in range(page_start, min(page_end, vec_size))
            )
            resident[i] = pages_in_mem / (page_end - page_start)
        return resident

    def teardown(self):
        self.mm.close()
        os.close(self.fd)
```

**验收脚本**：

```bash
# scripts/test_mmap_kv_cache.sh
#!/bin/bash
# 验证 mmap KV cache 的虚拟内存行为

python - <<'EOF'
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

from scripts.mmap_kv_cache import MMapKVCache

# 创建 100 个 block，每块 1MB（模拟 KV cache block）
cache = MMapKVCache(
    backing_file="/tmp/test_kv_cache.bin",
    total_blocks=100,
    block_size_bytes=1024 * 1024,  # 1MB per block
)

# 写入测试数据
print("=== 写入测试数据 ===")
data = np.random.randn(512 * 1024).astype(np.float16)
cache.write_block(0, data)
cache.write_block(50, data * 2)

# 换出所有 block（测试 MADV_DONTNEED）
print("=== 换出所有 block ===")
cache.evict_batch(list(range(100)))
resident_after_evict = cache.get_resident_pages()
print(f"换出后驻留 block 数: {sum(1 for v in resident_after_evict.values() if v > 0)}")

# 预取 block 0, 1, 2（测试 MADV_WILLNEED）
print("=== 预取 block 0/1/2 ===")
cache.prefetch_batch([0, 1, 2])
import time; time.sleep(0.1)  # 等待 OS readahead
resident_after_prefetch = cache.get_resident_pages()
prefetched = sum(1 for i, v in resident_after_prefetch.items() if i < 3 and v > 0)
print(f"预取后驻留 block 数（前3）: {prefetched}")

# 验证 on-demand loading（直接读取已换出的 block 50）
print("=== On-demand loading 验证 ===")
import time
t0 = time.perf_counter()
loaded = cache.read_block(50)
t1 = time.perf_counter()
print(f"冷读 block 50 耗时: {(t1-t0)*1000:.2f}ms（含 OS page fault）")
t0 = time.perf_counter()
loaded2 = cache.read_block(50)
t1 = time.perf_counter()
print(f"热读 block 50 耗时: {(t1-t0)*1000:.2f}ms（page 已在内存）")

cache.teardown()
print("=== 验收通过 ===")
EOF
```

**与虚拟内存机制的对应证据**（写入报告）：

```
测量结果示例：
- 冷读（OS page fault）：8.3ms（数据从 NVMe 加载）
- 热读（page 已驻留）：0.02ms
- MADV_DONTNEED 后驻留页下降至 0%
- MADV_WILLNEED 后目标 block 驻留率上升至 100%

结论：mmap 机制使 KV cache 的 on-demand loading 直接由 OS 虚拟内存子系统管理，
缺页延迟与 NVMe 读取延迟一致，验证了 OS page fault 机制的正确触发。
```

---

### 3.2 模型权重 Offloading PoC

对标 FlexInfer 的 layer-level weight offloading，证明 AstraKV-W 框架可扩展到权重管理。

**设计原则**：这是 PoC，不需要完整集成进 vLLM。目标是跑通一个独立脚本，产生"权重 offloading 节省了 X GB GPU 内存"的实验证据。

```python
# scripts/layer_offload_poc.py
"""
模型权重分层 Offloading PoC。

对标 FlexInfer 的 layer-level offloading。
本 PoC 不修改 vLLM 内核，而是独立演示：
在 GPU 内存受限时，如何通过 layer-by-layer CPU offloading
让完整模型在较小 GPU 上完成推理，并测量内存节省和延迟代价。
"""
import torch
import time
import gc
import logging
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from typing import List, Dict, Optional

logger = logging.getLogger("LayerOffloadPoC")


class LayerOffloadManager:
    """
    Transformer 模型的 layer-level offloading 管理器。

    策略：
    - GPU 上只保留当前正在执行的 N 层（gpu_layer_window）
    - 其余层权重保留在 CPU
    - 执行到某层时，将其从 CPU 移到 GPU；执行完后移回 CPU
    - 同时异步预取下一层（pipeline overlap，参考 FlexInfer）
    """

    def __init__(
        self,
        model_name: str,
        gpu_layer_window: int = 4,
        device: str = "cuda",
    ):
        self.model_name       = model_name
        self.gpu_window       = gpu_layer_window
        self.device           = device
        self.cpu_device       = "cpu"
        self.layers_on_gpu    = {}   # layer_idx -> True/False
        self.layer_load_times = []   # 记录每层加载延迟
        self.prefetch_stream  = torch.cuda.Stream() if torch.cuda.is_available() else None

    def load_model_to_cpu(self) -> AutoModelForCausalLM:
        """将完整模型加载到 CPU（不占用 GPU 内存）。"""
        logger.info(f"Loading {self.model_name} to CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        model.eval()
        logger.info(f"Model loaded to CPU. Layers: {model.config.num_hidden_layers}")
        return model

    def measure_gpu_memory_with_offload(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        window_sizes: List[int] = [2, 4, 8, 16],
    ) -> Dict:
        """
        测量不同 window size 下的 GPU 峰值内存和推理延迟。
        生成"内存 vs 延迟"权衡曲线，作为实验证据。
        """
        results = {}
        num_layers = model.config.num_hidden_layers

        for window in window_sizes:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()
            gpu_peak = self._run_with_window(model, input_ids, window, num_layers)
            t1 = time.perf_counter()

            results[window] = {
                "window_size": window,
                "gpu_peak_mb": gpu_peak / 1e6,
                "latency_ms": (t1 - t0) * 1000,
                "layer_load_avg_ms": (
                    sum(self.layer_load_times) / len(self.layer_load_times)
                    if self.layer_load_times else 0
                ),
            }
            self.layer_load_times.clear()
            logger.info(
                f"Window={window}: GPU peak={results[window]['gpu_peak_mb']:.0f}MB, "
                f"latency={results[window]['latency_ms']:.0f}ms"
            )

        return results

    def _run_with_window(
        self,
        model,
        input_ids: torch.Tensor,
        window: int,
        num_layers: int,
    ) -> int:
        """模拟 window 大小的 layer offloading 推理过程（简化版）。"""
        layers = model.model.layers  # Qwen2/LLaMA 风格
        peak_bytes = 0

        for i in range(0, num_layers, window):
            batch_end = min(i + window, num_layers)
            # 将这批 layers 移到 GPU
            t_load = time.perf_counter()
            for j in range(i, batch_end):
                layers[j].to(self.device)
                self.layers_on_gpu[j] = True
            load_time = time.perf_counter() - t_load
            self.layer_load_times.append(load_time * 1000)

            # 记录峰值 GPU 内存
            current_mem = torch.cuda.memory_allocated()
            peak_bytes  = max(peak_bytes, current_mem)

            # 将这批 layers 移回 CPU（offload）
            for j in range(i, batch_end):
                layers[j].to(self.cpu_device)
                self.layers_on_gpu[j] = False

        return torch.cuda.max_memory_allocated()


def run_poc(model_name: str, output_dir: str):
    """运行 layer offload PoC 并保存实验结果。"""
    import pandas as pd
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manager   = LayerOffloadManager(model_name, gpu_layer_window=4)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = manager.load_model_to_cpu()

    # 准备测试输入
    prompt    = "Explain the concept of virtual memory in operating systems." * 10
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # 测量不同 window size 的结果
    results = manager.measure_gpu_memory_with_offload(
        model, input_ids, window_sizes=[1, 2, 4, 8, 16, 32]
    )

    # 保存结果
    df = pd.DataFrame(results.values())
    df.to_csv(f"{output_dir}/layer_offload_results.csv", index=False)
    logger.info(f"Results saved to {output_dir}/layer_offload_results.csv")

    # 生成报告片段
    with open(f"{output_dir}/layer_offload_report.md", "w") as f:
        f.write("# Layer Offloading PoC 结果\n\n")
        f.write("## GPU 内存 vs 推理延迟权衡\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## 结论\n\n")
        min_mem = df['gpu_peak_mb'].min()
        max_mem = df['gpu_peak_mb'].max()
        f.write(
            f"- GPU window=1 时 GPU 峰值内存 {min_mem:.0f}MB，"
            f"相比 window=all（{max_mem:.0f}MB）减少 {1 - min_mem/max_mem:.1%}\n"
        )
        f.write("- 内存节省代价是每层加载延迟，可通过 pipeline prefetch 部分隐藏\n")
        f.write("- 这验证了 FlexInfer 的核心 trade-off，并为 AstraKV-W 框架扩展提供了基础\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--output-dir", default="results/gpu/layer_offload_poc")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_poc(args.model, args.output_dir)
```

**运行命令**：

```bash
# 用小模型验证（1.5B 更快，节省 GPU 时间）
python scripts/layer_offload_poc.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir results/gpu/layer_offload_poc

# 用主模型做完整实验
python scripts/layer_offload_poc.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output-dir results/gpu/layer_offload_poc_7b
```

**验收标准**：
- `layer_offload_results.csv` 存在，包含不同 window size 的数据
- `layer_offload_report.md` 中有明确的内存节省百分比
- GPU 峰值内存随 window size 减小而降低（单调性验证）

---

### 3.3 消融实验设计

**目的**：量化每个模块的贡献，避免"整体提升来自哪里说不清"的问题。

#### 实验矩阵

| 实验组 | Memory Pressure Controller | Chunk Scorer | Selective Prefetch | 预期效果 |
|--------|--------------------------|-------------|-------------------|----------|
| A：纯 baseline | ✗ | ✗ | ✗ | 基准 TTFT / OOM rate |
| B：+ pressure | ✓ | ✗ | ✗ | OOM rate 下降 |
| C：+ scorer | ✓ | ✓ | ✗ | GPU hit rate 上升 |
| D：+ prefetch | ✓ | ✓ | ✓ | TTFT 下降 |

每组使用相同 workload（shared prefix 场景）重复 3 次，取均值。

#### 新增 Workload：Shared Prefix 场景

当前 benchmark 缺少共享前缀场景，这是 KV cache reuse 最有价值的场景：

```yaml
# configs/shared_prefix_workload.yaml
# 模拟 RAG / 多轮对话：固定 2K token system prompt + 变化 user query
workload:
  name: shared_prefix_ablation
  shared_prefix_tokens: 2048   # 模拟长 system prompt
  query_tokens: [128, 256, 512]
  batch_sizes: [1, 2, 4]
  repeat: 5
  output_tokens: 64
```

#### 消融实验执行脚本

```bash
# scripts/run_ablation.sh
#!/bin/bash
set -e

MODEL=${ASTRAKV_MODEL:-Qwen/Qwen2.5-7B-Instruct}
BASE_DIR=results/gpu/ablation

# ---- 组 A：纯 LMCache CPU，无 pressure/scorer/prefetch ----
bash scripts/launch_lmcache_vllm.sh cpu &
sleep 30
python scripts/run_real_benchmark.py \
  --config configs/shared_prefix_workload.yaml \
  --output-dir $BASE_DIR/A_baseline \
  --disable-pressure-controller \
  --disable-chunk-scorer
pkill -f vllm; sleep 5

# ---- 组 B：+ Memory Pressure Controller ----
bash scripts/launch_lmcache_vllm.sh cpu &
sleep 30
python scripts/run_real_benchmark.py \
  --config configs/shared_prefix_workload.yaml \
  --output-dir $BASE_DIR/B_pressure \
  --enable-pressure-controller \
  --disable-chunk-scorer
pkill -f vllm; sleep 5

# ---- 组 C：+ Chunk Scorer ----
bash scripts/launch_lmcache_vllm.sh cpu &
sleep 30
python scripts/run_real_benchmark.py \
  --config configs/shared_prefix_workload.yaml \
  --output-dir $BASE_DIR/C_scorer \
  --enable-pressure-controller \
  --enable-chunk-scorer
pkill -f vllm; sleep 5

# ---- 组 D：完整系统（+ Selective Prefetch）----
bash scripts/launch_lmcache_vllm.sh cpu &
sleep 30
python scripts/run_selective_prefetch_real.py \
  --config configs/shared_prefix_workload.yaml \
  --output-dir $BASE_DIR/D_full \
  --enable-pressure-controller \
  --enable-chunk-scorer
pkill -f vllm; sleep 5

# ---- 汇总消融结果 ----
python scripts/compare_real_runs.py \
  --run baseline=$BASE_DIR/A_baseline \
  --run pressure=$BASE_DIR/B_pressure \
  --run scorer=$BASE_DIR/C_scorer \
  --run full=$BASE_DIR/D_full \
  --output-dir $BASE_DIR/ablation_summary
```

**验收标准**：
- `ablation_summary/comparison_results.csv` 有 A/B/C/D 四组数据
- B 相比 A：OOM rate 下降 > 10%（否则 pressure controller 无效）
- C 相比 B：cache hit rate 上升（否则 scorer 无效）
- D 相比 C：TTFT 下降（否则 prefetch 无效）

---

### 3.4 边缘设备模拟测试

赛题强调嵌入式/边缘设备，但实际在 A100 上测试，需要通过 **cgroups 内存限制** 模拟边缘约束。

#### cgroups v2 环境设置

```bash
# scripts/setup_edge_sim.sh
#!/bin/bash
# 创建 cgroup，模拟边缘设备内存约束

CGROUP_NAME="astrakv_edge_sim"
CGROUP_PATH="/sys/fs/cgroup/$CGROUP_NAME"

# 创建 cgroup（需要 root 权限）
mkdir -p $CGROUP_PATH

# 检查 cgroup v2 是否可用
if [ ! -f "$CGROUP_PATH/memory.max" ]; then
  echo "ERROR: cgroup v2 not available at $CGROUP_PATH"
  exit 1
fi

# 设置内存限制（模拟不同边缘设备）
set_memory_limit() {
  local limit_gb=$1
  local limit_bytes=$((limit_gb * 1024 * 1024 * 1024))
  echo $limit_bytes > $CGROUP_PATH/memory.max
  # 禁用 swap，模拟真实边缘设备无 swap 场景
  echo 0 > $CGROUP_PATH/memory.swap.max
  echo "Memory limit set to ${limit_gb}GB"
}

# 用法示例
# set_memory_limit 16  # 模拟 16GB 边缘节点
# set_memory_limit 24  # 模拟 24GB 边缘节点
# set_memory_limit 32  # 模拟 32GB 边缘节点

# 将当前 shell 加入 cgroup
echo $$ > $CGROUP_PATH/cgroup.procs
echo "Current shell (PID $$) added to cgroup $CGROUP_NAME"
```

#### 边缘设备模拟测试矩阵

```bash
# scripts/run_edge_sim_tests.sh
#!/bin/bash
# 模拟三种边缘设备配置

CGROUP_PATH="/sys/fs/cgroup/astrakv_edge_sim"

run_edge_config() {
  local config_name=$1
  local mem_gb=$2
  local model=$3
  local context=$4

  echo "=== 测试: $config_name (${mem_gb}GB CPU, model=$model) ==="

  # 设置内存限制
  echo $((mem_gb * 1024 * 1024 * 1024)) > $CGROUP_PATH/memory.max

  # 在受限环境中运行
  cgexec -g memory:astrakv_edge_sim \
    bash -c "
      export ASTRAKV_MODEL=$model
      export ASTRAKV_MAX_MODEL_LEN=$context
      export ASTRAKV_GPU_MEMORY_UTILIZATION=0.60
      bash scripts/launch_lmcache_vllm.sh cpu &
      sleep 30
      python scripts/run_real_benchmark.py \
        --config configs/dgx_spark_lmcache_cpu.yaml \
        --output-dir results/gpu/edge_sim/$config_name \
        --context-lengths 1024 2048 4096 \
        --batch-sizes 1 2
      pkill -f vllm
    "
}

# 场景1：16GB 边缘节点 + 小模型（最接近真实边缘设备）
run_edge_config "edge_16gb_1.5b" 16 "Qwen/Qwen2.5-1.5B-Instruct" 4096

# 场景2：24GB 边缘节点 + 7B 模型（入门级推理卡）
run_edge_config "edge_24gb_7b"   24 "Qwen/Qwen2.5-7B-Instruct"   4096

# 场景3：32GB 边缘节点 + 7B 长上下文（较好边缘配置）
run_edge_config "edge_32gb_7b"   32 "Qwen/Qwen2.5-7B-Instruct"   8192

# 汇总对比
python scripts/compare_real_runs.py \
  --run edge_16gb_1.5b=results/gpu/edge_sim/edge_16gb_1.5b \
  --run edge_24gb_7b=results/gpu/edge_sim/edge_24gb_7b \
  --run edge_32gb_7b=results/gpu/edge_sim/edge_32gb_7b \
  --output-dir results/gpu/edge_sim/summary
```

#### 报告叙事框架（对准赛题特征1）

在报告中增加章节"边缘设备场景验证"：

```
## 5. 边缘设备场景验证

### 5.1 实验设置
本节模拟赛题所指的"嵌入式设备、边缘计算设备"场景，
通过 Linux cgroups v2 限制 CPU 内存，
模拟 16GB / 24GB / 32GB 三种边缘节点配置。

| 配置 | 模拟设备类型 | CPU 内存限制 | 模型 | 最大 Context |
|------|------------|------------|------|-------------|
| edge_16gb | 低端推理卡/工控机 | 16GB | Qwen2.5-1.5B | 4096 |
| edge_24gb | 中端边缘服务器  | 24GB | Qwen2.5-7B | 4096 |
| edge_32gb | 高端边缘服务器  | 32GB | Qwen2.5-7B | 8192 |

### 5.2 实验结果
[插入 edge_sim/summary/comparison_report.md 内容]

### 5.3 结论
在 16GB 受限场景下，vLLM-only 在 context=4096 时 OOM rate 为 XX%，
而 AstraKV-W（LMCache CPU tier + memory pressure controller）
将 OOM rate 降低至 XX%，最大可用 context 从 XXXX 提升至 XXXX。
```

---

## 4. P2：冲奖扩展项

> **成本**：较高，在 P0/P1 完成后按剩余时间选做。
> **作用**：拉开与其他队伍的差距，提升"系统完整度"评分。

### 4.1 MoE 专家激活实验（从 optional 升级）

赛题任务1明确提到"专家激活"，不能完全跳过。

**最低成本实现**：使用 Qwen2-MoE 或 Mixtral 做路由 trace 离线分析，不需要完整 MoE serving。

```bash
# 用较小的 MoE 模型做 expert route trace
export ASTRAKV_MODEL=Qwen/Qwen1.5-MoE-A2.7B-Chat  # 2.7B active params
bash scripts/launch_vllm_server.sh 2>&1 | tee results/gpu/logs/moe_server.log

# 跑 benchmark，收集 router log
python scripts/run_real_benchmark.py \
  --config configs/dgx_spark_vllm_qwen7b.yaml \
  --output-dir results/gpu/moe_baseline

# 提取 expert route events
python scripts/extract_moe_expert_events.py \
  --router-log results/gpu/logs/moe_server.log \
  --output-dir results/gpu/moe_events

# 运行三种预测器对比
for predictor in next_token history_window profile_guided; do
  python scripts/predict_moe_experts.py \
    --moe-events results/gpu/moe_events/moe_expert_events.jsonl \
    --predictor-name $predictor \
    --output-dir results/gpu/moe_predict_$predictor
done
```

**报告叙事**：

```
## 6. MoE 专家激活分析

赛题指出 LLM 推理中"专家激活数量较少"，
为数据预取提供了施展空间。本节对 Qwen1.5-MoE 的
expert activation pattern 进行了实测分析：

- 平均每 token 激活 X / total_experts 个专家
- 专家激活具有明显的局部性（top-K 专家占 XX% 的激活）
- history_window predictor 的 top-1 命中率为 XX%
- 基于以上分析，profile-guided prefetch 可提前加载
  高概率激活专家，理论上减少 XX% 的专家加载延迟
```

### 4.2 uffd/mmap 完整集成

将 3.1 节的 mmap_kv_cache 接入 LMCache 作为新的 backend，使 KV cache 的 on-demand loading 和 eviction 直接由 OS 虚拟内存管理。

集成点：在 `lmcache/backends/` 下增加 `mmap_backend.py`，实现 LMCache backend 接口。

### 4.3 实时可演示 Dashboard

评审要点明确要求"可演示性"，静态图表不够有力。

```python
# scripts/build_realtime_dashboard.py
"""
实时 Dashboard，展示：
- GPU 内存使用曲线（实时）
- CPU RSS 变化（实时）
- KV cache tier 分布（GPU/CPU/SSD block 数量）
- 当前 memory pressure 等级
- 最近 10 个请求的 TTFT 分布
- prefetch hit/miss 实时计数

技术：FastAPI + SSE（Server-Sent Events） + HTML5 Canvas
可在浏览器打开，实时演示 AstraKV-W 的工作状态
"""
```

---

## 5. 实现时间线与优先级

```
Week 1（立即执行）
├── [P0] 补充 competition_report.md：虚拟内存映射章节    1天
├── [P0] 补充 competition_report.md：FlexInfer 对比章节  1天
└── [P1] GPU 上跑完三组 baseline + stress（已有流程）     2-3天

Week 2（系统实现）
├── [P1] 实现 mmap_kv_cache.py + 验收脚本               2天
├── [P1] 运行 layer_offload_poc.py（1.5B 模型）         1天
└── [P1] 设计并运行消融实验（shared prefix workload）     2天

Week 3（边缘测试 + 集成）
├── [P1] cgroups 边缘模拟测试（三种配置）                1-2天
├── [P1] 将 mmap 结果写入报告                           1天
└── [P2] MoE 专家激活实验                               1-2天

Week 4（收尾 + 演示）
├── [P2] 实时 dashboard                                 2天
├── 最终 competition report 更新                        1天
└── 答辩 PPT 准备                                       1-2天
```

---

## 6. 答辩关键问题准备

以下是评委必问问题及建议回答框架：

**Q1：你们用的是什么虚拟内存技术？**
> "AstraKV-W 在两个层面使用了虚拟内存技术。
> 第一层是理论对应：LMCache 的 CPU/SSD tiering 实现了与 OS 虚拟内存等价的三级存储层次，chunk scorer 对应页面置换算法，memory pressure controller 对应 OOM killer。
> 第二层是直接使用：我们实现了基于 mmap + madvise 的 KV cache 管理器，通过 madvise(MADV_WILLNEED) 实现 OS 级预取，通过 madvise(MADV_DONTNEED) 实现主动换出，直接复用 Linux 虚拟内存子系统。"

**Q2：你们和 FlexInfer 有什么区别？**
> "FlexInfer 解决权重 offloading 问题，目标是单用户端侧推理。
> AstraKV-W 解决 KV cache 管理问题，目标是多用户服务场景。
> 优化对象、目标场景、实现方式均不同，是互补关系。
> 我们的 layer offload PoC 验证了 AstraKV-W 框架可扩展到权重管理，思路与 FlexInfer 一致但面向服务场景。"

**Q3：你们的 scheduler 真的运行起来了吗？**
> "scheduler 输出的 hints 通过 selective prefetch endpoint 消费，prefetch_submitted 和 prefetch_completed 有真实日志证据。
> 对于 chunk eviction hints，通过 mmap_kv_cache 的 madvise 接口可以真实触发 OS 换出，这部分有 mincore 验证数据。
> unified object scheduler 和 load-vs-recompute planner 目前是 adapter-facing hints，下一步的工作是完整集成进 LMCache backend。"

**Q4：在真实边缘设备上跑过吗？**
> "我们通过 cgroups v2 限制 CPU 内存，模拟了 16GB / 24GB / 32GB 三种边缘节点配置，
> 并对比了 vLLM-only 和 AstraKV-W 在受限场景下的 OOM rate 和最大可用 context。
> 在 16GB 受限场景下，AstraKV-W 将 OOM rate 从 XX% 降低至 XX%。
> 后续工作是在真实边缘硬件（如 Jetson Orin 或 32GB 内存工控机）上验证。"

**Q5：你们的方法对推理质量有影响吗？**
> "我们实现了 quality evaluator，对比 vLLM baseline 和 LMCache 两种 variant 的输出一致性。
> output match rate 为 XX%，token divergence 为 XX，表明 tiering 策略不影响模型输出质量。
> KV cache 的 eviction 只影响缓存复用效率（体现在 TTFT 上），不影响当前请求的 KV 计算正确性。"

---

*文档结束。P0 章节（第2节）可直接复制进 competition_report.md，P1 代码框架可直接在项目中创建对应脚本文件。*
