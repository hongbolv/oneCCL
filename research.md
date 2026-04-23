# oneCCL（oneAPI Collective Communications Library）深度研究报告

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 整体架构设计](#2-整体架构设计)
- [3. 目录结构与模块划分](#3-目录结构与模块划分)
- [4. 核心数据结构](#4-核心数据结构)
  - [4.6 Rank 详解](#46-rank-详解)
- [5. 初始化与生命周期管理](#5-初始化与生命周期管理)
- [6. 传输层抽象（ATL）](#6-传输层抽象atl)
- [7. 集合通信操作](#7-集合通信操作)
- [8. 调度器与执行引擎](#8-调度器与执行引擎)
- [9. 算法选择策略](#9-算法选择策略)
- [10. 集合通信算法详解](#10-集合通信算法详解)
- [11. GPU/SYCL 加速支持](#11-gpusycl-加速支持)
- [12. 性能优化机制](#12-性能优化机制)
- [13. 完整调用流程追踪：以 Allreduce 为例](#13-完整调用流程追踪以-allreduce-为例)
- [14. 构建系统与依赖管理](#14-构建系统与依赖管理)
- [15. 环境变量与运行时配置](#15-环境变量与运行时配置)
- [16. 总结](#16-总结)

---

## 1. 项目概述

### 1.1 什么是 oneCCL

oneCCL（oneAPI Collective Communications Library）是 Intel 开发的高性能集合通信库，专为分布式深度学习训练场景设计。它是 oneAPI 规范的一部分，由 UXL Foundation 治理。

**核心定位**：为分布式深度学习提供高效的集合通信原语（如 allreduce、allgather、broadcast 等），支持 CPU 和 GPU 异构计算环境。

### 1.2 关键特性

| 特性 | 说明 |
|------|------|
| **多后端支持** | 支持 MPI 和 OFI（libfabric）两种传输后端 |
| **异构计算** | 同时支持 CPU 和 GPU（通过 SYCL/Level Zero） |
| **多种算法** | 每种集合操作提供多种算法实现（ring、tree、recursive doubling 等） |
| **自动算法选择** | 根据消息大小、拓扑、硬件等自动选择最优算法 |
| **操作融合** | 可将多个小操作合并执行以减少通信开销 |
| **低精度支持** | 原生支持 BF16、FP16 数据类型的通信与规约 |
| **拓扑感知** | 自动发现硬件拓扑（XeLink、PCIe），优化通信路径 |

### 1.3 应用集成

oneCCL 已集成到以下主流深度学习框架中：
- **PyTorch**（通过 torch-ccl 扩展）
- **Horovod**（分布式训练框架）

### 1.4 项目信息

- **版本**：2021.17.2（Gold 版本）
- **许可证**：Apache License 2.0
- **编程语言**：C++（C++11 标准），辅以 C、CMake、Shell
- **支持平台**：Ubuntu 18+，GCC 4.8.5+，Intel oneAPI DPC++/C++ 编译器

---

## 2. 整体架构设计

### 2.1 分层架构

oneCCL 采用清晰的分层架构设计，从上到下分为以下层次：

```
┌─────────────────────────────────────────────────────────┐
│                   用户应用层 (Application)                │
│         PyTorch / Horovod / 自定义 MPI 应用              │
├─────────────────────────────────────────────────────────┤
│                   公共 API 层 (Public API)               │
│    ccl::allreduce / ccl::broadcast / ccl::barrier ...   │
│              include/oneapi/ccl.hpp                      │
├─────────────────────────────────────────────────────────┤
│              集合操作调度层 (Collective Dispatch)          │
│     算法选择 → 调度构建 → 融合优化 → 并行化               │
│          src/coll/ + src/sched/ + src/fusion/            │
├─────────────────────────────────────────────────────────┤
│              执行引擎层 (Execution Engine)                │
│       Worker 线程池 → 优先级队列 → 条目执行               │
│              src/exec/ + src/sched/entry/                │
├─────────────────────────────────────────────────────────┤
│           传输抽象层 (Abstract Transport Layer, ATL)       │
│              atl_base_transport 接口                      │
│       ┌──────────────┬──────────────┐                    │
│       │   MPI 后端    │   OFI 后端    │                    │
│       │ src/atl/mpi/ │ src/atl/ofi/ │                    │
│       └──────────────┴──────────────┘                    │
├─────────────────────────────────────────────────────────┤
│               硬件/网络层 (Hardware/Network)              │
│     Ethernet │ InfiniBand │ Intel XeLink │ PCIe          │
└─────────────────────────────────────────────────────────┘
```

### 2.2 核心组件关系

```
                         ccl::init()
                             │
                             ▼
                      ┌─────────────┐
                      │ global_data │ ← 全局单例，管理所有核心组件
                      └──────┬──────┘
           ┌────────┬────────┼────────┬──────────┐
           ▼        ▼        ▼        ▼          ▼
      ┌────────┐┌────────┐┌──────┐┌────────┐┌─────────┐
      │executor││ sched  ││ algo ││fusion  ││  topo   │
      │(执行器)││ cache  ││select││manager ││ manager │
      └───┬────┘└────────┘└──────┘└────────┘└─────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌────────┐ ┌────────┐
│worker 0│ │worker N│  ← 工作线程
└───┬────┘ └───┬────┘
    │          │
    ▼          ▼
┌─────────────────┐
│  ATL Transport  │ ← 传输层（MPI 或 OFI）
└─────────────────┘
```

---

## 3. 目录结构与模块划分

### 3.1 顶层目录

```
oneCCL/
├── CMakeLists.txt          # 主构建配置
├── README.md               # 项目说明
├── INSTALL.md              # 安装指南
├── LICENSE                 # Apache 2.0 许可证
├── include/                # 公共 API 头文件
├── src/                    # 核心源代码
├── examples/               # 示例程序
├── tests/                  # 功能测试
├── deps/                   # 外部依赖
├── cmake/                  # CMake 构建脚本
├── doc/                    # 文档（Sphinx RST + Doxygen）
└── pkgconfig/              # pkg-config 配置
```

### 3.2 源代码模块

```
src/
├── atl/                    # 传输抽象层 (Abstract Transport Layer)
│   ├── mpi/               #   MPI 后端实现
│   ├── ofi/               #   OFI/libfabric 后端实现
│   └── util/              #   传输工具（PMI、KVS）
│       └── pm/            #     进程管理接口
├── coll/                   # 集合操作核心
│   ├── algorithms/        #   算法实现
│   │   ├── allreduce/     #     全规约算法
│   │   ├── allgatherv/    #     全收集算法
│   │   ├── broadcast/     #     广播算法
│   │   ├── reduce_scatter/#     规约散射算法
│   │   ├── barrier/       #     屏障算法
│   │   ├── alltoall/      #     全交换算法
│   │   ├── send/          #     点对点发送
│   │   └── recv/          #     点对点接收
│   ├── selection/         #   算法选择逻辑
│   ├── attr/              #   操作属性
│   └── group/             #   组操作
├── comm/                   # 通信器管理
├── sched/                  # 调度器
│   ├── entry/             #   调度条目类型
│   │   └── factory/       #     条目工厂
│   ├── buffer/            #   缓冲区管理
│   ├── cache/             #   调度缓存
│   ├── queue/             #   调度队列
│   └── ze/                #   Level Zero 相关调度
├── exec/                   # 执行引擎
│   └── thread/            #   工作线程
├── comp/                   # 计算/压缩组件
│   ├── bf16/              #   BFloat16 支持
│   └── fp16/              #   Float16 支持
├── fusion/                 # 操作融合优化
├── parallelizer/           # 并行化逻辑
├── topology/               # 硬件拓扑发现
├── common/                 # 公共工具
│   ├── global/            #   全局初始化
│   ├── datatype/          #   数据类型
│   ├── event/             #   事件系统
│   ├── stream/            #   流抽象
│   ├── request/           #   请求跟踪
│   └── utils/             #   通用工具
├── native_device_api/      # 设备 API 封装
│   └── sycl/              #   SYCL 后端
├── kernels/                # SYCL GPU 内核
└── hwloc/                  # 硬件拓扑发现封装
```

---

## 4. 核心数据结构

### 4.1 全局数据管理 (`global_data`)

文件：`src/common/global/global.hpp`

```cpp
class global_data {
    std::unique_ptr<ccl_datatype_storage> dtypes;           // 数据类型注册表
    std::unique_ptr<ccl_reduction_type_storage> redtype_storage; // 规约操作注册表
    std::unique_ptr<ccl_executor> executor;                  // 执行引擎
    std::unique_ptr<ccl_sched_cache> sched_cache;           // 调度缓存
    std::unique_ptr<ccl_parallelizer> parallelizer;         // 并行化器
    std::unique_ptr<ccl_fusion_manager> fusion_manager;     // 融合管理器
    std::unique_ptr<ccl_algorithm_selector_wrapper> algorithm_selector; // 算法选择器
    std::unique_ptr<ccl_hwloc_wrapper> hwloc_wrapper;       // 硬件拓扑
};
```

`global_data` 是整个库的核心单例对象，管理所有共享的子系统组件。

### 4.2 数据类型 (`datatype`)

文件：`include/oneapi/ccl/types.hpp`

```cpp
enum class datatype : int {
    int8, uint8, int16, uint16,
    int32, uint32, int64, uint64,
    float16,    // 半精度浮点
    float32,    // 单精度浮点
    float64,    // 双精度浮点
    bfloat16    // Brain Float 16（深度学习常用）
};
```

### 4.3 规约操作 (`reduction`)

```cpp
enum class reduction : int {
    sum = 0,        // 求和
    prod = 1,       // 乘积
    min = 2,        // 最小值
    max = 3,        // 最大值
    avg = 4,        // 平均值
    custom = 5      // 用户自定义
};
```

### 4.4 集合操作参数 (`ccl_coll_param`)

文件：`src/coll/coll_param.hpp`

```cpp
struct ccl_coll_param {
    ccl_coll_type ctype;        // 操作类型（allreduce, broadcast 等）
    ccl_coll_algo hint_algo;    // 建议使用的算法
    ccl_buffer send_buf;        // 发送缓冲区
    ccl_buffer recv_buf;        // 接收缓冲区
    size_t count;               // 元素数量
    ccl_datatype dtype;         // 数据类型
    ccl::reduction reduction;   // 规约操作
    int root;                   // 根节点（用于 broadcast/reduce）
};
```

### 4.5 通信器 (`ccl_comm`)

文件：`src/comm/comm.hpp`

通信器是 oneCCL 中最重要的抽象之一，它封装了一组参与通信的进程（rank）：
- 管理 rank 编号和映射关系
- 支持通信器拆分（`comm_split`）
- 维护拓扑感知的子通信器（pair_comm、even_comm、node_comm、r2r_comm）

### 4.6 Rank 详解

#### 4.6.1 什么是 Rank

在分布式计算中，**rank** 是分配给每个参与进程的唯一整数标识符，范围从 `0` 到 `size - 1`（其中 `size` 是通信器中的进程总数）。可以把 rank 理解为"进程编号"——在一个由多台机器（或多个 GPU）组成的分布式系统中，每个参与计算的进程都有一个独一无二的 rank。

**类比**：如果把分布式训练比作一个团队协作项目，每个团队成员（进程）都有一个工号（rank）。成员之间通过工号来标识"我要把数据发给谁"或"我从谁那里接收数据"。

#### 4.6.2 Rank 在 oneCCL 中的数据结构

**进程坐标结构**（文件：`src/atl/atl_def.h`）：

```cpp
typedef struct atl_proc_coord {
    int global_idx;           // 全局 rank 索引（在整个系统中的编号）
    int global_count;         // 全局进程总数
    int local_idx;            // 本地 rank 索引（在同一台机器上的编号）
    int local_count;          // 同一台机器上的进程数
    std::vector<int> global2local_map;  // 全局 rank → 本地索引映射
    size_t hostname_hash;     // 主机标识哈希值
} atl_proc_coord_t;
```

**通信器中的 rank 管理**（文件：`src/atl/atl_base_comm.hpp`）：

```cpp
class atl_base_comm {
protected:
    int rank;                          // 当前进程在此通信器中的 rank
    int size;                          // 通信器中的进程总数
    int parent_rank;                   // 在父通信器中的 rank（用于通信器拆分）
    int parent_size;                   // 父通信器大小

    std::vector<int> rank2rank_map;    // 本通信器 rank → 父通信器 rank 映射
    std::vector<int> rank2proc_map;    // 本通信器 rank → 物理进程索引映射
    atl_proc_coord_t coord;            // 进程坐标信息
};
```

**拓扑感知的 rank 信息**（文件：`src/topology/topo_manager.hpp`）：

```cpp
struct topo_rank_info {
    int rank;               // 全局 rank
    int host_idx;           // 所在主机索引
    int local_proc_idx;     // 主机上的本地进程索引
    char uuid[35];          // GPU 设备 UUID（用于 GPU rank）
};
```

#### 4.6.3 Rank 的多层语义

在 oneCCL 中，同一个物理进程可以在不同的上下文中拥有不同的 rank 值：

| 术语 | 含义 | 使用场景 |
|------|------|----------|
| **rank**（通信器内） | 进程在当前通信器中的编号（0 到 size-1） | 集合操作中标识参与者 |
| **global_idx** | 进程在全局系统中的绝对编号 | 系统级进程追踪 |
| **local_idx** | 进程在同一台主机上的编号 | 节点内通信优化 |
| **parent_rank** | 进程在父通信器中的编号 | 通信器拆分后保持映射关系 |

**示意图：通信器拆分后的 rank 变化**

```
全局通信器（4个进程，2台机器）:
┌──────────────────────────────────────────┐
│  rank=0      rank=1     rank=2    rank=3 │
│ (机器A)     (机器A)    (机器B)   (机器B)  │
└──────────────────────────────────────────┘
            │ comm_split（按机器拆分）
            ▼
节点内通信器 (机器A):        节点内通信器 (机器B):
┌───────────────────┐     ┌───────────────────┐
│ rank=0    rank=1  │     │ rank=0    rank=1  │
│(全局0)   (全局1)  │     │(全局2)   (全局3)  │
└───────────────────┘     └───────────────────┘

注意：拆分后同一个进程在不同通信器中有不同的 rank
     例如全局 rank=2 的进程，在机器B的节点内通信器中 rank=0
```

#### 4.6.4 Rank 在集合操作中的作用

rank 决定了每个进程在集合操作中的角色和数据归属：

**1. Allreduce 中的 rank**
```
所有 rank 贡献自己的数据，规约后所有 rank 得到相同的结果

rank 0: [1, 2, 3]  ─┐
rank 1: [4, 5, 6]  ─┤── allreduce(sum) ──→ 所有 rank 得到 [10, 14, 18]
rank 2: [5, 7, 9]  ─┘
```

**2. Broadcast 中的 rank**
```
root rank 的数据广播给所有其他 rank

rank 0 (root): [A, B, C]  ──broadcast──→  rank 0: [A, B, C]
rank 1:        [?, ?, ?]                   rank 1: [A, B, C]
rank 2:        [?, ?, ?]                   rank 2: [A, B, C]
```

**3. Reduce-Scatter 中的 rank**
```
规约后，结果的不同部分分配给不同 rank

rank 0: [1, 2, 3]  ─┐                     rank 0: [10]  (第0块的规约结果)
rank 1: [4, 5, 6]  ─┤── reduce_scatter ──→ rank 1: [14]  (第1块的规约结果)
rank 2: [5, 7, 9]  ─┘                     rank 2: [18]  (第2块的规约结果)
```

**4. Ring 算法中 rank 决定通信伙伴**
```
在 Ring Allreduce 中，rank 决定环形拓扑中的相邻关系：

    rank 0 ──→ rank 1 ──→ rank 2 ──→ rank 3
      ↑                                 │
      └─────────────────────────────────┘

每个 rank 只与相邻的 rank 通信：
  rank i 发送给 (i+1) % size
  rank i 接收来自 (i-1+size) % size
```

#### 4.6.5 用户代码中的 Rank 使用

文件：`examples/cpu/cpu_allreduce_test.cpp`

```cpp
int main() {
    ccl::init();

    int size, rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // 获取总进程数
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // 获取当前进程的 rank

    // 使用 rank 和 size 创建 CCL 通信器
    auto comm = ccl::create_communicator(size, rank, kvs);

    // 通过通信器查询 rank
    rank = comm.rank();   // 获取 CCL rank
    size = comm.size();   // 获取通信器大小

    // 每个 rank 用自己的 rank 值初始化数据
    std::vector<int> send_buf(count, rank);  // rank 0 填充 0, rank 1 填充 1, ...

    // 所有 rank 参与 allreduce
    ccl::allreduce(send_buf.data(), recv_buf.data(), count,
                   ccl::reduction::sum, comm).wait();

    // 通常由 rank 0 负责输出结果
    if (rank == 0) {
        std::cout << "结果验证通过" << std::endl;
    }
}
```

**运行方式**（启动 4 个 rank）：
```bash
mpirun -n 4 ./cpu_allreduce_test
#   -n 4 表示启动 4 个进程，rank 分别为 0, 1, 2, 3
```

#### 4.6.6 GPU 场景下的 Rank

在 GPU 场景中，每个 rank 通常关联一个 GPU 设备。rank 的拓扑信息用于优化通信路径：

```
节点 0 (2 GPU):
├── rank 0 → GPU 0 (UUID: xxx-aaa)  ─┐
│                                     ├─ XeLink 直连（节点内高速通信）
├── rank 1 → GPU 1 (UUID: xxx-bbb)  ─┘
│
节点 1 (2 GPU):
├── rank 2 → GPU 0 (UUID: yyy-aaa)  ─┐
│                                     ├─ XeLink 直连
└── rank 3 → GPU 1 (UUID: yyy-bbb)  ─┘

节点间通信: rank 0 ←──网络（OFI/MPI）──→ rank 2
```

oneCCL 利用 rank 的拓扑信息构建分层通信器：
- **pair_comm**：XeLink 直连的 GPU 对（如 rank 0 和 rank 1）
- **node_comm**：同一节点内的所有 rank
- **r2r_comm**：跨节点的对应 rank（如 rank 0 和 rank 2）

这使得 topo 算法能够先在节点内利用 XeLink 高速通信，再通过网络进行节点间通信，最大化整体通信效率。

---

## 5. 初始化与生命周期管理

### 5.1 初始化流程

```
应用程序调用 ccl::init()
         │
         ▼
environment::instance()  ──→  创建全局环境单例
         │
         ▼
global_data::init()
    ├── env.parse()           ──→  解析环境变量
    ├── dtypes = new storage  ──→  初始化数据类型
    ├── api_wrappers_init()   ──→  加载动态库（libfabric、MPI）
    │     ├── ofi_api_init()  ──→  加载 libfabric
    │     └── mpi_api_init()  ──→  加载 MPI（如果启用）
    ├── executor = new exec   ──→  创建执行引擎
    ├── sched_cache = new     ──→  初始化调度缓存
    ├── parallelizer = new    ──→  初始化并行化器
    ├── algorithm_selector    ──→  初始化算法选择器
    └── hwloc_wrapper = new   ──→  初始化硬件拓扑
         │
         ▼
atl_comm_manager::create()
    ├── 根据 CCL_ATL_TRANSPORT 选择后端
    ├── 创建 atl_ofi_comm 或 atl_mpi_comm
    ├── init_transport() ──→ 初始化传输层
    └── 交换端点地址（通过 KVS/PMI）
         │
         ▼
topo_manager::init()
    ├── 交换 rank 信息
    ├── 确定主机亲和性
    ├── 构建拓扑域（SYCL/Level Zero）
    └── 计算通信器拆分颜色
         │
         ▼
Ready ──→ 可以执行集合操作
```

### 5.2 终止化流程

```cpp
global_data::reset() {
    executor.reset();                    // 停止工作线程
    reset_resize_dependent_objects();     // 释放通信相关资源
    reset_resize_independent_objects();   // 释放全局资源
    pmix_api_fini();                     // 终止 PMIx
    api_wrappers_fini();                 // 卸载动态库
}
```

---

## 6. 传输层抽象（ATL）

### 6.1 架构设计

ATL（Abstract Transport Layer）是 oneCCL 的网络通信抽象层，定义了统一的传输接口。

文件：`src/atl/atl_base_transport.hpp`

```cpp
class atl_base_transport {
    // 点对点通信
    virtual atl_status_t send(const void* buf, size_t len, int dst, uint64_t tag, atl_req_t& req) = 0;
    virtual atl_status_t recv(void* buf, size_t len, int src, uint64_t tag, atl_req_t& req) = 0;

    // 集合通信
    virtual atl_status_t allreduce(const void* send, void* recv, size_t count, ...) = 0;
    virtual atl_status_t allgatherv(...) = 0;
    virtual atl_status_t barrier(...) = 0;
    virtual atl_status_t broadcast(...) = 0;

    // 远程内存访问
    virtual atl_status_t mr_reg(const void* buf, size_t len, atl_mr_t** mr) = 0;
    virtual atl_status_t read(void* buf, size_t len, atl_mr_t* mr, ...) = 0;
    virtual atl_status_t write(const void* buf, size_t len, atl_mr_t* mr, ...) = 0;
};
```

### 6.2 MPI 后端

文件：`src/atl/mpi/atl_mpi.cpp`（约 1104 行）

**初始化过程**：
1. 调用 `MPI_Init_thread(MPI_THREAD_MULTIPLE)` 请求完全线程安全的 MPI
2. 查询线程支持级别
3. 通过 `MPI_Comm_rank` / `MPI_Comm_size` 获取进程信息
4. 配置进度模式（POLL 或 CHECK）

**特点**：
- 使用标准 MPI 异步操作（`MPI_Isend`、`MPI_Irecv`、`MPI_Iallreduce` 等）
- 完整的数据类型映射（`atl2mpi_dtype`）
- 自定义 MPI_Op 支持 BF16/FP16 规约
- 不支持 RMA（远程内存访问）和内存注册

**数据结构**：
```cpp
typedef struct {
    MPI_Request native_req;         // MPI 请求句柄
    atl_mpi_comp_state_t comp_state; // 完成状态
} atl_mpi_req_t;

typedef struct {
    MPI_Comm mpi_comm;              // MPI 通信器
} atl_mpi_ep_t;
```

### 6.3 OFI 后端

文件：`src/atl/ofi/atl_ofi.cpp`（约 1524 行）

**初始化过程**：
1. 解析 `FI_PROVIDER` 环境变量确定网络提供者
2. 通过 PMI 获取进程信息
3. 创建 FI hints（指定端点类型 `FI_EP_RDM`、能力 `FI_TAGGED` 等）
4. 打开网络提供者（支持多 NIC）
5. 可选启用 SHM 提供者（节点内通信优化）
6. 通过 KVS/PMI 交换端点地址

**特点**：
- 使用 libfabric 的 Tagged Messaging（`fi_tsendv`/`fi_trecvv`）
- 支持多种网络：Verbs（InfiniBand）、PSM3、CXI
- 支持 RMA（远程内存访问）和内存注册（`fi_mr_reg`）
- 支持 HMEM（GPU 内存直接传输）
- 大部分集合操作在上层软件实现，OFI 只提供点对点原语

**提供者架构**：
```
OFI 后端
├── 网络提供者 (Network Provider)    ← 节点间通信
│   └── Verbs / PSM3 / CXI / etc.
└── SHM 提供者 (SHM Provider)        ← 节点内通信（可选）
    └── 共享内存直接传输
```

### 6.4 传输后端选择

文件：`src/common/env/env_parser.cpp`

选择逻辑：

1. **显式指定**：通过 `CCL_ATL_TRANSPORT` 环境变量（值为 `ofi` 或 `mpi`）
2. **自动检测**：检查 MPI 启动器环境变量（`MPI_LOCALRANKID`、`PMI_RANK` 等）
   - 检测到 MPI 启动器 → 使用 MPI
   - 未检测到 → 默认使用 OFI
3. **回退机制**：如果选定的后端初始化失败，尝试另一个后端

### 6.5 KVS（键值存储）机制

文件：`src/atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/`

KVS 是 oneCCL 用于进程协调的核心机制：
- **地址交换**：各进程通过 KVS 交换网络端点地址
- **Rank 映射**：建立全局 rank 到本地进程的映射
- **主机名交换**：确定哪些进程在同一节点上

支持的 KVS 模式：
```cpp
enum class kvs_mode : int {
    pmi,           // PMI-1 标准
    mpi,           // 通过 MPI 通信器交换
    pmix_ofi,      // PMIx + OFI
    pmix_ofi_shm   // PMIx + OFI + SHM
};
```

---

## 7. 集合通信操作

### 7.1 支持的操作类型

| 操作 | 说明 | 典型用途 |
|------|------|----------|
| **allreduce** | 全规约：所有 rank 贡献数据，规约结果分发给所有 rank | 梯度聚合 |
| **allgather/allgatherv** | 全收集：收集所有 rank 的数据并分发 | 参数同步 |
| **broadcast** | 广播：一个 rank 的数据发送给所有 rank | 模型初始化 |
| **reduce** | 规约：所有 rank 的数据规约到 root rank | 损失计算 |
| **reduce_scatter** | 规约散射：规约后将结果分散到各 rank | 梯度分片 |
| **alltoall/alltoallv** | 全交换：每个 rank 向每个 rank 发送不同数据 | 数据重分布 |
| **barrier** | 屏障同步：等待所有 rank 到达同步点 | 阶段同步 |
| **send/recv** | 点对点通信 | 自定义通信模式 |

### 7.2 API 接口设计

文件：`include/oneapi/ccl/api_functions.hpp`（约 2157 行）

每个集合操作都提供统一的 C++ API：

```cpp
ccl::event allreduce(
    const void* send_buf,       // 发送缓冲区
    void* recv_buf,              // 接收缓冲区
    size_t count,                // 元素数量
    ccl::datatype dtype,         // 数据类型
    ccl::reduction reduction,    // 规约操作
    const ccl::communicator& comm, // 通信器
    const ccl::stream& stream,   // 计算流（可选，用于 GPU）
    const ccl::allreduce_attr& attr, // 操作属性
    const vector_class<ccl::event>& deps  // 依赖事件
);
```

**设计特点**：
- 返回 `ccl::event` 对象，支持异步操作
- 可指定依赖事件（`deps`），实现操作间的依赖管理
- 通过 `stream` 参数支持 GPU 计算流集成
- 通过属性（`attr`）支持细粒度控制（如算法提示、优先级）

---

## 8. 调度器与执行引擎

### 8.1 调度器架构

oneCCL 使用调度器模式（Scheduler Pattern）将集合操作分解为一系列可执行的条目（Entry）。

#### 8.1.1 调度（Schedule）

文件：`src/sched/sched.hpp`

```cpp
class ccl_sched : public ccl_sched_base {
    std::deque<sched_entry_ptr> entries;           // 执行条目序列
    std::vector<std::shared_ptr<ccl_sched>> subscheds; // 子调度

    // 生命周期方法
    static ccl_sched_ptr create(const ccl_coll_param& param, const ccl_coll_attr& attr);
    void commit(ccl_parallelizer* parallelizer);
    ccl_request* start(ccl_executor* exec);
};
```

#### 8.1.2 调度条目类型

文件：`src/sched/entry/`

oneCCL 定义了 22 种调度条目类型：

| 类别 | 条目类型 | 说明 |
|------|----------|------|
| **通信** | `send_entry` | 点对点发送 |
| | `recv_entry` | 点对点接收 |
| | `recv_reduce_entry` | 接收并规约（融合操作） |
| | `recv_copy_entry` | 接收并复制 |
| **计算** | `reduce_local_entry` | 本地规约（无通信） |
| | `copy_entry` | 本地内存复制 |
| **内存** | `register_entry` | 内存注册（RDMA） |
| | `deregister_entry` | 内存注销 |
| **同步** | `sync_entry` | 子调度间同步 |
| | `barrier_entry` | 通信屏障 |
| | `deps_entry` | 依赖管理 |
| | `wait_value_entry` | 等待特定值 |
| **高级** | `subsched_entry` | 嵌套子调度 |
| | `coll_entry` | 集合操作条目 |
| | `function_entry` | 自定义函数调用 |
| | `write_entry` | 远程内存写入 |
| | `probe_entry` | 消息探测 |

#### 8.1.3 条目状态机

每个调度条目遵循以下状态转换：

```
NOT_STARTED ──→ AGAIN ──→ STARTED ──→ COMPLETE
     │                        │           │
     │                        └──→ AGAIN ──┘  (需要重试)
     │
     └──→ COMPLETE_ONCE  (一次性执行)
```

执行流程（`sched_entry::do_progress()`）：
1. 检查是否已完成
2. 如果 NOT_STARTED：获取信用（流量控制），调用 `start()`
3. 如果 STARTED：调用 `update()` 检查进度
4. 如果 COMPLETE：归还信用

### 8.2 调度缓存

文件：`src/sched/cache/cache.hpp`

为了避免重复构建相同的调度，oneCCL 使用 Hash 缓存：

```cpp
class ccl_sched_cache {
    find_or_create(ccl_sched_key&& key, const Lambda& create_fn) {
        // 自旋锁保护哈希表
        // 查找匹配的 key
        // 命中：增加引用计数，返回已有调度
        // 未命中：调用 create_fn() 创建新调度并缓存
    }
};
```

**缓存键**由以下因素组成：通信器、数据类型、缓冲区指针、操作属性等。

### 8.3 执行引擎

文件：`src/exec/exec.hpp`

```cpp
class ccl_executor {
    std::vector<std::unique_ptr<ccl_worker>> workers;  // 工作线程池

    void start(ccl_sched* sched);    // 分发调度到工作线程
    void wait(const ccl_request* req); // 等待操作完成
    bool test(const ccl_request* req); // 非阻塞检查完成
    void do_work();                    // 推进执行进度
};
```

#### 8.3.1 工作线程分发

```cpp
void ccl_executor::start(ccl_sched* sched) {
    auto& partial_scheds = sched->get_subscheds();
    size_t worker_idx = get_worker_idx_by_sched_id(partial_scheds[0].get());

    // 轮询分配子调度到工作线程
    for (auto& sub : partial_scheds) {
        workers[worker_idx]->add(sub.get());
        worker_idx = (worker_idx + 1) % workers.size();
    }
}
```

#### 8.3.2 工作线程执行循环

文件：`src/exec/thread/worker.cpp`

```cpp
ccl::status ccl_worker::do_work(size_t& processed_count) {
    // 1. 优先处理严格顺序队列
    process_strict_sched_queue();

    // 2. 处理优先级队列
    process_sched_queue(processed_count);
    //   └── peek() 获取最高优先级的 bin
    //   └── 遍历 bin 中的调度
    //       └── sched->do_progress() 执行条目
}
```

### 8.4 优先级队列

文件：`src/sched/queue/queue.hpp`

```cpp
class ccl_sched_queue {
    sched_bin_list_t bins;    // priority → ccl_sched_bin 映射

    struct ccl_sched_bin {
        size_t priority;              // 优先级
        size_t atl_ep;                // 关联的 ATL 端点
        ccl_sched_list sched_list;    // 调度列表
    };
};
```

每个调度根据优先级分配到不同的 bin，每个 bin 有专用的 ATL 端点以避免通信竞争。

### 8.5 请求与完成跟踪

文件：`src/common/request/request.hpp`

```cpp
class ccl_request {
    std::atomic_int completion_counter;  // 完成计数器

    void set_counter(int count);   // 设置为子调度数量
    void complete();                // 计数器递减
    bool is_completed() const;      // 计数器 == 0 表示完成
};
```

---

## 9. 算法选择策略

### 9.1 选择架构

文件：`src/coll/selection/`

oneCCL 采用基于消息大小的查找表进行算法选择，每种集合操作有独立的选择器。

#### 9.1.1 选择流程

```
ccl_coll_build_allreduce()
    │
    ▼
构建 ccl_selector_param
    ├── count（元素数量）
    ├── dtype（数据类型）
    ├── comm（通信器信息）
    ├── stream（设备流）
    └── buffer 类型（host/device）
    │
    ▼
algorithm_selector->get<ccl_coll_allreduce>(param)
    │
    ▼
查找 selection_table
    ├── main_table（主选择表）
    ├── fallback_table（回退表）
    └── scaleout_table（跨节点表）
    │
    ▼
返回选定的算法枚举值
```

#### 9.1.2 Allreduce 算法选择表

文件：`src/coll/selection/selector_allreduce.cpp`

| 传输后端 | 消息大小 | 默认算法 |
|----------|----------|----------|
| **SYCL + Level Zero（GPU）** | 全部 | `topo`（拓扑感知） |
| **OFI（CPU）** | < 8KB（短消息） | `recursive_doubling` |
| **OFI（CPU）** | 8KB - 1MB（中等消息） | `nreduce` |
| **OFI（CPU）** | > 1MB（大消息） | `ring` |
| **MPI** | 全部 | `direct`（委托给 MPI） |
| **跨节点** | 全部 | `ring` |

#### 9.1.3 算法约束条件

| 算法 | 约束条件 |
|------|----------|
| `rabenseifner` | `count >= pof2`（2 的幂次） |
| `ring_rma` | 需要 RMA 支持（`enable_rma`） |
| `nreduce` | `count / comm_size >= 1`（每个 rank 至少 1 个元素） |
| `direct` | OFI 传输时禁用 |
| `topo` | 仅 GPU（需要 SYCL + Level Zero） |
| `2d` | 跨节点配置时禁用 |

---

## 10. 集合通信算法详解

### 10.1 Allreduce 算法

#### 10.1.1 Ring Allreduce

**原理**：将 allreduce 分解为 reduce-scatter + allgather 两个阶段。

```
Phase 1: Reduce-Scatter（环形规约散射）
  每个 rank 在 (N-1) 轮中依次发送和接收数据块，
  每轮对收到的数据块进行本地规约。

Phase 2: Allgather（环形全收集）
  每个 rank 在 (N-1) 轮中将规约完成的数据块
  沿环形拓扑传播给所有 rank。

通信轮数: 2(N-1)
适用场景: 中大消息，稳定的延迟特性
```

**实现**（`allreduce.cpp` L442-538）：
```cpp
ccl_coll_build_reduce_scatter_block(sched, send_buf, recv_buf, ...);
sched->add_barrier();
ccl_coll_build_ring_allgatherv(sched, ...);
```

#### 10.1.2 Recursive Doubling（递归倍增）

**原理**：在 log₂(N) 轮中，每个 rank 与距离为 2^k 的伙伴交换并规约数据。

```
轮 0: rank i 与 rank (i XOR 1) 交换    ← 距离 1
轮 1: rank i 与 rank (i XOR 2) 交换    ← 距离 2
轮 2: rank i 与 rank (i XOR 4) 交换    ← 距离 4
...
轮 k: rank i 与 rank (i XOR 2^k) 交换

通信轮数: log₂(N)
适用场景: 短消息（延迟敏感）
```

**位操作实现**：
```cpp
mask = 0x1;
while (mask < pof2) {
    newdst = newrank ^ mask;   // XOR 确定通信伙伴
    // 与伙伴交换数据并规约
    mask <<= 1;               // 下一层
}
```

#### 10.1.3 Rabenseifner 算法

**原理**：结合递归倍增的 reduce-scatter 和 allgather 阶段，特别适合元素数量为 2 的幂次的情况。

```
Phase 1: 递归倍增 Reduce-Scatter
  log₂(N) 轮，每轮数据量减半

Phase 2: 递归倍增 Allgather
  log₂(N) 轮，每轮数据量加倍

通信量: 2(N-1)/N × count × sizeof(dtype)
适用场景: 大消息，进程数为 2 的幂次
```

#### 10.1.4 Nreduce（分段规约）

**原理**：将数据分为多个段，使用中间缓冲区进行流水线处理。

```
数据分段（默认 2MB 段大小）
每段独立执行规约操作
可配置: CCL_ALLREDUCE_NREDUCE_SEGMENT_SIZE

适用场景: 中等消息（8KB - 1MB）
```

#### 10.1.5 2D 算法（层次化）

**原理**：将通信分为两个维度（节点内 + 节点间），分别优化。

```
维度 1: 节点内通信（node_comm）
维度 2: 节点间通信（r2r_comm）

步骤:
1. reduce_scatter(维度1)   ← 节点内规约
2. allreduce(维度2)        ← 节点间全规约
3. allgatherv(维度1)       ← 节点内全收集

可通过 CCL_ALLREDUCE_2D_SWITCH_DIMS 切换维度顺序
```

#### 10.1.6 Topo 算法（GPU 拓扑感知）

**原理**：利用 GPU 硬件拓扑（XeLink、MDFi）进行最优通信。

```
利用的通信器层次:
├── pair_comm  ← XeLink 直连 GPU 对
├── even_comm  ← 偶数编号 GPU 组
├── node_comm  ← 节点内所有 GPU
└── r2r_comm   ← 跨节点 GPU

特性:
- 使用 MDFi 流水线内核
- 双向 XeLink 通信
- IPC 内存句柄实现 GPU 间直接访问
```

### 10.2 Broadcast 算法

#### 10.2.1 Naive Broadcast

```
Root 直接向每个 rank 发送: O(N) 通信
适用场景: 小规模集群
```

#### 10.2.2 二项树 Broadcast

```
使用二项树拓扑，log₂(N) 轮
每轮倍增覆盖范围
适用场景: 通用
```

#### 10.2.3 Scatter + Ring + Allgather

```
Phase 1: Scatter（数据分散）
Phase 2: Ring 传播
Phase 3: Allgather（收集完整数据）

适用场景: 大消息
```

### 10.3 Barrier 算法

#### Dissemination Barrier

```cpp
mask = 0x1;
while (mask < size) {
    dst = (rank + mask) % size;     // 发送目标
    src = (rank - mask + size) % size; // 接收来源
    send(dst); recv(src);
    mask <<= 1;
}
```

通信轮数：⌈log₂(N)⌉

### 10.4 Reduce-Scatter 算法

支持 Block 和 Strided 两种布局：
- **Block**：数据分为 N 块，每个 rank 规约后保留一块
- **Ring**：环形迭代规约，适合大消息
- **Topo**：GPU 拓扑感知的流水线规约

---

## 11. GPU/SYCL 加速支持

### 11.1 GPU 算法分类

GPU 上的集合操作根据消息大小分为三类：

| 分类 | 消息大小 | 实现策略 |
|------|----------|----------|
| **Small** | < 64KB | ESIMD 内核 + 原子操作同步 |
| **Medium** | 64KB - 1MB | 平衡内核开销和同步 |
| **Large** | > 1MB | 流水线操作 + 类型特化 |

### 11.2 Small Allreduce（ESIMD 实现）

文件：`src/coll/algorithms/allreduce/sycl/allreduce_small_sycl.hpp`

```
实现策略:
1. 通过 IPC 内存句柄建立 GPU 间直接内存访问
2. 每个 rank 分配: data_size + 128B(同步字节) 的缓冲区
3. 三重缓冲区（Triple Buffer）用于阶段重叠
4. 使用 ESIMD 内核执行规约:
   - SIMD 宽度 = 256B / sizeof(data_type)
   - MAX_THREAD = 512 EU × 8 threads/EU
   - 原子操作进行 GPU 间同步
```

### 11.3 Large Allreduce（类型特化）

文件：`src/coll/algorithms/allreduce/sycl/allreduce_large_sycl_*.cpp`

为不同数据类型提供独立的内核实现：
- `allreduce_large_sycl_fp16.cpp`
- `allreduce_large_sycl_bf16.cpp`
- `allreduce_large_sycl_fp32.cpp`
- `allreduce_large_sycl_int32.cpp`

### 11.4 GPU 硬件特性利用

| 特性 | 用途 |
|------|------|
| **XeLink** | GPU 间直接高带宽通信 |
| **MDFi** | 多维结构接口，用于流水线内核 |
| **IPC Memory Handles** | 跨进程 GPU 内存直接访问 |
| **Atomic Operations** | 小消息场景下的 GPU 间同步 |
| **ESIMD** | 显式 SIMD 编程，最大化计算吞吐 |

---

## 12. 性能优化机制

### 12.1 操作融合（Fusion）

文件：`src/fusion/fusion.hpp`

```cpp
class ccl_fusion_manager {
    size_t bytes_threshold;     // 数据量阈值
    size_t count_threshold;     // 操作数阈值
    duration cycle;              // 融合等待超时

    void add(ccl_sched* sched);  // 添加到待融合队列
    void execute();               // 检查是否触发融合
};
```

**工作原理**：
1. 小操作进入 `postponed_queue` 等待队列
2. 当满足以下条件之一时触发融合：
   - 累积数据量达到 `bytes_threshold`
   - 操作数达到 `count_threshold`
   - 等待超时（`cycle`）到期
   - 用户标记为紧急（`urgent`）
3. 构建融合调度，一次执行多个操作

### 12.2 调度缓存

对于相同参数的重复集合操作（如每个训练迭代的梯度聚合），调度缓存避免了重复的调度构建开销：
- 基于 Hash 的快速查找
- 引用计数管理缓存生命周期
- 支持重新缓存（key 变更时）

### 12.3 BF16/FP16 低精度优化

文件：`src/comp/bf16/bf16.hpp`、`src/comp/fp16/fp16.hpp`

**CPU 优化**：
- 使用 AVX512BF16 指令进行 BF16 规约
- 使用 F16C 指令进行 FP16/FP32 转换
- 规约时可选择在 FP32 精度下累加以减少精度损失

**GPU 优化**：
- 类型特化的 SYCL 内核
- 硬件原生 BF16 支持

### 12.4 In-Place 操作优化

当发送和接收缓冲区相同时（`send_buf == recv_buf`）：
- 跳过初始数据复制（`copy_entry`）
- 调整缓冲区偏移避免数据覆盖
- 减少内存占用和带宽消耗

### 12.5 流量控制

调度条目使用信用机制（`flow_control.take_credit()`/`return_credit()`）限制同时活跃的条目数量，避免资源耗尽。

### 12.6 工作线程亲和性

- 支持自动或手动设置 CPU 亲和性（`CCL_WORKER_AFFINITY`）
- 利用 hwloc 库发现 NUMA 拓扑
- 将工作线程绑定到最优的 CPU 核心

---

## 13. 完整调用流程追踪：以 Allreduce 为例

以下追踪一个 allreduce 操作从用户调用到完成的完整路径：

### Step 1: 用户 API 调用

```cpp
// 用户代码
auto event = ccl::allreduce(send_buf, recv_buf, count,
                            ccl::datatype::float32,
                            ccl::reduction::sum,
                            comm, stream, attr, deps);
```

### Step 2: API 层分发

文件：`src/ccl_api_functions.cpp` L455-468

```cpp
event allreduce(...) {
    impl_dispatch disp;
    return disp(comm)->allreduce(
        send_buf, recv_buf, count, dtype, reduction,
        disp(op_stream), attr, deps);
}
```

`impl_dispatch` 从 C++ 包装对象中提取底层实现（`ccl_comm*`）。

### Step 3: 集合操作创建

文件：`src/coll/coll.cpp` L156-472

```
ccl_coll_create()
    │
    ├── 构建 ccl_selector_param（count, dtype, comm, stream, buffer_type）
    ├── 通过算法选择器获取最优算法
    ├── 创建或从缓存获取调度（ccl_sched）
    ├── 检查是否可以融合
    ├── 并行化处理（parallelizer->commit()）
    └── 提交给执行器（executor->start()）
```

### Step 4: 算法选择

文件：`src/coll/coll.cpp` L611-690

```cpp
auto algo = global_data::get().algorithm_selector
    ->get<ccl_coll_allreduce>(param);

switch (algo) {
    case ccl_coll_allreduce_direct:
        ccl_coll_build_direct_allreduce(...); break;
    case ccl_coll_allreduce_ring:
        ccl_coll_build_ring_allreduce(...); break;
    case ccl_coll_allreduce_rabenseifner:
        ccl_coll_build_rabenseifner_allreduce(...); break;
    case ccl_coll_allreduce_topo:
        ccl_coll_build_topo_allreduce(...); break;
    // ... 其他算法
}
```

### Step 5: 调度构建（以 Ring 为例）

文件：`src/coll/algorithms/allreduce/allreduce.cpp` L442-538

```
ccl_coll_build_ring_allreduce()
    │
    ├── 检测 in-place 模式
    ├── Phase 1: 构建 reduce_scatter_block 条目
    │   └── entry_factory::create<send_entry>(...)
    │   └── entry_factory::create<recv_entry>(...)
    │   └── entry_factory::create<reduce_local_entry>(...)
    │
    ├── sched->add_barrier()  ← 阶段间同步
    │
    └── Phase 2: 构建 ring_allgatherv 条目
        └── entry_factory::create<send_entry>(...)
        └── entry_factory::create<recv_entry>(...)
        └── entry_factory::create<copy_entry>(...)
```

### Step 6: 调度提交与并行化

```
sched->commit(parallelizer)
    │
    ├── parallelizer->process(sched)
    │   ├── process_base()          ← 创建子调度
    │   ├── process_pre_post_copies() ← GPU 数据传输
    │   ├── process_output_event()  ← SYCL 事件处理
    │   └── process_deps()          ← 依赖管理
    │
    └── 添加 sync_entry 确保子调度同步
```

### Step 7: 执行器启动

```
executor->start(sched)
    │
    ├── 获取子调度列表
    ├── 计算起始 worker 索引
    └── 轮询分配子调度到 worker 线程
        ├── worker[0].add(subsched[0])
        ├── worker[1].add(subsched[1])
        └── ...
```

### Step 8: Worker 线程执行

```
ccl_worker::do_work()
    │
    ├── process_strict_sched_queue()  ← 严格顺序队列
    └── process_sched_queue()         ← 优先级队列
        │
        └── sched->do_progress()
            │
            └── 遍历 entries:
                ├── entry[0].do_progress()  ← send_entry::start()
                │   └── atl_comm->send(buf, len, dst, tag, req)
                │       └── MPI_Isend() 或 fi_tsendv()
                │
                ├── entry[1].do_progress()  ← recv_entry::start()
                │   └── atl_comm->recv(buf, len, src, tag, req)
                │       └── MPI_Irecv() 或 fi_trecvv()
                │
                ├── entry[2].do_progress()  ← reduce_local_entry
                │   └── 本地 reduce 计算
                │
                └── ... （每个条目独立推进）
```

### Step 9: 完成与通知

```
所有条目完成
    │
    ├── 每个子调度完成时: request->complete()
    │   └── completion_counter.fetch_sub(1)
    │
    ├── 所有子调度完成: completion_counter == 0
    │
    └── request->is_completed() 返回 true
        │
        └── 用户代码:
            event.wait();  // 阻塞等待
            // 或
            event.test();  // 非阻塞检查
```

### 完整流程图

```
用户代码: ccl::allreduce()
    │
    ▼
API 层: impl_dispatch → ccl_comm::allreduce()
    │
    ▼
调度层: ccl_coll_create()
    ├── 算法选择: algorithm_selector->get()
    ├── 缓存查找: sched_cache->find_or_create()
    ├── 调度构建: ccl_coll_build_ring_allreduce()
    │   └── 创建 send/recv/reduce 条目
    ├── 并行化: parallelizer->process()
    │   └── 创建子调度 + 同步条目
    └── 融合检查: fusion_manager->can_fuse()
    │
    ▼
执行层: executor->start(sched)
    └── 分发子调度到 worker 线程
    │
    ▼
Worker 线程: do_work() 循环
    └── 逐条目执行 do_progress()
        ├── send_entry → ATL send
        ├── recv_entry → ATL recv
        └── reduce_entry → 本地计算
    │
    ▼
传输层: atl_base_transport
    ├── MPI: MPI_Isend/MPI_Irecv
    └── OFI: fi_tsendv/fi_trecvv
    │
    ▼
完成通知: request->complete() → event 就绪
```

---

## 14. 构建系统与依赖管理

### 14.1 构建步骤

```bash
cd oneCCL
mkdir build && cd build
cmake ..
make -j install
```

### 14.2 主要构建选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `CMAKE_BUILD_TYPE` | Release | 构建类型 |
| `CMAKE_INSTALL_PREFIX` | `_install/` | 安装目录 |
| `BUILD_EXAMPLES` | TRUE | 构建示例程序 |
| `BUILD_FT` | TRUE | 构建功能测试 |
| `ENABLE_MPI` | TRUE | 启用 MPI 支持 |
| `ENABLE_OFI_HMEM` | TRUE | 启用 OFI HMEM（GPU 内存）支持 |
| `ENABLE_ITT` | TRUE | 启用 Intel VTune 性能分析 |
| `ENABLE_PMIX` | TRUE | 启用 PMIx 进程管理 |
| `ENABLE_OMP` | TRUE | 启用 OpenMP（节点内并行） |
| `ENABLE_STUB_BACKEND` | TRUE | 启用 Stub 后端 |

### 14.3 外部依赖

| 依赖 | 用途 | 类型 |
|------|------|------|
| **Intel MPI** | MPI 通信后端 | 必需（二选一） |
| **libfabric (OFI)** | 网络通信抽象 | 必需（二选一） |
| **hwloc** | 硬件拓扑发现 | 必需 |
| **Level Zero** | GPU API 抽象 | GPU 支持 |
| **PMIX** | 进程管理 | 可选 |
| **UMF** | 统一内存框架 | 可选 |
| **ITT** | Intel 性能分析工具 | 可选 |

### 14.4 安装目录结构

```
_install/intel64/
├── lib/                      # 库文件
├── include/oneapi/ccl/       # API 头文件
├── bin/                      # 可执行文件
├── examples/                 # 示例程序
├── tests/                    # 测试程序
├── env/                      # 环境设置脚本 (setvars.sh)
├── lib/cmake/oneCCL/         # CMake 配置文件
├── lib/ccl/kernels/          # SYCL GPU 内核 (.spv)
└── share/doc/ccl/            # 文档和许可证
```

---

## 15. 环境变量与运行时配置

### 15.1 传输配置

| 环境变量 | 值 | 说明 |
|----------|------|------|
| `CCL_ATL_TRANSPORT` | `ofi` / `mpi` | 选择传输后端 |
| `FI_PROVIDER` | 提供者名称 | 指定 OFI 网络提供者 |

### 15.2 工作线程配置

| 环境变量 | 说明 |
|----------|------|
| `CCL_WORKER_COUNT` | 工作线程数量 |
| `CCL_WORKER_AFFINITY` | CPU 亲和性（`auto` 或 CPU 列表） |

### 15.3 算法配置

| 环境变量 | 说明 |
|----------|------|
| `CCL_ALLREDUCE` | 指定 allreduce 算法 |
| `CCL_ALLGATHERV` | 指定 allgatherv 算法 |
| `CCL_BROADCAST` | 指定 broadcast 算法 |
| `CCL_REDUCE_SCATTER` | 指定 reduce_scatter 算法 |
| `CCL_ALLREDUCE_SHORT_MSG_SIZE` | 短消息阈值（默认 ~8KB） |
| `CCL_ALLREDUCE_MEDIUM_MSG_SIZE` | 中等消息阈值（默认 ~1MB） |
| `CCL_ALLREDUCE_NREDUCE_SEGMENT_SIZE` | Nreduce 分段大小 |
| `CCL_ALLREDUCE_2D_SWITCH_DIMS` | 2D 算法维度切换 |
| `CCL_ALLREDUCE_2D_CHUNK_COUNT` | 2D 算法分块数量 |

### 15.4 融合配置

| 环境变量 | 说明 |
|----------|------|
| `CCL_FUSION` | 启用/禁用融合 |
| `CCL_FUSION_BYTES_THRESHOLD` | 融合数据量阈值 |
| `CCL_FUSION_COUNT_THRESHOLD` | 融合操作数阈值 |
| `CCL_FUSION_CYCLE_MS` | 融合等待超时（毫秒） |

### 15.5 GPU 配置

| 环境变量 | 说明 |
|----------|------|
| `CCL_REDUCE_SCATTER_MONOLITHIC_PIPELINE_KERNEL` | MDFi 流水线内核 |
| `CCL_ALLGATHERV_MONOLITHIC_PIPELINE_KERNEL` | Allgatherv 流水线内核 |
| `CCL_ENABLE_ZE_BIDIR_ALGO` | 双向 XeLink 算法 |
| `CCL_ZE_MULTI_WORKERS` | 多 worker GPU 扩展 |

---

## 16. 总结

### 16.1 设计亮点

1. **清晰的分层架构**：API 层、调度层、执行层、传输层各司其职，通过定义良好的接口解耦
2. **调度器模式**：将复杂的集合操作分解为可组合的条目序列，便于实现多种算法
3. **传输抽象**：统一的 ATL 接口使得添加新的传输后端只需实现 `atl_base_transport`
4. **自适应算法选择**：根据消息大小、硬件拓扑、传输后端自动选择最优算法
5. **缓存与融合**：调度缓存和操作融合大幅减少重复操作的开销
6. **异构计算支持**：通过 SYCL/Level Zero 无缝支持 CPU 和 GPU

### 16.2 关键设计模式

| 模式 | 应用 |
|------|------|
| **单例模式** | `global_data`、`atl_base_transport`（共享传输实例） |
| **工厂模式** | `entry_factory`（创建调度条目）、`atl_comm_manager`（创建通信器） |
| **策略模式** | 算法选择器（根据参数选择不同算法实现） |
| **观察者/事件模式** | `ccl_request` + `ccl::event` 的异步完成通知 |
| **生产者-消费者模式** | 调度队列 + Worker 线程的分发机制 |
| **组合模式** | 子调度（`subsched_entry`）允许递归嵌套 |

### 16.3 性能优化总结

```
消息大小维度:
├── 短消息 (< 8KB): 递归倍增 → 最小化延迟
├── 中等消息 (8KB-1MB): Nreduce/分段 → 平衡延迟和带宽
└── 大消息 (> 1MB): Ring → 最大化带宽利用

硬件维度:
├── CPU: Ring / Recursive Doubling / Rabenseifner
├── GPU (Small): ESIMD + 原子同步
├── GPU (Medium): 平衡内核开销
└── GPU (Large): 流水线 + 类型特化

拓扑维度:
├── 节点内: SHM 提供者 / XeLink 直连
├── 节点间: OFI 网络 / MPI
└── 层次化: 2D 算法（节点内 + 节点间分离）

运行时优化:
├── 调度缓存: 避免重复构建
├── 操作融合: 合并小操作
├── In-Place: 减少内存复制
└── 流量控制: 防止资源耗尽
```

### 16.4 适用场景

oneCCL 最适合以下场景：
- **分布式深度学习训练**：梯度聚合（allreduce）、模型同步（broadcast）
- **Intel 硬件生态**：在 Intel CPU + Intel GPU 上获得最佳性能
- **大规模集群**：拓扑感知算法在多节点场景下优势明显
- **PyTorch/Horovod 集成**：通过成熟的插件直接使用

---

## 17. 案例分析：跨 ZE_AFFINITY_MASK 的 IPC 通信 Hang 问题

### 17.1 问题描述

在 4 张 Intel B60 GPU 上运行 `test_xccl_cross_affinity.py` 测试脚本，三种模式的表现不同：

| 模式 | ZE_AFFINITY_MASK 设置 | 结果 |
|------|----------------------|------|
| **A (baseline)** | 不设置（所有进程看到全部 4 GPU） | ✅ 正常 |
| **B (cross_affinity)** | Rank 0,1 → "0,1"；Rank 2,3 → "2,3" | ❌ Hang |
| **C (same_affinity)** | 所有进程 → "0,1,2,3" | ✅ 正常 |

### 17.2 根因分析

#### 17.2.1 ZE_AFFINITY_MASK 的作用机制

`ZE_AFFINITY_MASK` 是 Level Zero 运行时的环境变量，在 `zeDeviceGet()` 阶段过滤可见设备。oneCCL 在初始化时调用此 API 获取设备列表：

```cpp
// src/common/global/ze/ze_data.cpp, 第 81-84 行
uint32_t device_count{};
ZE_CALL(zeDeviceGet, (drivers.at(i), &device_count, nullptr));
std::vector<ze_device_handle_t> devs(device_count);
ZE_CALL(zeDeviceGet, (drivers.at(i), &device_count, devs.data()));
```

设置不同 `ZE_AFFINITY_MASK` 后，各进程看到的设备列表不同：

```
模式 B 中的设备视图：
┌─────────────────────────────────────────────────────────┐
│ Rank 0,1 (ZE_AFFINITY_MASK="0,1")                      │
│   zeDeviceGet → 返回 2 个设备                           │
│   devices[0] = 物理 GPU 0                               │
│   devices[1] = 物理 GPU 1                               │
│   contexts[0] = 关联 GPU 0,1 的上下文                    │
├─────────────────────────────────────────────────────────┤
│ Rank 2,3 (ZE_AFFINITY_MASK="2,3")                      │
│   zeDeviceGet → 返回 2 个设备                           │
│   devices[0] = 物理 GPU 2（但本地索引为 0）              │
│   devices[1] = 物理 GPU 3（但本地索引为 1）              │
│   contexts[0] = 关联 GPU 2,3 的上下文                    │
└─────────────────────────────────────────────────────────┘
```

#### 17.2.2 IPC Handle 交换流程中的断裂点

oneCCL 的 `all_reduce` 在 GPU 模式下通过 IPC（Inter-Process Communication）在进程间共享 GPU 内存。完整流程如下：

**步骤 1：发送端创建 IPC Handle**

```cpp
// src/sched/entry/ze/ze_handle_exchange_entry.cpp, 第 130-137 行
// Rank 0: 在自己的 context 上调用 zeMemGetIpcHandle
mem_info = get_mem_info(mem_ptr);
sched->get_memory().handle_manager.get_handle(
    mem_info.first, &ipc_handle, &handle_id);
```

Rank 0 在 GPU 0 上分配内存并生成 IPC handle。

**步骤 2：附加上下文和设备元数据**

```cpp
// src/sched/entry/ze/ze_handle_exchange_entry.cpp, 第 234-254 行
ccl::ze::get_buffer_context_and_device(mem_ptr, &remote_context, &remote_device, &mem_alloc_props);
ccl::ze::get_context_global_id(remote_context, &remote_context_id);
ccl::ze::get_device_global_id(remote_device, &remote_device_id);
payload.remote_context_id = remote_context_id;  // 例如: 0
payload.remote_device_id = remote_device_id;    // 例如: 0
```

Rank 0 将 `remote_context_id=0`（自己的 context）和 `remote_device_id=0`（物理 GPU 0，本地索引 0）写入 payload。

**步骤 3：通过 allgather 交换 payload**

```cpp
// src/sched/entry/ze/ze_handle_exchange_entry.cpp, 第 309-324 行
ccl::utils::allgather(comm->get_atl_comm(),
                      local_payloads.data(),
                      all_payloads.data(),
                      sizeof(payload_t) * in_buffers.size());
```

所有 rank 交换各自的 IPC payload（包含 handle、context_id、device_id 等）。

**步骤 4：接收端打开 IPC Handle（❌ 断裂点）**

```cpp
// src/sched/entry/ze/cache/ze_cache.cpp, 第 430-435 行
auto remote_context_id = std::get<...>(key);
auto remote_context = global_data::get().ze_data->contexts.at(remote_context_id);
ze_ipc_mem_handle_t handle = info.mem_to_ipc_handle();
ZE_CALL(zeMemOpenIpcHandle, (remote_context, device, handle, {}, &ptr));
```

当 Rank 2 尝试打开 Rank 0 的 IPC handle 时：
- `remote_context_id=0`：Rank 0 中指向 GPU 0,1 的上下文
- 但在 Rank 2 进程中，`contexts[0]` 指向 **GPU 2,3** 的上下文（因为 `ZE_AFFINITY_MASK="2,3"`）
- **上下文不匹配**：Rank 2 用一个关联 GPU 2,3 的上下文去打开一个在 GPU 0 上创建的 IPC handle

#### 17.2.3 Hang 的直接原因

```
Rank 0 创建 IPC handle                Rank 2 尝试打开 IPC handle
┌──────────────────────┐              ┌──────────────────────┐
│ context → GPU 0,1    │              │ context → GPU 2,3    │
│ device  → GPU 0      │              │ device  → GPU 2      │
│ 内存在 GPU 0 上      │──── IPC ────→│ 用 GPU 2,3 的 context│
│                      │   handle     │ 打开 GPU 0 的内存    │
│                      │              │                      │
│                      │              │ ❌ zeMemOpenIpcHandle │
│                      │              │    无法映射到本地设备  │
│                      │              │    → hang / 错误      │
└──────────────────────┘              └──────────────────────┘
```

`zeMemOpenIpcHandle()` 要求接收端的 context 能够访问 IPC handle 对应的物理设备。由于 Rank 2 的 Level Zero 运行时根本**看不到** GPU 0（被 `ZE_AFFINITY_MASK` 过滤掉了），这个调用无法完成，导致 hang。

#### 17.2.4 为什么模式 A 和模式 C 不 hang

| 模式 | 设备可见性 | contexts[0] 关联的物理设备 | IPC 能否跨 rank 打开 |
|------|-----------|--------------------------|---------------------|
| A（不设 mask） | 全部 4 GPU | GPU 0,1,2,3 | ✅ 所有 rank 的 context 都能访问所有 GPU |
| B（分组 mask） | Rank 0,1→GPU 0,1；Rank 2,3→GPU 2,3 | 不同组不同 | ❌ 跨组 context 无法访问对方设备 |
| C（统一 mask） | 全部 4 GPU | GPU 0,1,2,3 | ✅ 等同于不设 mask |

### 17.3 oneCCL 源码中的相关警告

oneCCL 的拓扑管理器已经意识到 narrow affinity mask 可能导致问题，但仅输出警告而非报错：

```cpp
// src/topology/topo_manager.cpp, 第 316-342 行
char* affinity_mask_env = getenv("ZE_AFFINITY_MASK");

if (!is_sub_vector(node_dev_uuids, comm_dev_uuids)) {
    LOG_WARN("comm_dev_uuids is not sub-vector of node_dev_uuids"
             ", this may happen due to narrow device affinity mask (",
             ((affinity_mask_env) ? affinity_mask_env : "default"), ")");
}
```

代码注释标注了 `TODO: make these checks mandatory`（第 318 行），说明开发者知道这是潜在问题，但尚未将其变为强制检查。

### 17.4 IPC Handle 交换的三种模式

oneCCL 支持三种 IPC 交换模式，但它们都受到跨 affinity mask 问题的影响：

```cpp
// src/common/global/ze/ze_fd_manager.hpp, 第 29-36 行
enum class ipc_exchange_mode : int {
    sockets,  // Unix domain sockets 交换 IPC handle
    drmfd,    // DRM file descriptor 转换
    pidfd,    // pidfd_open + pidfd_getfd
    none
};
```

- **sockets 模式**：直接交换 `ze_ipc_mem_handle_t`，接收端调用 `zeMemOpenIpcHandle`，同样需要正确的 context
- **drmfd 模式**：转换为 DRM GEM handle，但最终仍通过 `zeMemOpenIpcHandle` 打开，context 问题不变
- **pidfd 模式**：通过 `pidfd_open` + `pidfd_getfd` 复制文件描述符，但 `zeMemOpenIpcHandle` 的 context 约束依然存在

### 17.5 对 vLLM DP>1 场景的影响

这个问题直接影响 vLLM 中使用 `DP>1`（数据并行）+ TP（张量并行）的场景：

```
vLLM DP=2, TP=2 配置（4 GPU）：
┌─────────────────────────────────────────┐
│ DP Group 0 (ZE_AFFINITY_MASK="0,1")    │
│   Rank 0 → GPU 0  ─┐                   │
│   Rank 1 → GPU 1  ─┘ TP 组内通信 ✅    │
├─────────────────────────────────────────┤
│ DP Group 1 (ZE_AFFINITY_MASK="2,3")    │
│   Rank 2 → GPU 2  ─┐                   │
│   Rank 3 → GPU 3  ─┘ TP 组内通信 ✅    │
├─────────────────────────────────────────┤
│ DP 跨组通信（Rank 0 ↔ Rank 2）         │
│   ❌ IPC handle 无法跨 affinity mask    │
│   → all_reduce hang                     │
└─────────────────────────────────────────┘
```

TP 组内通信（同一 `ZE_AFFINITY_MASK` 内的 rank）正常工作，因为它们共享相同的设备可见性。但 DP 跨组通信需要跨越不同的 `ZE_AFFINITY_MASK` 边界，触发上述 IPC 问题。

### 17.6 解决方案

根因在于：oneCCL 的 GPU IPC 路径（`zeMemOpenIpcHandle`）要求接收方进程的 Level Zero context 能访问发送方 IPC handle 对应的物理 GPU，而不同的 `ZE_AFFINITY_MASK` 破坏了这一前提。以下从**即时可用**到**长期改进**列出解决方案：

#### 方案 1（推荐）：应用层 — 不使用 ZE_AFFINITY_MASK 做进程隔离

**原理**：不设置 `ZE_AFFINITY_MASK`，让所有进程都能看到全部 GPU，然后通过 `torch.xpu.set_device(local_rank)` 在应用层控制每个 rank 绑定哪张 GPU。

**vLLM 的改法**：在 vLLM 的 worker 启动逻辑中，不要为不同的 DP group 设置不同的 `ZE_AFFINITY_MASK`，改用 `local_rank` 映射：

```python
# 修改前（vLLM 当前行为 — 导致 hang）:
# DP group 0: ZE_AFFINITY_MASK="0,1"  → rank 0,1 只看到 2 GPU
# DP group 1: ZE_AFFINITY_MASK="2,3"  → rank 2,3 只看到 2 GPU

# 修改后（推荐）:
# 不设置 ZE_AFFINITY_MASK，或统一设置 ZE_AFFINITY_MASK="0,1,2,3"
# 通过 local_rank 绑定设备：
#   rank 0 → torch.xpu.set_device(0)
#   rank 1 → torch.xpu.set_device(1)
#   rank 2 → torch.xpu.set_device(2)
#   rank 3 → torch.xpu.set_device(3)
```

**验证**：这正是测试脚本中模式 A（baseline）和模式 C（same_affinity）正常工作的原因。

**优点**：零代码改动（oneCCL 侧），仅需修改应用层的设备分配逻辑。
**缺点**：所有进程都能看到全部 GPU，内存隔离不如 `ZE_AFFINITY_MASK` 严格。

#### 方案 2：应用层 — 使用 ONEAPI_DEVICE_SELECTOR 替代 ZE_AFFINITY_MASK

**原理**：`ONEAPI_DEVICE_SELECTOR` 在 SYCL 运行时层过滤设备，但**不影响 Level Zero 的设备枚举**。oneCCL 内部直接使用 Level Zero API（`zeDeviceGet`），因此 `ONEAPI_DEVICE_SELECTOR` 不会影响 IPC handle 的 context 映射。

```bash
# 替代方案（如果应用层需要设备过滤）
export ONEAPI_DEVICE_SELECTOR="level_zero:0,1"  # 只对 SYCL 层生效
# oneCCL 内部的 zeDeviceGet 仍然能看到全部 GPU
```

**注意**：需要验证 oneCCL 是否在所有路径都通过 SYCL 还是直接调用 Level Zero。从源码看，`src/common/utils/sycl_utils.cpp` 第 31 行有 `ONEAPI_DEVICE_SELECTOR` 检查，但 IPC 路径直接使用 Level Zero API，因此此方案的有效性取决于具体驱动版本。

#### 方案 3：oneCCL 配置 — 禁用 ZE IPC 路径强制走 host 内存

**原理**：通过环境变量关闭 oneCCL 的 Level Zero GPU IPC 优化，强制使用 host 内存作为中转，绕过 `zeMemOpenIpcHandle` 的 context 限制。

```bash
# 方法 A：完全禁用 Level Zero 加速（回退到纯 host 路径）
export CCL_ZE_ENABLE=0

# 方法 B：禁用 IPC handle 缓存（不解决根因，但可排除缓存相关问题）
export CCL_ZE_CACHE_OPEN_IPC_HANDLES=0

# 方法 C：切换 IPC 交换模式为 sockets（可能绕过部分 context 问题）
export CCL_ZE_IPC_EXCHANGE=sockets
```

对应源码位置：
- `CCL_ZE_ENABLE`：`src/common/env/env.cpp` 第 350 行，`ze_enable(1)` 默认开启
- `CCL_ZE_IPC_EXCHANGE`：`src/common/env/env.cpp` 第 355 行，默认 `pidfd` 模式
- `CCL_ZE_CACHE_OPEN_IPC_HANDLES`：`src/common/env/env.cpp` 第 330 行

**优点**：不需要修改任何代码，纯环境变量配置。
**缺点**：`CCL_ZE_ENABLE=0` 会禁用所有 GPU 直通优化，all_reduce 性能会**显著下降**（数据需经 GPU→Host→Host→GPU 两次拷贝）。

#### 方案 4：oneCCL 代码改进 — IPC handle 交换时检测并回退

**原理**：在 `ze_handle_exchange_entry.cpp` 的 `fill_payload` 阶段，增加对发送方和接收方 `ZE_AFFINITY_MASK` 的一致性检测。如果不一致，自动回退到 host 内存路径。

**改动点**（`src/sched/entry/ze/ze_handle_exchange_entry.cpp`）：

```cpp
// 在 common_fd_mode_exchange() 或 create_local_ipc_handles() 中增加检测：
void ze_handle_exchange_entry::validate_affinity_masks() {
    // 1. 收集所有 rank 的 ZE_AFFINITY_MASK
    char* local_mask = getenv("ZE_AFFINITY_MASK");
    std::string mask_str = local_mask ? local_mask : "";
    
    // 2. 通过 allgather 交换 mask 信息
    // 3. 如果存在不一致的 mask，设置标志禁用 IPC 路径
    // 4. 回退到 host staging buffer 路径
}
```

**改动点**（`src/sched/ze/ze_handle_manager.cpp` 第 302-318 行 `open_handle`）：

```cpp
void ipc_handle_manager::open_handle(ipc_handle_desc& info, void** ptr, bool to_cache) {
    // 增加：检查 remote context 是否能访问目标设备
    // 如果不能，回退到 host 内存中转
    if (!can_access_remote_device(info)) {
        // 通过 host staging buffer 复制数据
        use_host_staging_path(info, ptr);
        return;
    }
    // ... 原有 IPC 路径
}
```

**优点**：对应用层完全透明，自动检测并处理。
**缺点**：需要修改 oneCCL 核心代码，需要上游接受 patch。

#### 方案 5：Level Zero 驱动层 — 支持跨 affinity mask 的 IPC

**原理**：修改 Level Zero 驱动，使 `zeMemOpenIpcHandle` 能够在目标进程的 context 中映射任意物理 GPU 的内存，即使该 GPU 不在当前进程的 `ZE_AFFINITY_MASK` 中。

**现状**：这需要 Intel GPU 驱动团队的支持，是长期解决方案。

#### 方案对比总结

| 方案 | 改动位置 | 性能影响 | 复杂度 | 推荐度 |
|------|---------|---------|--------|--------|
| 1. 不设 ZE_AFFINITY_MASK | 应用层（vLLM） | 无 | ⭐ | ⭐⭐⭐⭐⭐ |
| 2. 用 ONEAPI_DEVICE_SELECTOR | 应用层 | 无 | ⭐⭐ | ⭐⭐⭐ |
| 3. CCL_ZE_ENABLE=0 | 环境变量 | 严重下降 | ⭐ | ⭐⭐（仅调试） |
| 4. oneCCL 自动回退 | oneCCL 代码 | 跨组通信下降 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 5. 驱动层修复 | Level Zero 驱动 | 无 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐（长期） |

**立即可行的最佳方案**是方案 1：修改 vLLM 的 worker 启动逻辑，不为不同 DP group 设置不同的 `ZE_AFFINITY_MASK`，而是统一让所有进程看到全部 GPU，通过 `set_device()` 控制绑定。

---

*本报告基于 oneCCL 2021.17.2 版本源代码分析完成。*
