# SGLang Architecture Deep Dive

A comprehensive guide to the SGLang codebase structure, design, and implementation.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Component Architecture Diagram](#2-component-architecture-diagram)
3. [Python Runtime (SRT)](#3-python-runtime-srt)
4. [Frontend Language DSL](#4-frontend-language-dsl)
5. [CUDA Kernel Library (sgl-kernel)](#5-cuda-kernel-library-sgl-kernel)
6. [Model Gateway (Rust)](#6-model-gateway-rust)
7. [Data Flow Diagrams](#7-data-flow-diagrams)
8. [Key Files Reference](#8-key-files-reference)

---

## 1. Overview

SGLang is a high-performance LLM serving framework organized as a monorepo with four major components:

| Component | Language | Location | Purpose |
|-----------|----------|----------|---------|
| **SRT (Serving Runtime)** | Python | `python/sglang/srt/` | Core inference engine with scheduling, memory management, model execution |
| **Frontend Language** | Python | `python/sglang/lang/` | DSL for structured LLM programming |
| **sgl-kernel** | C++/CUDA | `sgl-kernel/` | Optimized CUDA kernels for attention, MoE, quantization |
| **Model Gateway** | Rust | `sgl-model-gateway/` | High-performance routing, load balancing, API gateway |

### Technology Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Client Applications                            │
│                    (HTTP REST, gRPC, Python SDK)                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      SGLang Model Gateway (Rust)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │Load Balancer│  │Circuit Break│  │  Tokenizer  │  │ Tool Parser │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      SGLang Runtime (Python)                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     HTTP/gRPC Server                             │   │
│  │  (FastAPI + uvicorn / tonic gRPC)                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │  Tokenizer  │  │  Scheduler  │  │Detokenizer  │  │   Engine    │   │
│  │   Manager   │  │   Process   │  │   Manager   │  │  Interface  │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Model Runner                                │   │
│  │  (TP/PP/DP Workers, CUDA Graphs, Attention Backends)            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      sgl-kernel (CUDA)                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │Flash Attn│ │   MoE    │ │Quantized │ │ RMSNorm  │ │Speculative│     │
│  │  + MLA   │ │ Routing  │ │  GEMMs   │ │  + RoPE  │ │ Decoding │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         GPU Hardware                                    │
│            (NVIDIA H100/A100, AMD MI300X, Intel, TPU)                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                                   SGLang Monorepo                                  │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                          python/sglang/                                      │  │
│  │  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐   │  │
│  │  │         lang/               │  │              srt/                    │   │  │
│  │  │  ┌─────────────────────┐   │  │  ┌────────────────────────────────┐  │   │  │
│  │  │  │ api.py (DSL funcs)  │   │  │  │ entrypoints/                   │  │   │  │
│  │  │  │ ir.py (IR nodes)    │   │  │  │   http_server.py (FastAPI)     │  │   │  │
│  │  │  │ interpreter.py      │   │  │  │   grpc_server.py (tonic)       │  │   │  │
│  │  │  │ tracer.py           │   │  │  │   engine.py (Python API)       │  │   │  │
│  │  │  │ chat_template.py    │   │  │  └────────────────────────────────┘  │   │  │
│  │  │  └─────────────────────┘   │  │  ┌────────────────────────────────┐  │   │  │
│  │  │  ┌─────────────────────┐   │  │  │ managers/                      │  │   │  │
│  │  │  │ backend/            │   │  │  │   scheduler.py (batching)      │  │   │  │
│  │  │  │   runtime_endpoint  │   │  │  │   tokenizer_manager.py         │  │   │  │
│  │  │  │   openai.py         │   │  │  │   detokenizer_manager.py       │  │   │  │
│  │  │  │   anthropic.py      │   │  │  │   schedule_batch.py            │  │   │  │
│  │  │  └─────────────────────┘   │  │  │   tp_worker.py                 │  │   │  │
│  │  └─────────────────────────────┘  │  └────────────────────────────────┘  │   │  │
│  │                                    │  ┌────────────────────────────────┐  │   │  │
│  │                                    │  │ mem_cache/                     │  │   │  │
│  │                                    │  │   radix_cache.py               │  │   │  │
│  │                                    │  │   memory_pool.py               │  │   │  │
│  │                                    │  └────────────────────────────────┘  │   │  │
│  │                                    │  ┌────────────────────────────────┐  │   │  │
│  │                                    │  │ model_executor/                │  │   │  │
│  │                                    │  │   model_runner.py              │  │   │  │
│  │                                    │  │   forward_batch_info.py        │  │   │  │
│  │                                    │  └────────────────────────────────┘  │   │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                    │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────────────┐ │
│  │        sgl-kernel/             │  │         sgl-model-gateway/              │ │
│  │  ┌───────────────────────────┐ │  │  ┌─────────────────────────────────────┐│ │
│  │  │ csrc/                     │ │  │  │ src/                                ││ │
│  │  │   attention/ (MLA, merge) │ │  │  │   core/ (worker, registry)         ││ │
│  │  │   moe/ (routing, expert)  │ │  │  │   policies/ (cache_aware, etc)     ││ │
│  │  │   gemm/ (FP8,INT8,INT4)   │ │  │  │   routers/ (grpc, http, openai)    ││ │
│  │  │   elementwise/ (norm,act) │ │  │  │   tokenizer/ (HF, tiktoken)        ││ │
│  │  │   speculative/            │ │  │  │   tool_parser/ (function calls)    ││ │
│  │  └───────────────────────────┘ │  │  │   reasoning_parser/                ││ │
│  │  ┌───────────────────────────┐ │  │  └─────────────────────────────────────┘│ │
│  │  │ python/sgl_kernel/        │ │  │  ┌─────────────────────────────────────┐│ │
│  │  │   attention.py            │ │  │  │ bindings/python/                   ││ │
│  │  │   moe.py                  │ │  │  │   sglang_router/                   ││ │
│  │  │   gemm.py                 │ │  │  │     launch_router.py               ││ │
│  │  │   elementwise.py          │ │  │  └─────────────────────────────────────┘│ │
│  │  └───────────────────────────┘ │  └─────────────────────────────────────────┘ │
│  └─────────────────────────────────┘                                              │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Python Runtime (SRT)

### 3.1 Process Architecture

The SRT uses a multi-process architecture for isolation and performance:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Main Process                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                  HTTP Server (FastAPI + uvicorn)                  │   │
│  │  /generate, /v1/chat/completions, /v1/embeddings, /health        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     TokenizerManager                              │   │
│  │  - Tokenize input text                                           │   │
│  │  - Manage request state (rid_to_state)                           │   │
│  │  - Stream output to clients                                      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │ ZMQ IPC
         ┌───────────────────┼───────────────────┐
         ▼                   │                   ▼
┌─────────────────────┐      │      ┌─────────────────────────────────────┐
│  Scheduler Process  │      │      │     DetokenizerManager Process      │
│  (GPU Worker)       │      │      │                                     │
│  ┌───────────────┐  │      │      │  - Convert token IDs to text        │
│  │   Scheduler   │  │◄─────┴──────│  - Incremental streaming output     │
│  │   - Batching  │  │             │  - Handle special tokens            │
│  │   - Prefill   │  │             └─────────────────────────────────────┘
│  │   - Decode    │  │
│  └───────────────┘  │
│  ┌───────────────┐  │
│  │  TpWorker     │  │
│  │  (TP rank 0)  │  │
│  └───────────────┘  │
│  ┌───────────────┐  │
│  │ ModelRunner   │  │
│  │ - Forward     │  │
│  │ - CUDA Graphs │  │
│  └───────────────┘  │
└─────────────────────┘
```

### 3.2 Scheduler Architecture

The Scheduler is the core of SRT, implemented via multiple mixins:

```python
class Scheduler(
    SchedulerOutputProcessorMixin,          # Output processing
    SchedulerUpdateWeightsMixin,            # Weight updates
    SchedulerProfilerMixin,                 # Profiling
    SchedulerMetricsMixin,                  # Metrics collection
    SchedulerDisaggregationDecodeMixin,     # PD disaggregation (decode)
    SchedulerDisaggregationPrefillMixin,    # PD disaggregation (prefill)
    SchedulerMultiplexMixin,                # PD multiplexing
    SchedulerRuntimeCheckerMixin,           # Runtime validation
    SchedulerPPMixin,                       # Pipeline parallelism
    SchedulerDPAttnMixin,                   # Data parallel attention
):
```

**Scheduling Flow:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Scheduler Main Loop                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. _get_next_batch() - Fetch tokenized requests from TokenizerManager   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. _add_reqs() - Add new requests to batch (priority scheduling)        │
│    - Check memory availability                                          │
│    - Allocate KV cache blocks                                           │
│    - Insert into RadixCache                                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. _schedule() - Create ScheduleBatch                                   │
│    - Separate prefill vs decode requests                                │
│    - Apply batch size constraints                                       │
│    - Prepare GPU tensors                                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. _step() - Execute forward pass                                       │
│    - ScheduleBatch → ModelWorkerBatch → ForwardBatch                   │
│    - Call model_runner.forward()                                        │
│    - Sample next tokens                                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. Process outputs - Send to DetokenizerManager via ZMQ                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Memory Management - RadixAttention

SGLang's key innovation is the RadixCache for prefix sharing:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RadixCache Structure                            │
│                                                                         │
│                              [ROOT]                                     │
│                                │                                        │
│                    ┌───────────┴───────────┐                           │
│                    ▼                       ▼                           │
│              ["System:"]            ["User:"]                          │
│                    │                       │                           │
│              ┌─────┴─────┐           ┌─────┴─────┐                     │
│              ▼           ▼           ▼           ▼                     │
│         ["You are"]  ["Be"]    ["Hello"]   ["What is"]                │
│              │                       │           │                     │
│              ▼                       ▼           ▼                     │
│         ["helpful"]            [KV cache]  ["2+2?"]                   │
│              │                                   │                     │
│              ▼                                   ▼                     │
│         [KV cache]                          [KV cache]                │
│                                                                         │
│  Legend:                                                               │
│    [text] = RadixKey (token sequence)                                  │
│    [KV cache] = Cached key-value tensors for that prefix               │
└─────────────────────────────────────────────────────────────────────────┘
```

**Cache Operations:**

| Operation | Description |
|-----------|-------------|
| `match(tokens)` | Find longest matching prefix, return cached KV |
| `allocate(tokens, value)` | Insert new KV cache entry |
| `evict(strategy)` | Free memory using LRU/LFU/FIFO strategy |
| `lock_ref(node)` | Prevent eviction during active request |

### 3.4 Forward Modes

```python
class ForwardMode(IntEnum):
    EXTEND = auto()          # Prefill: extend existing sequence
    DECODE = auto()          # Decode: generate one token
    MIXED = auto()           # Chunked prefill: EXTEND + DECODE together
    IDLE = auto()            # DP attention worker idle
    TARGET_VERIFY = auto()   # Speculative: verify draft tokens
    DRAFT_EXTEND = auto()    # Speculative: extend draft model
    PREBUILT = auto()        # Disaggregated: KV already built
    SPLIT_PREFILL = auto()   # PD mux: split prefill across workers
```

### 3.5 Key Classes & Files

| Class | File | Purpose |
|-------|------|---------|
| `Scheduler` | `managers/scheduler.py` | Main scheduling logic, 2998 lines |
| `TokenizerManager` | `managers/tokenizer_manager.py` | Request handling, 2318 lines |
| `DetokenizerManager` | `managers/detokenizer_manager.py` | Token→text, 433 lines |
| `ModelRunner` | `model_executor/model_runner.py` | Inference execution, 2000+ lines |
| `RadixCache` | `mem_cache/radix_cache.py` | Prefix cache |
| `ScheduleBatch` | `managers/schedule_batch.py` | Batch representation |
| `ForwardBatch` | `model_executor/forward_batch_info.py` | GPU tensor container |
| `ServerArgs` | `server_args.py` | Configuration, 5346 lines |

---

## 4. Frontend Language DSL

### 4.1 DSL API Overview

The SGLang DSL provides a structured way to program LLM interactions:

```python
import sglang as sgl

@sgl.function
def multi_turn_qa(s, questions):
    s += sgl.system("You are a helpful assistant.")
    for q in questions:
        s += sgl.user(q)
        s += sgl.assistant(sgl.gen("answer", max_tokens=256))
    return s

# Execute
result = multi_turn_qa.run(
    questions=["What is Python?", "How do I learn it?"],
    backend=sgl.Runtime(model="meta-llama/Llama-3.1-8B")
)
```

### 4.2 IR (Intermediate Representation)

All DSL operations compile to `SglExpr` nodes:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      SglExpr Hierarchy                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SglExpr (base)                                                         │
│    │                                                                    │
│    ├── SglConstantText      # Static strings                           │
│    ├── SglGen               # Generation call with sampling params     │
│    ├── SglSelect            # Multiple choice selection                │
│    ├── SglRoleBegin/End     # Chat message boundaries                  │
│    ├── SglImage/SglVideo    # Multimodal inputs                        │
│    ├── SglVariable          # Reference to previous output             │
│    ├── SglExprList          # Sequential composition                   │
│    ├── SglFork/GetForkItem  # Parallel execution branching            │
│    ├── SglVarScopeBegin/End # Variable capture scopes                  │
│    ├── SglCommitLazy        # KV-cache commit                          │
│    └── SglSeparateReasoning # Reasoning token separation               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         User Code                                       │
│  @sgl.function                                                          │
│  def my_program(s):                                                     │
│      s += sgl.user("Hello")                                             │
│      s += sgl.gen("response")                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  SglFunction.run(backend, args)                                         │
│  → Creates StreamExecutor with background worker thread                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ProgramState                                                           │
│  → Wraps StreamExecutor for user-facing interface                       │
│  → Provides context managers: system(), user(), assistant()             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
┌────────────────────────────────┐  ┌────────────────────────────────────┐
│  _execute_role_begin/end()     │  │  _execute_gen()                    │
│  → Format with chat_template   │  │  → backend.generate(sampling_params)│
│  → Update messages_ list       │  │  → Store result in variables{}     │
└────────────────────────────────┘  └────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Backend (RuntimeEndpoint / OpenAI / Anthropic)                         │
│  → HTTP POST to inference server                                        │
│  → Return generated text + metadata                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Backend Abstraction

| Backend | File | Description |
|---------|------|-------------|
| `RuntimeEndpoint` | `backend/runtime_endpoint.py` | SGLang SRT via HTTP |
| `Runtime` | `backend/runtime_endpoint.py` | Auto-launches SRT server |
| `OpenAI` | `backend/openai.py` | OpenAI API client |
| `Anthropic` | `backend/anthropic.py` | Claude API client |
| `VertexAI` | `backend/vertexai.py` | Google VertexAI |
| `LiteLLM` | `backend/litellm.py` | Multi-provider wrapper |

### 4.5 Key Files

| File | Lines | Key Contents |
|------|-------|--------------|
| `api.py` | 292 | `gen()`, `function()`, `select()`, role functions |
| `ir.py` | 643 | `SglExpr` hierarchy, `SglFunction`, `SglSamplingParams` |
| `interpreter.py` | 1061 | `StreamExecutor`, `ProgramState`, `run_program()` |
| `tracer.py` | 279 | `TracerProgramState`, prefix extraction |
| `chat_template.py` | 668 | `ChatTemplate`, template registry |
| `choices.py` | 164 | Choice selection algorithms |

---

## 5. CUDA Kernel Library (sgl-kernel)

### 5.1 Kernel Categories

The library contains **111 CUDA kernel files** organized into major categories:

| Category | Files | Key Kernels | Python Module |
|----------|-------|-------------|---------------|
| **Attention** | 6 | MLA decode, state merge, cascade | `attention.py` |
| **Flash Attention** | 20+ | FA3, FA4, sparse variants | `flash_attn.py`, `flash_mla.py` |
| **MoE** | 16 | Routing, expert selection, grouping | `moe.py`, `cutlass_moe.py` |
| **GEMM/Quant** | 23 | FP8, INT8, FP4, INT4, AWQ, GPTQ | `gemm.py` |
| **Elementwise** | 8 | Norm, RoPE, activation, cast | `elementwise.py` |
| **Expert Spec** | 4 | FP8, MX-FP8 for SM100+ | `expert_specialization.py` |
| **Speculative** | 4 | Tree sampling, drafting | `speculative.py` |
| **Allreduce** | 4 | Custom, MSCCLPP | `allreduce.py` |
| **Other** | 26 | Mamba, memory, kvcache, grammar | Various |

### 5.2 Architecture Support Matrix

| GPU | SM | Flash Attn 3 | Flash Attn 4 | FlashMLA | NVFP4 |
|-----|----|----|-----|----------|-------|
| A100 | 80 | Yes | No | No | No |
| RTX 4090 | 89 | Yes | No | No | No |
| H100 | 90 | Yes | Yes* | Yes | Yes* |
| H200 | 90a | Yes | Yes* | Yes | Yes* |
| B100/B200 | 100+ | No | Yes | Yes | Yes |

*Requires CUDA 12.8+

### 5.3 Build System

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       CMake Build Flow                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Fetch dependencies (CUTLASS, FlashInfer, Flash-Attention)          │
│                              │                                          │
│                              ▼                                          │
│  2. Compile SM90 library → sgl_kernel/sm90/common_ops.so               │
│     (Fast math, default for H100)                                      │
│                              │                                          │
│                              ▼                                          │
│  3. Compile SM100 library → sgl_kernel/sm100/common_ops.so             │
│     (Precise math, for Blackwell)                                      │
│                              │                                          │
│                              ▼                                          │
│  4. Optional: FA3 → sgl_kernel/flash_ops.so (CUDA 12.4+)              │
│                              │                                          │
│                              ▼                                          │
│  5. Optional: FlashMLA → sgl_kernel/flashmla_ops.so (CUDA 12.8+)      │
│                              │                                          │
│                              ▼                                          │
│  6. Runtime: load_utils.py detects GPU, loads appropriate library      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.4 MoE Kernel Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MoE Forward Pass                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: [batch, hidden_dim]                                            │
│                              │                                          │
│                              ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ 1. topk_softmax() / topk_sigmoid()                            │     │
│  │    Select top-K experts per token                             │     │
│  │    Output: expert_ids[batch, K], weights[batch, K]            │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ 2. moe_align_block_size()                                     │     │
│  │    Align tokens to expert blocks (power-of-2)                 │     │
│  │    Output: sorted_expert_ids, sorted_weights                   │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ 3. prepare_moe_input()                                        │     │
│  │    Reorder input for grouped GEMM                             │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ 4. fp8_blockwise_scaled_grouped_mm() / moe_wna16_marlin_gemm()│     │
│  │    Quantized grouped expert matrix multiplications            │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ 5. moe_sum() / moe_sum_reduce()                               │     │
│  │    Aggregate expert outputs with weights                      │     │
│  │    Output: [batch, hidden_dim]                                │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.5 Key Python Bindings

```python
# Attention
from sgl_kernel import cutlass_mla_decode, merge_state

# Elementwise
from sgl_kernel import rmsnorm, fused_add_rmsnorm, rotary_embedding

# GEMM/Quantization
from sgl_kernel import fp8_scaled_mm, int8_scaled_mm, gptq_marlin_gemm

# MoE
from sgl_kernel import topk_softmax, moe_align_block_size, moe_sum

# Speculative
from sgl_kernel import tree_speculative_sampling_target_only
```

---

## 6. Model Gateway (Rust)

### 6.1 Module Structure

```
src/
├── core/                    # Worker abstraction & lifecycle
│   ├── worker.rs            # Worker trait (43+ methods)
│   ├── worker_registry.rs   # HashRing consistent hashing
│   ├── worker_manager.rs    # Worker lifecycle management
│   ├── circuit_breaker.rs   # Failure tracking & recovery
│   └── retry.rs             # Exponential backoff with jitter
│
├── policies/                # Load balancing strategies
│   ├── cache_aware.rs       # Approximate radix tree + LB
│   ├── round_robin.rs       # Sequential selection
│   ├── random.rs            # Uniform random
│   ├── power_of_two.rs      # 2-choice load-aware
│   └── prefix_hash.rs       # Token-based consistent hashing
│
├── routers/                 # Request routing
│   ├── grpc/                # gRPC pipeline (7-stage)
│   ├── http/                # HTTP request routing
│   ├── openai/              # OpenAI-compatible API
│   └── router_manager.rs    # Multi-router orchestration
│
├── tokenizer/               # Text ↔ tokens
├── tool_parser/             # Function call extraction
├── reasoning_parser/        # Reasoning block extraction
├── observability/           # Metrics & tracing
└── config/                  # Configuration types
```

### 6.2 Request Processing Pipeline

The gRPC router uses a 7-stage pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      gRPC Request Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  HTTP POST /v1/chat/completions                                        │
│  { model: "llama", messages: [...] }                                   │
│                              │                                          │
│                              ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ Stage 1: PreparationStage                                     │     │
│  │   - Tokenize request text                                     │     │
│  │   - Extract metadata                                          │     │
│  │   Output: tokens[], text, metadata                            │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ Stage 2: WorkerSelectionStage                                 │     │
│  │   - policy.select_worker(workers, info)                       │     │
│  │   - Apply health checks & circuit breaker                     │     │
│  │   Output: WorkerSelection (Single or PrefillDecode)           │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ Stage 3: ClientAcquisitionStage                               │     │
│  │   - Get gRPC client for selected worker(s)                    │     │
│  │   Output: ClientSelection with active gRPC clients            │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ Stage 4: RequestBuildingStage                                 │     │
│  │   - ChatCompletionRequest → gRPC GenerateRequest              │     │
│  │   Output: ProtoRequest (gRPC protobuf)                        │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ Stage 5: DispatchMetadataStage                                │     │
│  │   - Track request ID, timing, tracing context                 │     │
│  │   Output: DispatchMetadata                                    │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ Stage 6: RequestExecutionStage                                │     │
│  │   - client.generate(proto_request)                            │     │
│  │   - Handle streaming responses                                │     │
│  │   Output: ExecutionResult                                     │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ Stage 7: ResponseProcessingStage                              │     │
│  │   - Detokenize output                                         │     │
│  │   - Parse tool calls / reasoning blocks                       │     │
│  │   Output: ChatCompletionResponse                              │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│  HTTP 200 { choices: [...], usage: {...} }                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Load Balancing Policies

| Policy | Algorithm | Best For |
|--------|-----------|----------|
| `cache_aware` | Approximate radix tree matching + load balance | Default, cache locality |
| `round_robin` | Sequential cycling through workers | Fair distribution |
| `random` | Uniform random selection | Chaos testing |
| `power_of_two` | Pick 2 random, choose lower load | Load-aware routing |
| `prefix_hash` | Consistent hashing on token prefix | Session affinity |

**Cache-Aware Policy Detail:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  Cache-Aware Policy Decision Flow                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: request tokens, available workers                              │
│                              │                                          │
│                              ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ For each worker: compute prefix match using approximate       │     │
│  │ radix tree (per-model, per-worker trees)                      │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│         ┌────────────────────┴────────────────────┐                    │
│         │ match_rate > cache_threshold?           │                    │
│         └────────────────────┬────────────────────┘                    │
│                   Yes        │        No                               │
│                    │         │         │                               │
│                    ▼         │         ▼                               │
│  ┌────────────────────┐      │  ┌────────────────────────────────┐    │
│  │ Select worker with │      │  │ Check load imbalance:          │    │
│  │ highest match rate │      │  │ max_load - min_load > threshold │    │
│  └────────────────────┘      │  └────────────────────────────────┘    │
│                              │         │                               │
│                              │    Yes  │  No                           │
│                              │    │    │   │                           │
│                              │    ▼    │   ▼                           │
│                              │  ┌──────┴───────────────────────┐      │
│                              │  │ Shortest│ Smallest tree      │      │
│                              │  │ queue   │ (most cache space) │      │
│                              │  └──────────────────────────────┘      │
│                              │                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Key Rust Structures

```rust
// Worker abstraction
pub trait Worker: Send + Sync + Debug {
    fn url(&self) -> &str;
    fn is_healthy(&self) -> bool;
    fn load(&self) -> usize;
    fn circuit_breaker(&self) -> &CircuitBreaker;
    fn model_id(&self) -> &str;
}

// Load balancing interface
pub trait LoadBalancingPolicy: Send + Sync + Debug {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo)
        -> Option<usize>;
    fn on_request_complete(&self, worker_url: &str, success: bool);
    fn name(&self) -> &'static str;
}

// Request context through pipeline
pub struct RequestContext {
    pub input: RequestInput,
    pub state: ProcessingState,
}

// Configuration
pub enum RoutingMode {
    Regular { worker_urls: Vec<String> },
    PrefillDecode {
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
    },
    OpenAI { worker_urls: Vec<String> },
}
```

---

## 7. Data Flow Diagrams

### 7.1 Complete Request Flow (HTTP → GPU → Response)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              Complete Request Flow                               │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Client                                                                          │
│    │                                                                             │
│    │ POST /v1/chat/completions                                                  │
│    ▼                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    Model Gateway (Rust) [Optional]                         │ │
│  │  1. Load balance across workers                                            │ │
│  │  2. Circuit breaker / retry                                                │ │
│  │  3. Tokenize (native Rust)                                                 │ │
│  │  4. Route to selected worker                                               │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│    │                                                                             │
│    │ HTTP/gRPC                                                                  │
│    ▼                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                       HTTP Server (FastAPI)                                │ │
│  │  - Parse request JSON                                                      │ │
│  │  - Validate parameters                                                     │ │
│  │  - Create GenerateReqInput                                                 │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│    │                                                                             │
│    ▼                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                      TokenizerManager                                      │ │
│  │  1. Tokenize input text → token_ids                                       │ │
│  │  2. Create TokenizedGenerateReqInput                                       │ │
│  │  3. Assign request ID (rid)                                                │ │
│  │  4. Send to Scheduler via ZMQ PUSH                                         │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│    │                                                                             │
│    │ ZMQ IPC                                                                    │
│    ▼                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                         Scheduler                                          │ │
│  │  1. Receive via ZMQ PULL                                                   │ │
│  │  2. Create Req object                                                      │ │
│  │  3. Check RadixCache for prefix match                                      │ │
│  │  4. Allocate KV cache blocks                                               │ │
│  │  5. Add to pending_requests                                                │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│    │                                                                             │
│    ▼                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    Batch Scheduling                                        │ │
│  │  1. Select requests for batch (prefill + decode)                           │ │
│  │  2. Check memory constraints                                               │ │
│  │  3. Create ScheduleBatch                                                   │ │
│  │  4. Transform to ForwardBatch (GPU tensors)                                │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│    │                                                                             │
│    ▼                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                       ModelRunner                                          │ │
│  │  ┌──────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                    Forward Pass                                      │ │ │
│  │  │  1. Embedding lookup                                                 │ │ │
│  │  │  2. For each layer:                                                  │ │ │
│  │  │     - RMSNorm (sgl-kernel)                                           │ │ │
│  │  │     - Self-attention with RadixAttention (FlashAttention/MLA)        │ │ │
│  │  │     - RMSNorm                                                        │ │ │
│  │  │     - MLP / MoE (quantized GEMM kernels)                            │ │ │
│  │  │  3. Final norm + LM head                                             │ │ │
│  │  │  4. Sampling (top-k, top-p, temperature)                            │ │ │
│  │  └──────────────────────────────────────────────────────────────────────┘ │ │
│  │  Output: next_token_ids                                                   │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│    │                                                                             │
│    │ ZMQ IPC                                                                    │
│    ▼                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    DetokenizerManager                                      │ │
│  │  1. Receive token IDs via ZMQ                                              │ │
│  │  2. Incremental detokenization                                             │ │
│  │  3. Handle special tokens                                                  │ │
│  │  4. Send text chunks to TokenizerManager                                   │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│    │                                                                             │
│    │ ZMQ IPC                                                                    │
│    ▼                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                     TokenizerManager                                       │ │
│  │  1. Collect output chunks                                                  │ │
│  │  2. Stream to client (if streaming=True)                                   │ │
│  │  3. Format final response                                                  │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│    │                                                                             │
│    ▼                                                                             │
│  Client receives response                                                        │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Memory Layout

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            GPU Memory Layout                                     │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                         Model Weights                                      │ │
│  │  - Embedding: [vocab_size, hidden_dim]                                    │ │
│  │  - Layers: QKV, O, MLP weights (potentially quantized)                    │ │
│  │  - LM Head: [hidden_dim, vocab_size]                                      │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                         KV Cache (Radix Tree)                              │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │ │
│  │  │ Prefix Cache (Shared)                                               │  │ │
│  │  │  - System prompt KV blocks                                          │  │ │
│  │  │  - Common prefixes across requests                                  │  │ │
│  │  └─────────────────────────────────────────────────────────────────────┘  │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │ │
│  │  │ Per-Request KV Blocks                                               │  │ │
│  │  │  - Unique tokens per request                                        │  │ │
│  │  │  - Allocated dynamically                                            │  │ │
│  │  └─────────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                         Request Pool                                       │ │
│  │  - ReqToTokenPool: Maps request_id → token indices                        │ │
│  │  - TokenToKVPool: Maps token index → KV cache location                    │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                       Activation Buffers                                   │ │
│  │  - Attention intermediate: [batch, seq_len, heads, head_dim]              │ │
│  │  - MLP intermediate: [batch, seq_len, intermediate_dim]                   │ │
│  │  - Residual buffers                                                       │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                      Temporary Buffers                                     │ │
│  │  - Sampling workspace                                                      │ │
│  │  - FlashAttention workspace                                                │ │
│  │  - CUDA graph capture buffers                                             │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Key Files Reference

### 8.1 Python Runtime (SRT)

| File | Lines | Description |
|------|-------|-------------|
| `srt/entrypoints/http_server.py` | 1804 | FastAPI HTTP server |
| `srt/entrypoints/grpc_server.py` | 1039 | gRPC server |
| `srt/entrypoints/engine.py` | 1019 | Python API |
| `srt/managers/scheduler.py` | 2998 | Core scheduler |
| `srt/managers/tokenizer_manager.py` | 2318 | Request handling |
| `srt/managers/detokenizer_manager.py` | 433 | Token→text |
| `srt/managers/schedule_batch.py` | ~800 | Batch structures |
| `srt/model_executor/model_runner.py` | 2000+ | Model execution |
| `srt/model_executor/forward_batch_info.py` | ~500 | GPU tensors |
| `srt/mem_cache/radix_cache.py` | ~400 | Prefix cache |
| `srt/server_args.py` | 5346 | Configuration |

### 8.2 Frontend Language

| File | Lines | Description |
|------|-------|-------------|
| `lang/api.py` | 292 | DSL functions |
| `lang/ir.py` | 643 | IR nodes |
| `lang/interpreter.py` | 1061 | Execution engine |
| `lang/tracer.py` | 279 | Program analysis |
| `lang/chat_template.py` | 668 | Chat formatting |
| `lang/backend/runtime_endpoint.py` | 544 | SRT backend |
| `lang/backend/openai.py` | 475 | OpenAI backend |

### 8.3 sgl-kernel

| Directory | Files | Description |
|-----------|-------|-------------|
| `csrc/attention/` | 6 | Attention kernels |
| `csrc/moe/` | 16 | MoE kernels |
| `csrc/gemm/` | 23 | Quantized GEMMs |
| `csrc/elementwise/` | 8 | Norm, RoPE, etc |
| `python/sgl_kernel/` | 27 | Python bindings |

### 8.4 Model Gateway

| File | Description |
|------|-------------|
| `src/main.rs` | CLI & startup |
| `src/core/worker.rs` | Worker abstraction |
| `src/core/worker_registry.rs` | HashRing routing |
| `src/policies/cache_aware.rs` | Cache-aware policy |
| `src/routers/grpc/pipeline.rs` | Request pipeline |
| `src/routers/grpc/router.rs` | gRPC routing |
| `src/server.rs` | HTTP server setup |

---

## Summary

SGLang is a sophisticated, production-grade LLM serving system with:

- **High Performance**: RadixAttention prefix caching, custom CUDA kernels, zero-overhead scheduling
- **Flexibility**: Multiple serving modes (HTTP, gRPC), parallelism strategies (TP/PP/DP/EP), quantization options
- **Scalability**: Prefill/decode disaggregation, multi-node deployment, load-balanced routing
- **Usability**: DSL for structured LLM programming, OpenAI-compatible APIs

The architecture separates concerns cleanly:
- **Gateway (Rust)**: Routing, load balancing, tokenization
- **Runtime (Python)**: Scheduling, memory management, model execution
- **Kernels (CUDA)**: Optimized GPU operations
- **Frontend (Python DSL)**: User-facing programming interface
