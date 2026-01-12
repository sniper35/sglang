# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

SGLang is a monorepo with three main components:
- **python/** - Main SGLang Python runtime for LLM/VLM serving
- **sgl-kernel/** - CUDA kernel library for optimized inference
- **sgl-model-gateway/** - High-performance Rust routing gateway

## Build Commands

### Python Package (sglang)
```bash
cd python
pip install -e .                    # Development install
pip install -e ".[dev]"             # With test dependencies
```

### sgl-kernel (CUDA kernels)
```bash
cd sgl-kernel
make build                          # Build wheel (uses all cores)
make build MAX_JOBS=2               # Limit parallelism
make test                           # Run kernel tests
make format                         # Format C++/Python code
```

### sgl-model-gateway (Rust)
```bash
cd sgl-model-gateway
cargo build --release               # Build Rust binary
cargo test                          # Run Rust tests
cargo clippy --all-targets --all-features -- -D warnings  # Lint

# Python bindings
cd bindings/python
maturin develop                     # Fast dev mode
maturin build --release --features vendored-openssl  # Production wheel
```

## Testing

### Python Tests
Tests use both unittest and pytest. Run individual test files directly:
```bash
cd test/srt
python3 test_srt_endpoint.py                              # Run file
python3 test_srt_endpoint.py TestSRTEndpoint.test_simple_decode  # Single test
python3 run_suite.py --suite per-commit                   # Run suite
```

For pytest tests:
```bash
pytest test/ -v
pytest sgl-model-gateway/e2e_test/ -v
```

### Adding Tests to CI
- Place tests in `test/srt/` or `test/lang/`
- Reference in `run_suite.py` for CI pickup (e.g., `per-commit-1-gpu` suite)
- Include `unittest.main()` or `sys.exit(pytest.main([__file__]))`
- Use CI registration markers:
```python
from sglang.test.ci.ci_register import register_cuda_ci
register_cuda_ci(est_time=80, suite="stage-a-test-1")
```

## Code Quality

Pre-commit hooks handle formatting:
```bash
pre-commit run --all-files
```

Tools: isort, black, ruff (F401/F821 only), clang-format (C++/CUDA), codespell

**Exclusions**: Files matching `python/sglang/srt/grpc/*_pb2*.py` are auto-generated protobuf files.

## Running Servers

```bash
# HTTP server
python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B

# Router/gateway
python3 -m sglang_router.launch_router --worker-urls http://localhost:8000

# CLI
sglang --help
```

## Architecture Overview

### SRT (SGLang Runtime) - python/sglang/srt/
The serving runtime with:
- **entrypoints/**: HTTP server (`http_server.py`), gRPC server (`grpc_server.py`), engine interface
- **managers/**: Scheduler, token management, detokenizer
- **layers/**: Attention mechanisms (RadixAttention), MoE, quantization
- **models/**: Model implementations by architecture
- **mem_cache/**: RadixAttention prefix caching with C++ radix tree
- **speculative/**: Speculative decoding implementations
- **disaggregation/**: Prefill/decode separation

### Frontend Language - python/sglang/lang/
DSL for structured LLM programming: `sglang.gen()`, `sglang.function()`, chat templates

### Model Gateway - sgl-model-gateway/
Rust-based routing layer:
- Load balancing policies: random, round_robin, cache_aware, power_of_two
- Prefill/decode disaggregation routing
- Native Rust tokenizer and reasoning parser
- OpenAI-compatible API endpoints
- MCP client integration

### Kernel Library - sgl-kernel/
Custom CUDA kernels:
- Add new kernels in `csrc/`, expose in `include/sgl_kernel_ops.h`
- Create torch extension in `csrc/common_extension.cc`
- Expose Python interface in `python/sgl_kernel/`

## Key Configuration

Server configuration is in `python/sglang/srt/server_args.py` - a large dataclass with extensive options for parallelism, quantization, caching, and hardware backends.

## Hardware Support

NVIDIA GPUs (primary), AMD GPUs (ROCm), Intel CPUs, Google TPUs, Ascend NPUs. Different build configurations exist for each platform (see `pyproject_rocm.toml`, `pyproject_cpu.toml` in sgl-kernel).
