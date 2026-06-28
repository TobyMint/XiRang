# third_party — 外部依赖仓库

本目录存放 XiRang 依赖的**外部开源仓库**（vLLM、后续的 OpenClaw 等）。
每个外部项目占一个子目录，互不干扰，避免污染 XiRang 自身的代码结构。

## 为什么这样组织

- XiRang 自身的代码（`xirang/`、`benchmarks/`、`scripts/`、`tests/`、`docs/`）与外部依赖物理隔离；
- 后续接入 OpenClaw 等其它部分时，只需在此目录新增一个子目录，不改动 XiRang 主结构；
- 整个 `third_party/` 被 `.gitignore` 忽略，**不进入 XiRang 仓库**，保持仓库小而干净；
- 版本可复现性由 `scripts/setup_third_party.sh` 钉死（精确到 tag/commit）。

## 初始化

```bash
./scripts/setup_third_party.sh
```

会按固定版本 clone：

| 子目录 | 仓库 | 版本 |
|---|---|---|
| `vllm/` | https://github.com/vllm-project/vllm | `v0.10.2` |
| `openclaw/` | （后续接入） | TBD |

## 运行 vLLM

vLLM 含编译的 CUDA 算子，**不能直接 `python` 跑源码**。运行方式：

- 推荐（预编译 wheel）：`pip install vllm==0.10.2`，再 `./scripts/start_vllm.sh`；
- 如需魔改源码：`pip install -e third_party/vllm`（需 CUDA 工具链编译）。

源码 clone 在 `third_party/vllm/` 主要用于：阅读、对照版本、以及后期"接入 vLLM 内部"的魔改（见 `docs/design.md` 路线）。

## 模型

模型不放在本目录，统一在 `/data/models/`，当前使用 `/data/models/Qwen3-4B`。
