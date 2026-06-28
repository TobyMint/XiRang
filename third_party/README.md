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

| 子目录 | 仓库 | 版本 | 技术栈 |
|---|---|---|---|
| `vllm/` | https://github.com/vllm-project/vllm | `v0.10.2` | Python (CUDA) |
| `openclaw/` | https://github.com/openclaw/openclaw | `v2026.6.10` | TypeScript (Node/pnpm) |

## 运行 vLLM

vLLM 含编译的 CUDA 算子，**不能直接 `python` 跑源码**。运行方式：

- 编译安装（editable，已封装）：`./scripts/build_vllm.sh`（在 `xirang` conda 环境编译 vLLM 0.10.2）；
- 启动：`./scripts/start_vllm.sh`（默认 `/data/models/Qwen3-4B`）。

源码 clone 在 `third_party/vllm/` 用于：editable 安装、对照版本、以及后期"接入 vLLM 内部"的魔改（见 `docs/design.md` 路线）。

## 运行 OpenClaw

OpenClaw 是 TypeScript/pnpm monorepo，**不是 Python**。运行前置：

- Node ≥ 22.19.0（本机当前 v20，需升级，建议用 nvm 或 `conda install -n xirang nodejs=22`）
- pnpm 11.2.2（`corepack enable && corepack prepare pnpm@11.2.2 --activate`）
- 构建：`cd third_party/openclaw && pnpm install && pnpm build`

### 与 vLLM / XiRang 的对接

OpenClaw **原生支持 vLLM provider**（`openai-completions` API，见 `third_party/openclaw/docs/providers/vllm.md`）。
benchmark 的两个模式只差 `baseUrl`：

| 模式 | OpenClaw 配置 `models.providers.vllm.baseUrl` |
|---|---|
| baseline (naive) | `http://127.0.0.1:8001/v1`（直连 vLLM） |
| xirang | `http://127.0.0.1:8010/v1`（经 XiRang proxy） |

OpenClaw 配置示例（`vllm` provider 指向 XiRang proxy，Qwen3 thinking）：

```json5
{
  models: {
    providers: {
      vllm: {
        baseUrl: "http://127.0.0.1:8010/v1",
        apiKey: "${VLLM_API_KEY}",   // 任意非空值即可（本地无鉴权）
        api: "openai-completions",
        models: [
          {
            id: "/data/models/Qwen3-4B",   // 与 vLLM 启动 --model 一致
            name: "Qwen3-4B (via XiRang)",
            reasoning: true,
            compat: { thinkingFormat: "qwen-chat-template" },
            contextWindow: 8192,
            maxTokens: 2048,
          },
        ],
      },
    },
  },
}
```

## 模型

模型不放在本目录，统一在 `/data/models/`，当前使用 `/data/models/Qwen3-4B`。
