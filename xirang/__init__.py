"""XiRang: 显存感知的 Agent 内存管理中间层。

XiRang 作为一个 HTTP proxy 位于 Agent 框架与 vLLM (OpenAI-compatible) 之间，
拦截 /v1/chat/completions 请求，在转发前完成 prompt 重写、工具结果压缩、
thinking 删除等内存优化，并记录 GPU/延迟指标用于 benchmark。
"""

__version__ = "0.1.0"
