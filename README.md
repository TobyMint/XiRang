# XiRang

面向智能体的内存管理系统设计与实现。

第三届研究生操作系统大赛 · 研究创新赛道 · 高校赛题。

## 赛题简介

随着基于大语言模型的智能体系统发展，其推理范式已由传统的单轮生成扩展为涵盖规划、执行与反思的长生命周期复杂过程。智能体推理在内存使用上呈现显著不同的特征：KV Cache 持续累积、上下文内容高度冗余、推理路径可能出现分支，以及工具调用带来大规模中间数据等问题。

本赛题要求基于开源技术栈，实现一个面向智能体推理过程的内存管理系统，在保证推理效果的前提下，通过对 KV Cache、上下文结构及显存分配机制的系统性优化，有效降低内存占用并提升整体推理效率。

## 优化方向

- **KV Cache 生命周期管理**：面向长生命周期推理的缓存管理策略，实现 KV 的复用、淘汰或分层存储。
- **分支推理内存共享**：针对智能体多路径决策过程，设计 KV Cache 的共享与 Copy-on-Write 机制。
- **Prompt 与上下文压缩**：对 system prompt、工具描述等内容去重与精简，消除冗余信息。
- **工具调用数据优化**：对工具调用产生的大规模中间数据进行结构化存储或按需加载。
- **分层内存与异构存储优化**：探索 GPU 显存、主存与外部存储间的分层内存体系，实现冷热分离与动态迁移。
- **异构 AI 加速硬件支持**：适配 CUDA、DTK、CANN 等异构计算平台及国产 AI 加速硬件。

## 技术栈

- 操作系统：openEuler / openKylin / OpenHarmony 等（至少一个国内主流开源操作系统）
- 推理框架：vLLM 或 llama.cpp（可在其基础上扩展）
- 开源大模型：Qwen、MiniCPM 等

## 目录结构

```
XiRang/
└── docs/           # 文档（赛题要求等）
```

## 相关资料

- [vLLM 文档](https://docs.vllm.ai/en/stable/index.html) · [vLLM 源码](https://github.com/vllm-project/vllm)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [MiniCPM3-4B](https://www.modelscope.cn/models/OpenBMB/MiniCPM3-4B)
- [LangChain](https://github.com/langchain-ai/langchain) · [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
