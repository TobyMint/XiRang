# openEuler 部署指南

赛题要求基于 openEuler / openKylin / OpenHarmony 等至少一个国内主流开源操作系统开发。以下以 openEuler 22.03 LTS 为例。

## 1. 系统准备

```bash
# openEuler 22.03 LTS，确认内核与 GPU 驱动
cat /etc/os-release
uname -r
nvidia-smi   # 需先装 NVIDIA 驱动 + CUDA
```

## 2. 安装 Python 与依赖

openEuler 自带 Python 3.9+。建议用 venv：

```bash
sudo dnf install -y python3 python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# 真实后端额外安装 vLLM（需 GPU + CUDA）:
# pip install vllm
```

## 3. 验证 GPU 监控

```bash
python -c "from xirang.metrics.gpu_monitor import GPUMonitor; m=GPUMonitor(); print(m.snapshot())"
# 无 GPU 时返回 0 值并自动降级，proxy 仍可运行（用于功能验证）。
```

## 4. 启动服务

```bash
# vLLM 后端
./scripts/start_vllm.sh Qwen/Qwen2.5-1.5B-Instruct

# XiRang proxy（另一终端）
./scripts/start_xirang.sh

# 健康检查
curl http://127.0.0.1:8000/health
```

## 5. 跑 benchmark

```bash
./scripts/run_all.sh benchmarks/workloads/long_tool_call.jsonl
```

结果见 `runs/compare/`。

## 6. 作为系统服务（可选）

`/etc/systemd/system/xirang.service`：

```ini
[Unit]
Description=XiRang proxy
After=network.target

[Service]
WorkingDirectory=/opt/XiRang
Environment=XIRANG_BACKEND=http://127.0.0.1:8001
ExecStart=/opt/XiRang/.venv/bin/python -m xirang.proxy.server
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now xirang
```

## 7. 国产化算力适配（路线）

v0 仅 CUDA（pynvml）。`gpu_monitor.py` 已隔离硬件访问，后续可增加 DTK/CANN 后端实现，对外接口不变。
