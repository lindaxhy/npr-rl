# NPR-RL

使用 **GRPO（Group Relative Policy Optimization）** 对谜题-答案类任务进行强化学习微调（RLVR）。基于 Qwen2.5 与 LoRA，支持按难度采样的实验选项。

## 功能

- **GRPO 训练**：组内相对优势、KL 约束、参考模型
- **LoRA + 4bit**：显存友好，可训 1.5B 级模型
- **奖励**：从模型输出中抽取答案，与标准答案规范化后 0/1 判对
- **采样模式**：`uniform`（均匀）或 `difficulty`（按通过率 EMA 的 p*(1-p) 加权）

## 环境

- Python 3.8+
- CUDA（推荐）

## 安装

```bash
pip install -r requirements.txt
```

## 数据格式

`data.json` 为 JSON 数组，每项包含：

- `challenge`：题目/谜题文本
- `answer`：标准答案（用于计算 0/1 reward）

示例：

```json
[
  {"challenge": "2+2=?", "answer": "4"},
  {"challenge": "What is the capital of France?", "answer": "Paris"}
]
```

## 运行

```bash
python train_grpo.py --data_path data.json --output_dir ./grpo_model
```

### 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model_name` | Qwen/Qwen2.5-1.5B-Instruct | 基座模型 |
| `--data_path` | data.json | 训练数据路径 |
| `--output_dir` | ./grpo_model | 保存目录 |
| `--group_size` | 8 | GRPO 每组采样数 |
| `--lr` | 1e-6 | 学习率 |
| `--kl_beta` | 0.02 | KL 惩罚系数 |
| `--max_new_tokens` | 128 | 每步最大生成长度 |
| `--max_steps` | 1000 | 训练步数 |
| `--sampling_mode` | uniform | 采样方式：`uniform` / `difficulty` |
| `--difficulty_alpha` | 1.0 | difficulty 模式下的权重指数 |
| `--ema_momentum` | 0.1 | 每题通过率 EMA 动量 |

### 按难度采样（difficulty）

当 `--sampling_mode difficulty` 时，会维护每题通过率的 EMA，并按 `p*(1-p)` 加权采样，使中等难度题目被更多选中。

## 输出

- 训练后的 LoRA 权重保存在 `--output_dir`
- 控制台每 `--log_every` 步打印 loss、reward、effective_ratio、avg_pass_ema 等

## License

MIT
