# 专利说明书文本提取工具

将发明专利说明书 PDF（单个或 ZIP 压缩包）在线转换为高质量 JSONL 格式数据，用于本地大模型预训练。

## 功能

- 📄 **上传单个 PDF** → 即时提取 → 下载 JSONL
- 📦 **上传 ZIP 压缩包**（内含多个 PDF）→ 后台批量并行处理 → 下载合并的 JSONL
- 🧹 **智能清洗**：自动去除专利页脚（CN号、页码、说明书章节名）、扉页元数据（申请号/日、申请人、发明人等）、代理人信息、通用法律套话
- 📐 **语义分段**：按专利章节拆分 + 招投标相关度过滤 + 120~2000字段长控制
- ⚡ **自适应并行加速**：自动检测 CPU 核数、内存、GPU，动态分配最优并行 Worker 数
- 🐳 **Docker 一键部署**

## 输出格式

每行一条 JSON，适用于大模型预训练：

```json
{"text": "本发明涉及一种电子招投标管理系统，其包括招投标基本信息管理模块、竞价模块和候选管理模块。", "category": "专利文献"}
{"text": "1. 一种招投标方法，其特征在于，包括以下步骤：招标人通过标书发布模块发布招标信息；投标人通过客户端查看招标信息并提交投标文件。", "category": "专利文献"}
```

## 快速开始

### 方式一：直接运行（本地开发）

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn app.main:app --reload --port 8020

# 打开浏览器
open http://localhost:8020
```

### 方式二：Docker 部署

```bash
# 构建并启动
docker compose up -d

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

## 使用说明

1. 打开浏览器访问 `http://localhost:8020`
2. 拖拽或点击选择 PDF / ZIP 文件
3. 点击「开始提取」
4. 等待处理完成，下载 JSONL 文件
5. 文件会保存一份在 `outputs/` 目录下

## 自适应并行加速

系统启动时自动检测宿主机器资源，选择最优并行策略：

| 环境 | CPU | 内存 | GPU | 并行 Worker | 策略 |
|------|-----|------|-----|-------------|------|
| 低配机器 | ≤2 核 | ≤4 GB | 无 | 1 | 串行处理 |
| 中配服务器 | 8 核 | 16 GB | 无 | 4~6 | 多进程并行 |
| 高配服务器 | 16+ 核 | 48+ GB | 无 | 8 | 多进程并行 |
| GPU 服务器 | 任意 | 任意 | NVIDIA CUDA | 1~2 | GPU 批量推理 |

- **CPU 模式**：通过 `ProcessPoolExecutor` 启动多个 Worker 进程，每个进程拥有独立的 PaddleOCR 实例，并行处理不同 PDF
- **GPU 模式**：检测到 CUDA 时自动启用 GPU 推理，每次 `predict()` 批量处理多页图像
- **单 PDF 请求**：OCR 在后台进程执行，不阻塞 FastAPI 事件循环

## 数据清洗说明

提取的文本经过多阶段清洗，确保预训练数据质量：

1. **OCR 行合并**：修复 `\n` 导致的中文句子截断
2. **编号附着**：`[0055]\n内容` → `[0055] 内容`
3. **页脚移除**：去除 `CN 102955986 A + 机明节 + 1/3页` 等专利页脚噪声
4. **扉页元数据移除**：申请公布号/日、申请号/日、申请人、发明人、代理人、专利权人、Int.Cl. 分类号等
5. **公司/人名行移除**：`中油物采信息技术有限公司`、`孙玉华 郭建光` 等
6. **通用套话过滤**：`以上所述仅为本发明的较佳实施例` 等专利法律套话
7. **短片段丢弃**：低于 120 字符的无效样本直接丢弃
8. **招投标相关度过滤**：不含招投标关键词的片段自动过滤

## 技术栈

- **后端**: Python FastAPI
- **OCR**: PaddleOCR v3.7+（PP-OCRv6，中文）
- **PDF 处理**: PyMuPDF (fitz)
- **并行**: `concurrent.futures.ProcessPoolExecutor`
- **资源检测**: psutil + sysctl + /proc/meminfo
- **部署**: Docker + docker-compose

## 项目结构

```
├── app/
│   ├── main.py                 # FastAPI 入口
│   ├── config.py               # 配置 + 资源检测 + 并行规划
│   ├── routers/extraction.py   # API 端点 + 并行 Worker
│   ├── services/
│   │   ├── pdf_extractor.py    # PDF→PNG 图像 (PyMuPDF)
│   │   ├── ocr_engine.py       # PaddleOCR 封装（支持独立实例）
│   │   └── text_processor.py   # 多阶段清洗 + 分段 + 质量过滤
│   ├── schemas/models.py       # 数据模型
│   ├── templates/index.html    # 前端页面
│   └── static/style.css        # 样式
├── uploads/                    # 临时上传（gitignored）
├── outputs/                    # 输出 JSONL（gitignored）
├── tests/
│   └── test_text_processor.py  # 31 个单元测试
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
