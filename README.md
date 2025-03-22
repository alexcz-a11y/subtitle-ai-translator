# 字幕智能翻译工具 (Subtitle AI Translator)

这是一个基于AI的字幕翻译工具，能够分析字幕内容、上下文、电影类型，并提供更为准确的翻译结果。利用向量化技术和场景感知，本工具可以生成高质量的字幕翻译，保留原作风格和情感表达。

## 核心特性

- **智能内容分析**：自动推断电影类型、情感基调和提取关键术语
- **向量化场景识别**：使用嵌入向量技术识别场景转换点，保持翻译连贯性
- **相似场景关联**：识别相似场景并提供上下文关联，确保术语和风格一致性
- **并行多线程处理**：高效处理大型字幕文件
- **专业字幕校准**：自动调整翻译字幕格式，确保与源字幕精确对应
- **灵活的扩展性**：支持自定义任何OpenAI兼容的模型和参数

## 技术亮点

- 使用语义向量嵌入对字幕内容进行深度理解
- 基于余弦相似度的场景边界检测算法
- 相似场景识别与上下文关联增强翻译一致性
- 智能分块和合并策略优化API调用效率
- 多线程并行处理提高翻译速度
- 自适应的字幕排版优化

## 安装

1. 克隆此仓库
```bash
git clone https://github.com/yourusername/subtitle-ai-translator.git
cd subtitle-ai-translator
```

2. 创建并激活虚拟环境(可选但推荐)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 创建配置文件
```bash
cp .env.example .env
```

5. 编辑`.env`文件，设置API密钥和可选模型参数
```
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1  # 可选，默认为OpenAI官方API
EMBEDDING_MODEL=text-embedding-3-large     # 可选，默认为text-embedding-3-large
TRANSLATION_MODEL=gpt-4o-2024-11-20        # 可选，默认为gpt-4o-2024-11-20
ANALYSIS_MODEL=gpt-4o-2024-11-20           # 可选，默认为gpt-4o-2024-11-20
```

## 快速开始

### 基本翻译

```bash
python subtitle_translator.py --input your_subtitle.srt --lang zh-CN
```

这将翻译字幕文件到中文，并将结果保存在`translations/your_subtitle/your_subtitle.zh-CN.srt`。

### 高性能配置

```bash
python subtitle_translator.py --input your_subtitle.srt --workers 10 --vectorize-workers 10 --batch-delay 0.1 --chunk-size 2 --vector-chunk-size 2 --align
```

此配置适用于大型字幕文件，使用更多线程加速处理，同时启用字幕校准功能。

### 自定义模型

```bash
python subtitle_translator.py --input your_subtitle.srt --translation-model gpt-4-turbo --embedding-model text-embedding-3-large
```

使用特定的模型进行翻译和向量化，适用于需要更高质量输出的场景。

## 详细参数说明

```bash
python subtitle_translator.py [参数]
```

### 必选参数

- `--input`, `-i` : 输入字幕文件路径

### 可选参数

#### 输出控制
- `--output`, `-o` : 输出翻译文件路径（默认自动生成）
- `--output-dir` : 翻译结果输出目录（默认为`translations/`）
- `--lang`, `-l` : 目标语言代码（默认为`zh-CN`）
- `--save-analysis`, `-s` : 保存内容分析结果到JSON文件

#### 性能调优
- `--workers`, `-w` : 翻译并发工作线程数量（默认为4）
- `--vectorize-workers`, `-vw` : 向量化并发工作线程数量（默认为4）
- `--batch-delay` : 批次间延迟(秒)（默认为0.2秒）
- `--chunk-size`, `-c` : 翻译块大小（默认为10）
- `--vector-chunk-size`, `-v` : 向量化块大小（默认为3）
- `--force-analysis` : 强制重新分析内容，即使有缓存

#### 缓存和相似度
- `--use-cache`, `-u` : 使用缓存的向量数据
- `--cache-dir`, `-d` : 缓存目录（默认为`.cache`）
- `--similar-chunks`, `-k` : 相似块数量（默认为2）

#### 模型选择
- `--embedding-model` : 指定嵌入模型名称
- `--translation-model` : 指定翻译模型名称
- `--analysis-model` : 指定分析模型名称

#### 字幕校准
- `--align` : 启用字幕校准，优化翻译字幕的排版和格式

## 项目结构

```
subtitle-ai-translator/
├── subtitle_translator.py     # 主程序
├── requirements.txt           # 依赖列表
├── .env.example               # 环境变量示例
├── README.md                  # 项目说明
├── utils/                     # 工具模块
│   ├── ai_service.py          # AI服务接口
│   ├── subtitle_parser.py     # 字幕解析器
│   ├── vector_tools.py        # 向量处理工具
│   └── subtitle_alignment.py  # 字幕校准工具
├── translations/              # 翻译结果目录
└── .cache/                    # 缓存目录
```

## 高级功能

### 场景一致性增强

本工具实现了智能场景边界检测和相似场景关联功能，确保翻译在场景转换时保持连贯，并在相似场景中保持术语和风格一致性。

```bash
# 激活相似场景分析的示例命令
python subtitle_translator.py --input your_subtitle.srt --similar-chunks 3 --align
```

### 批量处理多个字幕

可以创建一个简单的批处理脚本，处理多个字幕文件：

```bash
# batch_translate.sh 示例
#!/bin/bash
for file in subtitles/*.srt; do
  python subtitle_translator.py --input "$file" --workers 5 --align
done
```

### 翻译质量优化建议

1. **为专业术语创建术语库**：系统会自动从内容分析中提取关键术语，保持一致性。
2. **调整模型参数**：对于专业内容，建议使用GPT-4系列模型。
3. **优化块大小**：对于对话密集型内容，可使用较小的块大小（2-3）；对于描述密集型内容，可使用较大的块大小（5-10）。
4. **启用字幕校准**：几乎所有场景都建议使用`--align`选项，确保字幕格式专业。

## 贡献指南

欢迎对此项目做出贡献！请遵循以下步骤：

1. Fork此仓库
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送到分支：`git push origin feature/amazing-feature`
5. 提交Pull Request

## 许可证

本项目采用MIT许可证 - 详情请查看[LICENSE](LICENSE)文件

## 联系方式

如有问题或建议，请通过以下方式联系我们：

- 提交GitHub Issues
- 电子邮件：[your-email@example.com](mailto:your-email@example.com)

---

**注意**：使用此工具时，请确保遵守OpenAI的使用政策和相关服务条款。 