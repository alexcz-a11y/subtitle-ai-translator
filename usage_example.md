# 字幕翻译工具使用示例

## 准备工作

1. 安装依赖：
   ```bash
   # 创建虚拟环境
   python -m venv venv
   
   # 激活虚拟环境
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   
   # 安装依赖
   pip install -r requirements.txt
   ```

2. 配置API：
   复制 `.env.example` 到 `.env` 并编辑该文件，填入您的API密钥和自定义API端点：
   ```bash
   cp .env.example .env
   # 然后编辑 .env 文件
   ```

## 基本用法

### 简单翻译

将英文字幕翻译为中文：

```bash
python subtitle_translator.py --input examples/example.srt
```

### 指定输出文件

```bash
python subtitle_translator.py --input examples/example.srt --output translated_subtitle.srt
```

### 指定目标语言

```bash
python subtitle_translator.py --input examples/example.srt --lang ja
```

### 保存内容分析结果

```bash
python subtitle_translator.py --input examples/example.srt --save-analysis
```

## 高级用法

### 调整处理块大小

对于长字幕，可以调整每次处理的字幕数量：

```bash
python subtitle_translator.py --input examples/example.srt --chunk-size 20
```

### 使用向量化功能

利用向量化提高翻译质量（新增功能）：

```bash
# 设置向量化块的大小
python subtitle_translator.py --input examples/example.srt --vector-chunk-size 5

# 使用缓存的向量数据（加速后续翻译）
python subtitle_translator.py --input examples/example.srt --use-cache

# 指定缓存目录
python subtitle_translator.py --input examples/example.srt --cache-dir ".vectors_cache"

# 设置用于增强上下文的相似块数量
python subtitle_translator.py --input examples/example.srt --similar-chunks 3
```

### 完整示例（使用所有高级功能）

```bash
python subtitle_translator.py \
  --input examples/example.srt \
  --output examples/example.zh-CN.srt \
  --lang zh-CN \
  --chunk-size 15 \
  --vector-chunk-size 5 \
  --similar-chunks 3 \
  --use-cache \
  --save-analysis
```

## 自定义API端点

在 `.env` 文件中，可以设置自定义的API端点：

```
OPENAI_API_BASE=https://your-custom-openai-api-endpoint.com/v1
```

## 自定义模型

在 `.env` 文件中，可以更改使用的模型：

```
EMBEDDING_MODEL=text-embedding-3-large
TRANSLATION_MODEL=gpt-4o-2024-11-20
ANALYSIS_MODEL=gpt-4o-2024-11-20
```

## 向量化技术说明

本工具使用了先进的向量化技术来提高翻译质量：

1. **全文向量化**：将整个字幕文本转换为语义向量
2. **分块向量化**：将字幕分成小块并分别向量化
3. **术语向量化**：为专业术语生成向量表示
4. **上下文匹配**：翻译时查找语义相似的上下文
5. **术语匹配**：根据向量相似度识别相关专业术语

这些技术相结合，能够：

- 更准确地理解字幕内容
- 保持上下文一致性
- 更准确地翻译专业术语
- 提高翻译的专业性和准确性 