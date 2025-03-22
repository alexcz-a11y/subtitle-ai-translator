#!/bin/bash
# 批量翻译脚本示例

# 确保工作目录正确
cd "$(dirname "$0")/.."

# 激活虚拟环境（如果使用）
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 处理目录中的所有.srt文件
INPUT_DIR="./examples"
OUTPUT_DIR="./translations"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 遍历所有.srt文件
for file in "$INPUT_DIR"/*.srt; do
    if [ -f "$file" ]; then
        # 获取文件名（不包含路径和扩展名）
        filename=$(basename "$file" .srt)
        
        echo "翻译文件: $filename.srt"
        
        # 运行翻译命令
        python subtitle_translator.py \
            --input "$file" \
            --output-dir "$OUTPUT_DIR" \
            --lang zh-CN \
            --workers 4 \
            --chunk-size 5 \
            --vector-chunk-size 3 \
            --similar-chunks 2 \
            --align
    fi
done

echo "批量翻译完成！所有翻译结果保存在 $OUTPUT_DIR 目录中。" 