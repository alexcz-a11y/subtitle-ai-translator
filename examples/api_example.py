#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
字幕翻译API调用示例脚本
"""

import os
import sys
import dotenv
from pathlib import Path

# 添加父级目录到路径以便导入模块
sys.path.insert(0, str(Path(__file__).parent.parent))

# 加载环境变量
dotenv.load_dotenv()

from utils.ai_service import AIService
from utils.subtitle_parser import SubtitleParser
from utils.vector_tools import VectorManager


def simple_translation_example():
    """简单翻译示例"""
    # 初始化AI服务
    ai_service = AIService()
    
    # 简单文本翻译
    text = "Hello, this is a sample text for translation."
    target_lang = "zh-CN"
    
    print(f"原文: {text}")
    result = ai_service.translate_text(text, target_lang)
    print(f"翻译: {result}")
    
    # 格式化上下文翻译
    context = "This is a sci-fi movie about advanced technology."
    text_with_context = f"Context: {context}\nText: {text}"
    
    result_with_context = ai_service.translate_text(text_with_context, target_lang)
    print(f"带上下文翻译: {result_with_context}")


def subtitle_translation_example():
    """字幕翻译示例"""
    # 初始化解析器和AI服务
    parser = SubtitleParser()
    ai_service = AIService()
    
    # 读取示例字幕
    example_path = Path(__file__).parent / "example.srt"
    subtitles = parser.parse_srt(example_path)
    
    # 提取文本内容
    texts = [sub.text for sub in subtitles]
    
    # 合并为批次（示例：每3条合并为一批）
    batch_size = 3
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    
    # 翻译每一批
    target_lang = "zh-CN"
    translated_batches = []
    
    for i, batch in enumerate(batches):
        print(f"翻译批次 {i+1}/{len(batches)}...")
        
        # 合并批次文本
        batch_text = "\n".join([f"{j+1}. {text}" for j, text in enumerate(batch)])
        
        # 翻译
        translated = ai_service.translate_text(batch_text, target_lang)
        translated_batches.append(translated)
        
        print(f"批次 {i+1} 翻译完成")
    
    # 输出结果
    print("\n翻译结果示例:")
    for batch in translated_batches:
        print(batch)
        print("---")


def vector_similarity_example():
    """向量相似度示例"""
    # 初始化向量管理器
    vector_manager = VectorManager()
    
    # 示例文本段落
    paragraphs = [
        "量子计算使用量子比特进行计算。",
        "量子纠缠是量子力学的基本特性。",
        "量子算法可以解决传统计算机难以解决的问题。",
        "量子比特可以同时表示0和1。",
        "量子计算可以加速某些类型的计算任务。"
    ]
    
    # 生成向量
    vectors = vector_manager.create_embeddings(paragraphs)
    
    # 查询文本
    query = "量子计算的优势是什么？"
    query_vector = vector_manager.create_single_embedding(query)
    
    # 查找最相似的段落
    similarities = vector_manager.calculate_similarities(query_vector, vectors)
    
    # 输出结果
    print(f"查询: {query}")
    print("\n相似度排名:")
    
    # 排序并输出结果
    sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    for i, idx in enumerate(sorted_indices):
        print(f"{i+1}. 相似度: {similarities[idx]:.4f} - {paragraphs[idx]}")


if __name__ == "__main__":
    print("=== 简单翻译示例 ===")
    simple_translation_example()
    
    print("\n=== 字幕翻译示例 ===")
    subtitle_translation_example()
    
    print("\n=== 向量相似度示例 ===")
    vector_similarity_example() 