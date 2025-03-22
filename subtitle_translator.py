#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
字幕智能翻译工具主程序

性能优化建议:
1. 增大批处理大小和减少批次延迟可以提高速度，但需注意API限制
2. 在资源允许的情况下，增加worker_count可以显著提高翻译速度
3. 场景边界检测可以优化为使用二分法，提高大型字幕文件的处理速度
4. 对于非常大的字幕文件(>3000行)，可考虑扩大chunk_size以减少API调用次数
5. 向量缓存的使用可以显著提高重复处理速度，建议在重复翻译时启用
"""

import os
import json
import argparse
import time
import re
import shutil
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import traceback
import concurrent.futures
import threading
from datetime import datetime

from utils.subtitle_parser import SubtitleParser, Subtitle
from utils.ai_service import AIService
from utils.vector_tools import VectorTools


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AI驱动的智能字幕翻译工具")
    parser.add_argument("--input", "-i", required=True, help="输入字幕文件路径")
    parser.add_argument("--output", "-o", help="输出翻译文件路径")
    parser.add_argument("--lang", "-l", default="zh-CN", help="目标语言")
    parser.add_argument("--chunk-size", "-c", type=int, default=10, help="处理块大小")
    parser.add_argument("--vector-chunk-size", "-v", type=int, default=3, help="向量化块大小")
    parser.add_argument("--save-analysis", "-s", action="store_true", help="保存内容分析结果")
    parser.add_argument("--use-cache", "-u", action="store_true", help="使用缓存的向量数据")
    parser.add_argument("--cache-dir", "-d", default=".cache", help="缓存目录")
    parser.add_argument("--similar-chunks", "-k", type=int, default=2, help="每个块使用的相似块数量")
    parser.add_argument("--output-dir", type=str, default="translations", help="翻译结果输出目录")
    parser.add_argument("--workers", "-w", type=int, default=4, help="翻译并发工作线程数量")
    parser.add_argument("--vectorize-workers", "-vw", type=int, default=4, help="向量化并发工作线程数量")
    parser.add_argument("--batch-delay", type=float, default=0.2, help="多线程翻译时每批次延迟(秒)，防止API限制")
    parser.add_argument("--embedding-model", type=str, help="指定嵌入模型名称，覆盖环境变量设置")
    parser.add_argument("--translation-model", type=str, help="指定翻译模型名称，覆盖环境变量设置")
    parser.add_argument("--analysis-model", type=str, help="指定分析模型名称，覆盖环境变量设置")
    parser.add_argument("--force-analysis", action="store_true", help="强制重新分析内容")
    parser.add_argument("--align", action="store_true", help="启用字幕校准功能，确保翻译字幕与源字幕精确对应并优化排版")
    
    return parser.parse_args()


def get_movie_name(file_path):
    """从文件路径中提取电影名称"""
    base_name = os.path.basename(file_path)
    # 移除扩展名
    movie_name = os.path.splitext(base_name)[0]
    # 移除常见后缀如.eng等
    movie_name = re.sub(r'\.(eng|chs|cht|jp|kor)$', '', movie_name, flags=re.IGNORECASE)
    return movie_name


def ensure_dir(directory):
    """确保目录存在，如不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")
    return directory


def translate_chunk(args):
    """
    翻译单个字幕块的工作函数，用于多线程处理
    
    Args:
        args: 包含翻译所需参数的元组
        
    Returns:
        翻译结果和块索引的元组
    """
    i, chunk, ai_service, chunk_embedding, chunk_embeddings, chunks, terminology_embeddings, content_analysis, args_obj, chunk_subtitles = args
    
    try:
        # 查找相似的块
        similar_chunk_indices = []
        similar_chunks = []
        if chunk_embeddings and len(chunk_embeddings) > 0:
            similar_chunk_indices = VectorTools.static_find_similar_chunks(
                chunk_embedding, 
                chunk_embeddings,
                top_k=args_obj.similar_chunks
            )
            similar_chunks = [chunks[idx] for idx in similar_chunk_indices if idx < len(chunks)]
        
        # 提取相关术语
        relevant_terms = []
        if terminology_embeddings and len(terminology_embeddings) > 0:
            relevant_terms = VectorTools.static_extract_key_terms(
                chunk_embedding,
                terminology_embeddings,
                threshold=0.75
            )
        
        # 使用增强上下文的翻译方法
        translated_chunk = ai_service.translate_with_context(
            chunk, 
            target_language=args_obj.lang,
            content_analysis=content_analysis,
            similar_chunks=similar_chunks,
            relevant_terms=relevant_terms
        )
        
        # 拆分翻译结果
        lines = translated_chunk.strip().split("\n")
        chunk_translations = []
        
        # 匹配翻译结果与原始字幕
        current_sub_idx = 0
        current_line_group = []
        
        for line in lines:
            # 跳过空行
            if not line.strip():
                if current_line_group:
                    chunk_translations.append("\n".join(current_line_group))
                    current_line_group = []
                    current_sub_idx += 1
                continue
            
            # 跳过可能的序号或时间码行
            if not re.match(r'^\d+\.?$', line) and not re.match(r'^\d{2}:\d{2}:\d{2}', line):
                current_line_group.append(line)
        
        # 添加最后一组
        if current_line_group:
            chunk_translations.append("\n".join(current_line_group))
        
        # 确保翻译数量与当前块中的字幕数量匹配
        expected_count = len(chunk_subtitles)
        
        if len(chunk_translations) != expected_count:
            # 尝试基于行数匹配
            if len(lines) >= expected_count:
                chunk_translations = []
                lines_per_subtitle = len(lines) // expected_count
                for j in range(expected_count):
                    start_idx = j * lines_per_subtitle
                    end_idx = start_idx + lines_per_subtitle
                    if j == expected_count - 1:  # 最后一个字幕
                        end_idx = len(lines)
                    sub_lines = lines[start_idx:end_idx]
                    chunk_translations.append("\n".join([l for l in sub_lines if l.strip()]))
            else:
                # 如果不匹配，则使用空字符串填充或截断
                if len(chunk_translations) < expected_count:
                    chunk_translations.extend([""] * (expected_count - len(chunk_translations)))
                else:
                    chunk_translations = chunk_translations[:expected_count]
        
        return (i, chunk_translations)
    except Exception as e:
        print(f"⚠️ 翻译块 {i+1} 时出错: {e}")
        # 填充空白翻译
        expected_count = len(chunk_subtitles)
        return (i, [""] * expected_count)


def translate_subtitles(subtitle_parser, ai_service, vector_tools, args, content_analysis=None, chunk_embeddings=None, chunks=None):
    """翻译字幕内容
    
    性能优化建议:
    - 对于>500行的字幕，可考虑将batch_size增加到100，提高吞吐量
    - worker_count可根据系统CPU核心数和网络状况适当增加，但注意API速率限制
    - 如需极致速度，可将batch_delay减少至0，但需确保API不会限流
    - 大型字幕文件可增加scene_boundary_threshold的值(>5秒)以减少场景边界数量
    - 将similar_chunk检测频率从10调整为20可进一步减少API调用
    
    Args:
        subtitle_parser: 字幕解析器
        ai_service: AI服务
        vector_tools: 向量工具
        args: 命令行参数
        content_analysis: 内容分析结果
        chunk_embeddings: 块嵌入向量
        chunks: 文本块
        
    Returns:
        翻译后的字幕文本
    """
    print(f"🔠 开始翻译字幕到 {args.lang}... (70%)")
    
    # 获取所有字幕
    subtitles = subtitle_parser.subtitles
    total = len(subtitles)
    
    # 根据命令行参数调整工作线程数和延迟
    # 提高性能：使用更多线程，更大批次，更小延迟
    worker_count = min(args.workers, 20)  # 线程数增加到20，大幅增加并发
    batch_delay = max(args.batch_delay, 0.0)  # 减少批次间延迟，依赖API自己的限流
    batch_size = 50  # 显著增大批处理量，减少循环次数
    
    # 优化：预处理关键帧和场景转换点，提高上下文理解
    # 识别可能的场景边界
    scene_boundaries = []
    prev_time = None
    for i, subtitle in enumerate(subtitles):
        if prev_time is not None:
            # 获取当前字幕的开始时间（秒）
            start_parts = str(subtitle.start).split(':')
            start_seconds = int(start_parts[0]) * 3600 + int(start_parts[1]) * 60 + float(start_parts[2].replace(',', '.'))
            
            # 获取前一字幕的结束时间（秒）
            end_parts = prev_time.split(':')
            end_seconds = int(end_parts[0]) * 3600 + int(end_parts[1]) * 60 + float(end_parts[2].replace(',', '.'))
            
            # 如果间隔超过5秒，可能是场景转换
            if start_seconds - end_seconds > 5:
                scene_boundaries.append(i)
        
        prev_time = str(subtitle.end)
    
    print(f"📊 检测到 {len(scene_boundaries)} 个可能的场景转换点")
    
    # 是否使用多线程翻译
    if worker_count > 1:
        # 创建任务队列
        tasks = []
        
        # 每个字幕都是一个翻译任务
        for i, subtitle in enumerate(subtitles):
            original_text = subtitle.text
            
            # 跳过空字幕
            if not original_text.strip():
                continue
                
            # 获取相似块（优化：同时考虑前后相邻字幕作为上下文）
            similar_chunks = []
            
            # 添加相邻字幕作为直接上下文
            for j in range(max(0, i-3), min(len(subtitles), i+4)):
                if j != i and subtitles[j].text.strip():
                    similar_chunks.append(subtitles[j].text)
            
            # 获取语义相似的块
            try:
                # 每隔10个字幕计算一次向量相似度（原来是20）
                # 减少相似度计算频率但增加相邻字幕的直接使用
                if i % 10 == 0 and chunk_embeddings:
                    text_embedding = ai_service.get_embedding(original_text)
                    similar_indices = vector_tools.find_similar_chunks(text_embedding, chunk_embeddings, args.similar_chunks)
                    for idx in similar_indices:
                        if chunks and idx < len(chunks):
                            chunk_text = " ".join(chunks[idx])
                            if chunk_text not in similar_chunks:  # 避免重复
                                similar_chunks.append(chunk_text)
            except Exception as e:
                print(f"⚠️ 警告: 获取相似块失败: {e}")
            
            # 增加场景信息作为上下文辅助
            is_scene_boundary = i in scene_boundaries
            
            tasks.append({
                "index": i,
                "subtitle": subtitle,
                "text": original_text,
                "similar_chunks": similar_chunks,
                "is_scene_boundary": is_scene_boundary
            })
        
        # 优化：对字幕文本进行分组，注意保持场景连贯性
        # 将连续的短字幕组合起来一次性翻译，提高效率
        combined_tasks = []
        temp_group = []
        temp_text = ""
        temp_indices = []
        
        for task in tasks:
            # 如果是场景边界，先结束当前组
            if task["is_scene_boundary"] and temp_group:
                combined_tasks.append({
                    "indices": temp_indices,
                    "text": temp_text.strip(),
                    "similar_chunks": temp_group[0]["similar_chunks"],  # 使用第一个任务的相似块
                    "subtitles": [t["subtitle"] for t in temp_group]
                })
                temp_group = []
                temp_text = ""
                temp_indices = []
            
            # 如果当前文本长度加上新任务不超过2000字符，则组合
            # 但单个组最多不超过5个字幕，避免过度组合导致翻译质量下降
            if len(temp_text) + len(task["text"]) < 2000 and len(temp_group) < 5:
                temp_group.append(task)
                temp_text += "\n\n" + task["text"]
                temp_indices.append(task["index"])
            else:
                # 如果组里有内容，先保存当前组
                if temp_group:
                    combined_tasks.append({
                        "indices": temp_indices,
                        "text": temp_text.strip(),
                        "similar_chunks": temp_group[0]["similar_chunks"],  # 使用第一个任务的相似块
                        "subtitles": [t["subtitle"] for t in temp_group]
                    })
                # 重新开始新的组
                temp_group = [task]
                temp_text = task["text"]
                temp_indices = [task["index"]]
        
        # 添加最后一组
        if temp_group:
            combined_tasks.append({
                "indices": temp_indices,
                "text": temp_text.strip(),
                "similar_chunks": temp_group[0]["similar_chunks"],
                "subtitles": [t["subtitle"] for t in temp_group]
            })
        
        print(f"📊 优化：将 {len(tasks)} 条字幕组合为 {len(combined_tasks)} 个翻译任务")
        
        # 使用线程池并行翻译
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = {}
            
            print(f"🧵 使用 {worker_count} 个工作线程进行翻译（批次大小: {batch_size}, 延迟: {batch_delay}秒）...")
            print(f"🤖 使用翻译模型: {ai_service.translation_model}")
            
            with tqdm(total=total, desc="翻译进度") as pbar:
                # 分批次处理
                for i in range(0, len(combined_tasks), batch_size):
                    batch = combined_tasks[i:i+batch_size]
                    
                    # 创建当前批次的任务
                    futures = []
                    for task in batch:
                        future = executor.submit(
                            ai_service.translate_subtitle_chunk,
                            task["text"],
                            args.lang,
                            content_analysis,
                            task["similar_chunks"]
                        )
                        futures.append((future, task["indices"], task["subtitles"]))
                    
                    # 收集结果
                    for future, indices, batch_subtitles in futures:
                        try:
                            translation = future.result()
                            
                            if translation:
                                # 拆分翻译结果回多个字幕
                                if len(indices) > 1:
                                    # 优化：使用更智能的拆分方法
                                    # 首先尝试按照空行拆分
                                    parts = translation.split("\n\n")
                                    
                                    # 如果拆分出的部分与原字幕数量匹配
                                    if len(parts) == len(indices):
                                        for j, idx in enumerate(indices):
                                            results[idx] = parts[j]
                                    else:
                                        # 尝试按句子拆分
                                        sentences = re.split(r'([。！？…])', translation)
                                        # 合并句子与标点
                                        real_sentences = []
                                        for k in range(0, len(sentences)-1, 2):
                                            if k+1 < len(sentences):
                                                real_sentences.append(sentences[k] + sentences[k+1])
                                            else:
                                                real_sentences.append(sentences[k])
                                        
                                        # 如果句子数量与字幕数量匹配或接近
                                        if abs(len(real_sentences) - len(indices)) <= 1:
                                            # 尽量均匀分配句子到字幕
                                            sentences_per_subtitle = max(1, len(real_sentences) // len(indices))
                                            for j, idx in enumerate(indices):
                                                start = j * sentences_per_subtitle
                                                end = min(start + sentences_per_subtitle, len(real_sentences))
                                                if j == len(indices) - 1:  # 最后一个字幕拿剩下所有句子
                                                    end = len(real_sentences)
                                                if start < len(real_sentences):
                                                    results[idx] = "".join(real_sentences[start:end])
                                                else:
                                                    results[idx] = ""
                                        else:
                                            # 按比例拆分文本
                                            total_chars = len(translation)
                                            total_original_chars = sum([len(sub.text) for sub in batch_subtitles])
                                            
                                            start_pos = 0
                                            for j, idx in enumerate(indices):
                                                # 按原始文本长度比例分配翻译结果
                                                original_ratio = len(batch_subtitles[j].text) / total_original_chars
                                                char_count = int(total_chars * original_ratio)
                                                
                                                # 确保不会越界
                                                end_pos = min(start_pos + char_count, total_chars)
                                                
                                                # 提取对应的翻译片段
                                                results[idx] = translation[start_pos:end_pos].strip()
                                                start_pos = end_pos
                                else:
                                    # 单个字幕直接使用整个翻译结果
                                    results[indices[0]] = translation
                            else:
                                # 如果翻译结果为空，使用原文
                                for idx in indices:
                                    results[idx] = tasks[idx]["text"] if idx < len(tasks) else ""
                                    
                            # 更新进度条
                            pbar.update(len(indices))
                            pbar.set_description(f"翻译进度 ({int(pbar.n/pbar.total*100)}%)")
                        except Exception as e:
                            print(f"⚠️ 翻译失败: {e}")
                            # 失败时使用原文
                            for idx in indices:
                                if idx < len(tasks):
                                    results[idx] = tasks[idx]["text"]
                            pbar.update(len(indices))
                    
                    # 每批次之间添加极短延迟，避免API限制
                    if i + batch_size < len(combined_tasks) and batch_delay > 0:
                        time.sleep(batch_delay)
            
            # 更新字幕翻译结果
            for idx, translation in results.items():
                if idx < len(subtitles):
                    subtitles[idx].text = translation
    else:
        # 单线程翻译 - 但优化处理方式
        print("使用单线程翻译...")
        print(f"🤖 使用翻译模型: {ai_service.translation_model}")
        
        # 将字幕分组以减少API调用，但考虑场景边界
        groups = []
        current_group = []
        current_text = ""
        
        for i, subtitle in enumerate(subtitles):
            if not subtitle.text.strip():
                continue
            
            # 如果是场景边界，先结束当前组
            if i in scene_boundaries and current_group:
                groups.append(current_group)
                current_group = []
                current_text = ""
                
            # 如果不超过2000字符且不超过5个字幕，添加到当前组
            if len(current_text) + len(subtitle.text) < 2000 and len(current_group) < 5:
                # 收集相邻字幕作为上下文
                similar_chunks = []
                for j in range(max(0, i-3), min(len(subtitles), i+4)):
                    if j != i and subtitles[j].text.strip():
                        similar_chunks.append(subtitles[j].text)
                
                current_group.append((i, subtitle, similar_chunks))
                current_text += "\n\n" + subtitle.text
            else:
                # 保存当前组并开始新组
                if current_group:
                    groups.append(current_group)
                
                # 收集相邻字幕作为上下文
                similar_chunks = []
                for j in range(max(0, i-3), min(len(subtitles), i+4)):
                    if j != i and subtitles[j].text.strip():
                        similar_chunks.append(subtitles[j].text)
                
                current_group = [(i, subtitle, similar_chunks)]
                current_text = subtitle.text
        
        # 添加最后一组
        if current_group:
            groups.append(current_group)
            
        print(f"📊 优化：将 {len(subtitles)} 条字幕组合为 {len(groups)} 个翻译任务")
        
        with tqdm(total=total, desc="翻译进度") as pbar:
            for group in groups:
                # 组合文本
                combined_text = "\n\n".join([sub.text for _, sub, _ in group])
                
                # 获取所有相似块并合并去重
                all_similar_chunks = []
                for _, _, similar_chunks in group:
                    for chunk in similar_chunks:
                        if chunk not in all_similar_chunks:
                            all_similar_chunks.append(chunk)
                
                # 限制相似块数量，优先保留前后相邻的
                similar_chunks = all_similar_chunks[:6]
                
                # 获取向量相似的块
                try:
                    # 为每组计算一次相似度
                    if chunk_embeddings:
                        text_embedding = ai_service.get_embedding(combined_text[:1000])  # 限制长度
                        similar_indices = vector_tools.find_similar_chunks(text_embedding, chunk_embeddings, args.similar_chunks)
                        for idx in similar_indices:
                            if chunks and idx < len(chunks):
                                chunk_text = " ".join(chunks[idx])
                                if chunk_text not in similar_chunks:  # 避免重复
                                    similar_chunks.append(chunk_text)
                except Exception as e:
                    print(f"⚠️ 警告: 获取相似块失败: {e}")
                
                # 翻译组合文本
                try:
                    translation = ai_service.translate_subtitle_chunk(
                        combined_text,
                        args.lang,
                        content_analysis,
                        similar_chunks
                    )
                    
                    if translation:
                        # 拆分翻译结果
                        if len(group) > 1:
                            # 优化：使用更智能的拆分方法
                            parts = translation.split("\n\n")
                            
                            # 如果拆分部分与字幕数量匹配
                            if len(parts) == len(group):
                                for j, (idx, sub, _) in enumerate(group):
                                    sub.text = parts[j]
                            else:
                                # 尝试按句子拆分
                                sentences = re.split(r'([。！？…])', translation)
                                # 合并句子与标点
                                real_sentences = []
                                for k in range(0, len(sentences)-1, 2):
                                    if k+1 < len(sentences):
                                        real_sentences.append(sentences[k] + sentences[k+1])
                                    else:
                                        real_sentences.append(sentences[k])
                                
                                # 如果句子数量与字幕数量匹配或接近
                                if abs(len(real_sentences) - len(group)) <= 1:
                                    # 尽量均匀分配句子到字幕
                                    sentences_per_subtitle = max(1, len(real_sentences) // len(group))
                                    for j, (_, sub, _) in enumerate(group):
                                        start = j * sentences_per_subtitle
                                        end = min(start + sentences_per_subtitle, len(real_sentences))
                                        if j == len(group) - 1:  # 最后一个字幕拿剩下所有句子
                                            end = len(real_sentences)
                                        if start < len(real_sentences):
                                            sub.text = "".join(real_sentences[start:end])
                                else:
                                    # 按比例分配翻译结果
                                    total_chars = len(translation)
                                    total_original_chars = sum([len(sub.text) for _, sub, _ in group])
                                    
                                    start_pos = 0
                                    for _, sub, _ in group:
                                        original_ratio = len(sub.text) / total_original_chars
                                        char_count = int(total_chars * original_ratio)
                                        
                                        end_pos = min(start_pos + char_count, total_chars)
                                        sub.text = translation[start_pos:end_pos].strip()
                                        start_pos = end_pos
                        else:
                            # 单个字幕直接使用整个翻译
                            group[0][1].text = translation
                except Exception as e:
                    print(f"⚠️ 翻译失败: {e}")
                
                # 更新进度条
                pbar.update(len(group))
                pbar.set_description(f"翻译进度 ({int(pbar.n/pbar.total*100)}%)")
    
    # 返回翻译后的字幕解析器
    return subtitle_parser


def save_srt(subtitle_parser, output_path, encoding='utf-8'):
    """正确格式保存SRT字幕文件，避免格式问题
    
    Args:
        subtitle_parser: 字幕解析器对象
        output_path: 输出文件路径
        encoding: 文件编码，默认UTF-8
    """
    with open(output_path, 'w', encoding=encoding) as f:
        for i, item in enumerate(subtitle_parser.subtitles):
            # 确保索引是连续的
            item.index = i + 1
            
            # 格式化时间码行
            timestamp_line = f"{item.start} --> {item.end}"
            
            # 确保文本没有前后多余的空行
            text = item.text.strip()
            
            # 写入标准格式的字幕条目
            f.write(f"{item.index}\n")
            f.write(f"{timestamp_line}\n")
            f.write(f"{text}\n")
            
            # 在每个字幕条目之间只添加一个空行
            if i < len(subtitle_parser.subtitles) - 1:
                f.write("\n")


def main():
    """主程序"""
    parser = argparse.ArgumentParser(description='字幕翻译工具')
    parser.add_argument('--input', required=True, help='输入字幕文件路径')
    parser.add_argument('--output', help='输出翻译文件路径')
    parser.add_argument('--lang', default='zh-CN', help='目标语言，默认中文')
    parser.add_argument('--chunk-size', type=int, default=3, help='分块大小')
    parser.add_argument('--vector-chunk-size', type=int, default=5, help='向量化分块大小')
    parser.add_argument('--save-analysis', action='store_true', help='是否保存分析结果')
    parser.add_argument('--force-analysis', action='store_true', help='强制重新分析内容')
    parser.add_argument('--use-cache', action='store_true', help='使用缓存加速')
    parser.add_argument('--cache-dir', help='缓存目录')
    parser.add_argument('--similar-chunks', type=int, default=5, help='匹配相似块数量')
    parser.add_argument('--output-dir', help='输出目录')
    parser.add_argument('--workers', type=int, default=1, help='翻译工作线程数')
    parser.add_argument('--vectorize-workers', type=int, default=2, help='向量化工作线程数')
    parser.add_argument('--batch-delay', type=float, default=1.0, help='批处理延迟(秒)')
    parser.add_argument('--embedding-model', help='指定嵌入模型名称')
    parser.add_argument('--translation-model', help='指定翻译模型名称')
    parser.add_argument('--analysis-model', help='指定分析模型名称')
    parser.add_argument('--align', action='store_true', help='启用字幕校准功能，确保翻译字幕与源字幕精确对应并优化排版')

    args = parser.parse_args()

    try:
        # 检查输入文件是否存在
        if not os.path.exists(args.input):
            print(f"错误: 输入文件 '{args.input}' 不存在")
            return

        # 获取电影名称（不含扩展名）
        base_name = os.path.basename(args.input)
        movie_name = os.path.splitext(base_name)[0]
        
        # 设置输出目录
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.join("translations", movie_name)
            
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            print(f"创建目录: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            
        print(f"🎬 正在处理字幕文件: {args.input}")
        print(f"📂 输出目录: {output_dir}")
        
        # 设置输出文件路径
        if args.output:
            output_file = args.output
        else:
            output_file = os.path.join(output_dir, f"{movie_name}.{args.lang}.srt")
            
        # 设置缓存目录
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(".cache", movie_name)
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化进度
        progress = 0
        print(f"⚙️ 初始化服务... ({progress}%)")
        
        # 初始化字幕解析器
        subtitle_parser = SubtitleParser(args.input)
        
        # 初始化AI服务
        ai_service = AIService(
            embedding_model=args.embedding_model,
            translation_model=args.translation_model,
            analysis_model=args.analysis_model
        )
        
        # 初始化向量工具
        vector_tools = VectorTools()
        
        print(f"🤖 使用嵌入模型: {ai_service.embedding_model}")
        print(f"🤖 使用翻译模型: {ai_service.translation_model}")
        print(f"🤖 使用分析模型: {ai_service.analysis_model}")
        
        # 获取字幕元数据
        metadata = subtitle_parser.get_metadata()
        print(f"ℹ️ 字幕信息: {metadata['subtitle_count']} 条字幕, 总时长: {metadata['duration']}")
        
        progress = 5
        print(f"⚙️ 初始化完成 ({progress}%)")
        
        # 1. 分析字幕文本
        progress = 10
        print(f"🔍 正在分析字幕内容... ({progress}%)")
        
        subtitle_text = subtitle_parser.get_all_text()
        
        # 生成字幕文本的嵌入向量
        progress = 15
        print(f"🧠 正在生成文本嵌入向量... ({progress}%)")
        print(f"🤖 使用嵌入模型: {ai_service.embedding_model}")
        
        # 获取全文的嵌入向量
        full_embedding = ai_service.get_embedding(subtitle_text)
        
        progress = 20
        if len(full_embedding) > 0:
            embedding_dim = len(full_embedding)
            print(f"✅ 成功生成嵌入向量，维度: {embedding_dim} ({progress}%)")
        
        # 使用向量工具进行字幕分块和向量化
        progress = 22
        print(f"📊 将字幕分块并向量化 (块大小: {args.chunk_size})... ({progress}%)")
        
        # 将字幕分块并计算向量相似度
        chunks_data = vector_tools.chunk_and_vectorize(
            subtitle_parser.get_subtitle_chunks(args.chunk_size),
            ai_service.get_embedding,
            chunk_size=args.chunk_size,
            max_workers=args.vectorize_workers
        )
        
        chunks = chunks_data["chunks"]
        chunk_embeddings = chunks_data["embeddings"]
        chunk_count = chunks_data["total_chunks"]
        
        progress = 25
        print(f"💾 向量数据已缓存，共 {chunk_count} 个块 ({progress}%)")
        
        # 如果没有任何块，使用备用方法
        if chunk_count == 0:
            print("⚠️ 警告: 向量化失败，使用备用方法...")
            chunks = [subtitle_text]
            if full_embedding:
                chunk_embeddings = [full_embedding]
            else:
                chunk_embeddings = []
                full_embedding = []
        
        # 2. 分析内容类型
        content_analysis = None
        analysis_file = os.path.join(output_dir, f"{movie_name}.analysis.json")
        
        # 如果存在分析文件且不强制重新分析，则读取
        if os.path.exists(analysis_file) and not args.force_analysis:
            try:
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    content_analysis = json.load(f)
                print(f"📊 使用现有内容分析结果 (40%)")
            except Exception as e:
                print(f"⚠️ 警告: 读取分析文件失败: {e}")
                content_analysis = None
        
        # 如果没有内容分析，则进行分析
        if not content_analysis:
            try:
                progress = 30
                print(f"📝 正在分析影片类型和内容... ({progress}%)")
                print(f"🤖 使用分析模型: {ai_service.analysis_model}")
                
                content_analysis = ai_service.analyze_content(subtitle_text)
                
                # 保存分析结果
                if args.save_analysis:
                    with open(analysis_file, 'w', encoding='utf-8') as f:
                        json.dump(content_analysis, f, ensure_ascii=False, indent=2)
                    print(f"📊 内容分析结果已保存至 {analysis_file} (40%)")
                
                # 打印分析结果摘要
                print("影片分析结果:")
                if content_analysis and "genre" in content_analysis:
                    print(f"🎭 影片类型: {content_analysis['genre']}")
                if content_analysis and "plot_summary" in content_analysis:
                    print(f"📖 剧情摘要: {content_analysis['plot_summary']}")
                    
                progress = 50
                print(f"✅ 内容分析完成 ({progress}%)")
            except Exception as e:
                print(f"⚠️ 警告: 内容分析失败: {e}")
                progress = 50
        else:
            progress = 50
        
        # 3. 提取关键术语
        progress = 60
        print(f"🔑 提取关键术语... ({progress}%)")
        
        # 使用分析结果中的术语或从文本中提取
        if content_analysis and "terminology" in content_analysis:
            # 如果分析结果中有术语，直接使用
            if isinstance(content_analysis["terminology"], dict):
                terms = list(content_analysis["terminology"].keys())
            else:
                terms = [term.split(":")[0].strip() for term in content_analysis["terminology"] if ":" in term]
                if not terms:
                    terms = content_analysis["terminology"]
        else:
            # 否则尝试从文本中提取
            terms = ai_service.extract_key_terms(subtitle_text)
            
        print(f"发现 {len(terms)} 个关键术语")
        
        # 4. 开始翻译字幕
        progress = 70
        print(f"🔠 开始翻译字幕到 {args.lang}... ({progress}%)")
        
        # 检测场景边界
        scene_boundaries = vector_tools.find_scene_boundaries(chunk_embeddings, threshold=0.7)
        print(f"📊 检测到 {len(scene_boundaries)} 个可能的场景转换点")
        
        # 识别相似场景以增强翻译一致性
        similar_scenes = vector_tools.find_similar_scenes(
            chunk_embeddings, 
            scene_boundaries, 
            similarity_threshold=0.75
        )
        
        # 准备翻译任务
        subtitles = subtitle_parser.subtitles
        tasks = []
        
        # 创建字幕索引到块索引的映射
        subtitle_to_chunk_mapping = {}
        
        # 构建翻译任务列表
        for i, subtitle in enumerate(subtitles):
            # 获取块索引
            chunk_idx = i // args.chunk_size
            subtitle_to_chunk_mapping[i] = chunk_idx
            
            # 获取相似块
            similar_chunk_indices = []
            if chunk_idx < len(chunks):
                similar_chunk_indices = vector_tools.find_similar_chunks(
                    chunk_embeddings[chunk_idx], 
                    chunk_embeddings, 
                    args.similar_chunks
                )
            
            similar_text = []
            for idx in similar_chunk_indices:
                if idx < len(chunks):
                    similar_text.append(" ".join(chunks[idx]))
            
            # 创建任务
            tasks.append({
                "index": i,
                "text": subtitle.text,
                "similar_chunks": similar_text,
                "subtitle": subtitle
            })
        
        progress = 70
        print(f"🔠 开始翻译字幕到 {args.lang}... ({progress}%)")
        
        # 再次检测场景边界但使用更保守的阈值，确保更高质量的分组
        conservative_boundaries = vector_tools.find_scene_boundaries(chunk_embeddings, threshold=0.5)
        print(f"📊 检测到 {len(conservative_boundaries)} 个可能的场景转换点")
        
        # 优化：按场景边界组合任务以减少API调用
        combined_tasks = []
        temp_group = []
        temp_text = ""
        temp_indices = []
        
        # 场景边界集合，用于判断是否需要分割
        boundary_set = set()
        for b in conservative_boundaries:
            # 将块索引转换为字幕索引范围
            start_subtitle = max(0, b * args.chunk_size - 1)
            end_subtitle = min(len(subtitles), (b + 1) * args.chunk_size + 1)
            for idx in range(start_subtitle, end_subtitle):
                boundary_set.add(idx)
        
        # 组合任务
        for i, task in enumerate(tasks):
            # 如果当前字幕索引是场景边界，或达到最大组大小，开始新组
            if i in boundary_set or (temp_group and len(temp_group) >= args.chunk_size * 2):
                # 保存当前组
                if temp_group:
                    combined_tasks.append({
                        "indices": temp_indices,
                        "text": temp_text.strip(),
                        "similar_chunks": temp_group[0]["similar_chunks"],
                        "subtitles": [t["subtitle"] for t in temp_group]
                    })
                # 开始新组
                temp_group = [task]
                temp_text = task["text"]
                temp_indices = [task["index"]]
            else:
                # 添加到当前组
                temp_group.append(task)
                temp_text += "\n\n" + task["text"]
                temp_indices.append(task["index"])
        
        # 添加最后一组
        if temp_group:
            combined_tasks.append({
                "indices": temp_indices,
                "text": temp_text.strip(),
                "similar_chunks": temp_group[0]["similar_chunks"],
                "subtitles": [t["subtitle"] for t in temp_group]
            })
        
        print(f"📊 优化：将 {len(tasks)} 条字幕组合为 {len(combined_tasks)} 个翻译任务")
        
        # 增强翻译任务，添加相似场景信息
        enhanced_tasks = vector_tools.enhance_translation_consistency(
            combined_tasks,
            similar_scenes,
            subtitle_to_chunk_mapping
        )
        
        # 批处理参数
        batch_size = 50  # 每批处理的任务数
        batch_delay = args.batch_delay  # 批次间延迟
        total = len(subtitles)  # 总字幕数
        
        # 使用线程池并行翻译
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = {}
            
            print(f"🧵 使用 {args.workers} 个工作线程进行翻译（批次大小: {batch_size}, 延迟: {batch_delay}秒）...")
            print(f"🤖 使用翻译模型: {ai_service.translation_model}")
            
            with tqdm(total=total, desc="翻译进度") as pbar:
                # 分批次处理
                for i in range(0, len(enhanced_tasks), batch_size):
                    batch = enhanced_tasks[i:i+batch_size]
                    
                    # 创建当前批次的任务
                    futures = []
                    for task in batch:
                        # 提取上下文参考，如果有的话
                        context_refs = task.get("context_references", None)
                        
                        future = executor.submit(
                            ai_service.translate_subtitle_chunk,
                            task["text"],
                            args.lang,
                            content_analysis,
                            task["similar_chunks"],
                            context_refs
                        )
                        futures.append((future, task["indices"], task["subtitles"]))
                    
                    # 收集结果
                    for future, indices, batch_subtitles in futures:
                        try:
                            translation = future.result()
                            
                            if translation:
                                # 拆分翻译结果回多个字幕
                                if len(indices) > 1:
                                    # 优化：使用更智能的拆分方法
                                    # 首先尝试按照空行拆分
                                    parts = translation.split("\n\n")
                                    
                                    # 如果拆分出的部分与原字幕数量匹配
                                    if len(parts) == len(indices):
                                        for j, idx in enumerate(indices):
                                            results[idx] = parts[j]
                                    else:
                                        # 尝试按句子拆分
                                        sentences = re.split(r'([。！？…])', translation)
                                        # 合并句子与标点
                                        real_sentences = []
                                        for k in range(0, len(sentences)-1, 2):
                                            if k+1 < len(sentences):
                                                real_sentences.append(sentences[k] + sentences[k+1])
                                            else:
                                                real_sentences.append(sentences[k])
                                        
                                        # 如果句子数量与字幕数量匹配或接近
                                        if abs(len(real_sentences) - len(indices)) <= 1:
                                            # 尽量均匀分配句子到字幕
                                            sentences_per_subtitle = max(1, len(real_sentences) // len(indices))
                                            for j, idx in enumerate(indices):
                                                start = j * sentences_per_subtitle
                                                end = min(start + sentences_per_subtitle, len(real_sentences))
                                                if j == len(indices) - 1:  # 最后一个字幕拿剩下所有句子
                                                    end = len(real_sentences)
                                                if start < len(real_sentences):
                                                    results[idx] = "".join(real_sentences[start:end])
                                                else:
                                                    results[idx] = ""
                                        else:
                                            # 按比例拆分文本
                                            total_chars = len(translation)
                                            total_original_chars = sum([len(sub.text) for sub in batch_subtitles])
                                            
                                            start_pos = 0
                                            for j, idx in enumerate(indices):
                                                # 按原始文本长度比例分配翻译结果
                                                original_ratio = len(batch_subtitles[j].text) / total_original_chars
                                                char_count = int(total_chars * original_ratio)
                                                
                                                # 确保不会越界
                                                end_pos = min(start_pos + char_count, total_chars)
                                                
                                                # 提取对应的翻译片段
                                                results[idx] = translation[start_pos:end_pos].strip()
                                                start_pos = end_pos
                                else:
                                    # 单个字幕直接使用整个翻译结果
                                    results[indices[0]] = translation
                            else:
                                # 如果翻译结果为空，使用原文
                                for idx in indices:
                                    results[idx] = tasks[idx]["text"] if idx < len(tasks) else ""
                                    
                            # 更新进度条
                            pbar.update(len(indices))
                            pbar.set_description(f"翻译进度 ({int(pbar.n/pbar.total*100)}%)")
                        except Exception as e:
                            print(f"⚠️ 翻译失败: {e}")
                            # 失败时使用原文
                            for idx in indices:
                                if idx < len(tasks):
                                    results[idx] = tasks[idx]["text"]
                            pbar.update(len(indices))
                    
                    # 每批次之间添加极短延迟，避免API限制
                    if i + batch_size < len(enhanced_tasks) and batch_delay > 0:
                        time.sleep(batch_delay)
            
            # 更新字幕翻译结果
            for idx, translation in results.items():
                if idx < len(subtitles):
                    subtitles[idx].text = translation
        
        # 如果启用了字幕校准功能
        if args.align:
            progress = 90
            print(f"🔄 正在校准字幕格式和排版... ({progress}%)")
            
            # 导入字幕校准工具
            from utils.subtitle_alignment import SubtitleAligner
            
            # 创建源字幕解析器的副本以用于对比
            source_parser = SubtitleParser(args.input)
            
            # 创建字幕校准器
            subtitle_aligner = SubtitleAligner(ai_service)
            
            # 执行字幕校准
            aligned_parser = subtitle_aligner.align_subtitles(source_parser, subtitle_parser)
            
            # 使用校准后的字幕解析器替换原字幕解析器
            subtitle_parser = aligned_parser
        
        # 5. 保存翻译结果
        progress = 95
        print(f"💾 正在保存翻译结果到 {output_file}... ({progress}%)")
        
        # 保存SRT格式的字幕
        save_srt(subtitle_parser, output_file, encoding='utf-8')
        print(f"字幕已保存至: {output_file}")
        
        # 额外保存原始字幕
        original_srt_path = os.path.join(output_dir, base_name)
        shutil.copy2(args.input, original_srt_path)
        print(f"📄 原始字幕已复制到 {original_srt_path}")
        
        # 保存处理日志
        log_file = os.path.join(output_dir, f"{movie_name}.log.json")
        log_data = {
            "input_file": args.input,
            "output_file": output_file,
            "target_language": args.lang,
            "subtitle_count": len(subtitles),
            "processed_at": datetime.now().isoformat(),
            "duration": metadata["duration"],
            "embedding_model": ai_service.embedding_model,
            "translation_model": ai_service.translation_model,
            "analysis_model": ai_service.analysis_model,
            "used_alignment": args.align
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        print(f"📝 处理日志已保存至 {log_file}")
        
        # 完成
        progress = 100
        print(f"✅ 翻译完成! ({progress}%)")
        print(f"📂 所有输出文件已保存至 {output_dir}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    exit(main()) 