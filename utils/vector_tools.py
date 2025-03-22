import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
import json
import os
from datetime import datetime
import pickle
import concurrent.futures
from tqdm import tqdm
import hashlib


class VectorTools:
    """向量处理和利用工具"""
    
    def __init__(self, cache_dir: str = ".cache"):
        """
        初始化向量工具
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            相似度得分(0-1)
        """
        return VectorTools.static_compute_similarity(vec1, vec2)
    
    @staticmethod
    def static_compute_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度（静态方法，用于多线程环境）
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            相似度得分(0-1)
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # 计算余弦相似度
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # 防止除零错误
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        return dot_product / (norm_v1 * norm_v2)
    
    @staticmethod
    def _vectorize_chunk(chunk_with_index: Tuple[int, List[str]], embedding_func: Callable) -> Tuple[int, List[str], List[float]]:
        """处理并向量化单个块（用于多线程）
        
        Args:
            chunk_with_index: 元组 (索引, 文本块)
            embedding_func: 向量化函数
            
        Returns:
            元组 (索引, 文本块, 向量)
        """
        idx, chunk = chunk_with_index
        text = " ".join(chunk)
        embedding = embedding_func(text)
        return (idx, chunk, embedding)
    
    def chunk_and_vectorize(self, texts: List[str], embedding_func: Callable, chunk_size: int = 3, max_workers: int = 3) -> Dict[str, Any]:
        """将文本分块并向量化
        
        Args:
            texts: 文本列表
            embedding_func: 向量化函数
            chunk_size: 每块的文本数量
            max_workers: 并发工作线程数
            
        Returns:
            Dict: 包含块和对应的向量
        """
        # 优化：跳过空文本，减少处理量
        non_empty_texts = [t for t in texts if t and t.strip()]
        
        # 分块 - 优化分块策略，确保语义完整性
        chunks = []
        for i in range(0, len(non_empty_texts), chunk_size):
            # 提取当前块中的文本
            current_chunk = non_empty_texts[i:i+chunk_size]
            chunks.append(current_chunk)
        
        # 多线程向量化，使用线程池优化
        chunk_embeddings = [None] * len(chunks)
        
        # 创建带索引的任务列表
        tasks = [(i, chunk) for i, chunk in enumerate(chunks)]
        
        # 批处理优化：每次处理多个块，减少API调用次数
        batch_size = min(10, len(chunks))  # 每批最多10个块
        
        # 创建进度条
        with tqdm(total=len(chunks), desc="向量化进度") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 分批提交任务
                for i in range(0, len(tasks), batch_size):
                    batch_tasks = tasks[i:i+batch_size]
                    
                    # 提交当前批次的任务
                    futures = []
                    for task in batch_tasks:
                        future = executor.submit(self._vectorize_chunk, task, embedding_func)
                        futures.append(future)
                    
                    # 处理当前批次的结果
                    for future in concurrent.futures.as_completed(futures):
                        idx, chunk, embedding = future.result()
                        chunk_embeddings[idx] = embedding
                        chunks[idx] = chunk
                        # 更新进度条
                        pbar.update(1)
                        pbar.set_description(f"向量化进度 ({int(pbar.n/pbar.total*100)}%)")
        
        return {
            "chunks": chunks,
            "embeddings": chunk_embeddings,
            "total_chunks": len(chunks)
        }
    
    def find_similar_chunks(self, query_embedding: List[float], chunk_embeddings: List[List[float]], k: int = 2) -> List[int]:
        """查找与查询向量最相似的k个块
        
        Args:
            query_embedding: 查询向量
            chunk_embeddings: 块向量列表
            k: 返回的相似块数量
            
        Returns:
            List[int]: 相似块的索引
        """
        return VectorTools.static_find_similar_chunks(query_embedding, chunk_embeddings, k)
    
    @staticmethod
    def static_find_similar_chunks(query_embedding: List[float], chunk_embeddings: List[List[float]], top_k: int = 2) -> List[int]:
        """静态方法：查找与查询向量最相似的k个块（用于多线程环境）
        
        Args:
            query_embedding: 查询向量
            chunk_embeddings: 块向量列表
            top_k: 返回的相似块数量
            
        Returns:
            List[int]: 相似块的索引
        """
        if not chunk_embeddings or not query_embedding:
            return []
            
        similarities = []
        for i, embedding in enumerate(chunk_embeddings):
            if embedding:  # 确保嵌入存在
                similarity = VectorTools.static_compute_similarity(query_embedding, embedding)
                similarities.append((i, similarity))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前k个相似块的索引
        return [idx for idx, _ in similarities[:top_k]]
    
    def extract_key_terms(self, 
                         full_embedding: List[float], 
                         term_embeddings: Dict[str, List[float]], 
                         threshold: float = 0.8) -> List[str]:
        """
        根据向量相似度提取关键术语
        
        Args:
            full_embedding: 完整字幕的向量表示
            term_embeddings: 术语及其向量的字典
            threshold: 相似度阈值
            
        Returns:
            相关度高的术语列表
        """
        return VectorTools.static_extract_key_terms(full_embedding, term_embeddings, threshold)
    
    @staticmethod
    def static_extract_key_terms(full_embedding: List[float], 
                               term_embeddings: Dict[str, List[float]], 
                               threshold: float = 0.8) -> List[str]:
        """
        根据向量相似度提取关键术语（静态方法，用于多线程环境）
        
        Args:
            full_embedding: 完整字幕的向量表示
            term_embeddings: 术语及其向量的字典
            threshold: 相似度阈值
            
        Returns:
            相关度高的术语列表
        """
        relevant_terms = []
        
        for term, term_emb in term_embeddings.items():
            similarity = VectorTools.static_compute_similarity(full_embedding, term_emb)
            if similarity >= threshold:
                relevant_terms.append(term)
        
        return relevant_terms
    
    def get_file_hash(self, file_path: str) -> str:
        """计算文件的MD5哈希值
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: MD5哈希值
        """
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    
    def cache_embeddings(self, file_path: str, embeddings_data: Dict[str, Any]) -> None:
        """缓存嵌入向量数据
        
        Args:
            file_path: 原始文件路径
            embeddings_data: 嵌入向量数据
        """
        file_hash = self.get_file_hash(file_path)
        cache_path = os.path.join(self.cache_dir, f"{file_hash}.json")
        
        # 准备缓存数据
        cache_data = {
            "file_path": file_path,
            "file_hash": file_hash,
            "chunks": embeddings_data["chunks"],
            "embeddings": embeddings_data["embeddings"],
            "total_chunks": len(embeddings_data["chunks"])
        }
        
        if "full_embedding" in embeddings_data:
            cache_data["full_embedding"] = embeddings_data["full_embedding"]
            
        if "vector_chunk_size" in embeddings_data:
            cache_data["vector_chunk_size"] = embeddings_data["vector_chunk_size"]
        
        # 保存缓存
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False)
    
    def load_cached_embeddings(self, file_path: str) -> Dict[str, Any]:
        """加载缓存的嵌入向量数据
        
        Args:
            file_path: 原始文件路径
            
        Returns:
            Dict: 缓存的嵌入向量数据，如果没有缓存则返回None
        """
        file_hash = self.get_file_hash(file_path)
        cache_path = os.path.join(self.cache_dir, f"{file_hash}.json")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # 验证缓存
                if cache_data.get("file_hash") == file_hash:
                    return cache_data
            except Exception as e:
                print(f"加载缓存失败: {e}")
        
        return None
    
    def get_context_enhanced_subtitle(self, 
                                    subtitle_text: str, 
                                    similar_chunks: List[str], 
                                    key_terms: List[str]) -> str:
        """
        生成增强上下文的字幕文本
        
        Args:
            subtitle_text: 原字幕文本
            similar_chunks: 相似块内容
            key_terms: 关键术语
            
        Returns:
            增强上下文的字幕文本
        """
        context = ""
        
        # 添加相似块内容作为上下文
        if similar_chunks:
            context += "相关上下文:\n"
            for i, chunk in enumerate(similar_chunks[:3]):  # 最多使用3个相似块
                context += f"上下文 {i+1}: {chunk}\n"
        
        # 添加关键术语
        if key_terms:
            context += "\n关键术语:\n"
            context += ", ".join(key_terms)
        
        # 将原文和上下文组合
        enhanced_text = f"{subtitle_text}\n\n{context}"
        
        return enhanced_text
    
    def find_scene_boundaries(self, embeddings: List[List[float]], threshold: float = 0.7) -> List[int]:
        """
        通过向量相似度检测可能的场景边界
        
        Args:
            embeddings: 嵌入向量列表
            threshold: 相似度阈值，低于此值被视为场景边界
            
        Returns:
            可能的场景转换点索引列表
        """
        if not embeddings or len(embeddings) < 2:
            return []
            
        boundaries = []
        
        # 计算相邻向量之间的相似度
        for i in range(1, len(embeddings)):
            prev_embedding = embeddings[i-1]
            curr_embedding = embeddings[i]
            
            # 计算余弦相似度
            similarity = self.compute_similarity(prev_embedding, curr_embedding)
            
            # 如果相似度低于阈值，可能是场景转换
            if similarity < threshold:
                boundaries.append(i)
                
        return boundaries
        
    def find_similar_scenes(self, 
                            chunk_embeddings: List[List[float]], 
                            scene_boundaries: List[int], 
                            similarity_threshold: float = 0.75) -> Dict[int, List[int]]:
        """
        找出字幕中相似的场景，用于提高翻译一致性
        
        Args:
            chunk_embeddings: 块的嵌入向量列表
            scene_boundaries: 场景边界索引列表
            similarity_threshold: 场景相似度阈值
            
        Returns:
            场景映射字典，键为场景索引，值为相似场景索引列表
        """
        print(f"🔍 识别相似场景，以提高翻译一致性...")
        
        # 处理有效边界
        boundaries = sorted(list(set([0] + scene_boundaries + [len(chunk_embeddings)])))
        
        # 提取每个场景的平均向量
        scene_vectors = []
        scene_ranges = []
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i+1]
            
            # 提取当前场景的所有向量
            scene_chunks = chunk_embeddings[start:end]
            if not scene_chunks:
                continue
                
            # 计算平均向量代表整个场景
            scene_vector = np.mean(scene_chunks, axis=0).tolist()
            scene_vectors.append(scene_vector)
            scene_ranges.append((start, end))
        
        # 计算场景间相似度并找出相似场景
        similar_scenes = {}
        for i, scene_i in enumerate(scene_vectors):
            similar_to_i = []
            
            for j, scene_j in enumerate(scene_vectors):
                if i == j:
                    continue
                    
                similarity = self.compute_similarity(scene_i, scene_j)
                if similarity > similarity_threshold:
                    similar_to_i.append(j)
            
            if similar_to_i:
                similar_scenes[i] = similar_to_i
        
        # 转换为块索引的映射
        chunk_similarities = {}
        for scene_idx, similar_scenes_idx in similar_scenes.items():
            start, end = scene_ranges[scene_idx]
            
            # 为当前场景中的每个块建立相似映射
            for chunk_idx in range(start, end):
                similar_chunks = []
                
                # 添加所有相似场景中的块
                for sim_scene_idx in similar_scenes_idx:
                    sim_start, sim_end = scene_ranges[sim_scene_idx]
                    similar_chunks.extend(range(sim_start, sim_end))
                
                chunk_similarities[chunk_idx] = similar_chunks
        
        if chunk_similarities:
            print(f"✅ 找到 {len(chunk_similarities)} 个块具有相似场景")
        else:
            print("⚠️ 未找到明显的相似场景")
            
        return chunk_similarities
    
    def enhance_translation_consistency(self, 
                                       translation_tasks: List[Dict], 
                                       similar_scenes: Dict[int, List[int]],
                                       chunk_mapping: Dict[int, int]) -> List[Dict]:
        """
        增强翻译一致性，为相似场景添加上下文参考
        
        Args:
            translation_tasks: 翻译任务列表
            similar_scenes: 相似场景映射
            chunk_mapping: 字幕索引到块索引的映射
            
        Returns:
            增强后的翻译任务列表
        """
        print("🔄 增强翻译一致性...")
        enhanced_tasks = []
        
        # 整理字幕索引到相似场景的映射
        subtitle_to_similar_chunks = {}
        for subtitle_idx, chunk_idx in chunk_mapping.items():
            if chunk_idx in similar_scenes:
                subtitle_to_similar_chunks[subtitle_idx] = similar_scenes[chunk_idx]
        
        # 为每个任务添加相似场景信息
        for task in translation_tasks:
            task_copy = task.copy()
            
            # 检查任务中的字幕是否有相似场景
            for idx in task["indices"]:
                if idx in subtitle_to_similar_chunks:
                    # 找出任务中需要添加到上下文的相似块
                    similar_chunk_indices = subtitle_to_similar_chunks[idx]
                    context_references = []
                    
                    # 将相似块的信息添加到翻译上下文中
                    for chunk_idx in similar_chunk_indices[:3]:  # 限制最多3个参考块
                        for ref_task in translation_tasks:
                            if any(ref_idx in chunk_mapping and chunk_mapping[ref_idx] == chunk_idx for ref_idx in ref_task["indices"]):
                                context_references.append({
                                    "text": ref_task["text"],
                                    "relevance": "high",
                                    "reason": "相似场景"
                                })
                    
                    # 如果找到相似内容，添加到任务中
                    if context_references:
                        if "context_references" not in task_copy:
                            task_copy["context_references"] = []
                        task_copy["context_references"].extend(context_references)
                        break  # 一个任务只需要添加一次相似场景
            
            enhanced_tasks.append(task_copy)
        
        print(f"✅ 已为 {sum(1 for t in enhanced_tasks if 'context_references' in t)} 个翻译任务增强上下文一致性")
        return enhanced_tasks 