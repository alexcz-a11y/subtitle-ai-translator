import os
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import json
import re

# 加载环境变量
load_dotenv()

class AIService:
    """AI服务接口"""
    
    def __init__(self, embedding_model=None, translation_model=None, analysis_model=None):
        """
        初始化AI服务
        
        Args:
            embedding_model: 嵌入模型名称，如果为None则使用环境变量
            translation_model: 翻译模型名称，如果为None则使用环境变量
            analysis_model: 分析模型名称，如果为None则使用环境变量
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE")
        
        # 优先使用参数指定的模型，否则使用环境变量
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.translation_model = translation_model or os.getenv("TRANSLATION_MODEL", "gpt-3.5-turbo")
        self.analysis_model = analysis_model or os.getenv("ANALYSIS_MODEL", "gpt-3.5-turbo")
        
        if not self.api_key:
            raise ValueError("未设置OPENAI_API_KEY环境变量")
        
        # 增强API性能优化
        if self.api_base:
            self.client = OpenAI(
                api_key=self.api_key, 
                base_url=self.api_base,
                timeout=120,  # 增加超时时间到2分钟，防止大批量请求超时
                max_retries=10,  # 增加重试次数到10次，提高可靠性
                default_headers={"User-Agent": "SubtitleTranslator/1.0"}  # 自定义头部便于API识别
            )
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                timeout=120,
                max_retries=10,
                default_headers={"User-Agent": "SubtitleTranslator/1.0"}
            )
        
        # 用于缓存分析出的术语向量
        self.term_embeddings = {}
        
        # 用于缓存翻译结果，避免重复翻译
        self.translation_cache = {}
        
        # 记录使用的模型
        print(f"🤖 使用嵌入模型: {self.embedding_model}")
        print(f"🤖 使用翻译模型: {self.translation_model}")
        print(f"🤖 使用分析模型: {self.analysis_model}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        max_retries = 5
        retry_delay = 2
        
        # 如果文本超长，分段处理并合并向量
        max_chars = 4000  # 约5000-6000 tokens，足够安全
        if len(text) > max_chars:
            print(f"⚠️ 文本过长 ({len(text)}字符)，将分段处理并合并向量")
            # 分段处理文本
            segments = []
            for i in range(0, len(text), max_chars):
                segments.append(text[i:i+max_chars])
            
            # 获取每段文本的向量
            segment_embeddings = []
            for i, segment in enumerate(segments):
                print(f"正在处理第 {i+1}/{len(segments)} 段文本...")
                for attempt in range(max_retries):
                    try:
                        response = self.client.embeddings.create(
                            model=self.embedding_model,
                            input=segment
                        )
                        segment_embeddings.append(response.data[0].embedding)
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"获取嵌入向量失败，正在重试 ({attempt+1}/{max_retries}): {e}")
                            time.sleep(retry_delay)
                        else:
                            raise ValueError(f"获取嵌入向量失败: {e}")
            
            # 合并向量（简单平均）
            if segment_embeddings:
                combined_embedding = np.mean(segment_embeddings, axis=0)
                return combined_embedding.tolist()
            else:
                raise ValueError("没有成功获取任何段落的嵌入向量")
        
        # 对于正常长度的文本，直接处理
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"获取嵌入向量失败，正在重试 ({attempt+1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                else:
                    raise ValueError(f"获取嵌入向量失败: {e}")
    
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本的嵌入向量
        
        Args:
            texts: 输入文本列表
            
        Returns:
            嵌入向量列表
        """
        max_retries = 5  # 增加重试次数
        retry_delay = 1  # 减少重试延迟
        max_chars = 4000  # 每段最大字符数
        
        # 缓存结果避免重复计算
        cached_results = []
        texts_to_embed = []
        original_indices = []
        
        # 检查哪些文本需要计算向量
        for i, text in enumerate(texts):
            # 使用MD5作为缓存键
            cache_key = f"emb_{hash(text) % 10000000}"
            if cache_key in self.term_embeddings:
                cached_results.append((i, self.term_embeddings[cache_key]))
            else:
                texts_to_embed.append(text)
                original_indices.append(i)
        
        if not texts_to_embed:
            # 所有向量已缓存
            results = [None] * len(texts)
            for i, emb in cached_results:
                results[i] = emb
            return results
            
        # 批量调用API获取向量
        # 将长文本列表分成更小的批次，每批最多20个文本
        all_embeddings = []
        for i in range(0, len(texts_to_embed), 20):
            batch = texts_to_embed[i:i+20]
            processed_batch = []
            
            # 处理每个文本，确保不超过长度限制
            for text in batch:
                if len(text) > max_chars:
                    # 分段处理超长文本
                    segments = []
                    for j in range(0, len(text), max_chars):
                        segments.append(text[j:j+max_chars])
                    
                    # 获取每段的向量
                    segment_embeddings = []
                    for segment in segments:
                        for attempt in range(max_retries):
                            try:
                                response = self.client.embeddings.create(
                                    model=self.embedding_model,
                                    input=segment
                                )
                                segment_embeddings.append(response.data[0].embedding)
                                break
                            except Exception as e:
                                if attempt < max_retries - 1:
                                    print(f"获取嵌入向量失败，正在重试: {e}")
                                    time.sleep(retry_delay)
                                else:
                                    raise ValueError(f"获取嵌入向量失败: {e}")
                    
                    # 合并段落向量
                    if segment_embeddings:
                        combined_embedding = np.mean(segment_embeddings, axis=0)
                        all_embeddings.append(combined_embedding.tolist())
                    else:
                        # 如果没有获取到任何向量，使用零向量
                        raise ValueError("没有成功获取任何段落的嵌入向量")
                else:
                    processed_batch.append(text)
            
            # 对正常长度的文本批量处理
            if processed_batch:
                for attempt in range(max_retries):
                    try:
                        response = self.client.embeddings.create(
                            model=self.embedding_model,
                            input=processed_batch
                        )
                        all_embeddings.extend([item.embedding for item in response.data])
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"获取批量嵌入向量失败，正在重试 ({attempt+1}/{max_retries}): {e}")
                            time.sleep(retry_delay)
                        else:
                            raise ValueError(f"获取批量嵌入向量失败: {e}")
        
        # 合并结果
        results = [None] * len(texts)
        
        # 添加已缓存的结果
        for i, emb in cached_results:
            results[i] = emb
        
        # 添加新计算的结果并缓存
        for i, idx in enumerate(original_indices):
            if i < len(all_embeddings):  # 防止索引越界
                results[idx] = all_embeddings[i]
                
                # 缓存结果
                cache_key = f"emb_{hash(texts[idx]) % 10000000}"
                self.term_embeddings[cache_key] = all_embeddings[i]
        
        return results
    
    def analyze_content(self, text, max_length=15000):
        """分析字幕内容，识别类型、主题和关键术语
        
        Args:
            text: 字幕文本
            max_length: 最大处理文本长度
            
        Returns:
            Dict: 内容分析结果
        """
        # 限制文本长度
        text = text[:max_length]
        
        try:
            prompt = """分析以下字幕文本，并提供以下信息：
1. 影片类型/风格（如动作、喜剧、恐怖、科幻等）
2. 剧情简要概述
3. 关键人物及其特点
4. 出现的专业术语或特殊词汇
5. 情感基调

请以JSON格式返回，使用以下结构：
{
  "genre": "电影类型",
  "plot_summary": "剧情概述",
  "characters": ["角色1", "角色2"],
  "terminology": ["术语1", "术语2"],
  "tone": "情感基调"
}

只返回JSON结构，不要添加其他解释。"""
            
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {"role": "system", "content": "你是一位专业的影视内容分析专家，擅长从字幕中分析影片内容。"},
                    {"role": "user", "content": f"{prompt}\n\n字幕文本:\n{text}"}
                ],
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            
            # 解析JSON结果
            try:
                analysis = json.loads(result)
                return analysis
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
                # 返回基本结构
                return {
                    "genre": "未知",
                    "plot_summary": "无法解析剧情",
                    "characters": [],
                    "terminology": [],
                    "tone": "未知"
                }
                
        except Exception as e:
            print(f"内容分析失败: {e}")
            raise
    
    def build_terminology_embeddings(self, content_analysis: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        为分析出的术语生成嵌入向量
        
        Args:
            content_analysis: 内容分析结果
            
        Returns:
            术语及其向量的字典
        """
        # 如果没有术语，返回空字典
        if not content_analysis or "terminology" not in content_analysis:
            return {}
        
        terminology = content_analysis["terminology"]
        terms = list(terminology.keys())
        
        if not terms:
            return {}
        
        try:
            # 批量获取术语的嵌入向量
            term_embeddings = {}
            batch_embeddings = self.get_batch_embeddings(terms)
            
            for i, term in enumerate(terms):
                term_embeddings[term] = batch_embeddings[i]
            
            # 缓存结果
            self.term_embeddings = term_embeddings
            return term_embeddings
        except Exception as e:
            print(f"警告: 生成术语向量失败: {e}")
            return {}
    
    def translate_subtitle(self, 
                          subtitle_text: str,
                          target_language: str = "zh-CN",
                          content_analysis: Optional[Dict[str, Any]] = None) -> str:
        """
        翻译字幕文本
        
        Args:
            subtitle_text: 字幕文本
            target_language: 目标语言
            content_analysis: 内容分析结果
            
        Returns:
            翻译后的文本
        """
        context = ""
        if content_analysis:
            context = f"""
            电影类型与主题: {content_analysis.get('genre', '')}
            主要内容: {content_analysis.get('content', '')}
            专业术语: {content_analysis.get('terminology', {})}
            人物关系: {content_analysis.get('characters', {})}
            特殊语境: {content_analysis.get('context', '')}
            """
        
        system_prompt = f"""
        你是一个专业的字幕翻译专家，请将以下字幕翻译成{target_language}。
        
        翻译指南:
        1. 保持原意的同时，使译文自然流畅，符合目标语言习惯
        2. 根据影片类型和场景调整用词和语气
        3. 保留专业术语的准确性
        4. 注意人物对话的语气和个性
        5. 考虑文化差异，适当本地化
        
        以下是关于影片的背景信息，请在翻译时参考:
        {context}
        
        请只返回翻译后的文本，不要添加任何其他内容。
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.translation_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": subtitle_text}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"翻译失败: {e}")
    
    def translate_with_context(self, subtitle_text, target_language="zh-CN", content_analysis=None, similar_chunks=None, relevant_terms=None):
        """使用上下文翻译字幕
        
        Args:
            subtitle_text: 字幕文本
            target_language: 目标语言
            content_analysis: 内容分析结果
            similar_chunks: 相似文本块
            relevant_terms: 相关术语
            
        Returns:
            翻译后的文本
        """
        # 如果字幕文本为空，直接返回
        if not subtitle_text or subtitle_text.strip() == "":
            return subtitle_text

        # 构建上下文
        context = ""
        
        # 添加内容分析上下文（丰富处理以提高传递给模型的信息量）
        if content_analysis:
            # 为翻译提供更详细的内容背景
            if 'genre' in content_analysis:
                context += f"类型: {content_analysis['genre']}\n"
            
            if 'plot_summary' in content_analysis:
                context += f"剧情概要: {content_analysis['plot_summary']}\n"
            
            if 'main_characters' in content_analysis:
                # 提供主要角色信息以保持人物称呼一致性
                characters = content_analysis['main_characters']
                if characters and len(characters) > 0:
                    character_text = ", ".join([f"{name}" for name in characters])
                    context += f"主要角色: {character_text}\n"
            
            # 确保术语翻译一致
            if 'terminology' in content_analysis:
                terms = content_analysis['terminology']
                if terms and len(terms) > 0:
                    term_text = ", ".join([f"{term}" for term in terms])
                    context += f"术语: {term_text}\n"
            
            # 提供情感和语气信息以保持翻译风格
            if 'tone' in content_analysis:
                context += f"语气: {content_analysis['tone']}\n"
                
            # 添加剧情设定和场景描述（如果有）
            if 'setting' in content_analysis:
                context += f"场景设定: {content_analysis['setting']}\n"
        
        # 系统提示，强调翻译准确性和自然流畅
        system_prompt = f"""你是一位专业电影字幕翻译专家，精通{target_language}和英语。
请将英文字幕翻译成{target_language}，遵循以下原则：
1. 准确性：保持原意准确传达，包括专业术语、习语、文化特定表达和双关语
2. 自然流畅：确保译文符合{target_language}的语言习惯和表达方式
3. 风格匹配：保持原作的风格、语气和语域特征
4. 情感表达：准确传达原文的情感和语气，包括愤怒、幽默、讽刺等
5. 文化适应：适当调整文化特定内容，使目标语言观众能理解
6. 保留粗口和俚语：准确翻译粗口、脏话和俚语，保持原作的语气和强度
7. 简洁性：字幕应简洁明了，便于观众快速阅读

注意，电影字幕翻译与普通文本翻译不同，要考虑观看体验和视听一致性。
"""

        # 处理字幕文本，去除序号和时间戳
        clean_text = re.sub(r'^\d+\s+\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}\s+', '', subtitle_text, flags=re.MULTILINE)
        
        # 增加相似段落上下文，优先考虑紧邻的对话
        similar_context = ""
        if similar_chunks and len(similar_chunks) > 0:
            # 增加相似块数量，最多使用8个（原来是3个）
            max_chunks = min(8, len(similar_chunks))
            similar_text = "\n".join([chunk for chunk in similar_chunks[:max_chunks]])
            similar_context = f"相关对话上下文:\n{similar_text}\n"
        
        # 增加相关术语上下文
        terms_context = ""
        if relevant_terms and len(relevant_terms) > 0:
            # 增加最大相关术语数量
            max_terms = min(10, len(relevant_terms))
            terms_text = ", ".join([term for term in relevant_terms[:max_terms]])
            terms_context = f"相关术语: {terms_text}\n"
            
        # 构建最终用户提示
        user_prompt = f"""请翻译以下电影字幕到{target_language}，保持语气、幽默感和情感表达。

### 内容背景 ###
{context}

{similar_context}
{terms_context}

### 要翻译的字幕 ###
{clean_text}

只需返回译文，无需解释。保留原文的段落格式和换行。保持专业电影字幕风格，符合目标语言的表达习惯。
对于粗口、俚语等情感强烈的表达，请保持原有的语气强度和表达效果。"""

        # 调用API进行翻译 - 增加多次重试机制
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 重试次数
        max_retries = 5
        retry_count = 0
        retry_delay = 2  # 初始延迟2秒
        
        # 主要翻译方法 - 多次重试
        while retry_count < max_retries:
            try:
                # 优化API调用参数，提高翻译质量
                response = self.client.chat.completions.create(
                    model=self.translation_model,
                    messages=messages,
                    temperature=0.1,  # 降低温度，提高一致性
                    timeout=90 + retry_count * 30,  # 随着重试次数增加超时时间
                    max_tokens=4000  # 增加最大令牌数，确保完整输出
                )
                
                # 安全提取翻译结果 - 增强对None的检查
                translation = ""
                if (response and hasattr(response, 'choices') and 
                    len(response.choices) > 0 and 
                    hasattr(response.choices[0], 'message') and 
                    hasattr(response.choices[0].message, 'content') and 
                    response.choices[0].message.content is not None):
                    
                    translation = response.choices[0].message.content.strip()
                
                # 检查翻译结果是否为空
                if translation:
                    if retry_count > 0:
                        print(f"✅ 第{retry_count+1}次尝试成功翻译")
                    return translation
                else:
                    print(f"⚠️ 警告: 第{retry_count+1}次尝试翻译结果为空")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"🔄 等待{retry_delay}秒后重试...")
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 1.5, 10)  # 指数退避，最大延迟10秒
                    continue
                    
            except Exception as e:
                print(f"⚠️ 第{retry_count+1}次翻译失败: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"🔄 等待{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.5, 10)  # 指数退避，最大延迟10秒
                else:
                    print("❌ 主要翻译方法多次尝试失败，切换至备用方法")
                    break
        
        # 如果主要翻译方法多次尝试都失败，使用备用翻译方法
        print("ℹ️ 使用备用翻译方法...")
        
        # 备用翻译方法 - 使用更简单的提示
        try:
            # 使用更简单的提示进行翻译尝试
            simple_messages = [
                {"role": "system", "content": f"将以下英文字幕翻译成{target_language}，保持原意和风格。"},
                {"role": "user", "content": clean_text}
            ]
            
            fallback_response = self.client.chat.completions.create(
                model=self.translation_model,
                messages=simple_messages,
                temperature=0.2,
                timeout=60
            )
            
            # 安全提取备用翻译结果
            simple_translation = ""
            if (fallback_response and hasattr(fallback_response, 'choices') and 
                len(fallback_response.choices) > 0 and 
                hasattr(fallback_response.choices[0], 'message') and 
                hasattr(fallback_response.choices[0].message, 'content') and 
                fallback_response.choices[0].message.content is not None):
                
                simple_translation = fallback_response.choices[0].message.content.strip()
            
            if simple_translation:
                print("✅ 使用备用方法成功翻译")
                return simple_translation
            else:
                print("⚠️ 备用翻译结果为空")
            
        except Exception as fallback_error:
            print(f"⚠️ 备用翻译也失败: {str(fallback_error)}")
        
        # 如果所有尝试都失败，返回原文
        print("⚠️ 所有翻译方法都失败，返回原文")
        return clean_text
    
    def translate_subtitle_chunk(self, 
                               text: str, 
                               target_language: str, 
                               content_analysis: Dict = None, 
                               similar_chunks: List[str] = None,
                               context_references: List[Dict] = None) -> str:
        """
        翻译字幕文本块
        
        Args:
            text: 要翻译的文本
            target_language: 目标语言代码
            content_analysis: 内容分析结果
            similar_chunks: 相似文本块
            context_references: 上下文参考，用于提高翻译一致性
            
        Returns:
            翻译后的文本
        """
        # 构建翻译提示
        system_prompt = f"""你是一位专业的影视字幕翻译专家，精通各种语言之间的字幕翻译。
你的任务是将下面的字幕文本翻译成{target_language}。

翻译准则：
1. 保持原文的风格、语气和表达方式
2. 适当本地化表达，使译文流畅自然
3. 保留专业术语和人名的准确性
4. 保持字幕的简洁，适合观众快速阅读
5. 注意语境，确保翻译符合影片整体风格
6. 保留原文中的标点符号格式
7. 保持专业术语的一致性
8. 如果出现多人对话，保持对话结构并适当标记

输出标准：
- 只返回翻译后的文本，不要添加任何解释或注释
- 不需要解释你的翻译选择
- 格式应与原文格式一致"""

        # 添加内容分析信息增强翻译上下文
        if content_analysis:
            system_prompt += f"""

影片背景信息：
- 影片类型：{content_analysis.get('film_type', '未知')}
- 剧情摘要：{content_analysis.get('plot_summary', '无')}"""

        # 如果有关键术语，添加术语表
        if "key_terms" in content_analysis and content_analysis["key_terms"]:
            system_prompt += "\n\n专业术语表（请在翻译中保持一致性）："
            for term in content_analysis["key_terms"]:
                system_prompt += f"\n- {term}"

        # 添加相似场景的上下文参考，增强翻译一致性
        if context_references and len(context_references) > 0:
            system_prompt += "\n\n相似场景参考（用于保持翻译风格和术语一致性）："
            for i, ref in enumerate(context_references[:3]):  # 最多使用3个参考
                system_prompt += f"\n参考{i+1}：\n```\n{ref['text']}\n```"
            system_prompt += "\n\n请特别注意与上述相似场景保持风格和术语的一致性。"
        # 添加相似块信息
        elif similar_chunks and len(similar_chunks) > 0:
            system_prompt += "\n\n相似内容参考（用于保持上下文一致性）："
            for i, chunk in enumerate(similar_chunks[:3]):  # 最多使用3个相似块
                system_prompt += f"\n内容{i+1}：\n```\n{chunk}\n```"
                
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.translation_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"请将以下字幕翻译成{target_language}：\n\n{text}"}
                    ],
                    temperature=0.3,  # 降低随机性，提高一致性
                    timeout=120,  # 2分钟超时
                    max_tokens=4000
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"翻译失败，正在重试 ({attempt+1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                else:
                    print(f"翻译失败: {e}")
                    return text  # 失败时返回原文
    
    def extract_terms_from_analysis(self, content_analysis: Dict[str, Any]) -> List[str]:
        """
        从分析结果中提取专业术语
        
        Args:
            content_analysis: 内容分析结果
            
        Returns:
            术语列表
        """
        if not content_analysis or "terminology" not in content_analysis:
            return []
        
        return list(content_analysis["terminology"].keys())
    
    def summarize_text(self, text: str, max_length: int = 50) -> str:
        """
        使用AI模型压缩或总结文本内容
        
        Args:
            text: 需要压缩的文本
            max_length: 压缩后的最大长度（字符数）
            
        Returns:
            压缩后的文本
        """
        if not text or len(text) <= max_length:
            return text
            
        try:
            system_prompt = f"你是一个字幕优化专家，需要将以下文本压缩到{max_length}个字符以内，同时保持原意。保持原文的风格和语气。"
            
            response = self.client.chat.completions.create(
                model=self.translation_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"原文：{text}\n\n压缩后（不超过{max_length}字符）："}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            summary = response.choices[0].message.content.strip()
            
            # 确保结果不超过最大长度
            if len(summary) > max_length:
                summary = summary[:max_length]
                
            return summary
            
        except Exception as e:
            print(f"⚠️ 文本压缩失败: {e}")
            # 如果AI压缩失败，则进行简单的截断
            return text[:max_length]
            
    def translate_text(self, text: str, target_lang: str = "zh-CN") -> str:
        """
        简单翻译文本，不包含上下文信息
        
        Args:
            text: 要翻译的文本
            target_lang: 目标语言
            
        Returns:
            翻译后的文本
        """
        try:
            system_prompt = f"你是一名专业的字幕翻译专家，请将以下文本翻译成{target_lang}。保持原意，同时使翻译自然流畅。"
            
            response = self.client.chat.completions.create(
                model=self.translation_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"原文：{text}\n\n译文："}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            translation = response.choices[0].message.content.strip()
            return translation
            
        except Exception as e:
            print(f"⚠️ 文本翻译失败: {e}")
            return "" 