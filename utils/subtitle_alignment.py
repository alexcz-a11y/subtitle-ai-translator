import re
from typing import List, Dict, Tuple, Optional
import pysrt
from .subtitle_parser import Subtitle, SubtitleParser
from .ai_service import AIService

class SubtitleAligner:
    """字幕校准工具，确保翻译字幕与源字幕精确对应并优化排版"""
    
    def __init__(self, ai_service: AIService):
        """
        初始化字幕校准器
        
        Args:
            ai_service: AI服务实例，用于分析对话场景和优化排版
        """
        self.ai_service = ai_service
        
    def align_subtitles(self, 
                         source_parser: SubtitleParser, 
                         translated_parser: SubtitleParser) -> SubtitleParser:
        """
        校准翻译字幕与源字幕，确保时间码和内容精确对应
        
        Args:
            source_parser: 源字幕解析器
            translated_parser: 翻译后字幕解析器
            
        Returns:
            校准后的字幕解析器对象
        """
        print("🔄 开始字幕校准...")
        
        # 1. 确保字幕数量一致
        if len(source_parser.subtitles) != len(translated_parser.subtitles):
            print(f"⚠️ 警告: 源字幕({len(source_parser.subtitles)}条)和译文字幕({len(translated_parser.subtitles)}条)数量不一致")
            # 如果数量不一致，根据索引和时间码进行匹配
            self._fix_subtitle_count_mismatch(source_parser, translated_parser)
        
        # 2. 复制源字幕的时间码到翻译字幕
        for i, source_sub in enumerate(source_parser.subtitles):
            if i < len(translated_parser.subtitles):
                translated_parser.subtitles[i].start = source_sub.start
                translated_parser.subtitles[i].end = source_sub.end
                translated_parser.subtitles[i].index = source_sub.index
        
        # 3. 分析对话场景并优化排版
        self._optimize_subtitle_layout(source_parser, translated_parser)
        
        print("✅ 字幕校准完成")
        return translated_parser
    
    def _fix_subtitle_count_mismatch(self, source_parser: SubtitleParser, translated_parser: SubtitleParser) -> None:
        """
        修复源字幕和翻译字幕数量不匹配的问题
        
        Args:
            source_parser: 源字幕解析器
            translated_parser: 翻译字幕解析器
        """
        print("🔄 正在修复字幕数量不匹配问题...")
        
        # 方法1: 如果翻译字幕数量少于源字幕，补充缺失的字幕
        if len(translated_parser.subtitles) < len(source_parser.subtitles):
            for i in range(len(translated_parser.subtitles), len(source_parser.subtitles)):
                source_sub = source_parser.subtitles[i]
                # 创建一个新的翻译字幕，暂时使用源字幕的文本
                translated_parser.subtitles.append(Subtitle(
                    index=source_sub.index,
                    start=source_sub.start,
                    end=source_sub.end,
                    text="[未翻译]"  # 标记为未翻译
                ))
            
            # 使用AI翻译这些缺失的字幕
            self._translate_missing_subtitles(source_parser, translated_parser)
        
        # 方法2: 如果翻译字幕数量多于源字幕，合并或删除多余的字幕
        elif len(translated_parser.subtitles) > len(source_parser.subtitles):
            # 使用AI决定如何合并多余的字幕
            self._merge_excess_subtitles(source_parser, translated_parser)
    
    def _translate_missing_subtitles(self, source_parser: SubtitleParser, translated_parser: SubtitleParser) -> None:
        """
        翻译缺失的字幕
        
        Args:
            source_parser: 源字幕解析器
            translated_parser: 翻译字幕解析器
        """
        missing_indices = []
        missing_texts = []
        
        # 收集所有标记为未翻译的字幕
        for i, sub in enumerate(translated_parser.subtitles):
            if sub.text == "[未翻译]":
                missing_indices.append(i)
                source_text = source_parser.subtitles[i].text
                missing_texts.append(source_text)
        
        if missing_indices:
            print(f"🔄 正在翻译{len(missing_indices)}条缺失的字幕...")
            
            # 批量翻译缺失的字幕
            translations = []
            for text in missing_texts:
                # 使用与主翻译相同的上下文信息进行翻译
                translation = self.ai_service.translate_text(text, "zh-CN")
                translations.append(translation)
            
            # 更新缺失的字幕
            for idx, trans in zip(missing_indices, translations):
                translated_parser.subtitles[idx].text = trans
            
            print(f"✅ 已完成{len(missing_indices)}条缺失字幕的翻译")
    
    def _merge_excess_subtitles(self, source_parser: SubtitleParser, translated_parser: SubtitleParser) -> None:
        """
        合并或删除多余的翻译字幕
        
        Args:
            source_parser: 源字幕解析器
            translated_parser: 翻译字幕解析器
        """
        print(f"🔄 正在处理{len(translated_parser.subtitles) - len(source_parser.subtitles)}条多余的字幕...")
        
        # 创建一个新的字幕列表，数量与源字幕一致
        new_subtitles = []
        
        # 为每个源字幕找到最匹配的翻译字幕
        for i, source_sub in enumerate(source_parser.subtitles):
            # 简单策略：优先选择时间码重叠最多的翻译字幕
            best_match_idx = self._find_best_matching_subtitle(source_sub, translated_parser.subtitles)
            
            if best_match_idx is not None:
                matched_sub = translated_parser.subtitles[best_match_idx]
                new_subtitles.append(Subtitle(
                    index=source_sub.index,
                    start=source_sub.start,
                    end=source_sub.end,
                    text=matched_sub.text
                ))
            else:
                # 如果没有找到匹配，使用AI翻译
                trans_text = self.ai_service.translate_text(source_sub.text, "zh-CN")
                new_subtitles.append(Subtitle(
                    index=source_sub.index,
                    start=source_sub.start,
                    end=source_sub.end,
                    text=trans_text
                ))
        
        # 更新翻译字幕解析器中的字幕列表
        translated_parser.subtitles = new_subtitles
        print("✅ 多余字幕处理完成")
    
    def _find_best_matching_subtitle(self, source_sub: Subtitle, translated_subs: List[Subtitle]) -> Optional[int]:
        """
        找到与源字幕最匹配的翻译字幕
        
        Args:
            source_sub: 源字幕
            translated_subs: 翻译字幕列表
            
        Returns:
            最匹配的字幕索引，如果没有找到则返回None
        """
        best_match_idx = None
        max_overlap = 0
        
        source_start = self._time_to_seconds(source_sub.start)
        source_end = self._time_to_seconds(source_sub.end)
        
        for i, trans_sub in enumerate(translated_subs):
            trans_start = self._time_to_seconds(trans_sub.start)
            trans_end = self._time_to_seconds(trans_sub.end)
            
            # 计算时间重叠
            overlap_start = max(source_start, trans_start)
            overlap_end = min(source_end, trans_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match_idx = i
        
        return best_match_idx
    
    def _time_to_seconds(self, time_str: str) -> float:
        """
        将时间字符串转换为秒数
        
        Args:
            time_str: 时间字符串，格式为 HH:MM:SS,mmm
            
        Returns:
            对应的秒数
        """
        # 处理不同的时间格式
        time_str = str(time_str).replace(',', '.').replace(';', '.')
        
        # 匹配时间格式
        match = re.match(r'(\d+):(\d+):(\d+)[.,](\d+)', time_str)
        if match:
            hours, minutes, seconds, milliseconds = map(int, match.groups())
            return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
        
        # 尝试匹配没有小时的格式 MM:SS,mmm
        match = re.match(r'(\d+):(\d+)[.,](\d+)', time_str)
        if match:
            minutes, seconds, milliseconds = map(int, match.groups())
            return minutes * 60 + seconds + milliseconds / 1000
        
        return 0
    
    def _optimize_subtitle_layout(self, source_parser: SubtitleParser, translated_parser: SubtitleParser) -> None:
        """
        根据对话场景优化字幕排版
        
        Args:
            source_parser: 源字幕解析器
            translated_parser: 翻译字幕解析器
        """
        print("🔄 正在根据对话场景优化字幕排版...")
        
        # 1. 识别对话场景
        scenes = self._identify_dialogue_scenes(source_parser)
        
        # 2. 根据场景优化字幕排版
        for scene in scenes:
            start_idx, end_idx = scene["start_idx"], scene["end_idx"]
            scene_type = scene["type"]
            
            # 根据场景类型应用不同的排版规则
            if scene_type == "dialog":
                self._optimize_dialog_scene(translated_parser, start_idx, end_idx)
            elif scene_type == "monologue":
                self._optimize_monologue_scene(translated_parser, start_idx, end_idx)
            elif scene_type == "action":
                self._optimize_action_scene(translated_parser, start_idx, end_idx)
        
        print("✅ 字幕排版优化完成")
    
    def _identify_dialogue_scenes(self, parser: SubtitleParser) -> List[Dict]:
        """
        识别对话场景，将字幕分为不同类型的场景
        
        Args:
            parser: 字幕解析器
            
        Returns:
            场景列表，每个场景包含起始索引、结束索引和场景类型
        """
        scenes = []
        current_scene = {"start_idx": 0, "type": "unknown"}
        
        for i in range(1, len(parser.subtitles)):
            prev_sub = parser.subtitles[i-1]
            curr_sub = parser.subtitles[i]
            
            # 检测场景切换
            time_gap = self._time_to_seconds(curr_sub.start) - self._time_to_seconds(prev_sub.end)
            
            # 如果时间间隔大于2秒，可能是场景切换
            if time_gap > 2.0:
                # 结束当前场景
                current_scene["end_idx"] = i - 1
                
                # 确定场景类型
                scene_text = " ".join([parser.subtitles[j].text for j in range(current_scene["start_idx"], current_scene["end_idx"] + 1)])
                current_scene["type"] = self._determine_scene_type(scene_text)
                
                scenes.append(current_scene)
                
                # 开始新场景
                current_scene = {"start_idx": i, "type": "unknown"}
        
        # 处理最后一个场景
        current_scene["end_idx"] = len(parser.subtitles) - 1
        scene_text = " ".join([parser.subtitles[j].text for j in range(current_scene["start_idx"], current_scene["end_idx"] + 1)])
        current_scene["type"] = self._determine_scene_type(scene_text)
        scenes.append(current_scene)
        
        return scenes
    
    def _determine_scene_type(self, scene_text: str) -> str:
        """
        确定场景类型：对话、独白或动作描述
        
        Args:
            scene_text: 场景文本
            
        Returns:
            场景类型: "dialog", "monologue" 或 "action"
        """
        # 简单启发式规则
        if ":" in scene_text or "-" in scene_text:
            return "dialog"  # 包含对话标记，可能是对话场景
        elif len(scene_text.split()) > 50:
            return "monologue"  # 文本较长，可能是独白
        else:
            return "action"  # 默认为动作描述
    
    def _optimize_dialog_scene(self, parser: SubtitleParser, start_idx: int, end_idx: int) -> None:
        """
        优化对话场景的字幕排版
        
        Args:
            parser: 字幕解析器
            start_idx: 起始字幕索引
            end_idx: 结束字幕索引
        """
        for i in range(start_idx, end_idx + 1):
            sub = parser.subtitles[i]
            
            # 对话优化规则
            # 1. 添加对话前缀 (-) 如果需要
            if not sub.text.startswith("-") and not sub.text.startswith("—") and ":" not in sub.text:
                # 确保文本不是空的
                if sub.text.strip():
                    sub.text = f"- {sub.text}"
            
            # 2. 处理多行对话
            lines = sub.text.split("\n")
            if len(lines) > 1:
                processed_lines = []
                for line in lines:
                    # 跳过空的对话行
                    if not line.strip() or line.strip() == "-" or line.strip() == "— " or line.strip() == "- ":
                        continue
                    
                    if line.strip() and not line.startswith("-") and not line.startswith("—") and ":" not in line:
                        processed_lines.append(f"- {line}")
                    else:
                        # 确保添加的行不是空的带破折号的行
                        if line.strip() and not (line.strip() == "-" or line.strip() == "— " or line.strip() == "- "):
                            processed_lines.append(line)
                
                # 确保至少有一行内容
                if processed_lines:
                    sub.text = "\n".join(processed_lines)
                else:
                    # 如果处理后没有内容，使用简单标记而不是空行
                    sub.text = "(无对话内容)"
    
    def _optimize_monologue_scene(self, parser: SubtitleParser, start_idx: int, end_idx: int) -> None:
        """
        优化独白场景的字幕排版
        
        Args:
            parser: 字幕解析器
            start_idx: 起始字幕索引
            end_idx: 结束字幕索引
        """
        for i in range(start_idx, end_idx + 1):
            sub = parser.subtitles[i]
            
            # 独白优化规则
            # 1. 移除不必要的对话标记
            if sub.text.startswith("- ") or sub.text.startswith("— "):
                sub.text = sub.text[2:]
            
            # 2. 处理中文引号
            sub.text = sub.text.replace('"', '"').replace('"', '"')
    
    def _optimize_action_scene(self, parser: SubtitleParser, start_idx: int, end_idx: int) -> None:
        """
        优化动作描述场景的字幕排版
        
        Args:
            parser: 字幕解析器
            start_idx: 起始字幕索引
            end_idx: 结束字幕索引
        """
        for i in range(start_idx, end_idx + 1):
            sub = parser.subtitles[i]
            
            # 动作描述优化规则
            # 1. 添加括号标记动作描述
            if not (sub.text.startswith("(") and sub.text.endswith(")")):
                # 检查是否已经有其他动作标记
                if not (sub.text.startswith("[") and sub.text.endswith("]")) and \
                   not (sub.text.startswith("{") and sub.text.endswith("}")):
                    sub.text = f"({sub.text})"
            
            # 2. 适当缩短过长的动作描述
            if len(sub.text) > 40:
                # 使用AI服务压缩长动作描述
                compressed_text = self.ai_service.summarize_text(sub.text, max_length=40)
                if compressed_text and len(compressed_text) < len(sub.text):
                    sub.text = compressed_text 