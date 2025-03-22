import pysrt
import re
import os
import codecs
from typing import List, Dict, Any, Tuple

class Subtitle:
    """字幕对象"""
    def __init__(self, index, start, end, text):
        self.index = index
        self.start = start
        self.end = end
        self.text = text


class SubtitleParser:
    """字幕解析工具类"""
    
    def __init__(self, file_path: str):
        """
        初始化字幕解析器
        
        Args:
            file_path: 字幕文件路径
        """
        self.file_path = file_path
        self.subtitles = []
        self.text_content = ""
        self.encoding = self._detect_encoding()
        self.load_subtitles()
    
    def _detect_encoding(self) -> str:
        """检测文件编码"""
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'big5', 'latin1']
        for encoding in encodings:
            try:
                with codecs.open(self.file_path, 'r', encoding=encoding) as f:
                    f.read()
                    return encoding
            except UnicodeDecodeError:
                continue
        return 'utf-8'  # 默认使用utf-8
    
    def load_subtitles(self) -> None:
        """加载字幕文件"""
        try:
            # 先尝试使用pysrt加载
            srt_subtitles = pysrt.open(self.file_path, encoding=self.encoding)
            self.subtitles = []
            for sub in srt_subtitles:
                self.subtitles.append(Subtitle(
                    index=sub.index,
                    start=sub.start,
                    end=sub.end,
                    text=sub.text
                ))
        except Exception as e:
            # 如果pysrt加载失败，尝试手动解析
            print(f"pysrt加载失败，尝试手动解析: {e}")
            self._parse_manually()
        
        if not self.subtitles:
            raise ValueError(f"无法加载字幕文件: {self.file_path}")
        
        # 提取纯文本内容
        self.text_content = "\n".join([sub.text for sub in self.subtitles])
        # 清理HTML标签
        self.text_content = re.sub(r'<[^>]+>', '', self.text_content)
    
    def _parse_manually(self) -> None:
        """手动解析SRT格式"""
        with codecs.open(self.file_path, 'r', encoding=self.encoding) as f:
            content = f.read()
        
        # 分割字幕块
        subtitle_blocks = re.split(r'\n\s*\n', content.strip())
        for block in subtitle_blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            
            try:
                # 提取索引
                index = int(lines[0].strip())
                
                # 提取时间码
                time_line = lines[1]
                time_match = re.search(r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})', time_line)
                if not time_match:
                    continue
                
                start_time = time_match.group(1).replace(',', '.')
                end_time = time_match.group(2).replace(',', '.')
                
                # 提取文本
                text = '\n'.join(lines[2:])
                
                self.subtitles.append(Subtitle(
                    index=index,
                    start=start_time,
                    end=end_time,
                    text=text
                ))
            except Exception as e:
                print(f"解析字幕块失败: {e} - {block}")
    
    def get_all_text(self) -> str:
        """获取所有字幕文本"""
        return self.text_content
    
    def get_subtitle_chunks(self, chunk_size: int = 10) -> List[str]:
        """
        将字幕分块以便进行处理
        
        Args:
            chunk_size: 每块的字幕条数
            
        Returns:
            字幕块列表
        """
        chunks = []
        total_subs = len(self.subtitles)
        
        for i in range(0, total_subs, chunk_size):
            end_idx = min(i + chunk_size, total_subs)
            chunk_text = "\n".join([
                f"{sub.index}. [{sub.start} --> {sub.end}] {sub.text}"
                for sub in self.subtitles[i:end_idx]
            ])
            chunks.append(chunk_text)
        
        return chunks
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        获取字幕文件的元数据
        
        Returns:
            包含元数据的字典
        """
        return {
            "filename": os.path.basename(self.file_path),
            "subtitle_count": len(self.subtitles),
            "duration": str(self.subtitles[-1].end) if self.subtitles else "00:00:00,000",
            "first_line": self.subtitles[0].text if self.subtitles else "",
            "encoding": self.encoding
        }
    
    def save_translated_subtitles(self, translations: List[str], output_path: str) -> None:
        """
        保存翻译后的字幕
        
        Args:
            translations: 翻译后的文本列表，与字幕条目一一对应
            output_path: 输出文件路径
        """
        if len(translations) != len(self.subtitles):
            raise ValueError(f"翻译数量({len(translations)})与字幕条数({len(self.subtitles)})不匹配")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, sub in enumerate(self.subtitles):
                    # 写入字幕序号
                    f.write(f"{sub.index}\n")
                    
                    # 写入时间码，保持原始格式
                    time_format = f"{sub.start} --> {sub.end}\n"
                    f.write(time_format)
                    
                    # 写入翻译文本
                    f.write(f"{translations[i]}\n\n")
            
            print(f"字幕已保存至: {output_path}")
        except Exception as e:
            raise ValueError(f"保存字幕失败: {e}")
            
    def save(self, output_path: str) -> None:
        """
        保存字幕到文件
        
        Args:
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for sub in self.subtitles:
                    # 写入字幕序号
                    f.write(f"{sub.index}\n")
                    
                    # 写入时间码，保持原始格式
                    time_format = f"{sub.start} --> {sub.end}\n"
                    f.write(time_format)
                    
                    # 写入文本
                    f.write(f"{sub.text}\n\n")
            
            print(f"字幕已保存至: {output_path}")
        except Exception as e:
            raise ValueError(f"保存字幕失败: {e}") 