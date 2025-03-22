#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
字幕翻译工具测试脚本
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pysrt

from utils.subtitle_parser import SubtitleParser
from utils.ai_service import AIService


class TestSubtitleParser(unittest.TestCase):
    """测试字幕解析器"""
    
    def setUp(self):
        """准备测试环境"""
        # 创建临时字幕文件
        self.temp_srt = tempfile.NamedTemporaryFile(suffix=".srt", delete=False)
        
        # 写入测试数据
        test_content = """1
00:00:01,000 --> 00:00:04,000
Hello, world!

2
00:00:05,000 --> 00:00:08,000
This is a test subtitle.

3
00:00:09,000 --> 00:00:12,000
How are you today?
"""
        self.temp_srt.write(test_content.encode('utf-8'))
        self.temp_srt.close()
    
    def tearDown(self):
        """清理测试环境"""
        os.unlink(self.temp_srt.name)
    
    def test_load_subtitles(self):
        """测试加载字幕"""
        parser = SubtitleParser(self.temp_srt.name)
        self.assertEqual(len(parser.subtitles), 3)
        self.assertEqual(parser.subtitles[0].text, "Hello, world!")
    
    def test_get_all_text(self):
        """测试获取所有文本"""
        parser = SubtitleParser(self.temp_srt.name)
        all_text = parser.get_all_text()
        self.assertIn("Hello, world!", all_text)
        self.assertIn("This is a test subtitle.", all_text)
        self.assertIn("How are you today?", all_text)
    
    def test_get_metadata(self):
        """测试获取元数据"""
        parser = SubtitleParser(self.temp_srt.name)
        metadata = parser.get_metadata()
        self.assertEqual(metadata["subtitle_count"], 3)
        self.assertEqual(metadata["filename"], os.path.basename(self.temp_srt.name))
        self.assertEqual(metadata["first_line"], "Hello, world!")
    
    def test_save_translated_subtitles(self):
        """测试保存翻译后的字幕"""
        parser = SubtitleParser(self.temp_srt.name)
        translations = ["你好，世界！", "这是一个测试字幕。", "你今天好吗？"]
        
        output_file = tempfile.NamedTemporaryFile(suffix=".srt", delete=False)
        output_file.close()
        
        try:
            parser.save_translated_subtitles(translations, output_file.name)
            
            # 检查输出文件
            translated_subs = pysrt.open(output_file.name)
            self.assertEqual(len(translated_subs), 3)
            self.assertEqual(translated_subs[0].text, "你好，世界！")
            self.assertEqual(translated_subs[1].text, "这是一个测试字幕。")
            self.assertEqual(translated_subs[2].text, "你今天好吗？")
        finally:
            os.unlink(output_file.name)


@patch('openai.Embedding.create')
@patch('openai.ChatCompletion.create')
class TestAIService(unittest.TestCase):
    """测试AI服务"""
    
    def setUp(self):
        """准备测试环境"""
        # 环境变量模拟
        self.env_patcher = patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_api_key',
            'OPENAI_API_BASE': 'https://test.api.com/v1',
            'EMBEDDING_MODEL': 'test-embedding-model',
            'TRANSLATION_MODEL': 'test-translation-model',
            'ANALYSIS_MODEL': 'test-analysis-model'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """清理测试环境"""
        self.env_patcher.stop()
    
    def test_get_embedding(self, mock_chat_completion, mock_embedding):
        """测试获取嵌入向量"""
        # 模拟API响应
        mock_embedding.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        
        ai_service = AIService()
        embedding = ai_service.get_embedding("Test text")
        
        self.assertEqual(embedding, [0.1, 0.2, 0.3])
        mock_embedding.assert_called_once()
    
    def test_analyze_content(self, mock_chat_completion, mock_embedding):
        """测试内容分析"""
        # 模拟API响应
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"genre": "sci-fi", "content": "A space adventure"}'
        mock_resp.usage = {"total_tokens": 100}
        mock_chat_completion.return_value = mock_resp
        
        ai_service = AIService()
        result = ai_service.analyze_content("Test content")
        
        self.assertEqual(result["analysis"], '{"genre": "sci-fi", "content": "A space adventure"}')
        self.assertEqual(result["usage"], {"total_tokens": 100})
        mock_chat_completion.assert_called_once()
    
    def test_translate_subtitle(self, mock_chat_completion, mock_embedding):
        """测试字幕翻译"""
        # 模拟API响应
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "翻译后的文本"
        mock_chat_completion.return_value = mock_resp
        
        ai_service = AIService()
        content_analysis = {"genre": "动作片", "content": "一部关于特工的电影"}
        result = ai_service.translate_subtitle("Test subtitle", "zh-CN", content_analysis)
        
        self.assertEqual(result, "翻译后的文本")
        mock_chat_completion.assert_called_once()


if __name__ == "__main__":
    unittest.main() 