import unittest
import os
from unittest.mock import patch, MagicMock
from youtube2blog import YouTubeHandler, BlogGenerator, FileHandler

class TestYouTubeHandler(unittest.TestCase):
    def setUp(self):
        self.handler = YouTubeHandler()
        
    @patch('youtube2blog.YouTubeTranscriptApi.get_transcript')
    @patch('youtube2blog.TextFormatter.format_transcript')
    def test_fetch_transcript(self, mock_format, mock_get_transcript):
        # モックデータの設定
        mock_get_transcript.return_value = [{'text': 'test transcript'}]
        mock_format.return_value = 'formatted transcript'
        
        # テスト実行
        result = self.handler.fetch_transcript('https://youtube.com/watch?v=dQw4w9WgXcQ')
        
        # アサーション
        self.assertEqual(result, 'formatted transcript')
        mock_get_transcript.assert_called_once()
        mock_format.assert_called_once()

    def test_get_video_id_valid_url(self):
        url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
        result = self.handler.get_video_id(url)
        self.assertEqual(result, 'dQw4w9WgXcQ')

    def test_get_video_id_invalid_url(self):
        with self.assertRaises(ValueError):
            self.handler.get_video_id('invalid_url')

class TestBlogGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = BlogGenerator()
        
    @patch('youtube2blog.OpenAI')
    def test_generate_blog_article(self, mock_openai):
        # テストモードを有効化
        os.environ['TEST_MODE'] = 'true'
        
        # モックの設定
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # テスト実行
        result = self.generator.generate_blog_article('test transcript', 'https://youtube.com/watch?v=22ikTBTK2Hg')
        
        # アサーション
        self.assertEqual(result, 'test blog content')
        
        # テストモードを無効化
        os.environ.pop('TEST_MODE', None)

class TestFileHandler(unittest.TestCase):
    def setUp(self):
        self.handler = FileHandler()
        
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_save_to_file(self, mock_open):
        # テスト実行
        self.handler.save_to_file('test content', 'test.txt')
        
        # アサーション
        mock_open.assert_called_once_with('test.txt', 'w', encoding='utf-8')
        
    def test_get_safe_filename(self):
        test_cases = [
            ('Test: File/Name*', 'Test FileName'),
            ('Normal File Name.txt', 'Normal File Name.txt'),
            ('Invalid<>:"/\\|?*Chars', 'InvalidChars')
        ]
        
        for input_name, expected in test_cases:
            with self.subTest(input_name=input_name):
                result = self.handler.get_safe_filename(input_name)
                self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
