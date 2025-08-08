import unittest
from datetime import datetime

import importlib


class TestYoutube2BlogLocalJA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 動的にモジュールを読み込み。依存が無ければスキップ
        try:
            cls.ja = importlib.import_module("youtube2blog_local")
        except Exception as e:
            raise unittest.SkipTest(f"youtube2blog_local の読み込みをスキップ: {e}")

    def test_get_video_id_various_urls(self):
        f = self.ja.get_video_id
        self.assertEqual(f("https://www.youtube.com/watch?v=dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(f("https://youtu.be/dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(f("https://www.youtube.com/embed/dQw4w9WgXcQ"), "dQw4w9WgXcQ")

    def test_generate_output_filenames_contains_date_and_id(self):
        filenames = self.ja._generate_output_filenames("https://youtu.be/aaaaaaaaaaa", False)
        today = datetime.now().strftime('%Y%m%d')
        self.assertTrue(filenames['text'].startswith(f"{today}_blog_local_"))
        self.assertIn("aaaaaaaaaaa", filenames['text'])
        self.assertTrue(filenames['text'].endswith("_article.txt"))

    def test_generate_blog_article_empty_transcript(self):
        article, err = self.ja.generate_blog_article([], "https://youtu.be/aaaaaaaaaaa", no_timestamps=True, language="ja")
        self.assertIsNone(article)
        self.assertIsNotNone(err)


class TestYoutube2BlogLocalZH(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.zh = importlib.import_module("youtube2blog_local_zh")
        except Exception as e:
            raise unittest.SkipTest(f"youtube2blog_local_zh の読み込みをスキップ: {e}")

    def test_get_video_id_various_urls(self):
        f = self.zh.get_video_id
        self.assertEqual(f("https://www.youtube.com/watch?v=dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(f("https://youtu.be/dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(f("https://www.youtube.com/embed/dQw4w9WgXcQ"), "dQw4w9WgXcQ")

    def test_generate_output_filenames_contains_date_and_id(self):
        filenames = self.zh._generate_output_filenames("https://youtu.be/bbbbbbbbbbb", False)
        today = datetime.now().strftime('%Y%m%d')
        self.assertTrue(filenames['text'].startswith(f"{today}_blog_local_zh_"))
        self.assertIn("bbbbbbbbbbb", filenames['text'])
        self.assertTrue(filenames['text'].endswith("_article.txt"))

    def test_generate_blog_article_empty_transcript(self):
        article, err = self.zh.generate_blog_article_zh([], "https://youtu.be/bbbbbbbbbbb", no_timestamps=True, language="zh")
        self.assertIsNone(article)
        self.assertIsNotNone(err)


if __name__ == "__main__":
    unittest.main()


