from youtube2blog import get_video_id, fetch_transcript, generate_blog_article, save_to_file
import pytest
from unittest.mock import patch, MagicMock

def test_get_video_id():
    # 正常系のテスト
    assert get_video_id("https://www.youtube.com/watch?v=u-8NkOx6OJc") == "u-8NkOx6OJc"
    
    # 異常系のテスト
    with pytest.raises(ValueError):
        get_video_id("invalid_url")

@patch('youtube2blog.YouTubeTranscriptApi.get_transcript')
def test_fetch_transcript(mock_get_transcript):
    mock_get_transcript.return_value = [{'text': 'test transcript'}]
    result = fetch_transcript('https://youtube.com/watch?v=u-8NkOx6OJc')
    assert 'test transcript' in result
