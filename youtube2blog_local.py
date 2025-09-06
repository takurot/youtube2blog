import argparse
import os
import tempfile
import time
from typing import Dict, List, Tuple
import whisper
import yt_dlp
import json
from datetime import datetime
from openai import OpenAI
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI()
LLM_MODEL = "chatgpt-4o-latest"

# 利用可能な音声のリスト
AVAILABLE_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"]

PROMPT_TEMPLATES = {
    "zh": {
        "word_count_instruction": "2. 请将全文控制在约{min_words}～{max_words}字。",
        "intro": (
            "请基于以下 YouTube 视频的逐字稿，撰写一篇中文解读型博客文章。"
            "所有段落（包括标题）必须使用简体中文输出，除品牌英文名/符号名（如 @cosme、Dior）外，不得混用日文或英文。"
            "若逐字稿或视频标题中包含日文，请翻译为自然的简体中文进行表达；标点请使用中文标点。"
        ),
        "common": (
            "写作要求:\n"
            "1. 采用第三人称与客观分析视角，穿插细节、示例与背景信息，突出重点并深入展开。\n"
            "{word_count_instruction}\n"
            "3. 在标题之后，直接写出视频的 URL（{youtube_url}）作为单独一行文本。\n"
            "4. 第一节使用二级标题“## 要点”，以要点式列出核心论点与洞见；每条须具体、信息密度高，使读者仅读此节即可把握全貌。\n"
            "5. 正文中可适度引入相关主题，给出独到见解、案例与应用，进行合理的延展与对比。\n"
            "6. 最后一节使用二级标题“## 总结”，用简洁有力的语言收束全文，并提出可执行建议。\n"
            "7. 不要使用加粗符号（*）；请合理使用“##”“###”等标题层级。\n"
            "8. 生成的正文中不要包含任何转录段落标记（例如: [Segment 001 ...]）。"
        ),
        "output_format": (
            "输出格式要求:\n"
            "请将生成的整篇博客文章以如下 JSON 结构输出:\n"
            "```json\n"
            "{\n"
            "  \"blog_article\": \"这里填写生成的整篇博客文章（使用 Markdown 格式）\"\n"
            "}\n"
            "```"
        ),
        "system": "你是一名专业中文博客作者，能够严格按照指示生成高质量内容，并以指定的 JSON 结构输出。",
    },
    "ja": {
        "word_count_instruction": "2. 記事全体の文字数は{min_words}〜{max_words}字程度にしてください。",
        "intro": "以下のYouTube動画の文字起こしを元に、日本語の解説ブログ記事を作成してください。",
        "common": (
            "ブログ記事の要件:\n"
            "1. 読者が分かりやすいように、より詳細な説明や具体的な例を交えながら構成し、第三者視点で重要なポイントを深く掘り下げて強調してください。\n"
            "{word_count_instruction}\n"
            "3. タイトルの次に動画のURL ({youtube_url}) を文字列としてそのまま記載してください。\n"
            "4. 最初の項目は「## ポイント」として、主な主張や論点の考察を箇条書きで記載し、ここだけ読めば概要がわかるようにしてください。各ポイントは具体的で、詳細な説明や背景情報も適宜含めてください。\n"
            "5. 文字起こし内容に関連するトピックを適宜マッシュアップし、独自の見解や意見、さらには具体的な事例や応用例を交えながら述べてください。\n"
            "6. 最後の項目は「## まとめ」として、記事全体の要点を簡潔に、かつ読者の行動を促すような形でまとめてください。\n"
            "7. 太字表現 (*) は使用しないでください。見出し (## や ###) は適切に使用してください。\n"
            "8. 生成するブログ記事の本文中には、文字起こしセグメントの情報（例: `[Segment 001 ...]`）を一切含めないでください。"
        ),
        "output_format": (
            "出力形式の要件:\n"
            "生成するブログ記事の全文を、以下のJSON形式で出力してください。\n"
            "```json\n"
            "{\n"
            "  \"blog_article\": \"ここに生成されたブログ記事の全文をマークダウン形式で記述...\"\n"
            "}\n"
            "```"
        ),
        "system": "あなたはプロのブロガーであり、指示された形式で情報を正確に出力できるアシスタントです。",
    },
}

def get_video_id(youtube_url: str) -> str:
    """YouTube URLから動画IDを抽出"""
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', youtube_url)
    if video_id_match:
        extracted_id = video_id_match.group(1)
        return extracted_id
    else:
        raise ValueError("無効なYouTube URLです。動画IDを取得できませんでした。")

def download_audio_from_youtube(youtube_url: str, output_dir: str = "temp_audio") -> Tuple[str, str]:
    """
    YouTubeから音声をダウンロードする
    Returns: (audio_file_path, video_title)
    """
    try:
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        
        # より単純で確実なファイル名パターンを使用
        video_id = get_video_id(youtube_url)
        output_template = os.path.join(output_dir, f"audio_{video_id}.%(ext)s")
        
        # yt-dlpの設定
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info("YouTube動画をダウンロード中: %s", youtube_url)

            # 動画情報を取得
            info = ydl.extract_info(youtube_url, download=False)
            video_title = info.get('title', 'Unknown')
            logger.info("動画タイトル: %s", video_title)

            # 音声をダウンロード
            ydl.download([youtube_url])

            logger.info("ダウンロード完了、ファイルを探索中: %s", output_dir)
            # ダウンロード後に実際にどんなファイルが作成されたかを確認
            files_in_dir = os.listdir(output_dir)
            logger.info("ディレクトリ内のファイル: %s", files_in_dir)
            
            # 予想されるファイル名
            expected_file = os.path.join(output_dir, f"audio_{video_id}.mp3")
            
            # ファイルが存在するか確認
            if os.path.exists(expected_file):
                logger.info("音声ファイルをダウンロード完了: %s", expected_file)
                return expected_file, video_title
            else:
                # mp3ファイルを探す（どんな名前でも）
                for file in files_in_dir:
                    if file.endswith('.mp3'):
                        full_path = os.path.join(output_dir, file)
                        logger.info("MP3ファイルを発見: %s", full_path)
                        return full_path, video_title
                
                # まだ見つからない場合、他の音声ファイル形式も探す
                audio_extensions = ['.mp3', '.m4a', '.wav', '.webm', '.ogg']
                for file in files_in_dir:
                    for ext in audio_extensions:
                        if file.endswith(ext):
                            full_path = os.path.join(output_dir, file)
                            logger.info("音声ファイルを発見: %s", full_path)
                            return full_path, video_title
                
                raise FileNotFoundError(f"音声ファイルが見つかりません。ディレクトリ: {output_dir}, ファイル一覧: {files_in_dir}")
                
    except Exception as e:
        error_msg = f"YouTubeからの音声ダウンロードに失敗しました: {str(e)}"
        logger.exception(error_msg)
        raise Exception(error_msg)

def transcribe_audio_with_whisper(audio_file_path: str, model_name: str = "base") -> tuple[List[Dict], str]:
    """
    Whisperを使って音声ファイルからトランスクリプトを生成
    Returns: (transcript_data (youtube_transcript_api互換形式), detected_language)
    """
    try:
        logger.info("Whisperモデル '%s' を読み込み中...", model_name)
        model = whisper.load_model(model_name)

        logger.info("音声ファイルを転写中: %s", audio_file_path)
        result = model.transcribe(audio_file_path, verbose=False)

        # 検出された言語を取得
        detected_language = result.get('language', 'unknown')
        logger.info("検出された言語: %s", detected_language)
        
        # youtube_transcript_api互換の形式に変換
        transcript_data = []
        for segment in result['segments']:
            transcript_data.append({
                'text': segment['text'].strip(),
                'start': segment['start'],
                'duration': segment['end'] - segment['start']
            })
        
        logger.info("転写完了: %d セグメント生成されました", len(transcript_data))
        return transcript_data, detected_language

    except Exception as e:
        error_msg = f"Whisperでの転写に失敗しました: {str(e)}"
        logger.exception(error_msg)
        raise Exception(error_msg)

def fetch_transcript_local(youtube_url: str, whisper_model: str = "base") -> dict:
    """
    ローカルでYouTubeトランスクリプトを生成
    Returns: {"error": None/str, "data": transcript_data, "language": "ja"}
    """
    try:
        with tempfile.TemporaryDirectory(prefix="youtube_transcript_") as temp_dir:
            audio_file, video_title = download_audio_from_youtube(youtube_url, temp_dir)
            transcript_data, detected_language = transcribe_audio_with_whisper(audio_file, whisper_model)

            if transcript_data:
                return {
                    "error": None,
                    "data": transcript_data,
                    "language": detected_language,  # Whisperが検出した実際の言語を使用
                    "video_title": video_title,
                }
            else:
                return {"error": "転写データが空です", "data": None}
    except Exception as e:
        error_msg = f"ローカル転写中にエラーが発生しました: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "data": None}

def generate_blog_article(
    transcript_data: list[dict],
    youtube_url: str,
    language: str = "ja",
    min_words: int = 2500,
    max_words: int = 3000,
) -> tuple[str | None, str | None]:
    """文字起こしデータを基に OpenAI API を使ってブログ記事を生成（日本語/中国語対応）"""
    if not transcript_data:
        return None, "文字起こしデータが空です。"

    formatted_transcript_for_llm = []
    for i, segment in enumerate(transcript_data):
        text = segment.get('text', '').strip()
        if text:
            formatted_transcript_for_llm.append(text)
    
    if not formatted_transcript_for_llm:
        return None, "有効な文字起こしセグメントがありませんでした。"
    
    transcript_string_for_llm = "\n".join(formatted_transcript_for_llm)

    # 言語に応じた指示を生成
    lang_key = "zh" if language in ("zh", "chinese", "zh-cn", "zh-tw") else "ja"
    config = PROMPT_TEMPLATES[lang_key]
    word_count_instruction = config["word_count_instruction"].format(min_words=min_words, max_words=max_words)
    intro_to_transcript_processing = config["intro"]
    common_blog_requirements = config["common"].format(word_count_instruction=word_count_instruction, youtube_url=youtube_url)
    output_format_requirements = config["output_format"]
    system_prompt = config["system"]

    prompt_content = f"""{intro_to_transcript_processing}
{common_blog_requirements}
{output_format_requirements}

文字起こし:
{transcript_string_for_llm}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_content},
    ]

    try:
        logger.info("LLMにブログ記事の生成をリクエストします...")
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=4095,
            response_format={"type": "json_object"}
        )
        
        raw_response_content = response.choices[0].message.content.strip()
        
        try:
            parsed_response = json.loads(raw_response_content)
            blog_article_text = parsed_response.get("blog_article")
            
            if not blog_article_text:
                logger.error("エラー: LLMからの応答形式が不正です（blog_articleが欠損）。")
                return None, "LLMからの応答形式が不正です（blog_articleが欠損）。"
            
            # セグメント注釈を削除
            blog_article_text = re.sub(r"\s*\[Segment[^\]]+\]\s*", "", blog_article_text)
            blog_article_text = re.sub(r"\n\s*\n", "\n\n", blog_article_text).strip()

            # 中国語出力時の日本語混在を検出（ひらがな/カタカナ）し、必要ならリライト
            if language in ("zh", "chinese", "zh-cn", "zh-tw"):
                if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", blog_article_text):
                    logger.info("検出: 中国語記事に日本語が混在。中国語のみに統一します…")
                    blog_article_text = _refine_to_simplified_chinese(blog_article_text)

            logger.info("ブログ記事を正常に生成・パースしました。")
            return blog_article_text, None

        except json.JSONDecodeError as json_e:
            logger.exception("LLMからの応答のJSONパースに失敗しました: %s", json_e)
            return None, f"LLM応答のJSONパース失敗: {json_e}"

    except Exception as e_llm:
        error_msg = f"ブログ記事の生成中にエラーが発生しました: {e_llm}"
        logger.error(error_msg)
        return None, error_msg


def _refine_to_simplified_chinese(draft_text: str) -> str:
    """混在した非中文（特に日文）を除去し、全文を簡体字中文に統一するフォールバック整形。"""
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是专业的中文编辑。请将用户提供的整段文本重写为纯简体中文，"
                    "不得包含任何日文（平假名/片假名）或英文句子。品牌名如 @cosme、Dior 可保留。"
                    "保持 Markdown 结构（标题/列表），所有文字与标点均使用简体中文；输出 JSON，键为 blog_article。"
                ),
            },
            {
                "role": "user",
                "content": "请重写为纯简体中文：\n\n" + draft_text,
            },
        ]
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=4095,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        refined = json.loads(raw).get("blog_article")
        if not refined:
            return draft_text
        refined = re.sub(r"\n\s*\n", "\n\n", refined).strip()
        # 最終チェック：まだ日本語が残っていれば原文を返す（安全策）
        if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", refined):
            return draft_text
        return refined
    except Exception:
        return draft_text

def save_to_file(content, filename):
    """生成されたブログ記事をファイルに保存"""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        logger.info("ブログ記事が %s に保存されました。", filename)
    except Exception as e:
        logger.error("ファイル保存中にエラーが発生しました: %s", e)

def _generate_output_filenames(youtube_url: str, output_language: str = "ja") -> dict[str, str]:
    """動画IDとオプションに基づいて出力ファイル名を生成（中国語は _zh サフィックス）"""
    try:
        video_id = get_video_id(youtube_url)
    except ValueError as e:
        logger.error("ファイル名生成エラー: %s", e)
        return {'text': None}
    
    today = datetime.now().strftime('%Y%m%d')
    lang_suffix = "_zh" if output_language in ("zh", "chinese", "zh-cn", "zh-tw") else ""
    base_name = f"{today}_blog_local{lang_suffix}_{video_id}"
    
    filenames = {
        'text': f"{base_name}_article.txt",
    }
    
    return filenames

def main():
    parser = argparse.ArgumentParser(description="YouTubeからローカルで文字起こしを生成してブログ記事を作成するスクリプト（日本語/中国語対応）")
    parser.add_argument("youtube_url", help="YouTube動画のURLを指定")
    parser.add_argument("--whisper-model", default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="使用するWhisperモデル (デフォルト: base)")
    parser.add_argument("--output-language", default="ja", 
                       choices=["ja", "zh"],
                       help="出力するブログ記事の言語 (ja または zh)")
    parser.add_argument("--min-words", type=int, default=2500, 
                       help="ブログ記事の最小目標文字数 (デフォルト: 2500)")
    parser.add_argument("--max-words", type=int, default=3000, 
                       help="ブログ記事の最大目標文字数 (デフォルト: 3000)")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # 出力ファイル名を生成
    output_filenames = _generate_output_filenames(args.youtube_url, args.output_language)
    if any(value is None for value in output_filenames.values()):
        logger.error("ファイル名の生成に失敗しました。処理を中断します。")
        return
    
    article_file = output_filenames['text']
    
    logger.info("\n=== YouTube動画からローカルで文字起こしを生成 ===")
    logger.info("動画URL: %s", args.youtube_url)
    logger.info("Whisperモデル: %s", args.whisper_model)
    logger.info("出力言語: %s", args.output_language)
    
    # ローカルでトランスクリプトを生成
    logger.info("\n文字起こしを生成中...")
    transcript_response = fetch_transcript_local(args.youtube_url, args.whisper_model)
    
    if transcript_response.get('error'):
        logger.error("文字起こし生成エラー: %s", transcript_response['error'])
        return

    transcript_data = transcript_response.get('data')
    if not transcript_data:
        logger.error("文字起こしデータが取得できませんでした（データが空です）。")
        return

    logger.info("文字起こし生成成功。セグメント数: %d", len(transcript_data))
    
    # ブログ記事を生成
    logger.info("\nブログ記事（%s）を生成中...", args.output_language)
    blog_article_text, error_message = generate_blog_article(
        transcript_data,
        args.youtube_url,
        language=args.output_language,
        min_words=args.min_words,
        max_words=args.max_words,
    )
    
    if error_message:
        logger.error("ブログ記事生成エラー: %s", error_message)
        if blog_article_text:
            logger.info("部分的なブログ記事を保存します...")
            save_to_file(blog_article_text, article_file)
        return

    if not blog_article_text:
        logger.error("ブログ記事の生成に失敗しました（テキストが空です）。")
        return

    logger.info("ブログ記事生成成功。")
    save_to_file(blog_article_text, article_file)

    elapsed_time = time.time() - start_time
    logger.info("\n処理が完了しました。所要時間: %.2f秒", elapsed_time)
    logger.info("ブログ記事保存先: %s", article_file)

if __name__ == "__main__":
    main() 
