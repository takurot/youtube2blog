import argparse
import re
import textwrap
import random
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import os
import subprocess
import hashlib
from datetime import datetime
from openai import OpenAI
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
from wordcloud import WordCloud
from janome.tokenizer import Tokenizer
import time
import xml.etree.ElementTree as ET

client = OpenAI()
LLM_MODEL = "chatgpt-4o-latest"

# 利用可能な音声のリスト
AVAILABLE_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"]

# client.api_key = os.getenv('DEEPSEEK_API_KEY')
# client.base_url = "https://api.deepseek.com/v1"
# LLM_MODEL = "deepseek-chat"

def get_video_id(youtube_url):
    """YouTube URLから動画IDを抽出"""
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', youtube_url)
    if video_id_match:
        return video_id_match.group(1)
    else:
        raise ValueError("無効なYouTube URLです。動画IDを取得できませんでした。")

def fetch_transcript(youtube_url, language="en"):
    """YouTube URLから文字起こしを取得"""
    video_id = get_video_id(youtube_url)
    try:
        # トランスクリプトを取得
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        
        # トランスクリプトが取得できた場合、TextFormatterでフォーマット
        if transcript:
            # トランスクリプトの形式をチェック
            if isinstance(transcript, list) and all(isinstance(item, dict) for item in transcript):
                formatter = TextFormatter()
                formatted_transcript = formatter.format_transcript(transcript)
                return formatted_transcript
            elif isinstance(transcript, dict):
                # dictオブジェクトで、textキーがない場合の処理
                if 'text' not in transcript:
                    print(f"デバッグ情報: 予期しないトランスクリプト形式: {transcript}")
                    # 利用可能なキーを使用してテキストを構築
                    if transcript:
                        return str(transcript)
                    else:
                        return f"動画ID {video_id} に対する {language} の字幕が見つかりませんでした。"
            else:
                # その他の予期しない形式の場合
                print(f"デバッグ情報: 予期しない形式のトランスクリプト: {type(transcript)}")
                return str(transcript)
        else:
            return f"動画ID {video_id} に対する {language} の字幕が見つかりませんでした。"
    except ET.ParseError as e:
        error_message = f"字幕データのパースに失敗しました（ParseError）: {str(e)}"
        print(error_message)
        return error_message
    except Exception as e:
        # より詳細なエラーメッセージを表示
        error_message = f"文字起こしを取得できませんでした: {str(e)}"
        print(f"デバッグ情報: {e.__class__.__name__}")
        print(f"デバッグ情報: トランスクリプトの型: {type(e)}")
        
        # 言語の自動検出を試みる
        try:
            # 利用可能な言語のリストを取得
            available_languages = YouTubeTranscriptApi.list_transcripts(video_id)
            language_list = [lang.language_code for lang in available_languages]
            if language_list:
                return f"指定された言語 '{language}' の字幕が見つかりませんでした。利用可能な言語: {', '.join(language_list)}"
            else:
                return "この動画には字幕がありません。"
        except Exception as inner_e:
            print(f"デバッグ情報: 言語リスト取得時のエラー: {inner_e}")
            return error_message

def generate_blog_article(transcript, youtube_url, language="ja"):
    """文字起こしデータを基にOpenAI APIを使って日本語ブログ記事を生成"""
    messages = [
        { 
            "role": "system", 
            "content": "You are a great blog writer." 
        },
        { 
            "role": "user", 
            "content": f"""以下の文字起こし内容を元に、2000〜2300字程度の日本語の解説ブログ記事を作成してください。
                        記事は、読者が分かりやすいように構成し、第三者視点で重要なポイントを強調してください。
                        タイトルの次に動画へのURL {youtube_url} をリンク形式でなく文字列でそのまま入れてください。
                        最初の項目は「ポイント」として、主な主張や論点の考察を箇条書きで記載して、ここだけ読めばおおよその概要がわかるようにしてください。
                        文字起こし内容に関連するトピックを適宜マッシュアップし、独自の見解や意見を述べてください。
                        最後の項目は「まとめ」として、尖った意見を述べてください。
                        太字(*)は使わないでください。見出し（#）は使ってください。

                        {transcript}"""
        }
    ]

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=5000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ブログ記事の生成中にエラーが発生しました: {e}"

def save_to_file(content, filename):
    """生成されたブログ記事をファイルに保存"""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"ブログ記事が {filename} に保存されました。")
    except Exception as e:
        print(f"ファイル保存中にエラーが発生しました: {e}")

def _create_audio_chunks(paragraph_text: str, max_length: int = 4000) -> list[str]:
    """Splits a long paragraph into smaller chunks respecting sentence boundaries."""
    if len(paragraph_text) <= max_length:
        return [paragraph_text]

    # Split into sentences
    sentences = re.split(r'(?<=[。．！？])', paragraph_text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Ensure sentence itself isn't too long (unlikely but possible)
        if len(sentence) > max_length:
             # If a single sentence is too long, just split it by length
             for i in range(0, len(sentence), max_length):
                 chunks.append(sentence[i:i+max_length])
             continue # Skip the rest of the logic for this super long sentence

        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence # Start new chunk with the current sentence

    if current_chunk: # Add the last chunk if it exists
        chunks.append(current_chunk)

    return chunks


def _synthesize_speech_for_chunk(
    chunk: str, 
    temp_file_path: str, 
    voice: str, 
    client: OpenAI, 
    base_wait_time: float,
    max_retries: int = 5,
    timeout_seconds: int = 60 # Timeout for receiving stream data
) -> bool:
    """Synthesizes speech for a single text chunk with retry logic."""
    if not chunk.strip():
        return False

    retry_count = 0
    success = False

    while retry_count < max_retries and not success:
        try:
            print(f"  - Processing chunk ({len(chunk)} chars) (Attempt {retry_count + 1}/{max_retries})...")

            # Exponential backoff for retries and general API calls
            current_wait = base_wait_time * (2 ** retry_count)
            if retry_count > 0:
                print(f"    Waiting {current_wait:.2f}s before retrying...")
                time.sleep(current_wait)
            # Apply a smaller base wait between all API calls to avoid rate limits
            elif os.path.exists(temp_file_path): # Check if it's not the very first chunk attempt overall
                 time.sleep(base_wait_time / 2) # Shorter wait for non-retry calls


            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=chunk,
                response_format="mp3",
                speed=1.0,
            ) as response:
                buffer = bytearray()
                start_time = time.time()

                try:
                    for data in response.iter_bytes(chunk_size=8192):
                        buffer.extend(data)
                        if time.time() - start_time > timeout_seconds:
                            print(f"    Warning: Stream data reception timed out after {timeout_seconds}s.")
                            break
                except Exception as stream_error:
                    print(f"    Error during stream reading: {stream_error}")
                    # Proceed if we have partial data, otherwise re-raise
                    if not buffer:
                        raise

                if not buffer:
                    print("    Warning: No data received from stream. Retrying...")
                    retry_count += 1
                    continue

                with open(temp_file_path, 'wb') as f:
                    f.write(buffer)

                if os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) > 1000: # Min 1KB check
                    print(f"    Chunk synthesized successfully: {os.path.getsize(temp_file_path)} bytes.")
                    success = True
                else:
                    print(f"    Warning: Generated file is too small or missing ({os.path.getsize(temp_file_path) if os.path.exists(temp_file_path) else 'missing'}). Retrying...")
                    retry_count += 1

        except Exception as chunk_error:
            print(f"    Error processing chunk: {chunk_error}")
            retry_count += 1
            # Wait longer before the next attempt if an exception occurred
            backoff_time = base_wait_time * (2 ** retry_count)
            print(f"    Waiting {backoff_time:.2f}s before next attempt after error...")
            time.sleep(backoff_time)

    if not success:
         print(f"  - Failed to synthesize chunk after {max_retries} attempts.")
    return success


def text_to_speech(content, filename, voice=None):
    """生成されたブログテキストをTTS APIで音声化し、mp3ファイルに保存
    長いテキストは複数のチャンクに分割して処理し、安定性を向上させる
    """
    try:
        speech_file_path = Path(filename)
        if voice is None:
            voice = random.choice(AVAILABLE_VOICES)
        print(f"選択された音声: {voice}")

        if not content or not content.strip():
            print("エラー: 入力テキストが空です。音声生成をスキップします。")
            return False
        print(f"処理するテキストの合計長: {len(content)} 文字")

        paragraphs = [p for p in content.split('\n') if p.strip()]
        if not paragraphs:
            print("エラー: 有効な段落がありません。音声生成をスキップします。")
            return False

        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True)
        temp_files = []
        
        # Base wait time between API calls to manage rate limits
        base_wait_time = 1.5 # seconds 

        print(f"テキストを {len(paragraphs)} 個の段落に分割して処理します")

        chunk_index = 0
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            
            print(f"Processing paragraph {i + 1}/{len(paragraphs)}...")
            
            # Split paragraph into smaller chunks if needed
            chunks = _create_audio_chunks(paragraph, max_length=4000)
            print(f"  Paragraph split into {len(chunks)} chunk(s).")

            for j, chunk in enumerate(chunks):
                temp_file = os.path.join(temp_dir, f"temp_{chunk_index}.mp3")
                chunk_index += 1
                
                if _synthesize_speech_for_chunk(chunk, temp_file, voice, client, base_wait_time):
                     if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                        temp_files.append(temp_file)
                     else:
                         print(f"  Warning: Synthesized chunk file {temp_file} is missing or empty despite success signal.")
                else:
                     print(f"  Warning: Failed to synthesize chunk {j+1} for paragraph {i+1}. Skipping.")


        if not temp_files:
            print("音声生成に失敗しました。有効な一時ファイルがありません。")
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                print(f"一時ファイルの削除中にエラーが発生しました: {cleanup_error}")
            return False

        # Combine temporary files
        if len(temp_files) > 1:
            concat_file = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_file, 'w', encoding='utf-8') as f:
                valid_files_for_concat = []
                for temp_file in temp_files:
                    if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                        f.write(f"file '{os.path.abspath(temp_file)}'\n")
                        valid_files_for_concat.append(temp_file)
                    else:
                         print(f"Warning: Skipping invalid/empty temp file during concatenation: {temp_file}")
                
                if not valid_files_for_concat:
                    print("Error: No valid temporary files found for concatenation.")
                    return False # Cannot concatenate if all files were invalid


            concat_cmd = f'ffmpeg -y -f concat -safe 0 -i "{concat_file}" -c copy "{speech_file_path}"'
            print("結合コマンドを実行中...")
            print(f"  {concat_cmd}")
            
            result = subprocess.run(concat_cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if result.returncode != 0:
                print(f"FFmpeg結合エラー:\nstdout: {result.stdout}\nstderr: {result.stderr}")
                # Fallback: Copy the largest valid file if concatenation fails
                if valid_files_for_concat:
                    largest_file = max(valid_files_for_concat, key=lambda f: os.path.getsize(f))
                    print(f"結合に失敗したため、最大の有効なファイル {largest_file} をコピーします。")
                    import shutil
                    try:
                         shutil.copy(largest_file, speech_file_path)
                    except Exception as copy_err:
                         print(f"フォールバックコピー中にエラー: {copy_err}")
                         return False # Copy failed too
                else:
                    print("結合失敗、有効なフォールバックファイルなし。")
                    return False
            else:
                print("FFmpeg結合成功。")
        elif len(temp_files) == 1:
            # Only one file, just copy it
            import shutil
            print(f"単一の音声ファイル {temp_files[0]} を最終ファイルにコピーします。")
            try:
                shutil.copy(temp_files[0], speech_file_path)
            except Exception as copy_err:
                 print(f"単一ファイルのコピー中にエラー: {copy_err}")
                 return False
        else:
            # Should not happen if initial check passed, but handle defensively
            print("エラー: 結合するファイルが見つかりません。")
            return False

        # Final check and cleanup
        if os.path.exists(speech_file_path) and os.path.getsize(speech_file_path) > 0:
            print(f"音声ファイルが {filename} に保存されました。サイズ: {os.path.getsize(speech_file_path)} バイト")
            import shutil
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print("一時ファイルを削除しました。")
                else:
                    print("一時ディレクトリが見つからないため、削除をスキップしました。")
            except Exception as cleanup_error:
                print(f"一時ファイルの削除中にエラーが発生しました: {cleanup_error}")
            return True
        else:
            print(f"エラー: 最終音声ファイル {filename} が存在しないか空です。")
            # Attempt cleanup even on failure
            import shutil
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                print(f"一時ファイルの削除中にエラーが発生しました: {cleanup_error}")
            return False

    except Exception as e:
        print(f"音声ファイル作成中に予期せぬエラーが発生しました: {e}")
        # Attempt cleanup on general exceptions too
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
             import shutil
             try:
                 if os.path.exists(temp_dir):
                     shutil.rmtree(temp_dir)
             except Exception as cleanup_error:
                 print(f"エラー発生後の一時ファイル削除中にエラー: {cleanup_error}")
        return False

def extract_title_from_blog(blog_content):
    """ブログ記事からタイトルを抽出"""
    try:
        lines = blog_content.strip().split('\n')
        for line in lines:
            # 先頭の # で始まる行を探す
            if line.startswith('# '):
                title = line.replace('# ', '')
                # ハッシュタグを除去
                title = re.sub(r'#\w+', '', title).strip()
                return title
        
        # タイトルが見つからない場合は最初の行を返す
        return lines[0]
    except Exception as e:
        print(f"タイトル抽出中にエラーが発生しました: {e}")
        return "自動生成ブログ"

def create_audio_text(blog_content):
    """ブログ記事から音声出力用のテキストを作成
    タイトル（ハッシュタグなし）を含め、YouTube URLやポイントの章を除いた
    より自然な音声出力用のテキストを生成します。
    マークダウン記号（#, ---など）も削除します。
    """
    try:
        lines = blog_content.strip().split('\n')
        audio_text_lines = []
        title = ""
        skip_mode = False
        points_section = False
        
        for line in lines:
            # タイトル行を処理
            if line.startswith('# ') and not title:
                # ハッシュタグを除去したタイトルを抽出
                title = line.replace('# ', '')
                title = re.sub(r'#\w+', '', title).strip()
                audio_text_lines.append(title)
                continue
            
            # YouTube URLの行をスキップ
            if 'youtube.com' in line or 'youtu.be' in line:
                continue
            
            # 「ポイント」セクションの開始を検出
            if line.startswith('## ポイント') or '# ポイント' in line:
                points_section = True
                skip_mode = True
                continue
            
            # 「ポイント」セクション以降の新しいセクションが始まったらスキップモードを解除
            if points_section and (line.startswith('## ') or line.startswith('# ')) and '# ポイント' not in line:
                points_section = False
                skip_mode = False
            
            # スキップモードでなければテキストに追加
            if not skip_mode:
                # 見出し（#）から「#」記号を削除
                if line.startswith('## ') or line.startswith('# '):
                    line = line.replace('## ', '').replace('# ', '')
                
                # その他のマークダウン記号を削除
                # 水平線（---や***）を削除
                if re.match(r'^[\-*_]{3,}$', line.strip()):
                    continue
                
                # リストマーカー（- や * や 1. など）を削除
                line = re.sub(r'^\s*[\-*+]\s+', '', line)
                line = re.sub(r'^\s*\d+\.\s+', '', line)
                
                # 強調（**太字**や*斜体*）を削除して中のテキストだけを残す
                line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
                line = re.sub(r'\*(.*?)\*', r'\1', line)
                
                # バッククォート（`コード`）を削除
                line = re.sub(r'`(.*?)`', r'\1', line)
                
                # リンク（[テキスト](URL)）からURLを削除してテキストだけを残す
                line = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', line)
                
                audio_text_lines.append(line)
        
        # 空白行を削除して結合
        audio_text = '\n'.join([line for line in audio_text_lines if line.strip()])
        return audio_text
    
    except Exception as e:
        print(f"音声出力用テキスト生成中にエラーが発生しました: {e}")
        return blog_content  # エラーが発生した場合は元のブログ記事をそのまま返す

def _build_ffmpeg_command(audio_file: str, output_file: str, image_file: str | None, is_shorts: bool, use_bgm: bool) -> str:
    """Builds the FFmpeg command string based on options."""
    # --- Video Input --- 
    if image_file and os.path.exists(image_file):
        image_input = f"-loop 1 -i \"{image_file}\""
        video_map = "0:v" # Map from the image file
    else:
        # Use lavfi black background if no image provided
        size = "1080x1920" if is_shorts else "1920x1080"
        image_input = f"-f lavfi -i color=c=black:s={size}:r=30" # Added rate for compatibility
        video_map = "0:v" # Map from the lavfi source

    # --- Audio Input & Filtering --- 
    audio_input = f"-i \"{audio_file}\"" # Main speech audio
    bgm_input = ""
    audio_filter_complex = ""
    audio_map = "1:a" # Default map from the main audio file

    if use_bgm:
        bgm_dir = "bgm"
        bgm_file_path = None
        if os.path.exists(bgm_dir):
            bgm_files = [f for f in os.listdir(bgm_dir) if f.endswith('.mp3') and not f.startswith('._')]
            if bgm_files:
                chosen_bgm = random.choice(bgm_files)
                bgm_file_path = os.path.join(bgm_dir, chosen_bgm)
                print(f"バックグラウンド音楽として {chosen_bgm} を使用します")
                bgm_input = f"-i \"{bgm_file_path}\""
                # Inputs: 0=Image/Lavfi, 1=Speech, 2=BGM
                # Loop BGM, set volume, mix with speech
                audio_filter_complex = (
                    f'-filter_complex "'
                    f'[2:a]aloop=loop=-1:size=2e+06,volume=0.15[bgm];' # Loop Input #2 (BGM), label as [bgm]
                    f'[1:a][bgm]amix=inputs=2:duration=first:dropout_transition=3[a]"' # Mix Input #1 (Speech) and [bgm], label as [a]
                )
                audio_map = "[a]" # Map from the mixed audio stream
            else:
                print("bgmディレクトリにMP3ファイルが見つかりません。BGMなしで続行します。")
        else:
            print("bgmディレクトリが見つかりません。BGMなしで続行します。")

    # If BGM is not used or not found, audio_map remains "1:a" (or adjusts if image is source 0)
    if not bgm_input:
        if image_input.startswith("-f lavfi"):
             audio_map = "1:a" # Speech is input 1
        else:
             audio_map = "1:a" # Speech is input 1 when image is input 0

    # --- Output Settings --- 
    output_settings = (
        f'-map {video_map} -map {audio_map} ' # Map the selected video and audio streams
        f'-c:v libx264 -preset veryfast -tune stillimage ' # Video codec optimized for still images
        f'-c:a aac -b:a 192k ' # Audio codec AAC, 192kbps bitrate
        f'-pix_fmt yuv420p ' # Pixel format for compatibility
        f'-shortest ' # Finish encoding when the shortest input stream ends (the speech)
        f'\"{output_file}\"' # Output file path
    )

    # Combine all parts
    # Note the order: video input(s), audio input(s), filters, output settings
    cmd = f'ffmpeg -y {image_input} {audio_input} {bgm_input} {audio_filter_complex} {output_settings}'
    
    return cmd


def create_video_from_audio(audio_file, output_file, image_file=None, is_shorts=False, use_bgm=True):
    """
    音声ファイルと静止画からmp4動画を作成（YouTube向け）
    バックグラウンド音楽も適用可能
    """
    try:
        # Build the FFmpeg command using the helper function
        cmd = _build_ffmpeg_command(audio_file, output_file, image_file, is_shorts, use_bgm)
        
        print("実行するコマンド:")
        print(f"  {cmd}")
        
        # Execute the FFmpeg command
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')

        if result.returncode == 0:
            print(f"動画ファイルが {output_file} に正常に保存されました。")
            return True
        else:
            print(f"動画ファイル作成中にFFmpegエラーが発生しました:")
            print(f"  Return Code: {result.returncode}")
            # Limit output length to avoid flooding console
            max_log_lines = 20 
            stderr_lines = result.stderr.splitlines()
            stdout_lines = result.stdout.splitlines()
            print(f"  Stderr (last {max_log_lines} lines):")
            for line in stderr_lines[-max_log_lines:]:
                 print(f"    {line}")
            print(f"  Stdout (last {max_log_lines} lines):")
            for line in stdout_lines[-max_log_lines:]:
                 print(f"    {line}")
            return False

    except Exception as e:
        print(f"動画ファイル作成中に予期せぬエラーが発生しました: {e}")
        # Print traceback for unexpected errors
        import traceback
        traceback.print_exc()
        return False

def _get_japanese_font_path() -> str | None:
    """Tries to find a suitable Japanese font path based on the OS."""
    # Common font paths
    font_paths = [
        # macOS
        '/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc',
        '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
        '/Library/Fonts/ヒラギノ角ゴシック W6.ttc',
        '/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
        # Windows
        'C:/Windows/Fonts/meiryo.ttc',
        'C:/Windows/Fonts/msgothic.ttc',
        'C:/Windows/Fonts/yu-gothb.ttc',
        # Linux (common locations/names)
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf',
        # A generic fallback
        '/System/Library/Fonts/Arial Unicode.ttf'
    ]

    for path in font_paths:
        if os.path.exists(path):
            print(f"Using font: {path}")
            return path

    print("Warning: Could not find a specific Japanese font. Defaulting to WordCloud's built-in font (may not render Japanese correctly).")
    return None

def _tokenize_and_count_japanese_words(text: str) -> dict[str, int]:
    """Tokenizes Japanese text and counts frequency of nouns, verbs, adjectives."""
    word_frequencies = {}
    try:
        # Ensure text is string
        text = str(text)
        
        t = Tokenizer()
        tokens = t.tokenize(text)
        
        valid_pos = ['名詞', '動詞', '形容詞'] # Target parts of speech
        min_word_length = 2 # Minimum word length to consider

        for token in tokens:
            pos = token.part_of_speech.split(',')[0]
            if pos in valid_pos:
                word = token.surface
                # Consider longer words to avoid noise from particles/short words
                if len(word) >= min_word_length: 
                    word_frequencies[word] = word_frequencies.get(word, 0) + 1
                    
    except Exception as tokenize_error:
        print(f"形態素解析中にエラーが発生しました: {tokenize_error}. Falling back to space-separated words.")
        # Fallback: Count space-separated words if tokenization fails
        for word in text.split():
            if len(word) >= min_word_length: # Apply length filter here too
                 word_frequencies[word] = word_frequencies.get(word, 0) + 1
                 
    return word_frequencies


def create_wordcloud_image(text, output_path, is_shorts=False):
    """テキストからワードクラウド画像を生成"""
    try:
        width, height = (1080, 1920) if is_shorts else (1920, 1080)
        print(f"Generating word cloud image ({width}x{height})...")

        # 1. Tokenize and count words using helper
        word_frequencies = _tokenize_and_count_japanese_words(text)
        if not word_frequencies:
            print("ワードクラウド生成に十分な単語が見つかりませんでした。")
            return False
        print(f"Found {len(word_frequencies)} unique words for word cloud.")

        # 2. Find font path using helper
        font_path = _get_japanese_font_path()

        # 3. Configure and generate WordCloud
        bg_color = "rgb(20, 60, 140)" # Blue background
        wc = WordCloud(
            background_color=bg_color,
            width=width,
            height=height,
            font_path=font_path, # Use found path or None
            colormap='YlOrRd', # Yellow-Orange-Red colormap
            min_font_size=12,
            prefer_horizontal=0.9,
            regexp=r"[\w\p{Han}\p{Hiragana}\p{Katakana}]+", # Regex for Japanese words
            scale=1.2, 
            relative_scaling=0.7, 
            random_state=42, 
            include_numbers=False, # Exclude numbers from cloud
            max_words=200 # Limit number of words in cloud
        )

        try:
            wc.generate_from_frequencies(word_frequencies)
            wc.to_file(output_path)
            print(f"ワードクラウド画像が {output_path} に保存されました。")
            return True
        except Exception as generate_error:
            print(f"ワードクラウド生成またはファイル保存中にエラーが発生しました: {generate_error}")
            return False

    except Exception as e:
        print(f"ワードクラウド画像生成プロセス全体でエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

def _generate_output_filenames(youtube_url: str, is_shorts: bool) -> dict[str, str]:
    """Generates a dictionary of output filenames based on video ID and options."""
    try:
        video_id = get_video_id(youtube_url)
    except ValueError as e:
        # Handle invalid URL early
        print(f"ファイル名生成エラー: {e}")
        # Return a dictionary with None values or raise, depending on desired handling
        # For now, let's return None to indicate failure
        return {
            'text': None, 'audio_text': None, 'audio': None, 
            'wordcloud': None, 'video': None
        }
        
    today = datetime.now().strftime('%Y%m%d')
    base_name = f"{today}_blog_{video_id}"
    
    filenames = {
        'text': f"{base_name}_article.txt",
        'audio_text': f"{base_name}_audio_text.txt",
        'audio': f"{base_name}_audio.mp3",
        'wordcloud': f"{base_name}_wordcloud.png",
    }
    
    # Video filename depends on is_shorts
    format_suffix = "_shorts" if is_shorts else ""
    filenames['video'] = f"{base_name}_video{format_suffix}.mp4"
    
    return filenames

def main():
    # 引数の設定
    parser = argparse.ArgumentParser(description="YouTube動画の字幕を取得し、それを基にブログ記事を生成するスクリプト")
    parser.add_argument("language", help="取得したい字幕の言語コード（例: 'en', 'ja'）")
    parser.add_argument("youtube_url", help="YouTube動画のURLを指定")
    parser.add_argument("--image", help="動画作成に使用する静止画像ファイルのパス", default=None)
    parser.add_argument("--shorts", action="store_true", help="YouTube Shorts形式（縦長動画）で出力する")
    parser.add_argument("--voice", choices=AVAILABLE_VOICES, help=f"使用する音声を指定 (選択肢: {', '.join(AVAILABLE_VOICES)}). 指定しない場合はランダム")
    parser.add_argument("--wordcloud", action="store_true", help="ワードクラウド画像を生成して使用する")
    parser.add_argument("--no-bgm", action="store_true", help="バックグラウンド音楽を使用しない")
    parser.add_argument("--blog-only", action="store_true", help="ブログ記事のみを生成し、音声・動画は生成しない")
    
    args = parser.parse_args()
    youtube_url = args.youtube_url
    language = args.language
    image_file = args.image
    is_shorts = args.shorts
    voice = args.voice
    use_wordcloud = args.wordcloud
    use_bgm = not args.no_bgm
    blog_only = args.blog_only

    # --- 1. Generate Filenames --- 
    print("出力ファイル名を生成中...")
    output_filenames = _generate_output_filenames(youtube_url, is_shorts)
    if not output_filenames.get('text'): # Check if filename generation failed (e.g., bad URL)
         print("処理を中止します。")
         return
    print(f"  ブログ記事ファイル: {output_filenames['text']}")
    if not blog_only:
        print(f"  音声テキストファイル: {output_filenames['audio_text']}")
        print(f"  音声ファイル: {output_filenames['audio']}")
        print(f"  動画ファイル: {output_filenames['video']}")
        if use_wordcloud and not image_file:
            print(f"  ワードクラウドファイル: {output_filenames['wordcloud']}")

    # --- 2. Fetch Transcript --- 
    print("\n文字起こしを取得中...")
    transcript = fetch_transcript(youtube_url, language)
    # Handle potential errors from fetch_transcript (which now returns error messages)
    if not isinstance(transcript, str) or "取得できませんでした" in transcript or "失敗しました" in transcript or "字幕がありません" in transcript:
        print(f"文字起こし取得エラー: {transcript}")
        return
    print("文字起こし取得成功。")

    # --- 3. Generate Blog Article --- 
    print("\nブログ記事を生成中...")
    blog_article = generate_blog_article(transcript, youtube_url, language="ja")
    if not isinstance(blog_article, str) or "エラーが発生しました" in blog_article:
         print(f"ブログ記事生成エラー: {blog_article}")
         return 
    print("ブログ記事生成成功。")

    # --- 4. Save Blog Article --- 
    print(f"\nブログ記事をファイルに保存中 ({output_filenames['text']})...")
    save_to_file(blog_article, output_filenames['text'])
    
    # Exit if only blog generation is requested
    if blog_only:
        print(f"\nブログ記事のみ生成モード完了: {output_filenames['text']}")
        return
    
    # --- 5. Create and Save Audio Text --- 
    print(f"\n音声出力用テキストを生成・保存中 ({output_filenames['audio_text']})...")
    audio_text = create_audio_text(blog_article)
    save_to_file(audio_text, output_filenames['audio_text'])
    
    # --- 6. Text to Speech --- 
    print(f"\nテキストを音声に変換中 ({output_filenames['audio']})...")
    if not text_to_speech(audio_text, output_filenames['audio'], voice):
        print("音声ファイルの生成に失敗したため、動画変換をスキップします。")
        return # Stop processing if audio failed

    # Check if audio file was actually created
    if not os.path.exists(output_filenames['audio']) or os.path.getsize(output_filenames['audio']) == 0:
        print(f"エラー: 音声ファイル {output_filenames['audio']} が存在しないか空です。動画変換をスキップします。")
        return
    print("音声ファイル生成成功。")

    # --- 7. Prepare Image for Video --- 
    video_image_path = image_file # Use provided image first
    if not video_image_path: # If no image provided via args
        if use_wordcloud:
            print(f"\nワードクラウド画像を生成中 ({output_filenames['wordcloud']})...")
            if create_wordcloud_image(blog_article, output_filenames['wordcloud'], is_shorts):
                video_image_path = output_filenames['wordcloud'] # Use generated wordcloud
            else:
                print("ワードクラウド生成に失敗しました。デフォルトの黒背景を使用します。")
                video_image_path = None # Fallback to black background
        else:
            print("\n画像指定なし、ワードクラウド無効のため、デフォルトの黒背景を使用します。")
            video_image_path = None # Use black background

    # --- 8. Create Video --- 
    print(f"\n音声ファイルを動画に変換中 ({output_filenames['video']})...")
    create_video_from_audio(
        output_filenames['audio'], 
        output_filenames['video'], 
        video_image_path, # This is either the user image, wordcloud, or None
        is_shorts, 
        use_bgm
    )
    
    print("\n処理が完了しました。")

if __name__ == "__main__":
    main()
