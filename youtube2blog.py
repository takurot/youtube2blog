import argparse
import re
import textwrap
import random
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
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
    """YouTube URLから文字起こしを取得し、セグメントのリスト（各要素はtext, start, durationを含む辞書）を返す"""
    video_id = get_video_id(youtube_url)
    try:
        # トランスクリプトを取得
        transcript_list_api = YouTubeTranscriptApi.list_transcripts(video_id)
        
        try:
            # 指定された言語のトランスクリプトを検索
            transcript = transcript_list_api.find_transcript([language])
        except NoTranscriptFound:
            print(f"指定された言語 '{language}' の文字起こしが見つかりません。利用可能な言語を試みます...")
            # 利用可能な言語で最初に見つかったものを試す (手動または生成されたもの)
            available_langs = [t.language_code for t in transcript_list_api]
            if not available_langs:
                 return {"error": "この動画には字幕がありません。", "data": None}
            
            found_transcript = False
            for lang_code in available_langs:
                try:
                    transcript = transcript_list_api.find_transcript([lang_code])
                    print(f"代わりに言語 '{lang_code}' を使用します。")
                    language = lang_code # Update language for consistency
                    found_transcript = True
                    break
                except NoTranscriptFound:
                    continue
            if not found_transcript:
                 return {"error": "利用可能な言語の文字起こしも見つかりませんでした。", "data": None}

        # トランスクリプトデータを取得 (辞書のリスト形式)
        transcript_data = transcript.fetch()
        
        if transcript_data:
            # 念のため形式をチェック (リストであり、各要素が辞書であること)
            if isinstance(transcript_data, list) and all(isinstance(item, dict) for item in transcript_data):
                print(f"文字起こしを正常に取得しました。言語: {language}, セグメント数: {len(transcript_data)}")
                return {"error": None, "data": transcript_data, "language": language}
            else:
                error_msg = f"予期しない形式のトランスクリプトデータ: {type(transcript_data)}"
                print(f"デバッグ情報: {error_msg}")
                return {"error": error_msg, "data": None}
        else:
            # これは通常発生しないはず (fetch()が空リストを返す場合など)
            error_msg = f"動画ID {video_id} ({language}) の文字起こしデータの取得に失敗しました（空データ）。"
            print(error_msg)
            return {"error": error_msg, "data": None}
            
    except TranscriptsDisabled:
        error_msg = f"動画ID {video_id} の文字起こしは無効になっています。"
        print(error_msg)
        return {"error": error_msg, "data": None}
    except ET.ParseError as e:
        error_message = f"字幕データのパースに失敗しました（ParseError）: {str(e)}"
        print(error_message)
        return {"error": error_message, "data": None}
    except Exception as e:
        error_message = f"文字起こし取得中に予期せぬエラーが発生しました: {str(e)}"
        print(f"デバッグ情報: {e.__class__.__name__}")
        return {"error": error_message, "data": None}

def generate_blog_article(transcript_data: list[dict], youtube_url: str, language: str = "ja") -> tuple[str | None, list | None, str | None]:
    """文字起こしデータ (セグメントのリスト) を基にOpenAI APIを使ってブログ記事とタイムスタンプ情報を生成。
    戻り値: (ブログ記事文字列, タイムスタンプ情報リスト, エラーメッセージ)
    """
    if not transcript_data:
        return None, None, "文字起こしデータが空です。"

    # LLMに渡すために文字起こしセグメントを整形
    # 各セグメントにIDを付与し、開始時間とテキストを含める
    formatted_transcript_for_llm = []
    for i, segment in enumerate(transcript_data):
        start_time = segment.get('start', 0)
        duration = segment.get('duration', 0)
        end_time = start_time + duration
        text = segment.get('text', '').strip()
        if text: # 空のテキストセグメントはスキップ
            formatted_transcript_for_llm.append(f"[Segment {i:03d} | Start: {start_time:.2f}s | End: {end_time:.2f}s] {text}")
    
    if not formatted_transcript_for_llm:
        return None, None, "有効な文字起こしセグメントがありませんでした。"

    transcript_string_for_llm = "\\n".join(formatted_transcript_for_llm)
    
    # LLMへの指示 (プロンプト)
    # タイムスタンプ情報をJSON形式で出力させるように依頼
    # LLMがJSONをうまく生成できない場合に備えて、特定の区切り文字を使うことも検討できる
    prompt_content = f"""以下のYouTube動画の文字起こしを元に、日本語の解説ブログ記事を作成してください。
文字起こしは各セグメントに [Segment ID | Start: 開始時間s | End: 終了時間s] の形式で情報が付与されています。

ブログ記事の要件:
1. 読者が分かりやすいように、より詳細な説明や具体的な例を交えながら構成し、第三者視点で重要なポイントを深く掘り下げて強調してください。
2. 記事全体の文字数は2000〜2500字程度にしてください。
3. タイトルの次に動画のURL ({youtube_url}) を文字列としてそのまま記載してください。
4. 最初の項目は「## ポイント」として、主な主張や論点の考察を箇条書きで記載し、ここだけ読めば概要がわかるようにしてください。各ポイントは具体的で、詳細な説明や背景情報も適宜含めてください。
5. 文字起こし内容に関連するトピックを適宜マッシュアップし、独自の見解や意見、さらには具体的な事例や応用例を交えながら述べてください。
6. 最後の項目は「## まとめ」として、記事全体の要点を簡潔に、かつ読者の行動を促すような形でまとめてください。
7. 太字表現 (*) は使用しないでください。見出し (## や ###) は適切に使用してください。

出力形式の要件:
生成するブログ記事と、各段落が参照した文字起こしセグメントの情報を、以下のJSON形式で出力してください。
```json
{{
  "blog_article": "ここに生成されたブログ記事の全文をマークダウン形式で記述...",
  "timeline_references": [
    {{
      "paragraph_index": <段落番号 (0始まりの整数)>,
      "paragraph_text_snippet": "<該当段落の冒頭20文字程度>",
      "source_segment_ids": ["Segment 001", "Segment 002", ...] // 参照した文字起こしセグメントのIDのリスト
    }},
    // ... 他の段落についても同様に記述
  ]
}}
```
- `blog_article`: 生成されたブログ記事の全文です。マークダウン形式で記述してください。
- `timeline_references`: ブログの各段落（主要なコンテンツブロック単位で判断）が、どの文字起こしセグメントに基づいているかの対応リストです。
  - `paragraph_index`: 段落の通し番号（0から始まる整数）。
  - `paragraph_text_snippet`: LLMが判断した段落の冒頭の短いスニペット（照合用）。
  - `source_segment_ids`: その段落の生成根拠となった文字起こしセグメントのID（例: "Segment 001"）をリストで記述してください。複数のセグメントを参照した場合は、全て含めてください。段落が特定のセグメントに直接対応しない場合（例: 全体的な導入やまとめ、LLM独自の考察部分）は、`source_segment_ids` を空リスト `[]` としてください。
  - 「ポイント」セクションの箇条書きの各項目も、可能であればそれぞれを1つの「段落」として扱い、参照セグメントを記録してください。難しい場合は、「ポイント」全体を一つの段落として扱っても構いません。
  - 見出し自体は `timeline_references` に含める必要はありません。本文の段落のみを対象とします。

文字起こし:
{transcript_string_for_llm}
"""

    messages = [
        { 
            "role": "system", 
            "content": "あなたはプロのブロガーであり、指示された形式で情報を正確に出力できるアシスタントです。" 
        },
        { 
            "role": "user", 
            "content": prompt_content
        }
    ]

    try:
        print("LLMにブログ記事とタイムスタンプ情報の生成をリクエストします...")
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.5, # 精度重視のため少し下げる
            max_tokens=4095, # JSON構造も含むため、十分な長さを確保 (GPT-4o max is 4096 for output)
            response_format={ "type": "json_object" } # GPT-4o and later
        )
        
        raw_response_content = response.choices[0].message.content.strip()
        
        # LLMからの応答（JSON文字列）をパース
        import json
        try:
            parsed_response = json.loads(raw_response_content)
            blog_article_text = parsed_response.get("blog_article")
            timeline_references_raw = parsed_response.get("timeline_references")

            if not blog_article_text or not isinstance(timeline_references_raw, list):
                print("エラー: LLMからの応答形式が不正です（blog_articleまたはtimeline_referencesが欠損）。")
                print(f"Raw response: {raw_response_content[:500]}...") # Log part of the raw response
                return None, None, "LLMからの応答形式が不正です。"

            # Clean the blog article text from any [Segment ...] annotations
            if blog_article_text:
                # Matches "[Segment" followed by any characters except "]" until "]"
                # Handles multi-digit segment numbers and potential extra spaces.
                blog_article_text = re.sub(r"\\s*\\[Segment[^]]+\\]\\s*", "", blog_article_text)
                # Also remove any remaining empty lines that might result from the substitution
                blog_article_text = re.sub(r"\\n\\s*\\n", "\\n\\n", blog_article_text).strip()

            # タイムスタンプ情報を整形 (source_segment_idsから実際の時間情報を復元)
            final_timeline_references = []
            source_segments_map = {f"Segment {i:03d}": seg for i, seg in enumerate(transcript_data) if seg.get('text','').strip()}

            for ref_item_raw in timeline_references_raw:
                blog_point_id = ref_item_raw.get("paragraph_index")
                snippet = ref_item_raw.get("paragraph_text_snippet", "")
                segment_ids = ref_item_raw.get("source_segment_ids", [])
                
                video_clips_for_point = []
                for seg_id in segment_ids:
                    original_segment = source_segments_map.get(seg_id)
                    if original_segment:
                        start_time = original_segment.get('start', 0)
                        duration = original_segment.get('duration', 0)
                        video_clips_for_point.append({
                            "start_time": start_time,
                            "end_time": start_time + duration,
                            "transcript_snippet": original_segment.get('text', '')[:100] # 冒頭100文字
                        })
                
                if blog_point_id is not None: # 必須項目
                    final_timeline_references.append({
                        "blog_point_id": blog_point_id,
                        "blog_point_text_snippet": snippet,
                        "video_clips": video_clips_for_point
                    })
            
            print("ブログ記事とタイムスタンプ情報を正常に生成・パースしました。")
            return blog_article_text, final_timeline_references, None

        except json.JSONDecodeError as json_e:
            print(f"LLMからの応答のJSONパースに失敗しました: {json_e}")
            print(f"Raw response: {raw_response_content[:1000]}...") # Log part of the raw response
            return None, None, f"LLM応答のJSONパース失敗: {json_e}"
        except Exception as e_parse:
            print(f"LLM応答の処理中に予期せぬエラーが発生しました: {e_parse}")
            return None, None, f"LLM応答処理エラー: {e_parse}"

    except Exception as e_llm:
        error_msg = f"ブログ記事の生成中にエラーが発生しました: {e_llm}"
        print(error_msg)
        return None, None, error_msg

def save_to_file(content, filename):
    """生成されたブログ記事をファイルに保存"""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"ブログ記事が {filename} に保存されました。")
    except Exception as e:
        print(f"ファイル保存中にエラーが発生しました: {e}")

def save_timestamps_to_file(timestamps_data: list, filename: str):
    """タイムスタンプ情報をJSONファイルに保存"""
    if not timestamps_data:
        print("タイムスタンプデータが空のため、ファイル保存をスキップします。")
        return
    try:
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(timestamps_data, f, ensure_ascii=False, indent=4)
        print(f"タイムスタンプ情報が {filename} に保存されました。")
    except Exception as e:
        print(f"タイムスタンプ情報のファイル保存中にエラーが発生しました: {e}")

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
            'wordcloud': None, 'video': None, 'timestamps': None
        }
        
    today = datetime.now().strftime('%Y%m%d')
    base_name = f"{today}_blog_{video_id}"
    
    filenames = {
        'text': f"{base_name}_article.txt",
        'audio_text': f"{base_name}_audio_text.txt",
        'audio': f"{base_name}_audio.mp3",
        'wordcloud': f"{base_name}_wordcloud.png",
        'video': f"{base_name}_video{'_shorts' if is_shorts else ''}.mp4",
        'timestamps': f"{base_name}_timestamps.json"  # タイムスタンプファイル名を追加
    }
    
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

    # --- 1. Setup --- 
    output_filenames = _generate_output_filenames(youtube_url, args.shorts)
    if any(value is None for value in output_filenames.values()):
        print("ファイル名の生成に失敗しました。処理を中断します。")
        return

    article_file = output_filenames['text']
    audio_text_file = output_filenames['audio_text']
    speech_file = output_filenames['audio']
    wordcloud_file = output_filenames['wordcloud']
    video_file = output_filenames['video']
    timestamps_file = output_filenames['timestamps'] # タイムスタンプファイル名を取得

    # --blog-onlyが指定された場合は、記事と音声合成用のテキストのみ生成
    # if blog_only:
    #     print(f"\nブログ記事のみ生成モード完了: {article_file}")
    #     return # ここにあった早期リターンを移動

    # --- 2. Fetch Transcript --- 
    print("\n文字起こしを取得中...")
    transcript_response = fetch_transcript(youtube_url, language)
    # Handle potential errors from fetch_transcript (which now returns error messages)
    if transcript_response.get('error'):
        print(f"文字起こし取得エラー: {transcript_response['error']}")
        return
    
    transcript_data = transcript_response.get('data')
    fetched_language = transcript_response.get('language', language) # LLMに渡す言語を更新

    if not transcript_data:
        print("文字起こしデータが取得できませんでした（データが空です）。")
        return
    
    print(f"文字起こし取得成功。言語: {fetched_language}")

    # --- 3. Generate Blog Article & Timestamps ---
    print("\nブログ記事とタイムスタンプ情報を生成中...")
    blog_article_text, timeline_references, error_message = generate_blog_article(transcript_data, youtube_url, language=fetched_language)
    
    if error_message:
        print(f"ブログ記事生成エラー: {error_message}")
        # エラーがあっても、部分的にテキストが生成されている可能性があるので、保存を試みる
        if blog_article_text:
            print("部分的なブログ記事を保存します...")
            save_to_file(blog_article_text, article_file)
        return

    if not blog_article_text:
        print("ブログ記事の生成に失敗しました（テキストが空です）。")
        return

    print("ブログ記事生成成功。")
    save_to_file(blog_article_text, article_file)

    # タイムスタンプ情報を保存
    if timeline_references:
        save_timestamps_to_file(timeline_references, timestamps_file)
    else:
        print("タイムスタンプ情報は生成されませんでした。")

    # --blog-only が指定された場合は、ここで処理を終了
    if blog_only:
        print(f"\nブログ記事生成完了 (音声・動画はスキップ): {article_file}")
        if os.path.exists(timestamps_file):
             print(f"タイムスタンプ情報も生成されました: {timestamps_file}")
        return

    # --- 4. Create and Save Audio Text --- 
    print(f"\n音声出力用テキストを生成・保存中 ({audio_text_file})...")
    audio_text = create_audio_text(blog_article_text)
    save_to_file(audio_text, audio_text_file)
    
    # --- 5. Text to Speech --- 
    print(f"\nテキストを音声に変換中 ({speech_file})...")
    if not text_to_speech(audio_text, speech_file, voice):
        print("音声ファイルの生成に失敗したため、動画変換をスキップします。")
        return # Stop processing if audio failed

    # Check if audio file was actually created
    if not os.path.exists(speech_file) or os.path.getsize(speech_file) == 0:
        print(f"エラー: 音声ファイル {speech_file} が存在しないか空です。動画変換をスキップします。")
        return
    print("音声ファイル生成成功。")

    # --- 6. Prepare Image for Video --- 
    video_image_path = image_file # Use provided image first
    if not video_image_path: # If no image provided via args
        if use_wordcloud:
            print(f"\nワードクラウド画像を生成中 ({wordcloud_file})...")
            if create_wordcloud_image(blog_article_text, wordcloud_file, is_shorts):
                video_image_path = wordcloud_file # Use generated wordcloud
            else:
                print("ワードクラウド生成に失敗しました。デフォルトの黒背景を使用します。")
                video_image_path = None # Fallback to black background
        else:
            print("\n画像指定なし、ワードクラウド無効のため、デフォルトの黒背景を使用します。")
            video_image_path = None # Use black background

    # --- 7. Create Video --- 
    print(f"\n音声ファイルを動画に変換中 ({video_file})...")
    create_video_from_audio(
        speech_file, 
        video_file, 
        video_image_path, # This is either the user image, wordcloud, or None
        is_shorts, 
        use_bgm
    )
    
    print("\n処理が完了しました。")

if __name__ == "__main__":
    main()
