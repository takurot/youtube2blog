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
            "content": f"""以下の文字起こし内容を元に、2000〜3000字程度の日本語の解説ブログ記事を作成してください。
                        記事は、読者が分かりやすいように構成し、第三者視点で重要なポイントを強調してください。
                        タイトルの次に動画へのURL {youtube_url} をリンク形式でなく文字列でそのまま入れてください。
                        最初の項目は「ポイント」として、主な主張や論点の考察を箇条書きで記載して、ここだけ読めばおおよその概要がわかるようにしてください。
                        文字起こし内容に関連するトピックを適宜マッシュアップし、独自の見解や意見を述べてください。
                        最後の項目は「まとめ」として、尖った意見を述べてください。
                        太字(*)は使わないでください。見出し（#）は使ってください。
                        タイトルには関連するハッシュタグを入れてください。

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

def text_to_speech(content, filename, voice=None):
    """生成されたブログテキストをTTS APIで音声化し、mp3ファイルに保存"""
    try:
        # Pathオブジェクトを使用してファイルパスを作成
        speech_file_path = Path(filename)
        
        # 音声が指定されていない場合はランダムに選択
        if voice is None:
            voice = random.choice(AVAILABLE_VOICES)
        
        print(f"選択された音声: {voice}")
        
        # 最新の推奨方法（with_streaming_response）を使用
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",  # 最新のTTSモデルを使用
            voice=voice,  # ランダムに選択された音声または指定された音声
            input=content,
        ) as response:
            # チャンクごとにファイルに書き込む方法
            with open(speech_file_path, 'wb') as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
        
        print(f"音声ファイルが {filename} に保存されました。")
        return True
    except Exception as e:
        print(f"音声ファイル作成中にエラーが発生しました: {e}")
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
                
                audio_text_lines.append(line)
        
        # 空白行を削除して結合
        audio_text = '\n'.join([line for line in audio_text_lines if line.strip()])
        return audio_text
    
    except Exception as e:
        print(f"音声出力用テキスト生成中にエラーが発生しました: {e}")
        return blog_content  # エラーが発生した場合は元のブログ記事をそのまま返す

def create_video_from_audio(audio_file, output_file, image_file=None, is_shorts=False):
    """音声ファイルと静止画からmp4動画を作成（YouTube向け）"""
    try:
        # デフォルトの黒背景を使用するか、指定された画像ファイルを使用
        if image_file and os.path.exists(image_file):
            # 指定された画像を使用
            image_input = f"-loop 1 -i \"{image_file}\""
        else:
            # ショート動画かどうかでサイズを変える
            if is_shorts:
                # 縦長の黒い背景（1080x1920）
                image_input = "-f lavfi -i color=c=black:s=1080x1920"
            else:
                # 横長の黒い背景（1920x1080）
                image_input = "-f lavfi -i color=c=black:s=1920x1080"
        
        # FFmpegコマンドを構築
        cmd = f'ffmpeg -y {image_input} -i "{audio_file}" -c:v libx264 -tune stillimage -c:a aac -b:a 192k -pix_fmt yuv420p -shortest "{output_file}"'
        
        # FFmpegコマンドを実行
        subprocess.call(cmd, shell=True)
        print(f"動画ファイルが {output_file} に保存されました。")
        return True
    except Exception as e:
        print(f"動画ファイル作成中にエラーが発生しました: {e}")
        return False

def create_wordcloud_image(text, output_path, is_shorts=False):
    """テキストからワードクラウド画像を生成"""
    try:
        # ショート動画かどうかで画像サイズを変える
        if is_shorts:
            # YouTube Shortsサイズ (9:16)
            width, height = 1080, 1920
        else:
            # 通常の16:9動画サイズ
            width, height = 1920, 1080
        
        # 日本語テキストを形態素解析して単語の頻度を数える
        t = Tokenizer()
        
        # テキストをUTF-8として明示的に処理
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        else:
            # 念のためstr型に変換
            text = str(text)
        
        try:
            tokens = t.tokenize(text)
            
            word_frequencies = {}
            # 名詞、動詞、形容詞のみを対象にする（助詞や記号などは除外）
            valid_pos = ['名詞', '動詞', '形容詞']
            
            for token in tokens:
                pos = token.part_of_speech.split(',')[0]  # 品詞の最初の部分
                
                # 特定の品詞のみをカウント
                if pos in valid_pos:
                    word = token.surface
                    if len(word) > 2:  # 3文字以上の単語のみを対象に
                        if word in word_frequencies:
                            word_frequencies[word] += 1
                        else:
                            word_frequencies[word] = 1
        except Exception as tokenize_error:
            print(f"形態素解析中にエラーが発生しました: {tokenize_error}")
            # 単語分割を行わず、スペースで区切られた単語としてカウント
            word_frequencies = {}
            for word in text.split():
                if len(word) > 2:
                    if word in word_frequencies:
                        word_frequencies[word] += 1
                    else:
                        word_frequencies[word] = 1
        
        # 背景色（青色ベース）
        bg_color = "rgb(20, 60, 140)"  # 青色
        
        # フォントパスを設定
        try:
            # macOSの場合のフォントパス
            font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc'
            if not os.path.exists(font_path):
                # 別のmacOSフォントパスを試す
                font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
                if not os.path.exists(font_path):
                    # Windowsの場合のフォントパス
                    font_path = 'C:\\Windows\\Fonts\\meiryo.ttc'
                    if not os.path.exists(font_path):
                        # 代替フォントパス
                        font_path = '/System/Library/Fonts/Arial Unicode.ttf'
                        if not os.path.exists(font_path):
                            print("適切なフォントが見つかりませんでした。デフォルトフォントを使用します。")
                            font_path = None
        except Exception as font_error:
            print(f"フォントパス設定中にエラーが発生しました: {font_error}")
            font_path = None
        
        try:
            # WordCloudオブジェクトを作成（マスクなし）
            wc = WordCloud(
                background_color=bg_color,
                width=width,
                height=height,
                font_path=font_path,
                colormap='YlOrRd',  # 黄色からオレンジ、赤へのグラデーション
                min_font_size=12,
                max_font_size=None,  # 自動調整
                random_state=42,
                prefer_horizontal=0.9,  # 90%の単語を水平に
                contour_width=0,  # 輪郭線なし
                scale=1.2,  # 単語の密度を調整
                relative_scaling=0.7,  # 単語のサイズが頻度にどの程度比例するか
                normalize_plurals=False,  # 複数形を正規化しない (日本語では不要)
                regexp=r"[\w\p{Han}\p{Hiragana}\p{Katakana}]+"  # 日本語に対応する正規表現
            )
        except Exception as wc_init_error:
            print(f"WordCloudオブジェクト初期化中にエラーが発生しました: {wc_init_error}")
            # 最小限の設定でWordCloudを作成
            wc = WordCloud(
                width=width, 
                height=height,
                background_color=bg_color
            )
        
        # 単語の頻度からワードクラウドを生成
        if word_frequencies:
            try:
                wc.generate_from_frequencies(word_frequencies)
                
                # 画像を直接ファイルに保存
                wc.to_file(output_path)
                print(f"ワードクラウド画像が {output_path} に保存されました。単語数: {len(word_frequencies)}")
                return True
            except Exception as generate_error:
                print(f"ワードクラウド生成中にエラーが発生しました: {generate_error}")
                return False
        else:
            print("ワードクラウド生成に十分な単語が見つかりませんでした。")
            return False
    
    except Exception as e:
        print(f"ワードクラウド画像生成中にエラーが発生しました: {e}")
        return False

def main():
    # 引数の設定
    parser = argparse.ArgumentParser(description="YouTube動画の字幕を取得し、それを基にブログ記事を生成するスクリプト")
    parser.add_argument("language", help="取得したい字幕の言語コード（例: 'en', 'ja'）")
    parser.add_argument("youtube_url", help="YouTube動画のURLを指定")
    parser.add_argument("--image", help="動画作成に使用する静止画像ファイルのパス", default=None)
    parser.add_argument("--shorts", action="store_true", help="YouTube Shorts形式（縦長動画）で出力する")
    parser.add_argument("--voice", choices=AVAILABLE_VOICES, help=f"使用する音声を指定 (選択肢: {', '.join(AVAILABLE_VOICES)}). 指定しない場合はランダム")
    parser.add_argument("--wordcloud", action="store_true", help="ワードクラウド画像を生成して使用する")
    
    args = parser.parse_args()
    youtube_url = args.youtube_url
    language = args.language
    image_file = args.image
    is_shorts = args.shorts
    voice = args.voice
    use_wordcloud = args.wordcloud

    # 文字起こしを取得
    print("文字起こしを取得中...")
    transcript = fetch_transcript(youtube_url, language)
    if "取得できませんでした" in transcript:
        print(transcript)
        return

    # ブログ記事を生成
    print("ブログ記事を生成中...")
    blog_article = generate_blog_article(transcript, youtube_url, language="ja")

    # ファイルに保存
    today = datetime.now().strftime('%Y%m%d')
    video_id = get_video_id(youtube_url)
    text_filename = f"{today}_blog_article_{video_id}.txt"
    save_to_file(blog_article, text_filename)
    
    # 音声出力用のテキストを生成
    print("音声出力用テキストを生成中...")
    audio_text = create_audio_text(blog_article)
    audio_text_filename = f"{today}_blog_audio_text_{video_id}.txt"
    save_to_file(audio_text, audio_text_filename)
    
    # テキストを音声に変換して保存
    print("テキストを音声に変換中...")
    audio_filename = f"{today}_blog_audio_{video_id}.mp3"
    
    # 音声出力用テキストを使用して音声を生成
    if text_to_speech(audio_text, audio_filename, voice):
        # 画像が指定されていない場合
        if not image_file:
            # ワードクラウドの生成（オプションが指定されている場合）
            if use_wordcloud:
                wordcloud_filename = f"{today}_wordcloud_{video_id}.png"
                print("ワードクラウド画像を生成中...")
                if create_wordcloud_image(blog_article, wordcloud_filename, is_shorts):
                    image_file = wordcloud_filename
                # ワードクラウド生成に失敗した場合はデフォルト画像を使用
            else:
                # デフォルト画像を使用
                print("ワードクラウドを使用せず、デフォルト画像または指定された画像を使用します")
        
        # 音声ファイルを動画に変換
        print("音声ファイルを動画に変換中...")
        format_suffix = "_shorts" if is_shorts else ""
        video_filename = f"{today}_blog_video{format_suffix}_{video_id}.mp4"
        create_video_from_audio(audio_filename, video_filename, image_file, is_shorts)

if __name__ == "__main__":
    main()
