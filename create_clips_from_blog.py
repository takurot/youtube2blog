import os
import re
# from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound # Keep for now if original video download is needed
import json # For loading timestamps
import subprocess # For FFmpeg
import argparse # For command line arguments

# Janome Tokenizer and related functions are no longer needed.
# TOKENIZER = None
# try:
#     TOKENIZER = Tokenizer()
# except Exception as e:
#     print(f"Janome Tokenizer の初期化に失敗しました: {e}")
#     print("Janomeがインストールされているか確認してください: pip install janome")

def get_video_id_from_filename(article_file_path: str) -> str | None:
    """
    Extracts the YouTube video ID from a blog article filename.
    Assumes filename format like: YYYYMMDD_blog_VIDEOID_article.txt
    """
    basename = os.path.basename(article_file_path)
    print(f"[Debug get_video_id] basename: {basename}") # DEBUG
    # Updated regex to be more specific to _article.txt if used for initial lookup
    # Expects YYYYMMDD_blog_VIDEOID_article.txt
    match = re.search(r'^(\d{8}_blog_([a-zA-Z0-9_-]+)_article)\.txt$', basename)
    if match:
        return match.group(2) # Group 2 is the VIDEOID
    
    # Fallback for a more general pattern if the first one (for _article.txt) fails
    # This could match YYYYMMDD_blog_VIDEOID_text.txt etc.
    match = re.search(r'^(\d{8}_blog_([a-zA-Z0-9_-]+)_(?:video|wordcloud|audio|text|article))\.txt$', basename)
    if match:
        return match.group(2) # Group 2 is still the VIDEOID

    # Fallback for even more general patterns if the filename structure is less predictable
    # (e.g. if the YYYYMMDD part is not digits or not 8 characters)
    match = re.search(r'([^_]+_blog_([a-zA-Z0-9_-]+)_[^_]+)\.txt$', basename)
    if match:
        return match.group(2)
    
    print(f"エラー: ファイル名から動画IDを抽出できませんでした: {basename}")
    return None

def get_base_filename_from_article(article_file_path: str) -> str | None:
    """
    Extracts the base part of the filename (e.g., YYYYMMDD_blog_VIDEOID) from article.txt
    """
    basename = os.path.basename(article_file_path)
    print(f"[Debug get_base_filename] basename: {basename}") # DEBUG
    match = re.search(r'([^_]+_blog_[a-zA-Z0-9_-]+)_article\.txt$', basename)
    if match:
        return match.group(1)
    print(f"エラー: 記事ファイル名からベース名を抽出できませんでした: {basename}")
    return None

def parse_blog_post(article_file_path: str) -> list[str]:
    """
    Reads a blog post file and splits its content into paragraphs.
    The goal is to produce a list of strings where each string corresponds
    to a paragraph/blog point that has a blog_point_id in the timestamp file.
    """
    if not os.path.exists(article_file_path):
        print(f"エラー: 記事ファイルが見つかりません: {article_file_path}")
        return []
    try:
        with open(article_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        potential_blocks = re.split(r'(?:\r?\n\s*){2,}', content.strip())
        
        blog_points = []
        title_skipped = False
        url_skipped = False

        for block_text in potential_blocks:
            block_text = block_text.strip()
            if not block_text:
                continue

            if not title_skipped and block_text.startswith("# "):
                title_skipped = True
                print(f"[parse_blog_post Debug] Skipping title: {block_text[:60]}...")
                continue

            if not url_skipped and ("youtube.com/watch?v=" in block_text or "youtu.be/" in block_text) and len(block_text.splitlines()) == 1:
                url_skipped = True
                print(f"[parse_blog_post Debug] Skipping URL: {block_text}")
                continue
            
            cleaned_block_text_for_check = block_text.lower().strip('#').strip()
            if cleaned_block_text_for_check == "ポイント" or cleaned_block_text_for_check == "まとめ":
                 print(f"[parse_blog_post Debug] Skipping section header: {block_text}")
                 continue

            text_to_add = block_text

            if text_to_add:
                print(f"[parse_blog_post Debug] Adding point: {text_to_add[:60]}...")
                blog_points.append(text_to_add)
            else:
                print(f"[parse_blog_post Debug] Skipping empty block after processing: {block_text[:60]}...")

        if not blog_points and content.strip():
            print("警告: 新しいブログポイントのパース処理ですべてのポイントが削除されました。")
            print("フォールバックとして、改行で分割し、基本的なフィルタリングのみ行います。")
            lines = content.strip().splitlines()
            temp_points = []
            first_line_skipped_fallback = False # Use a different variable name for fallback
            url_skipped_fallback = False      # Use a different variable name for fallback
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Fallback: Skip first line if it looks like a title
                if not first_line_skipped_fallback and line_idx == 0 and line.startswith("# "):
                    first_line_skipped_fallback = True
                    print(f"[parse_blog_post Fallback Debug] Skipping title: {line[:60]}...")
                    continue
                
                # Fallback: Skip URL line (simple check)
                if not url_skipped_fallback and ("youtube.com/watch?v=" in line or "youtu.be/" in line) and len(line.split()) < 5:
                     url_skipped_fallback = True
                     print(f"[parse_blog_post Fallback Debug] Skipping URL: {line}")
                     continue
                
                # Fallback: Skip section headers
                cleaned_line_for_check = line.lower().strip('#').strip()
                if cleaned_line_for_check == "ポイント" or cleaned_line_for_check == "まとめ":
                     print(f"[parse_blog_post Fallback Debug] Skipping section header: {line}")
                     continue
                
                print(f"[parse_blog_post Fallback Debug] Adding point: {line[:60]}...")
                temp_points.append(line)
            
            # This return should be inside the `if not blog_points and content.strip():` block
            return temp_points 

        # This is the primary return if the main logic succeeded
        return blog_points
        
    except Exception as e:
        print(f"記事ファイルの読み込みまたはパース中にエラーが発生しました: {e}")
        return []

# fetch_transcript_segments is no longer needed as timestamps come from JSON file.
# def fetch_transcript_segments(video_id: str, language: str = "ja", retries: int = 3, delay: int = 5) -> list[dict] | None:
#     ...

# Tokenization and Jaccard similarity are no longer the primary alignment method.
# def _tokenize_text(text: str) -> list[str]:
#     ...
# def _calculate_jaccard_similarity(set1: set[str], set2: set[str]) -> float:
#     ...
# def align_points_to_transcript(blog_points: list[str], transcript_segments: list[dict], ...) -> list[dict]:
#    ...

def load_timestamp_data(timestamp_file_path: str) -> list[dict] | None:
    """Loads and parses the timestamp JSON file."""
    if not os.path.exists(timestamp_file_path):
        print(f"エラー: タイムスタンプファイルが見つかりません: {timestamp_file_path}")
        return None
    try:
        with open(timestamp_file_path, 'r', encoding='utf-8') as f:
            timestamp_data = json.load(f)
        if not isinstance(timestamp_data, list):
            print(f"エラー: タイムスタンプファイルの形式が不正です（リストではありません）。Path: {timestamp_file_path}")
            return None
        print(f"タイムスタンプファイルを正常に読み込みました: {timestamp_file_path}")
        return timestamp_data
    except json.JSONDecodeError as e:
        print(f"タイムスタンプファイルのJSONパース中にエラー: {e}. Path: {timestamp_file_path}")
        return None
    except Exception as e:
        print(f"タイムスタンプファイルの読み込み中に予期せぬエラー: {e}. Path: {timestamp_file_path}")
        return None

def get_clips_info_from_timestamps(blog_points_from_article: list[str], timestamp_data: list[dict]) -> list[dict]:
    """
    Matches blog points to timestamp data and extracts video clip information.
    The primary source of truth for what clips to generate is the timestamp_data.
    blog_points_from_article is used to fetch the full text for the summary.
    Returns a list of dictionaries, each representing a blog point with its associated video segments.
    """
    print("\nブログポイントとタイムスタンプ情報の照合を開始します...")
    aligned_clips_info = []
    
    if not timestamp_data:
        print("タイムスタンプデータが空のため、処理をスキップします。")
        # If there's no timestamp data, we can't generate clips based on it.
        # Optionally, we could return blog_points_from_article with empty clips, but the goal is to use timestamp_data as the driver.
        return []

    # Iterate through the timestamp_data, as this defines what points have potential clips
    for ts_entry in timestamp_data:
        blog_point_id = ts_entry.get('blog_point_id')
        video_clips_from_ts = ts_entry.get("video_clips", [])
        snippet_from_ts = ts_entry.get("blog_point_text_snippet", "")

        if blog_point_id is None: # Should ideally not happen if JSON is well-formed
            print(f"警告: blog_point_id が見つからないタイムスタンプエントリをスキップします: {ts_entry}")
            continue

        # Try to get the full blog point text from the parsed article using the blog_point_id as an index
        point_text_from_article = ""
        if blog_points_from_article and 0 <= blog_point_id < len(blog_points_from_article):
            point_text_from_article = blog_points_from_article[blog_point_id]
            # f-string内のバックスラッシュを避けるために、改行置換を先に行う
            display_text_snippet = point_text_from_article[:30].replace('\n', ' ')
            print(f"  処理中 blog_point_id {blog_point_id}: 「{display_text_snippet}...」 (記事本文より)")
        else:
            # Fallback to snippet if full text is not available or index is out of bounds
            point_text_from_article = snippet_from_ts 
            display_text_snippet = snippet_from_ts[:30].replace('\n', ' ')
            print(f"  処理中 blog_point_id {blog_point_id}: 「{display_text_snippet}...」 (スニペットより - 記事本文の対応ポイントなし)")

        current_point_info = {
            "blog_point_id": blog_point_id,
            "blog_point_text": point_text_from_article, # Use full text if available, else snippet
            "clips": video_clips_from_ts # This can be an empty list
        }

        if video_clips_from_ts:
            print(f"    タイムスタンプエントリ (ID: {blog_point_id}) に基づいてクリップ情報を追加します。")
            for clip in video_clips_from_ts:
                print(f"      - クリップ: {clip['start_time']:.2f}s - {clip['end_time']:.2f}s")
        else:
            print(f"    タイムスタンプエントリ (ID: {blog_point_id}) には動画クリップ情報がありませんでした。")
            
        aligned_clips_info.append(current_point_info)
        
    return aligned_clips_info

def create_video_clips(video_id: str, clips_info: list[dict], output_dir: str = "output_clips", original_video_path: str | None = None):
    """
    Creates video clips using FFmpeg based on the aligned_clips_info.
    Requires the original YouTube video to be downloaded or its path provided.
    """
    if not clips_info:
        print("切り出すクリップ情報がありません。")
        return

    base_video_filename = f"{video_id}_original.mp4" # Default name for downloaded video

    if not original_video_path:
        print(f"オリジナル動画のパスが指定されていません。動画ID: {video_id} をyt-dlpでダウンロード試行します。")
        # Check if the video already exists locally (e.g., from a previous download)
        if os.path.exists(base_video_filename):
            print(f"ローカルに動画 {base_video_filename} が見つかりました。これを使用します。")
            original_video_path = base_video_filename
        else:
            try:
                # Download the video using yt-dlp
                # Format: best video with audio, mp4 preferred.
                # If specific resolution is needed, add e.g. -f "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
                yt_dlp_command = [
                    "yt-dlp",
                    "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best", # Prefer MP4
                    "-o", base_video_filename,
                    f"https://www.youtube.com/watch?v={video_id}"
                ]
                print(f"実行するダウンロードコマンド: {' '.join(yt_dlp_command)}")
                
                result = subprocess.run(yt_dlp_command, check=True, capture_output=True, text=True)
                print(f"動画ダウンロード成功: {base_video_filename}")
                print(f"""yt-dlp stdout:
{result.stdout}""")
                original_video_path = base_video_filename
            except FileNotFoundError:
                print("エラー: yt-dlpコマンドが見つかりません。yt-dlpがインストールされ、PATHに追加されていることを確認してください。")
                return
            except subprocess.CalledProcessError as e:
                print(f"yt-dlpでの動画ダウンロード中にエラーが発生しました。")
                print(f"  Return code: {e.returncode}")
                # stderrとstdoutをより詳細に出力
                print(f"  yt-dlp stderr:\n{e.stderr}")
                print(f"  yt-dlp stdout:\n{e.stdout}")
                print(f"動画をダウンロードできませんでした。処理を中断します。手動でダウンロードして --video_path で指定してください。")
                return
            except Exception as e:
                print(f"動画ダウンロード中に予期せぬエラーが発生しました: {e}")
                return
    
    if not os.path.exists(original_video_path):
        print(f"エラー: オリジナル動画ファイルが見つかりません: {original_video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n動画クリップの生成を開始します... (出力先: {output_dir})")
    print(f"[Debug create_video_clips] Total points to process: {len(clips_info)}") # DEBUG

    final_clips_index_data = [] # To store info for the JSON index after potential concatenation

    for i, point_data in enumerate(clips_info):
        blog_point_text = point_data["blog_point_text"]
        video_segments = point_data["clips"]
        current_blog_point_id = point_data.get("blog_point_id", i) 
        
        def extract_title(text: str) -> str:
            lines = text.strip().splitlines()
            for line in lines:
                line = line.strip()
                if line.startswith('#'):
                    return line.lstrip('#').strip()
            if lines:
                return lines[0][:30] + ("..." if len(lines[0]) > 30 else "")
            return text[:30] + ("..." if len(text) > 30 else "")
        blog_point_title = extract_title(blog_point_text)
        
        # f-string内のバックスラッシュを避けるために、表示用テキストを事前に準備
        display_blog_point_text = blog_point_text[:30].replace('\n', ' ')

        if not video_segments:
            print(f"  ブログポイントID {current_blog_point_id} (「{display_blog_point_text}...」) には動画クリップがありません。スキップします。")
            continue

        print(f"  ブログポイントID {current_blog_point_id} (「{blog_point_text[:30].replace('\n', ' ')}...」) のクリップを生成中...")
        
        temp_segment_files = []
        overall_start_time = None
        overall_end_time = None

        for j, segment in enumerate(video_segments):
            start_time = segment.get("start_time")
            end_time = segment.get("end_time")

            if start_time is None or end_time is None:
                print(f"    セグメント {j} の開始/終了時間が不完全です。スキップします。")
                continue
            
            if overall_start_time is None or start_time < overall_start_time:
                overall_start_time = start_time
            if overall_end_time is None or end_time > overall_end_time:
                overall_end_time = end_time

            temp_clip_basename = f"point_{current_blog_point_id:02d}_segment_{j:02d}_temp.mp4"
            temp_output_filename = os.path.join(output_dir, temp_clip_basename)
            
            cmd = [
                'ffmpeg', '-y',
                '-i', original_video_path, 
                '-ss', str(round(start_time, 3)),
                '-to', str(round(end_time, 3)),
                '-c:v', 'libx264', '-preset', 'medium', 
                '-c:a', 'aac', '-b:a', '192k',
                '-avoid_negative_ts', 'make_zero', 
                temp_output_filename
            ]
            
            print(f"    一時セグメントFFmpegコマンド: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore')
                if result.returncode == 0:
                    print(f"      一時セグメント成功: {temp_output_filename}")
                    temp_segment_files.append(temp_output_filename)
                else:
                    print(f"      一時セグメント失敗 (ffmpegエラー): {temp_output_filename}")
                    print(f"        stderr: {result.stderr[:500]}...")
            except FileNotFoundError:
                print("エラー: ffmpegコマンドが見つかりません。ffmpegがインストールされ、PATHに含まれていることを確認してください。")
                return 
            except Exception as e:
                print(f"      一時セグメント生成中に予期せぬエラー: {e}")
        
        # Process collected temp segments for this blog point
        final_clip_basename_for_point = f"point_{current_blog_point_id:02d}_clip.mp4"
        final_output_filename_for_point = os.path.join(output_dir, final_clip_basename_for_point)

        if not temp_segment_files:
            print(f"  ブログポイントID {current_blog_point_id} の一時セグメントが生成されませんでした。スキップします。")
            continue

        if len(temp_segment_files) == 1:
            # Single segment, just rename
            try:
                os.rename(temp_segment_files[0], final_output_filename_for_point)
                print(f"    単一セグメントのためリネーム: {temp_segment_files[0]} -> {final_output_filename_for_point}")
                final_clips_index_data.append({
                    "clip_filename": final_clip_basename_for_point,
                    "blog_point_id": current_blog_point_id,
                    "blog_point_summary": blog_point_text,
                    "blog_point_title": blog_point_title,
                    "start_time": overall_start_time, # Should be the segment's start time
                    "end_time": overall_end_time    # Should be the segment's end time
                })
            except OSError as e:
                print(f"    一時ファイルのリネーム中にエラー: {e}")
        
        elif len(temp_segment_files) > 1:
            # Multiple segments, concatenate using filter_complex concat (more robust)
            input_args = []
            filter_inputs = []
            for idx, temp_file in enumerate(temp_segment_files):
                input_args += ['-i', temp_file]
                filter_inputs += [f'[{idx}:v][{idx}:a]']
            n = len(temp_segment_files)
            filter_complex = f'{"".join(filter_inputs)}concat=n={n}:v=1:a=1[outv][outa]'
            concat_cmd = [
                'ffmpeg', '-y', *input_args,
                '-filter_complex', filter_complex,
                '-map', '[outv]', '-map', '[outa]',
                '-c:v', 'libx264', '-preset', 'medium',
                '-c:a', 'aac', '-b:a', '192k',
                '-avoid_negative_ts', 'make_zero',
                final_output_filename_for_point
            ]
            print(f"    結合FFmpegコマンド: {' '.join(str(x) for x in concat_cmd)}")
            result_concat = subprocess.run(concat_cmd, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore')
            if result_concat.returncode == 0:
                print(f"      結合成功: {final_output_filename_for_point}")
                final_clips_index_data.append({
                    "clip_filename": final_clip_basename_for_point,
                    "blog_point_id": current_blog_point_id,
                    "blog_point_summary": blog_point_text,
                    "blog_point_title": blog_point_title,
                    "start_time": overall_start_time, # Overall start for the concatenated clip
                    "end_time": overall_end_time    # Overall end for the concatenated clip
                })
            else:
                print(f"      結合失敗 (ffmpegエラー): {final_output_filename_for_point}")
                print(f"        stderr: {result_concat.stderr[:500]}...")

            # Clean up temporary segment files
            for temp_file in temp_segment_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                        print(f"      一時ファイルを削除: {temp_file}")
                    except OSError as e_remove:
                         print(f"      一時ファイルの削除中にエラー: {e_remove}")
            # No concat_list file to remove in this method
                        
    # After processing all clips, write the final index file
    if final_clips_index_data:
        index_file_path = os.path.join(output_dir, "clips_index.json")
        try:
            final_clips_index_data.sort(key=lambda x: x.get("blog_point_id", float('inf'))) # Sort by blog_point_id
            with open(index_file_path, 'w', encoding='utf-8') as f:
                json.dump(final_clips_index_data, f, ensure_ascii=False, indent=4)
            print(f"\n最終クリップインデックスファイルを保存しました: {index_file_path}")
        except Exception as e:
            print(f"\n最終クリップインデックスファイルの保存中にエラーが発生しました: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ブログ記事とタイムスタンプ情報に基づいて動画クリップを生成します。")
    parser.add_argument("article_file_path", help="解析するブログ記事のファイルパス (例: YYYYMMDD_blog_VIDEOID_article.txt)")
    parser.add_argument("--video_path", help="オリジナル動画ファイルのパス。指定しない場合はyt-dlpによるダウンロードを試みます。", default=None)
    parser.add_argument("--output_dir", help="生成されたクリップを保存するディレクトリ", default="output_clips")
    parser.add_argument("--timestamp_file", help="使用するタイムスタンプJSONファイルのパス。指定しない場合は記事ファイル名から自動的に推定します。", default=None)
    
    args = parser.parse_args()

    article_path = args.article_file_path
    video_path_arg = args.video_path # Can be None
    output_clips_dir = args.output_dir
    custom_timestamp_file_path = args.timestamp_file

    if not os.path.exists(article_path):
        print(f"エラー: 記事ファイルが見つかりません: {article_path}")
        exit(1)

    # 1. Extract Video ID from article filename (needed for potential download)
    video_id = get_video_id_from_filename(article_path)
    print(f"[Debug main] video_id extracted: {video_id}") # DEBUG
    if not video_id:
        print("記事ファイル名から動画IDを抽出できませんでした。ダウンロードや処理を続行できません。")
        exit(1)
    
    base_filename = get_base_filename_from_article(article_path)
    print(f"[Debug main] base_filename extracted: {base_filename}") # DEBUG
    if not base_filename:
        print("記事ファイル名からベースファイル名を特定できませんでした。処理を中止します。")
        exit(1)
        
    # Determine timestamp file path
    if custom_timestamp_file_path:
        timestamp_file_to_load = custom_timestamp_file_path
        if not os.path.exists(timestamp_file_to_load):
            print(f"エラー: 指定されたタイムスタンプファイルが見つかりません: {timestamp_file_to_load}")
            exit(1)
    else:
        timestamp_file_to_load = f"{base_filename}_timestamps.json" # Construct path

    print(f"処理対象:")
    print(f"  記事ファイル: {article_path}")
    print(f"  動画ID: {video_id}")
    print(f"  タイムスタンプファイル: {timestamp_file_to_load}")
    print(f"  オリジナル動画パス: {video_path_arg if video_path_arg else '未指定'}")
    print(f"  クリップ出力先: {output_clips_dir}")

    blog_points_list = parse_blog_post(article_path)
    if not blog_points_list:
        print("ブログ記事からポイントを抽出できませんでした。")
        exit(1)
    
    print(f"ブログから {len(blog_points_list)} 個のポイント（段落）を抽出しました。")
    # for i, p in enumerate(blog_points_list):
    #     print(f"  Point {i}: {p[:60]}...")

    timestamp_json_data = load_timestamp_data(timestamp_file_to_load)
    # `get_clips_info_from_timestamps` handles empty timestamp_json_data internally

    # New alignment logic using timestamps
    clips_to_create = get_clips_info_from_timestamps(blog_points_list, timestamp_json_data)

    if not clips_to_create:
        print("タイムスタンプ情報から作成すべきクリップが見つかりませんでした。")
    else:
        print(f"\n{len(clips_to_create)} 個のブログポイントに対応するクリップ情報が見つかりました。")
        create_video_clips(video_id, clips_to_create, output_clips_dir, original_video_path=video_path_arg)

    print("\nスクリプトの処理が完了しました。") 