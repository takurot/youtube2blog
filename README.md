## YouTube2Blog 実行スクリプトの使い方（刷新版）

このREADMEは `run*.sh` と `batch_youtube2blog.py` の最新の使い方のみを記載します。

### 前提

- macOS または Linux（zsh/bash）
- Python 3 がインストール済み
- FFmpeg がインストール済み（動画・音声処理で使用）
- 多くの `run*.sh` は初回実行時に `.env` 仮想環境を自動作成し、`requirements.txt` をインストールします

---

### run_youtube2blog_local.sh（ローカル転写で日本語ブログ）

YouTube動画をダウンロード→Whisperで文字起こし→日本語ブログ記事などを生成するローカル転写モード。

```bash
./run_youtube2blog_local.sh <youtube_url> [whisper_model] [options]
```

- 位置引数:
  - `youtube_url`: 対象のYouTube動画URL
  - `whisper_model`: Whisperモデル名（省略可、既定: `base`）
- 主なオプション（スクリプト経由で `youtube2blog_local.py` に渡されます）:
  - `--blog-only`: ブログ記事のみを生成
  - `--min-words N`: 最小文字数（例: 2500）
  - `--max-words N`: 最大文字数（例: 3000）

例:
```bash
./run_youtube2blog_local.sh 'https://www.youtube.com/watch?v=VIDEO_ID'
./run_youtube2blog_local.sh 'https://www.youtube.com/watch?v=VIDEO_ID' medium --blog-only
```

---

### run_youtube2blog_local_zh.sh（ローカル転写で中国語ブログ）

日本語版と同様ですが、出力言語が中国語（`zh`）。

```bash
./run_youtube2blog_local_zh.sh <youtube_url> [whisper_model] [options]
```

- 位置引数:
  - `youtube_url`: 対象のYouTube動画URL
  - `whisper_model`: Whisperモデル名（省略可、既定: `base`）
- 主なオプション:
  - `--min-words N`: 最小文字数（既定: 2000）
  - `--max-words N`: 最大文字数（既定: 2500）

例:
```bash
./run_youtube2blog_local_zh.sh 'https://www.youtube.com/watch?v=VIDEO_ID'
./run_youtube2blog_local_zh.sh 'https://www.youtube.com/watch?v=VIDEO_ID' base --min-words 2200 --max-words 2600
```

---

### run_process_video_to_clips.sh（ブログ→タイムスタンプ統合→動画クリップ自動作成）

1. `youtube2blog.py --blog-only` でブログ記事と初期タイムスタンプを生成
2. `merge_timestamp_clips.py` で段落とタイムスタンプをマージ
3. `create_clips_from_blog.py` で複数の短尺クリップを自動生成

```bash
./run_process_video_to_clips.sh VIDEO_ID [--lang ja|zh] [--video-path PATH] [--output-dir DIR]
```

- 必須:
  - `VIDEO_ID`: 例 `dQw4w9WgXcQ`
- オプション:
  - `--lang LANG_CODE`: 文字起こし・記事の言語（既定: `ja`）
  - `--video-path PATH`: 元動画のローカルパス（未指定時は `yt-dlp` で取得を試行）
  - `--output-dir DIR`: 出力ディレクトリのベース名（既定: `my_clips`。最終名は `my_clips_VIDEOID`）

例:
```bash
./run_process_video_to_clips.sh dQw4w9WgXcQ
./run_process_video_to_clips.sh dQw4w9WgXcQ --video-path /path/to/original.mp4 --output-dir clips
```

生成物（プロジェクト直下）:
- `<YYYYMMDD>_blog_<VIDEO_ID>_article.txt`
- `<YYYYMMDD>_blog_<VIDEO_ID>_timestamps.json`
- `<YYYYMMDD>_blog_<VIDEO_ID>_timestamps_merged.json`

クリップ出力:
- `my_clips_<VIDEO_ID>/` 配下に `clip_*.mp4` と `clips_index.json`

---

### run_make_epub_vol1.sh（既存記事群からEPUB作成）

`millionaires/backup/vol1` ディレクトリにある記事群から `output/millionaires_vol1.epub` を生成します。スクリプト内のタイトルや説明は必要に応じて編集してください。

```bash
./run_make_epub_vol1.sh
```

---

### run_batch_cosme_zh.sh（中国語ブログの一括実行サンプル）

内部で `run_youtube2blog_local_zh.sh` を複数のURLに対して順次呼び出すシンプルなスクリプトです。ファイルを開き、行を追加・削除して対象URLを調整してください。

```bash
./run_batch_cosme_zh.sh
```

注: シェバンが無い場合は `bash run_batch_cosme_zh.sh` のように実行してください。

---

### batch_youtube2blog.py（URLリストの一括処理）

URLリストファイルの各行にあるYouTube URLを順次処理して、`youtube2blog_local.py` を呼び出します。ログは `batch_youtube2blog.log` に出力されます。

```bash
python batch_youtube2blog.py <url_list_file> [options]
```

- 位置引数:
  - `url_list_file`: 1行に1URL／`#` 始まりはコメント、空行は無視

- オプション:
  - `--whisper-model {tiny,base,small,medium,large}`（既定: `base`）
  - `--output-language {ja,zh}`（既定: `ja`）
  - `--min-words INT`（既定: 3000）
  - `--max-words INT`（既定: 3500）
  - `--delay INT` 各URL処理間の待機秒（既定: 5）
  - `--continue-on-error` エラー発生時も続行

例:
```bash
python batch_youtube2blog.py url_list.txt
python batch_youtube2blog.py url_list.txt --whisper-model large --output-language zh
python batch_youtube2blog.py url_list.txt --continue-on-error --delay 10
```

URLリスト例（`url_list.txt`）:
```
# コメント行
https://www.youtube.com/watch?v=example1

https://www.youtube.com/watch?v=example2
```

---

### ヒント・トラブルシュート

- 初回実行で `.env` が作成され、`pip install -r requirements.txt` が走ります。以後は高速です。
- `ffmpeg` が未インストールだとエラーになります（macOS: `brew install ffmpeg`）。
- `run_process_video_to_clips.sh` の最後で macOS の `._` 隠しファイルをクリーンアップしています。

---

### extract_popular_top.sh（チャンネルの人気順TopN動画URLを抽出）

YouTubeチャンネルの「人気順」一覧から、上位N件の動画URLを取得します。`yt-dlp` を利用します。

```bash
./extract_popular_top.sh <CHANNEL_URL_OR_HANDLE> [-n COUNT] [-o OUTPUT]
```

- 引数:
  - `CHANNEL_URL_OR_HANDLE`: 例 `https://www.youtube.com/@EnesYilmazer/videos` または `https://www.youtube.com/@EnesYilmazer`
  - `-n, --count COUNT`: 取得件数（既定: 50）
  - `-o, --output FILE`: 出力先ファイル。省略時は標準出力

例:
```bash
# Top50を標準出力
./extract_popular_top.sh 'https://www.youtube.com/@EnesYilmazer/videos' -n 50

# Top50をファイル保存
./extract_popular_top.sh 'https://www.youtube.com/@EnesYilmazer' -n 50 -o top50_urls.txt

# Top100をファイル保存
./extract_popular_top.sh 'https://www.youtube.com/@EnesYilmazer/videos' -n 100 -o top100_urls.txt
```

メモ:
- `/videos` が付いていなくても自動で補完し、人気順クエリ `?view=0&sort=p&flow=grid` を付与します。
- 重複行は除去され、入力順が保持されます。
- 依存: `yt-dlp`, `sed`, `awk`（macOS: `brew install yt-dlp`）

