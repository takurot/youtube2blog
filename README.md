# YouTube2Blog

YouTube の字幕を取得し、OpenAI API を使用して詳細な日本語ブログ記事を生成する Python スクリプトです。さらに、音声ナレーション付きの動画も自動生成します。動画コンテンツをブログ記事やオーディオコンテンツに効率的に変換するためのツールです。

## 主な機能

- YouTube 動画から字幕を取得（URL 言語コードを指定）
- **🆕 ローカル転写機能**: YouTube動画をダウンロードしてWhisperで転写生成
- OpenAI API を使用して 2000〜3000 文字の日本語ブログ記事を生成
- ブログ記事をテキストファイルとして保存
- テキストを音声に変換（OpenAI Text-to-Speech API 使用）
- ブログコンテンツに基づくワードクラウド画像の生成
- 音声ファイルと静止画像から MP4 動画ファイルを生成
- 音声出力用の最適化されたテキスト生成（URL やポイントセクションを除外）
- 通常の 16:9 フォーマットまたは YouTube Shorts 用の縦型 9:16 フォーマットに対応

## 必要条件

- Python 3.7 以降
- Python ライブラリ:
  - `youtube-transcript-api`
  - `openai`
  - `pillow` (PIL)
  - `wordcloud`
  - `janome`
  - `numpy`
  - `yt-dlp` (ローカル転写機能用)
  - `openai-whisper` (ローカル転写機能用)
  - `pydub` (音声処理用)
- OpenAI API キー
- FFmpeg (動画生成用)

## インストール

1. リポジトリをクローン:

   ```bash
   git clone https://github.com/takurot/youtube2blog.git
   cd youtube2blog
   ```

2. 必要な依存関係をインストール:

   ```bash
   pip install -r requirements.txt
   ```

3. OpenAI API キーを環境変数として設定するか、スクリプト内で直接指定:

   - 環境変数として設定:
     ```bash
     export OPENAI_API_KEY="your_api_key_here"
     ```

4. FFmpeg をインストール（動画生成に必要）:
   - macOS: `brew install ffmpeg`
   - Ubuntu: `sudo apt install ffmpeg`
   - Windows: [FFmpeg ウェブサイト](https://ffmpeg.org/download.html)からダウンロード

## 使用方法

### 通常の字幕取得モード

以下のコマンドでスクリプトを実行:

```bash
python youtube2blog.py <言語コード> <YouTube動画URL> [オプション]
```

### 🆕 ローカル転写モード

YouTube Transcript APIの問題を回避し、より高品質な転写を得るためのローカル転写モードが利用できます：

```bash
python youtube2blog_local.py <YouTube動画URL> [オプション]
```

**または便利なシェルスクリプトを使用:**

```bash
./run_youtube2blog_local.sh <YouTube動画URL> [whisper_model] [オプション]
```

### 引数

**通常モード:**
- **`言語コード`**: 字幕の言語コード (例: `en`, `ja`)
- **`YouTube動画URL`**: 字幕を含む YouTube 動画の URL

**ローカル転写モード:**
- **`YouTube動画URL`**: 転写したい YouTube 動画の URL
- **`whisper_model`**: 使用するWhisperモデル (tiny, base, small, medium, large)

### オプション

**通常モード:**
- **`--image <画像パス>`**: 動画作成に使用するカスタム画像ファイルのパス
- **`--shorts`**: YouTube Shorts 形式（縦長 9:16）の動画を生成
- **`--voice <音声名>`**: 使用する音声を指定（指定しない場合はランダム選択）
  - 選択肢: "alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"
- **`--wordcloud`**: タイトル画像の代わりにワードクラウド画像を生成して使用
- **`--no-bgm`**: バックグラウンド音楽を使用しない
- **`--blog-only`**: ブログ記事のみを生成し、音声・動画は生成しない

**ローカル転写モード:**
- **`--whisper-model <モデル>`**: 使用するWhisperモデル (tiny, base, small, medium, large)
- **`--blog-only`**: ブログ記事のみを生成し、音声・動画は生成しない
- **`--min-words <数値>`**: ブログ記事の最小文字数 (デフォルト: 2000)
- **`--max-words <数値>`**: ブログ記事の最大文字数 (デフォルト: 2500)

### 例

**通常モード:**
1. 日本語字幕を取得してブログ記事と音声付き動画を生成:

```bash
python youtube2blog.py ja https://www.youtube.com/watch?v=example_video_id
```

2. ワードクラウド画像と特定の音声を使用して Shorts ビデオを生成:

```bash
python youtube2blog.py ja https://www.youtube.com/watch?v=example_video_id --shorts --wordcloud --voice nova
```

**ローカル転写モード:**
1. 基本的な使用（baseモデル使用）:

```bash
python youtube2blog_local.py https://www.youtube.com/watch?v=example_video_id
```

2. シェルスクリプトを使用してmediumモデルでブログ記事のみ生成:

```bash
./run_youtube2blog_local.sh https://www.youtube.com/watch?v=example_video_id medium --blog-only
```

3. 高品質モデルを使用:

```bash
python youtube2blog_local.py https://www.youtube.com/watch?v=example_video_id --whisper-model large
```

### 出力ファイル

スクリプトは以下のファイルを生成します（`<日付>`は実行日、`<動画ID>`は YouTube 動画 ID）:

1. **ブログ記事**: `<日付>_blog_article_<動画ID>.txt`
2. **音声出力用テキスト**: `<日付>_blog_audio_text_<動画ID>.txt`
3. **音声ファイル**: `<日付>_blog_audio_<動画ID>.mp3`
4. **画像ファイル**:
   - タイトル画像: `<日付>_title_image_<動画ID>.png` または
   - ワードクラウド画像: `<日付>_wordcloud_<動画ID>.png`
5. **動画ファイル**: `<日付>_blog_video_<動画ID>.mp4` または `<日付>_blog_video_shorts_<動画ID>.mp4`

## ファイル構造

```
youtube2blog/
│
├── youtube2blog.py        # メインスクリプト
├── requirements.txt       # 必要な依存関係
└── README.md              # このファイル
```

## 注意点

**通常モード:**
- 指定された YouTube 動画に字幕が利用可能である必要があります。そうでない場合、文字起こしは失敗します。
- 最近YouTube Transcript APIが一部の環境（特にクラウドサーバー）でブロックされる問題が発生しています。この場合はローカル転写モードを使用してください。

**ローカル転写モード:**
- 動画のダウンロードとWhisperによる転写処理が必要なため、通常モードよりも時間がかかります。
- Whisperモデルは初回使用時にダウンロードされるため、インターネット接続が必要です。
- 長い動画の場合、相当な処理時間と計算リソースが必要になります。
- より高品質な転写を得るにはlarge モデルを使用しますが、GPUが推奨されます。

**共通:**
- OpenAI API キーが必要で、API 使用料が発生する場合があります。
- 長い動画の場合、トークン制限により、トランスクリプトを小さな部分に分割するための追加のロジックが必要になることがあります。
- OpenAI の Text-to-Speech API は使用量に応じて課金される場合があります。
- FFmpeg がシステムにインストールされていない場合、動画生成機能は動作しません。

## ライセンス

このプロジェクトは MIT ライセンスの下でライセンスされています。詳細については、`LICENSE`ファイルを参照してください。

## 動画クリッピング機能

`youtube2blog.py` で生成されたブログ記事の主要なポイントに基づいて、元の YouTube 動画から短いクリップ動画を自動生成する機能です。ブログ記事作成からタイムスタンプのマージ、クリップ作成までの一連の処理を自動化するシェルスクリプト `run_process_video_to_clips.sh` を提供します。

### 目的

ブログ記事のハイライト部分に対応する動画クリップを自動で作成し、コンテンツの再利用性を高めます。

### 主なスクリプトと役割

- **`run_process_video_to_clips.sh`**:
    1. `youtube2blog.py --blog-only` を実行し、ブログ記事と初期タイムスタンプJSONファイルを生成します。
    2. `merge_timestamp_clips.py` を実行し、ブログ記事の段落とタイムスタンプ情報をマージしたJSONファイルを生成します。
    3. `create_clips_from_blog.py` を実行し、マージされたタイムスタンプ情報に基づいて動画クリップを生成します。
- **`youtube2blog.py`**: (このプロセスでは `--blog-only` モードで使用) 指定されたYouTube動画から字幕を取得し、ブログ記事と初期タイムスタンプファイルを出力します。
- **`merge_timestamp_clips.py`**: ブログ記事のテキスト構造（段落）と初期タイムスタンプ情報を照合し、各段落に対応する開始・終了時間を含むマージ済みタイムスタンプファイルを作成します。
- **`create_clips_from_blog.py`**: マージ済みタイムスタンプファイルとブログ記事、そして元の動画（ローカルパス指定またはyt-dlpによるダウンロード）を使用して、各ポイントに対応する動画クリップを切り出し、指定されたディレクトリに保存します。また、クリップの情報をまとめた `clips_index.json` も生成します。

### `run_process_video_to_clips.sh` の使用方法

**コマンド構文:**

```bash
./run_process_video_to_clips.sh VIDEO_ID [OPTIONS]
```

**必須引数:**

-   `VIDEO_ID`: クリップを作成したい YouTube 動画の ID (例: `dQw4w9WgXcQ`)

**オプション:**

-   `--lang LANG_CODE`: 文字起こしとブログ記事生成に使用する言語コード (デフォルト: `ja`)。
-   `--video-path PATH`: オリジナルの高解像度動画ファイルのローカルパス。このオプションを指定しない場合、スクリプトは `yt-dlp` を使用して動画のダウンロードを試みます。
-   `--output-dir DIR_BASE_NAME`: 生成されたクリップを保存するディレクトリのベース名。最終的なディレクトリ名は `DIR_BASE_NAME_VIDEOID` となります (デフォルトのベース名: `my_clips`)。
-   `--help`: ヘルプメッセージを表示します。

**実行例:**

```bash
# 基本的な実行 (言語: ja, 出力先: my_clips_VIDEOID/, 動画は自動ダウンロード)
./run_process_video_to_clips.sh dQw4w9WgXcQ

# オリジナル動画ファイルを指定して実行
./run_process_video_to_clips.sh dQw4w9WgXcQ --video-path /path/to/my_original_video.mp4

# 出力ディレクトリのベース名を指定
./run_process_video_to_clips.sh dQw4w9WgXcQ --output-dir processed_clips
```

**出力:**

-   指定された出力ディレクトリ (例: `my_clips_VIDEOID/`) 内に、ブログの各ポイントに対応するクリップ動画ファイル (例: `clip_01.mp4`, `clip_02.mp4`, ...) が生成されます。
-   出力ディレクトリ内に `clips_index.json` ファイルが生成され、各クリップのファイル名、元のブログ記事での段落インデックス、テキスト内容、開始・終了時間などの情報が記録されます。
-   中間ファイルとして、以下のファイルがプロジェクトルートに生成されます:
    -   `<日付>_blog_<VIDEO_ID>_article.txt`
    -   `<日付>_blog_<VIDEO_ID>_timestamps.json` (初期タイムスタンプ)
    -   `<日付>_blog_<VIDEO_ID>_timestamps_merged.json` (マージ済みタイムスタンプ)

### 今後の開発予定

-   **クリップへの情報付加（オプション）**: 切り出したクリップに、関連するブログのテキストを字幕としてオーバーレイ表示する機能などを検討します。

# YouTube2Blog ワードクラウド生成の修正方法

## 問題: ワードクラウドが生成されない

YouTube2Blogを実行したとき、以下のようなエラーが発生する場合があります:

```
ワードクラウド画像生成中にエラーが発生しました: 'utf-8' codec can't decode byte 0xb0 in position 37: invalid start byte
```

これはmatplotlibのスタイルファイルに関連するエンコーディングの問題です。WordCloudはmatplotlibに依存しており、隠しファイル（._から始まるファイル）を適切に処理できないことが原因です。

## 修正方法

### 1. 簡単な修正スクリプトを実行する

このリポジトリで提供されている`fix_all_matplotlib_styles.py`スクリプトを実行することで、問題を自動的に修正できます:

```bash
python fix_all_matplotlib_styles.py
```

または:

```bash
chmod +x fix_all_matplotlib_styles.py
./fix_all_matplotlib_styles.py
```

このスクリプトは問題のあるスタイルファイルを検出して修正します。

### 2. 手動で修正する場合

手動で修正したい場合は、以下のようにmatplotlibのスタイルディレクトリ内の隠しファイルを削除または修正します:

1. matplotlibのインストールパスを確認する:
   ```python
   import matplotlib
   print(matplotlib.__file__)
   ```

2. stylelib ディレクトリに移動:
   ```bash
   cd /path/to/matplotlib/mpl-data/stylelib/
   ```

3. 問題のあるファイルを削除または空のファイルに置き換え:
   ```bash
   # 例: 
   rm ._*
   rm .__*
   # または
   # echo "# Fixed" > .__classic_test_patch.mplstyle
   ```

### 3. matplotlib を再インストールする

上記の方法で解決しない場合は、matplotlibを再インストールすることで問題が解消する場合があります:

```bash
pip install --force-reinstall matplotlib==3.5.3
```

## テスト

修正が正常に適用されたかを確認するには、テストスクリプトを実行します:

```bash
python test_wordcloud_simple.py
```

正常に動作すれば、カレントディレクトリに日付付きのワードクラウド画像ファイルが生成されます。

## その他の対処法

もし上記の方法で問題が解決しない場合:

1. 仮想環境を再構築する:
   ```bash
   rm -rf .env
   python -m venv .env
   source .env/bin/activate
   pip install -r requirements.txt
   ```

2. WordCloudを使わずにタイトル画像のみを生成する（--wordcloud オプションを指定しない）:
   ```bash
   ./run_youtube2blog.sh VIDEO_ID
   ```

## 技術的詳細

この問題はmacOSで特に発生しやすく、macOSがリソースフォークのために生成する隠しファイル（._で始まるファイル）がUTF-8以外のエンコーディングを使用していることが原因です。これらのファイルはmatplotlibのスタイル設定の読み込み時に問題を引き起こします。

修正スクリプトはこれらの隠しファイルを検出し、UTF-8でエンコードされた空のファイルに置き換えることで問題を解決します。

## YouTube自動アップロード機能

生成したブログ記事と音声読み上げ動画をYouTubeに自動アップロードすることができます。

### セットアップ

1. 必要なライブラリをインストール:
```bash
pip install --upgrade google-api-python-client google-auth-oauthlib google-auth-httplib2
```

2. Google Cloud Platformで認証情報を取得:
   - [Google Cloud Console](https://console.cloud.google.com/) にアクセス
   - プロジェクトを作成またはすでにあるプロジェクトを選択
   - 「APIとサービス」→「ライブラリ」で「YouTube Data API v3」を検索し有効化
   - 「認証情報」→「認証情報を作成」→「OAuthクライアントID」を選択
   - アプリケーションタイプ: デスクトップアプリ
   - 作成された認証情報の「JSONをダウンロード」をクリック
   - ダウンロードしたJSONファイルを`client_secrets.json`としてプロジェクトのルートディレクトリに保存するか、環境変数で指定（下記参照）

3. 環境変数の設定（オプション）:
   - クライアントシークレットファイルのパスを環境変数で指定できます
   ```bash
   # Linuxまたは macOS
   export YOUTUBE_CLIENT_SECRET=/path/to/your/client_secrets.json
   
   # Windows (コマンドプロンプト)
   set YOUTUBE_CLIENT_SECRET=C:\path\to\your\client_secrets.json
   
   # Windows (PowerShell)
   $env:YOUTUBE_CLIENT_SECRET="C:\path\to\your\client_secrets.json"
   ```

### 使用方法

コマンドラインから特定のYouTube動画ID（元にしたコンテンツ）を指定して実行:

```bash
python youtube_uploader.py [YouTube動画ID]
```

または、引数なしで実行すると最新の生成ファイルを検索してアップロード:

```bash
python youtube_uploader.py
```

特定のYouTubeチャンネルを指定してアップロード:

```bash
python youtube_uploader.py --channel チャンネル名 [YouTube動画ID]
```

run_youtube2blog.shからまとめて実行する場合:

```bash
./run_youtube2blog.sh VIDEO_ID [LANGUAGE] [CHANNEL]
```

### 機能

- `*_blog_article_*.txt`ファイルからタイトルを抽出（ハッシュタグは除去）
- 動画の説明文に元動画のURLを含める
- 説明欄の最後にハッシュタグを追加
- `*_blog_audio_text_*.txt`ファイルを字幕としてアップロード
- `*_wordcloud_*.png`ファイルをサムネイルとして設定
- 動画は非公開状態で投稿（後で確認してから公開可能）

初回実行時はブラウザが開き、YouTubeアカウントでの認証が求められます。認証後は`token_チャンネル名.json`に認証情報が保存され、以降の実行では自動的に再利用されます。

## バックグラウンド音楽機能

生成される動画にバックグラウンド音楽を自動的に追加することができます。

### 機能概要

- `bgm`ディレクトリ内の音楽ファイル（MP3形式）からランダムに選択
- BGMは動画の長さに合わせて自動的にループ再生（ナレーションが終わるまで継続）
- 音声とBGMをミックス（メイン音声を優先した適切な音量バランス）
- 動画終了時に自然なフェードアウト

### 使用方法

1. プロジェクトフォルダ内に`bgm`ディレクトリを作成
2. 著作権フリーのMP3ファイルを配置

```bash
# BGMを有効にして実行（デフォルト）
python youtube2blog.py ja "https://www.youtube.com/watch?v=VIDEO_ID" --wordcloud

# BGMを無効にして実行
python youtube2blog.py ja "https://www.youtube.com/watch?v=VIDEO_ID" --wordcloud --no-bgm
```

### 注意事項

- BGMには著作権フリーの音楽を使用してください
- 音楽ファイル名に特殊文字が含まれている場合、処理に問題が生じる可能性があります
- 生成される動画のサイズはBGMを使用すると大きくなります
- 極端に短いMP3ファイルを使用すると、ループの継ぎ目が目立つ場合があります

## 音声テキスト処理の改善点

ブログ記事から音声用テキストを生成する際に、以下の処理を行い、自然な音声出力を実現します：

1. **不要なセクションの除去**
   - YouTube URLの行
   - 「ポイント」セクションの内容（箇条書きリスト）

2. **マークダウン記号の削除**
   - 見出し記号（#）
   - 水平線（---、***、___）
   - リストマーカー（-、*、+、1. など）
   - 強調記号（**太字**、*斜体*）
   - コード記号（`コード`）
   - リンク記号（[テキスト](URL)→テキストのみ残す）

これにより、音声合成エンジンがマークダウン記号を読み上げてしまう問題を防ぎ、より自然な音声出力が可能になります。

## コマンドラインオプション

- `--image`: 動画作成に使用する静止画像ファイルを指定
- `--shorts`: YouTube Shorts形式（縦長動画）で出力
- `--voice`: 使用する音声を指定（alloy, echo, fable, onyx, nova, shimmer）
- `--wordcloud`: ワードクラウド画像を生成して使用
- `--no-bgm`: バックグラウンド音楽を使用しない
- `--blog-only`: ブログ記事のみを生成し、音声・動画は生成しない

### 使用例

```bash
# 基本的な使用方法
python youtube2blog.py ja "https://www.youtube.com/watch?v=VIDEO_ID"

# ブログ記事のみを生成する場合
python youtube2blog.py ja "https://www.youtube.com/watch?v=VIDEO_ID" --blog-only

# ワードクラウド画像を使用して動画を生成
python youtube2blog.py ja "https://www.youtube.com/watch?v=VIDEO_ID" --wordcloud

# 特定の音声を指定して動画を生成
python youtube2blog.py ja "https://www.youtube.com/watch?v=VIDEO_ID" --voice nova

# YouTube Shorts形式（縦長動画）で出力
python youtube2blog.py ja "https://www.youtube.com/watch?v=VIDEO_ID" --shorts

# BGMなしで動画を生成
python youtube2blog.py ja "https://www.youtube.com/watch?v=VIDEO_ID" --no-bgm
```
