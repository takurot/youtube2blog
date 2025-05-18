#!/bin/bash

# process_video_to_clips.sh - ブログ記事作成、タイムスタンプマージ、クリップ作成を自動化

rm ./202*.*

# --- 設定 ---
PYTHON_EXEC="python" # Pythonの実行コマンド

# --- ヘルパー関数 ---
show_usage() {
    echo "使用方法: $0 VIDEO_ID [OPTIONS]"
    echo ""
    echo "必須引数:"
    echo "  VIDEO_ID          YouTube の video ID (例: AlfvDrbJH6c)"
    echo ""
    echo "オプション:"
    echo "  --lang LANG_CODE  文字起こしとブログ記事の言語コード (デフォルト: ja)"
    echo "  --video-path PATH オリジナル動画ファイルのパス。指定しない場合はyt-dlpによるダウンロードを試みます。"
    echo "  --output-dir DIR  生成されたクリップを保存するディレクトリ (デフォルト: my_clips_VIDEOID)"
    echo "  --help            このヘルプメッセージを表示"
    exit 0
}

# --- 初期値設定 ---
LANGUAGE="ja"
ORIGINAL_VIDEO_PATH=""
CLIPS_OUTPUT_DIR_BASE="my_clips"

# --- 引数解析 ---
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

VIDEO_ID="$1"
shift # VIDEO_ID を消費

while [ $# -gt 0 ]; do
    case "$1" in
        --lang)
            LANGUAGE="$2"
            shift 2
            ;;
        --video-path)
            ORIGINAL_VIDEO_PATH="$2"
            shift 2
            ;;
        --output-dir)
            CLIPS_OUTPUT_DIR_BASE="$2" # ベース名のみ受け取り、後でVIDEO_IDを付加
            shift 2
            ;;
        --help)
            show_usage
            ;;
        *)
            echo "不明なオプション: $1"
            show_usage
            exit 1
            ;;
    esac
done

if [ -z "$VIDEO_ID" ]; then
    echo "エラー: VIDEO_IDが指定されていません。"
    show_usage
    exit 1
fi

# --- 変数設定 ---
YOUTUBE_URL="https://www.youtube.com/watch?v=$VIDEO_ID"
TODAY_DATE=$(date +%Y%m%d)

BASE_FILENAME_PREFIX="${TODAY_DATE}_blog_${VIDEO_ID}"
ARTICLE_FILE="${BASE_FILENAME_PREFIX}_article.txt"
INITIAL_TIMESTAMPS_FILE="${BASE_FILENAME_PREFIX}_timestamps.json"
MERGED_TIMESTAMPS_FILE="${BASE_FILENAME_PREFIX}_timestamps_merged.json"

# クリップ出力ディレクトリ名にVIDEO_IDを付加
FINAL_CLIPS_OUTPUT_DIR="${CLIPS_OUTPUT_DIR_BASE}_${VIDEO_ID}"


echo "--- パラメータ ---"
echo "Video ID: ${VIDEO_ID}"
echo "YouTube URL: ${YOUTUBE_URL}"
echo "言語: ${LANGUAGE}"
echo "記事ファイル: ${ARTICLE_FILE}"
echo "初期タイムスタンプファイル: ${INITIAL_TIMESTAMPS_FILE}"
echo "マージ後タイムスタンプファイル: ${MERGED_TIMESTAMPS_FILE}"
echo "オリジナル動画パス: ${ORIGINAL_VIDEO_PATH:-（指定なし、ダウンロード試行）}"
echo "クリップ出力ディレクトリ: ${FINAL_CLIPS_OUTPUT_DIR}"
echo "--------------------"
echo ""

# --- 1. ブログ記事と初期タイムスタンプの生成 ---
echo "ステップ1: ブログ記事と初期タイムスタンプを生成中 (${ARTICLE_FILE}, ${INITIAL_TIMESTAMPS_FILE})..."
"$PYTHON_EXEC" youtube2blog.py "$LANGUAGE" "$YOUTUBE_URL" --blog-only
if [ $? -ne 0 ]; then
    echo "エラー: youtube2blog.py の実行に失敗しました。"
    exit 1
fi
if [ ! -f "$ARTICLE_FILE" ] || [ ! -f "$INITIAL_TIMESTAMPS_FILE" ]; then
    echo "エラー: 記事ファイルまたは初期タイムスタンプファイルが生成されませんでした。"
    echo "  期待された記事ファイル: $ARTICLE_FILE"
    echo "  期待されたタイムスタンプファイル: $INITIAL_TIMESTAMPS_FILE"
    exit 1
fi
echo "ステップ1完了。"
echo ""

# --- 2. タイムスタンプのマージ ---
echo "ステップ2: タイムスタンプ情報をマージ中 (${INITIAL_TIMESTAMPS_FILE} -> ${MERGED_TIMESTAMPS_FILE})..."
"$PYTHON_EXEC" merge_timestamp_clips.py "$INITIAL_TIMESTAMPS_FILE"
if [ $? -ne 0 ]; then
    echo "エラー: merge_timestamp_clips.py の実行に失敗しました。"
    exit 1
fi
if [ ! -f "$MERGED_TIMESTAMPS_FILE" ]; then
    echo "エラー: マージ済みタイムスタンプファイル (${MERGED_TIMESTAMPS_FILE}) が生成されませんでした。"
    exit 1
fi
echo "ステップ2完了。"
echo ""

# --- 3. クリップ動画の生成 ---
echo "ステップ3: クリップ動画を生成中 (出力先: ${FINAL_CLIPS_OUTPUT_DIR})..."
CREATE_CLIPS_ARGS=("$ARTICLE_FILE" --timestamp_file "$MERGED_TIMESTAMPS_FILE" --output_dir "$FINAL_CLIPS_OUTPUT_DIR")

if [ -n "$ORIGINAL_VIDEO_PATH" ]; then
    if [ ! -f "$ORIGINAL_VIDEO_PATH" ]; then
        echo "警告: 指定されたオリジナル動画パスが見つかりません: $ORIGINAL_VIDEO_PATH"
        echo "yt-dlpによるダウンロードを試みます。"
        # video_path引数は渡さないことでダウンロードを促す
    else
        CREATE_CLIPS_ARGS+=(--video_path "$ORIGINAL_VIDEO_PATH")
    fi
elif [ -z "$ORIGINAL_VIDEO_PATH" ]; then
     echo "オリジナル動画パスは指定されていません。yt-dlpによるダウンロードを試みます。"
fi


"$PYTHON_EXEC" create_clips_from_blog.py "${CREATE_CLIPS_ARGS[@]}"
if [ $? -ne 0 ]; then
    echo "エラー: create_clips_from_blog.py の実行に失敗しました。"
    exit 1
fi
echo "ステップ3完了。"
echo ""

echo "全ての処理が正常に完了しました。"
echo "クリップは ${FINAL_CLIPS_OUTPUT_DIR} に保存されているはずです。"
echo "クリップのインデックスファイル: ${FINAL_CLIPS_OUTPUT_DIR}/clips_index.json"

find . -name "._*" -exec rm {} \;

exit 0 