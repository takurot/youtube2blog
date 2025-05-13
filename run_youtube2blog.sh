#!/bin/bash

rm ./._*
rm ./202*.*

# youtube2blog.sh - YouTube の video ID を引数として youtube2blog.py を実行するスクリプト

# 使用方法の表示関数
show_usage() {
    echo "使用方法: $0 VIDEO_ID [LANGUAGE] [CHANNEL] [OPTIONS]"
    echo ""
    echo "引数:"
    echo "  VIDEO_ID     YouTube の video ID (例: 5uBAQrg4SoQ)"
    echo "  LANGUAGE     字幕の言語コード (デフォルト: ja)"
    echo "  CHANNEL      YouTubeアップロード用のチャンネル名 (省略可能)"
    echo ""
    echo "オプション:"
    echo "  --blog-only  ブログ記事のみを生成し、音声・動画は生成しない"
    echo ""
    echo "例:"
    echo "  $0 5uBAQrg4SoQ"
    echo "  $0 5uBAQrg4SoQ en"
    echo "  $0 5uBAQrg4SoQ en channel1"
    echo "  $0 5uBAQrg4SoQ en --blog-only"
    echo "  $0 5uBAQrg4SoQ en channel1 --blog-only"
    exit 1
}

# 引数のチェック
if [ $# -lt 1 ]; then
    show_usage
fi

# 第1引数は必須 (VIDEO_ID)
VIDEO_ID=$1
shift

# 第2引数は任意 (LANGUAGE)
LANGUAGE="en"
if [ $# -gt 0 ] && [[ $1 != --* ]]; then
    LANGUAGE=$1
    shift
fi

# 第3引数は任意 (CHANNEL)
CHANNEL=""
if [ $# -gt 0 ] && [[ $1 != --* ]]; then
    CHANNEL=$1
    shift
fi

# ブログ記事のみのモードかどうかをチェック
BLOG_ONLY=false
EXTRA_ARGS="--wordcloud"
while [ $# -gt 0 ]; do
    case "$1" in
        --blog-only)
            BLOG_ONLY=true
            EXTRA_ARGS="$EXTRA_ARGS --blog-only"
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            ;;
    esac
    shift
done

# YouTube URL の構築
YOUTUBE_URL="https://www.youtube.com/watch?v=$VIDEO_ID"

# コマンドの実行
echo "YouTube URL: $YOUTUBE_URL"
echo "言語コード: $LANGUAGE"

echo "ブログ記事の生成を開始します..."
python youtube2blog.py "$LANGUAGE" "$YOUTUBE_URL" $EXTRA_ARGS

# ブログ記事のみの場合は終了
if [ "$BLOG_ONLY" = true ]; then
    echo "ブログ記事のみを生成しました。処理を終了します。"
    exit 0
fi

# 生成されたmp4ファイルを確認
TODAY=$(date +%Y%m%d)

# youtube2blog.py のファイル名生成ロジックに合わせる
FORMAT_SUFFIX=""
# EXTRA_ARGS に --shorts が含まれているかチェック
if [[ "$EXTRA_ARGS" == *"--shorts"* ]]; then
    FORMAT_SUFFIX="_shorts"
fi
MP4_FILE="${TODAY}_blog_${VIDEO_ID}_video${FORMAT_SUFFIX}.mp4"
echo "MP4_FILE: $MP4_FILE"
# mp4ファイルが存在するかチェック
if [ -f "$MP4_FILE" ]; then
    echo "動画ファイルが生成されました: $MP4_FILE"
    echo "YouTubeへのアップロードを開始します..."
    
    # チャンネル指定がある場合
    if [ -n "$CHANNEL" ]; then
        python youtube_uploader.py --channel "$CHANNEL" "$VIDEO_ID"
    else
        python youtube_uploader.py "$VIDEO_ID"
    fi
    
    # アップロード結果のチェック
    if [ $? -eq 0 ]; then
        echo "YouTubeへのアップロードが完了しました。"
    else
        echo "YouTubeへのアップロードに失敗しました。"
    fi
else
    echo "動画ファイルが生成されていません。YouTubeへのアップロードをスキップします。"
fi

# exit 0 