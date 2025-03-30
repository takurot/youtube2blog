#!/bin/bash

rm ./._*
rm ./202*.*

# youtube2blog.sh - YouTube の video ID を引数として youtube2blog.py を実行するスクリプト

# 使用方法の表示関数
show_usage() {
    echo "使用方法: $0 VIDEO_ID [LANGUAGE] [CHANNEL]"
    echo ""
    echo "引数:"
    echo "  VIDEO_ID     YouTube の video ID (例: 5uBAQrg4SoQ)"
    echo "  LANGUAGE     字幕の言語コード (デフォルト: ja)"
    echo "  CHANNEL      YouTubeアップロード用のチャンネル名 (省略可能)"
    echo ""
    echo "例:"
    echo "  $0 5uBAQrg4SoQ"
    echo "  $0 5uBAQrg4SoQ en"
    echo "  $0 5uBAQrg4SoQ en channel1"
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
if [ $# -gt 0 ]; then
    LANGUAGE=$1
    shift
fi

# 第3引数は任意 (CHANNEL)
CHANNEL=""
if [ $# -gt 0 ]; then
    CHANNEL=$1
    shift
fi

# YouTube URL の構築
YOUTUBE_URL="https://www.youtube.com/watch?v=$VIDEO_ID"

# コマンドの実行
echo "YouTube URL: $YOUTUBE_URL"
echo "言語コード: $LANGUAGE"

echo "ブログ記事の生成を開始します..."
python youtube2blog.py "$LANGUAGE" "$YOUTUBE_URL" --wordcloud

# 生成されたmp4ファイルを確認
TODAY=$(date +%Y%m%d)
MP4_FILE="${TODAY}_blog_video_${VIDEO_ID}.mp4"

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

exit 0 