#!/usr/bin/env python3
import os
import re
import sys
import json
import time
import glob
import httplib2
from datetime import datetime
import google.oauth2.credentials
import google_auth_oauthlib.flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import argparse

# 認証スコープの設定
# YouTubeへの動画アップロード、字幕のアップロードに必要な権限
SCOPES = [
    'https://www.googleapis.com/auth/youtube.upload',
    'https://www.googleapis.com/auth/youtube.force-ssl'
]

# 環境変数からCLIENT_SECRET_FILEを取得、設定されていない場合はデフォルト値を使用
CLIENT_SECRET_FILE = os.environ.get('YOUTUBE_CLIENT_SECRET', 'client_secrets.json')

def get_authenticated_service(channel_name="default"):
    """OAuth 2.0認証フローを処理し、YouTubeAPIクライアントを返す"""
    creds = None
    
    # チャンネル別のトークンファイル名
    token_file = f'token_{channel_name}.json'
    
    # トークンファイルが存在する場合は読み込む
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_info(
            json.loads(open(token_file, 'r').read())
        )
    
    # 認証情報がない、または無効な場合は新たに取得
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # client_secrets.jsonが存在することを確認
            if not os.path.exists(CLIENT_SECRET_FILE):
                print(f"Error: {CLIENT_SECRET_FILE} が見つかりません。")
                print("以下のいずれかの方法で設定してください:")
                print("1. 環境変数 YOUTUBE_CLIENT_SECRET にファイルパスを設定")
                print("2. カレントディレクトリに client_secrets.json として保存")
                print("3. Google Cloud Consoleからクライアント認証情報をダウンロード")
                sys.exit(1)
            
            print(f"\n===== 重要: {channel_name} チャンネル用の認証 =====")
            print(f"ブラウザウィンドウが開きます。")
            print(f"{channel_name} チャンネルを所有するGoogleアカウントで")
            print(f"ログインしてください。")
            print("============================================\n")
            
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # 認証情報を保存
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
    
    # YouTube API サービスを構築
    return build('youtube', 'v3', credentials=creds)

def create_caption(youtube, video_id, caption_file, language='ja'):
    """動画に字幕を追加する"""
    try:
        # 字幕ファイルの内容を読み込む
        with open(caption_file, 'r', encoding='utf-8') as file:
            caption_content = file.read()
            
        # 字幕の挿入（まずはメタデータを追加）
        insert_result = youtube.captions().insert(
            part="snippet",
            body={
                "snippet": {
                    "videoId": video_id,
                    "language": language,
                    "name": "自動生成字幕",
                    "isDraft": False
                }
            },
            media_body=MediaFileUpload(caption_file, mimetype='text/plain', resumable=True)
        ).execute()
        
        print(f"字幕がアップロードされました: {insert_result['id']}")
        return True
        
    except HttpError as e:
        print(f"字幕のアップロード中にエラーが発生しました: {e}")
        return False

def get_title_from_blog_article(article_file):
    """ブログ記事ファイルからタイトルを取得する"""
    try:
        with open(article_file, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # タイトルは通常、最初の行または「# 」で始まる行
        lines = content.splitlines()
        for line in lines:
            if line.strip():
                # #で始まるマークダウン形式のタイトル
                if line.startswith('# '):
                    return line[2:].strip()
                # ハッシュタグを含む場合は除去
                title = re.sub(r'#\w+', '', line).strip()
                return title
        
        # タイトルが見つからない場合はファイル名を使用
        return os.path.basename(article_file).replace('.txt', '')
    
    except Exception as e:
        print(f"タイトル取得中にエラーが発生しました: {e}")
        return os.path.basename(article_file).replace('.txt', '')

def prepare_video_description(original_video_id):
    """動画の説明文を準備する"""
    # オリジナル動画のURLを含める
    base_description = f"オリジナル動画：https://www.youtube.com/watch?v={original_video_id}\n\n"
    
    # 追加の説明文
    base_description += "この動画は自動生成されたブログ記事の音声読み上げです。\n\n"
    
    # ハッシュタグを最後に追加
    base_description += "#AI自動生成 #ブログ記事 #音声読み上げ"
    
    return base_description

def upload_to_youtube(youtube, video_file, title, description, thumbnail_file=None, caption_file=None, tags=None, category=22, privacy_status="private"):
    """YouTubeに動画をアップロードする関数"""
    try:
        print(f"動画をアップロード中: {title}")
        
        # アップロードするメタデータを準備
        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': tags if tags else [],
                'categoryId': category
            },
            'status': {
                'privacyStatus': privacy_status
            }
        }
        
        # 動画ファイルをアップロード
        media = MediaFileUpload(video_file, resumable=True)
        
        # アップロード実行
        insert_request = youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=media
        )
        
        response = None
        while response is None:
            status, response = insert_request.next_chunk()
            if status:
                print(f"アップロード中... {int(status.progress() * 100)}%")
        
        video_id = response['id']
        print(f"アップロードが完了しました。ビデオID: {video_id}")
        
        # サムネイルがある場合はアップロード
        if thumbnail_file and os.path.exists(thumbnail_file):
            print("サムネイルをアップロード中...")
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_file)
            ).execute()
            print("サムネイルがアップロードされました")
        
        # 字幕ファイルがある場合は追加
        if caption_file and os.path.exists(caption_file):
            print("字幕ファイルをアップロード中...")
            create_caption(youtube, video_id, caption_file)
        
        return video_id
    
    except HttpError as e:
        print(f"YouTube APIエラー: {e}")
        return None
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

def find_latest_files(video_id=None):
    """最新の生成ファイルを検索"""
    if not video_id:
        # video_idが指定されていない場合は最新の日付のファイルを探す
        now = datetime.now()
        date_prefix = now.strftime("%Y%m%d")
    else:
        # 特定のvideo_idに対応するファイルを探す
        date_pattern = "[0-9]" * 8  # 8桁の数字 (YYYYMMDD)
        files = glob.glob(f"{date_pattern}_blog_article_{video_id}.txt")
        if not files:
            print(f"指定されたvideo_id({video_id})に対応するファイルが見つかりません")
            return None, None, None, None
        
        date_prefix = files[0].split("_")[0]
    
    # 各ファイルタイプの検索
    article_file = glob.glob(f"{date_prefix}_blog_article_{video_id or '*'}.txt")
    video_file = glob.glob(f"{date_prefix}_blog_video_{video_id or '*'}.mp4")
    audio_text_file = glob.glob(f"{date_prefix}_blog_audio_text_{video_id or '*'}.txt")
    wordcloud_file = glob.glob(f"{date_prefix}_wordcloud_{video_id or '*'}.png")
    
    # 最新のファイルを返す（存在する場合のみ）
    return (
        article_file[0] if article_file else None,
        video_file[0] if video_file else None,
        audio_text_file[0] if audio_text_file else None,
        wordcloud_file[0] if wordcloud_file else None
    )

def main():
    """メイン関数"""
    # コマンドライン引数の処理
    parser = argparse.ArgumentParser(description='YouTube動画アップローダー')
    parser.add_argument('video_id', nargs='?', help='YouTube動画ID')
    parser.add_argument('--channel', '-c', default='default', help='YouTubeチャンネル名 (token_CHANNEL.jsonとして保存)')
    args = parser.parse_args()
    
    # チャンネル名を使って認証
    youtube = get_authenticated_service(args.channel)
    
    # コマンドライン引数からvideo_idを取得
    video_id = args.video_id
    
    # 必要なファイルを見つける
    article_file, video_file, audio_text_file, wordcloud_file = find_latest_files(video_id)
    
    if not article_file or not video_file:
        print("必要なファイルが見つかりません。以下のファイルが必要です:")
        print("- *_blog_article_*.txt (タイトル取得用)")
        print("- *_blog_video_*.mp4 (アップロードする動画)")
        print("オプションで以下のファイルも使用されます:")
        print("- *_blog_audio_text_*.txt (字幕用)")
        print("- *_wordcloud_*.png (サムネイル用)")
        sys.exit(1)
    
    print(f"ブログ記事ファイル: {article_file}")
    print(f"動画ファイル: {video_file}")
    
    if audio_text_file:
        print(f"音声テキストファイル: {audio_text_file}")
    if wordcloud_file:
        print(f"ワードクラウド画像: {wordcloud_file}")
    
    # 記事からタイトルを取得
    title = get_title_from_blog_article(article_file)
    print(f"動画タイトル: {title}")
    
    # 動画IDをファイル名から抽出
    original_video_id = os.path.basename(article_file).split('_')[-1].replace('.txt', '')
    
    # 説明文を準備
    description = prepare_video_description(original_video_id)
    
    # YouTubeにアップロード
    result = upload_to_youtube(
        youtube=youtube,
        video_file=video_file,
        title=title,
        description=description,
        thumbnail_file=wordcloud_file,
        caption_file=audio_text_file,
        privacy_status="private"  # 非公開として設定
    )
    
    if result:
        print(f"アップロード成功: https://www.youtube.com/watch?v={result}")
        return 0
    else:
        print("アップロードに失敗しました")
        return 1

if __name__ == "__main__":
    main() 