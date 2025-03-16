import argparse
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import os
from datetime import datetime
from openai import OpenAI

client = OpenAI()
LLM_MODEL = "chatgpt-4o-latest"

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
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        formatter = TextFormatter()
        formatted_transcript = formatter.format_transcript(transcript)
        return formatted_transcript
    except Exception as e:
        return f"文字起こしを取得できませんでした: {e}"

def generate_blog_article(transcript, youtube_url, language="ja", persona=None, blog_style=None):
    """文字起こしデータを基にOpenAI APIを使って日本語ブログ記事を生成"""
    
    # ペルソナの設定（デフォルトは技術ブロガー）
    if persona is None:
        persona = "あなたは10年以上の経験を持つ女性エンジニアであり且つ技術ブロガーで、最新技術に精通しており、実際に様々な技術を試した経験があります。あなたは読者と対話するような親しみやすい文体で書きます。"
    
    # ブログスタイルの設定（デフォルトは標準的なブログ）
    if blog_style is None:
        blog_style = "個人的な体験談や実践例を含め、読者が共感できる内容にしてください。専門的な内容でも初心者にもわかりやすく説明し、時には独自の見解や予測も大胆に述べてください。"
    
    messages = [
        { 
            "role": "system", 
            "content": f"{persona} あなたの記事は常に独自の視点と実践的な内容で読者を惹きつけます。" 
        },
        { 
            "role": "user", 
            "content": f"""以下の文字起こし内容を元に、2000〜3000字程度の日本語の解説ブログ記事を作成してください。
                        {blog_style}
                        
                        記事の構成は以下のようにしてください：
                        
                        1. タイトルの次に動画へのURL {youtube_url} をリンク形式でなく文字列でそのまま入れてください。
                        
                        2. 「ポイント」セクションでは、主な主張や論点の考察を箇条書きで記載して、ここだけ読めばおおよその概要がわかるようにしてください。
                        
                        3. 「はじめに」セクションでは、あなた自身がこのトピックに出会った個人的なエピソードや、実際に試した体験談から始めてください。一人称で書き、読者に親近感を持たせてください。
                        
                        4. 本文では、文字起こし内容の単なる要約ではなく、以下の要素を含めてください：
                           - 実際に試した「【実験】」や「【実践例】」のセクションを少なくとも2つ
                           - 日本市場や日本企業に特化した分析や考察
                           - 具体的な数字や事例を用いた「【分析】」セクション
                           - 大胆な「【予測】」セクション（2-3年後の未来予測）
                        
                        5. 「読者の皆さんへの質問」セクションを設け、読者が考えるべき3つの質問を提示し、コメント欄での議論を促してください。
                        
                        6. 「まとめ」セクションでは、全体を簡潔にまとめつつ、あなた独自の視点や尖った意見を述べてください。
                        
                        7. 最後に「【実践】〜を試してみたい方へ」というセクションを追加し、読者が実際に試せる具体的なアイデアやプロンプト例を3つ提案してください。
                        
                        太字(*)は使わず、見出し（#）を適切に使用してください。
                        タイトルには関連するハッシュタグを入れてください。
                        
                        文字起こし内容：
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

def main():
    # 引数の設定
    parser = argparse.ArgumentParser(description="YouTube動画の字幕を取得し、それを基にブログ記事を生成するスクリプト")
    parser.add_argument("language", help="取得したい字幕の言語コード（例: 'en', 'ja'）")
    parser.add_argument("youtube_url", help="YouTube動画のURLを指定")
    parser.add_argument("--persona", help="ブログ記事の書き手のペルソナを指定（例: 'tech_blogger', 'business_analyst', 'educator'）")
    parser.add_argument("--style", help="ブログ記事のスタイルを指定（例: 'analytical', 'storytelling', 'tutorial'）")
    
    args = parser.parse_args()
    youtube_url = args.youtube_url
    language = args.language
    
    # ペルソナの設定
    personas = {
        "tech_blogger": "あなたは10年以上の経験を持つ技術ブロガーで、最新技術に精通しており、実際に様々な技術を試した経験があります。あなたは読者と対話するような親しみやすい文体で書きます。",
        "business_analyst": "あなたは企業戦略コンサルタントとして多くの企業の分析を行ってきた経験を持ちます。データに基づいた分析と実践的な提案を得意とし、ビジネスの視点から物事を考察します。",
        "educator": "あなたは教育者として複雑な概念を初心者にもわかりやすく説明することを得意としています。具体例を多用し、ステップバイステップで理解を深める文章を書きます。",
        "futurist": "あなたは未来学者として技術トレンドを分析し、大胆な予測を行うことを得意としています。過去の事例と現在のデータから、説得力のある未来予測を提示します。"
    }
    
    # スタイルの設定
    styles = {
        "analytical": "データや事実に基づいた分析を重視し、論理的な構成で内容を展開してください。具体的な数字や比較を用いて説得力を高めてください。",
        "storytelling": "物語形式で内容を展開し、読者を引き込むような文体で書いてください。個人的なエピソードや具体的な事例を多く含めてください。",
        "tutorial": "読者が実際に試せる手順やステップを詳細に説明し、実践的なガイドとなるような内容にしてください。具体的なプロンプト例や使い方のコツを含めてください。",
        "debate": "賛否両論を提示し、様々な視点から議論を展開してください。読者に考えるきっかけを与え、自分自身の意見も明確に述べてください。"
    }
    
    selected_persona = personas.get(args.persona) if args.persona else None
    selected_style = styles.get(args.style) if args.style else None

    # 文字起こしを取得
    print("文字起こしを取得中...")
    transcript = fetch_transcript(youtube_url, language)
    if "取得できませんでした" in transcript:
        print(transcript)
        return

    # ブログ記事を生成
    print("ブログ記事を生成中...")
    blog_article = generate_blog_article(transcript, youtube_url, language="ja", 
                                         persona=selected_persona, blog_style=selected_style)

    # ファイルに保存
    today = datetime.now().strftime('%Y%m%d')
    video_id = get_video_id(youtube_url)
    filename = f"{today}_blog_article_{video_id}.txt"
    save_to_file(blog_article, filename)

if __name__ == "__main__":
    main()
