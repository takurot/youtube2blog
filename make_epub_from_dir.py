#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

from ebooklib import epub
from bs4 import BeautifulSoup
import markdown


@dataclass
class ChapterSource:
    file_path: str
    title: str
    content_html: str
    file_name: str


def read_text_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def infer_title_from_text(content: str, fallback: str) -> str:
    # Try to find first markdown ATX header or first non-empty line
    lines = content.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            # remove leading #'s and surrounding spaces
            return re.sub(r'^#+\s*', '', stripped).strip()
        # If looks like a setext header style, skip until next line check
        # We'll simply use first non-empty line as title if no ATX
        return stripped[:120]
    return fallback


def convert_markdown_to_html(md: str) -> str:
    return markdown.markdown(md, extensions=[
        'extra', 'sane_lists', 'toc', 'tables'
    ])


def convert_text_to_html(txt: str) -> str:
    # Treat blank-line-separated paragraphs into <p>
    paragraphs = [p.strip() for p in txt.split('\n\n') if p.strip()]
    html_parts = [f'<p>{BeautifulSoup(p, "html.parser").text}</p>' for p in paragraphs]
    return '\n'.join(html_parts)


def strip_leading_redundant_title_from_txt(raw: str, title: str) -> str:
    lines = raw.splitlines()
    i = 0
    # skip initial empty lines
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines):
        first = lines[i].strip()
        # pattern like '## Title' or '# Title'
        if first.startswith('#'):
            candidate = re.sub(r'^#+\s*', '', first).strip()
            if candidate == title:
                i += 1
                # also skip one following blank line if present
                if i < len(lines) and not lines[i].strip():
                    i += 1
        elif first == title:
            i += 1
            if i < len(lines) and not lines[i].strip():
                i += 1
    return '\n'.join(lines[i:])


def load_chapters_from_dir(input_dir: str) -> List[ChapterSource]:
    entries = []
    for name in sorted(os.listdir(input_dir)):
        if name.startswith('.'):
            continue
        lower = name.lower()
        if not (lower.endswith('.txt') or lower.endswith('.md')):
            continue
        full = os.path.join(input_dir, name)
        if not os.path.isfile(full):
            continue
        raw = read_text_file(full)
        
        # Check if content contains markdown syntax
        has_markdown = any([
            re.search(r'^#+\s+', raw, re.MULTILINE),  # Headers
            re.search(r'\*\*.*?\*\*', raw),            # Bold
            re.search(r'\*.*?\*', raw),                # Italic
            re.search(r'^\s*[-*+]\s+', raw, re.MULTILINE),  # Lists
            re.search(r'^\s*\d+\.\s+', raw, re.MULTILINE),  # Numbered lists
        ])
        
        # Use markdown conversion if file is .md or contains markdown syntax
        if lower.endswith('.md') or has_markdown:
            title = infer_title_from_text(raw, fallback=os.path.splitext(name)[0])
            body_html = convert_markdown_to_html(raw)
        else:
            title = infer_title_from_text(raw, fallback=os.path.splitext(name)[0])
            raw_no_dup = strip_leading_redundant_title_from_txt(raw, title)
            body_html = convert_text_to_html(raw_no_dup)

        # Ensure title heading is included at top of chapter body
        soup = BeautifulSoup(body_html, 'html.parser')
        # If body already starts with an h1/h2 that equals title, keep; else insert
        first_heading = soup.find(['h1', 'h2'])
        should_insert = True
        if first_heading and first_heading.get_text(strip=True) == title:
            should_insert = False
        if should_insert:
            heading_tag = soup.new_tag('h1')
            heading_tag.string = title
            if soup.contents:
                soup.insert(0, heading_tag)
            else:
                soup.append(heading_tag)
        content_html = str(soup)

        safe_base = re.sub(r'[^A-Za-z0-9._-]+', '_', os.path.splitext(name)[0])
        file_name = f'{safe_base}.xhtml'
        entries.append(ChapterSource(
            file_path=full,
            title=title,
            content_html=content_html,
            file_name=file_name,
        ))
    return entries


def build_epub(input_dir: str, output_path: str, title: str, author: str, lang: str,
               description: Optional[str] = None, cover: Optional[str] = None) -> None:
    book = epub.EpubBook()
    book.set_identifier(f"{os.path.basename(output_path)}")
    book.set_title(title)
    book.set_language(lang)
    book.add_author(author)
    if description:
        # description as DC:description
        book.add_metadata('DC', 'description', description)

    if cover and os.path.isfile(cover):
        with open(cover, 'rb') as f:
            data = f.read()
        book.set_cover(os.path.basename(cover), data)

    # Load chapters
    chapters = load_chapters_from_dir(input_dir)
    if not chapters:
        raise RuntimeError(f"No chapter files (.txt/.md) found in {input_dir}")

    spine_items = ['nav']
    toc_items = []

    # Add default CSS for pleasant reading
    style = '''
    body { 
        font-family: -apple-system, BlinkMacSystemFont, "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif; 
        line-height: 1.8;
        padding: 1em;
    }
    h1 { 
        font-size: 1.6em; 
        margin: 0 0 0.6em 0;
        font-weight: bold;
        border-bottom: 2px solid #333;
        padding-bottom: 0.3em;
    }
    h2 { 
        font-size: 1.3em; 
        margin: 1.2em 0 0.4em 0;
        font-weight: bold;
    }
    h3 { 
        font-size: 1.1em; 
        margin: 1em 0 0.3em 0;
        font-weight: bold;
    }
    p { 
        line-height: 1.8; 
        margin: 0.5em 0;
    }
    strong, b { 
        font-weight: bold; 
    }
    em, i { 
        font-style: italic; 
    }
    ul, ol { 
        margin: 0.8em 0; 
        padding-left: 2em; 
    }
    li { 
        margin: 0.3em 0; 
        line-height: 1.6;
    }
    code { 
        font-family: monospace; 
        background-color: #f0f0f0; 
        padding: 0.1em 0.3em;
        border-radius: 3px;
    }
    pre { 
        background-color: #f5f5f5; 
        padding: 1em; 
        overflow-x: auto;
        border-radius: 5px;
    }
    blockquote { 
        border-left: 4px solid #ccc; 
        margin: 1em 0; 
        padding-left: 1em;
        color: #666;
    }
    '''
    nav_css = epub.EpubItem(uid="style_nav", file_name="style/nav.css", media_type="text/css", content=style)
    book.add_item(nav_css)

    for ch in chapters:
        c = epub.EpubHtml(title=ch.title, file_name=ch.file_name, lang=lang)
        c.content = ch.content_html
        c.add_item(nav_css)  # Link CSS to chapter
        book.add_item(c)
        spine_items.append(c)
        toc_items.append(c)

    book.toc = tuple(toc_items)

    # Required nav files
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    book.spine = spine_items

    epub.write_epub(output_path, book, {})


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build EPUB from directory of .txt/.md files')
    p.add_argument('input_dir', help='Input directory containing .txt/.md files')
    p.add_argument('--title', required=True)
    p.add_argument('--author', required=True)
    p.add_argument('--lang', default='ja')
    p.add_argument('--description', default=None)
    p.add_argument('--cover', default=None)
    p.add_argument('--output', required=True)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        build_epub(
            input_dir=args.input_dir,
            output_path=args.output,
            title=args.title,
            author=args.author,
            lang=args.lang,
            description=args.description,
            cover=args.cover,
        )
        print(f"Created EPUB: {args.output}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
