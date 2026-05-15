#!/usr/bin/env python3
"""
Sync Hexo posts to Next.js posts.ts
Reads all .md files from Hexo's _posts/, converts to posts.ts format
"""
import os, re, base64, datetime

POSTS_DIR = "/root/.openclaw/workspace/blog-next/src/lib/posts"

def parse_hexo_post(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()
    
    # Parse front matter
    meta = {}
    if raw.startswith('---'):
        parts = raw[3:].split('---', 2)
        fm_text = parts[1]
        content = parts[2].lstrip('\n')
        
        for line in fm_text.split('\n'):
            if ':' in line:
                key, val = line.split(':', 1)
                meta[key.strip()] = val.strip().strip("'\"")
    
    title = meta.get('title', os.path.basename(path))
    date_str = meta.get('date', '')
    tags_str = meta.get('tags', '')
    
    if isinstance(tags_str, str) and tags_str.startswith('['):
        tags = [t.strip() for t in tags_str.strip('[]').split(',')]
    else:
        tags = [tags_str] if tags_str else []
    
    # Extract date YYYY-MM-DD
    date_match = re.match(r'(\d{4}-\d{2}-\d{2})', date_str)
    date = date_match.group(1) if date_match else date_str[:10]
    
    # Generate slug from filename
    slug = os.path.basename(path).replace('.md', '')
    
    # Clean content - remove hexo-specific shortcodes
    content = re.sub(r'\{%.*?%\}', '', content)  # hexo tags
    content = re.sub(r'\[!.*?\].*\n', '', content)  # blockquotes
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = content.strip()
    
    return {
        'slug': slug,
        'title': title,
        'date': date,
        'tags': tags,
        'excerpt': content.split('\n')[2] if len(content.split('\n')) > 2 else content[:120],
        'content': content
    }

def write_posts_ts(posts):
    os.makedirs(POSTS_DIR, exist_ok=True)
    
    lines = [
        "export interface Post {",
        "  slug: string;",
        "  title: string;",
        "  date: string;",
        "  tags: string[];",
        "  excerpt: string;",
        "  content: string;",
        "}",
        "",
        "export const posts: Post[] = [",
    ]
    
    for p in posts:
        lines.append("  {")
        lines.append(f'    slug: "{p["slug"]}",')
        lines.append(f'    title: "{p["title"]}",')
        lines.append(f'    date: "{p["date"]}",')
        lines.append(f'    tags: {p["tags"]},')
        
        # excerpt as template literal (handle quotes)
        excerpt = p['excerpt'].replace('\\', '\\\\').replace('`', '\\`').replace('"', '\\"')
        lines.append(f'    excerpt: `{excerpt}`,')
        
        # content as template literal
        content = p['content'].replace('\\', '\\\\').replace('`', '\\`')
        content_lines = content.split('\n')
        content_str = '\n'.join(content_lines)
        lines.append(f'    content: `{content_str}`,')
        
        lines.append("  },")
    
    lines.append("];")
    
    with open(os.path.join(POSTS_DIR, "posts.ts"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Written {len(posts)} posts to {POSTS_DIR}/posts.ts")

def main():
    hexo_posts_dir = "/root/.openclaw/workspace/blog/source/_posts"
    posts = []
    
    for fname in sorted(os.listdir(hexo_posts_dir)):
        if fname.endswith('.md') and not fname.startswith('hello'):
            path = os.path.join(hexo_posts_dir, fname)
            try:
                post = parse_hexo_post(path)
                if len(post['content']) > 200:  # filter out hello-world
                    posts.append(post)
                    print(f"  ✓ {fname}: {post['title'][:40]}")
            except Exception as e:
                print(f"  ✗ {fname}: {e}")
    
    write_posts_ts(posts)
    print(f"\n✅ Synced {len(posts)} posts")

if __name__ == '__main__':
    main()