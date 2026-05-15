import os, re, json

def escape_js(s):
    return s.replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')

posts_dir = "/root/.openclaw/workspace/blog/source/_posts"
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

post_items = []
for fname in sorted(os.listdir(posts_dir)):
    if not fname.endswith('.md') or fname.startswith('hello'):
        continue
    path = os.path.join(posts_dir, fname)
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()
    
    after_first = raw[3:]
    end_fm_idx = after_first.find('\n---')
    if end_fm_idx == -1:
        continue
    
    fm_text = after_first[:end_fm_idx]
    content = after_first[end_fm_idx+4:].strip()
    
    meta = {}
    for line in fm_text.split('\n'):
        if ':' in line:
            k, v = line.split(':', 1)
            meta[k.strip()] = v.strip().strip('"\'')
    
    title = meta.get('title', fname)
    date_raw = meta.get('date', '')
    date_match = re.match(r'(\d{4}-\d{2}-\d{2})', date_raw)
    date = date_match.group(1) if date_match else date_raw[:10]
    
    tags_raw = meta.get('tags', '[]')
    if tags_raw.startswith('['):
        tags = [t.strip().strip('"\'') for t in tags_raw.strip('[]').split(',') if t.strip()]
    elif tags_raw:
        tags = [tags_raw.strip('"\'')]
    else:
        tags = []
    
    slug = fname.replace('.md', '')
    
    excerpt_lines = []
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and not stripped.startswith('|') and not stripped.startswith('```') and len(stripped) > 40:
            excerpt_lines.append(stripped)
            if len(excerpt_lines) >= 2:
                break
    excerpt = excerpt_lines[0][:180] if excerpt_lines else content[:180]
    
    content = re.sub(r'\{%.*?%\}', '', content)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    post_items.append({
        'slug': slug,
        'title': title,
        'date': date,
        'tags': tags,
        'excerpt': excerpt,
        'content': content
    })

for p in post_items:
    lines.append("  {")
    lines.append(f'    slug: "{p["slug"]}",')
    title_escaped = p["title"].replace('\\', '\\\\').replace('"', '\\"').replace("\n", " ")
    lines.append(f'    title: "{title_escaped}",')
    lines.append(f'    date: "{p["date"]}",')
    lines.append(f'    tags: {json.dumps(p["tags"])},')
    lines.append(f'    excerpt: `{escape_js(p["excerpt"])}`,')
    lines.append(f'    content: `{escape_js(p["content"])}`,')
    lines.append("  },")
    print(f"  ok {len(p['content']):5d}w {p['title'][:50]}")

lines.append("];")
with open("/root/.openclaw/workspace/blog-next/src/lib/posts/posts.ts", 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"\nWritten {len(post_items)} posts, total {sum(len(p['content']) for p in post_items)//2//500*500}w")