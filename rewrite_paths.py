#!/usr/bin/env python3
"""
Post-build: replace _next/static/ paths with static/chunks/ in all HTML and .txt files.
GitHub Pages intercepts _next/ path, so we rewrite everything.
"""
import os, shutil

out_dir = "/root/.openclaw/workspace/blog-next/out"
print("Rewriting _next/static paths...")

count = 0
for root, dirs, files in os.walk(out_dir):
    for fname in files:
        if not (fname.endswith('.html') or fname.endswith('.txt')):
            continue
        path = os.path.join(root, fname)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        # More specific pattern first (otherwise _next/static/chunks contains _next/static and gets double-replaced)
        new_content = content.replace('/_next/static/chunks/', '/static/chunks/')
        new_content = new_content.replace('/_next/static/', '/static/chunks/')
        if new_content != content:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            rel = os.path.relpath(path, out_dir)
            print(f"  Rewrote: {rel}")
            count += 1

print(f"✅ Rewrote {count} files")

# Copy chunks from _next/static/chunks/ to static/chunks/
os.makedirs(f"{out_dir}/static/chunks", exist_ok=True)
src_chunks = f"{out_dir}/_next/static/chunks"
dst_chunks = f"{out_dir}/static/chunks"

if os.path.exists(src_chunks):
    for f in os.listdir(src_chunks):
        src = os.path.join(src_chunks, f)
        dst = os.path.join(dst_chunks, f)
        shutil.copy2(src, dst)
    print(f"✅ Copied {len(os.listdir(src_chunks))} chunks to static/chunks/")
else:
    print("  Note: _next/static/chunks not found")