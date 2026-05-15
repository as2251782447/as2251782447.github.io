#!/usr/bin/env python3
"""
Deploy script: builds Next.js blog and pushes to GitHub via git commit.
Triggered after new content is added to posts.ts.
Usage: python3 deploy_blog.py
"""
import urllib.request, json, base64, os, subprocess, sys

TOKEN = "REDACTED_TOKEN"
REPO = "BiLuoNoBug/biluonobug.github.io"
BASE = f"https://api.github.com/repos/{REPO}"
BLOG_DIR = "/root/.openclaw/workspace/blog-next"
OUT_DIR = f"{BLOG_DIR}/out"

def api(method, path, data=None):
    url = f"{BASE}/{path}"
    h = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github.v3+json"}
    if data and not isinstance(data, bytes):
        data = json.dumps(data).encode()
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read()), e.code

def build():
    print("Building Next.js blog...")
    result = subprocess.run(["npm", "run", "build"], cwd=BLOG_DIR,
                           capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print("Build failed:", result.stderr[-500:])
        return False
    print("  Build OK")
    
    # Fix _next/static/ → static/chunks/ in all HTML
    print("  Rewriting asset paths...")
    rewrite = subprocess.run(["python3", f"{BLOG_DIR}/rewrite_paths.py"],
                             cwd=BLOG_DIR, capture_output=True, text=True)
    if rewrite.returncode != 0:
        print("  Path rewrite warning:", rewrite.stderr[-200:])
    else:
        print("  Path rewrite OK")
    
    return True

def deploy():
    # Get current HEAD
    ref_info, _ = api("GET", "git/ref/heads/main")
    head_sha = ref_info["object"]["sha"]
    commit_data, _ = api("GET", f"git/commits/{head_sha}")
    base_tree_sha = commit_data["tree"]["sha"]
    print(f"Base commit: {head_sha[:8]}")

    # Read all out/ files and create blobs
    entries = []
    skip = ('__next', '.txt', 'vercel.svg', 'window.svg', 'file.svg', 'globe.svg', 'next.svg')

    # Ensure .nojekyll exists (tells GitHub Pages not to run Jekyll)
    nojekyll_path = f"{OUT_DIR}/.nojekyll"
    with open(nojekyll_path, 'w') as f:
        f.write('')

    for root, dirs, files in os.walk(OUT_DIR):
        dirs[:] = [d for d in dirs if d not in ('__next', '_next', '_not-found')]
        for fname in files:
            if any(p in fname for p in skip):
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, OUT_DIR).replace('\\', '/')
            with open(fpath, 'rb') as f:
                content = f.read()
            blob_data, _ = api("POST", "git/blobs", {
                "content": base64.b64encode(content).decode(),
                "encoding": "base64"
            })
            entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob_data["sha"]})

    print(f"  {len(entries)} files to deploy")

    # Create tree
    tree_data, s = api("POST", "git/trees", {"base_tree": base_tree_sha, "tree": entries})
    if s not in (200, 201):
        print("Tree error:", tree_data)
        return False
    new_tree_sha = tree_data["sha"]
    print(f"  New tree: {new_tree_sha[:8]}")

    # Create commit
    commit_data, s = api("POST", "git/commits", {
        "message": "Blog update via auto-deploy",
        "tree": new_tree_sha,
        "parents": [head_sha]
    })
    if s not in (200, 201):
        print("Commit error:", commit_data)
        return False
    new_commit_sha = commit_data["sha"]
    print(f"  New commit: {new_commit_sha[:8]}")

    # Update branch
    ref_update, s = api("PATCH", "git/refs/heads/main", {"sha": new_commit_sha, "force": False})
    if s not in (200, 201):
        print("Ref update error:", ref_update)
        return False
    
    print(f"  Deployed! Commit: {new_commit_sha[:8]}")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--build-only":
        sys.exit(0 if build() else 1)
    if build() and deploy():
        print("✅ Deploy complete")
    else:
        print("❌ Deploy failed")
        sys.exit(1)