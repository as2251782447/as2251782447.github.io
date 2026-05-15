# MEMORY.md - Long-Term Memory

## Server

- **Host**: VM-0-3-ubuntu (Tencent Cloud Lighthouse)
- **OS**: Ubuntu 6.8.0, x86_64
- **CPU**: 4 cores
- **RAM**: 3.6GB (2.5GB available)
- **Disk**: 40GB (17GB used, 21GB available)
- **Software**: Node.js v22.22.2, Python 3.12.3, npm 10.9.7, curl 8.5.0

## OpenClaw

- **Version**: 2026.5.7
- **Config**: /root/.openclaw/openclaw.json
- **Workspace**: /root/.openclaw/workspace/
- **Gateway**: Port 19721, token auth, LAN bind

## Channels

- **Feishu**: Paired ✅ (open_id: ou_162e22e793703cfcb940fd6799b5b6c5)
- **LightClawBot**: Enabled

## Blog (biluonobug.github.io)

**URL**: https://biluonobug.github.io/
**Repo**: BiLuoNoBug/biluonobug.github.io
**Stack**: Next.js 16 + output:export (static HTML), not Hexo
**Source**: `/root/.openclaw/workspace/blog-next/`
**Posts source**: `/root/.openclaw/workspace/blog/source/_posts/`

**8 articles published**: HTMX, WASM, eBPF, VectorDB, Subquadratic Attention, AI Agent workflow, OpenClaw skills, digital employee

**Auto-deploy pipeline**:
1. `python3 /root/.openclaw/workspace/blog-next/convert_posts.py` — converts Hexo posts → posts.ts
2. `python3 /root/.openclaw/workspace/blog-next/deploy_blog.py` — builds + pushes via git commit to trigger GitHub Pages rebuild

**Asset path fix**: Next.js `_next/static/` paths rewritten to `/static/chunks/` in all HTML files after build (GitHub Pages intercepts `_next`). Run `python3 /root/.openclaw/workspace/blog-next/rewrite_paths.py` after build.

**Cron job** (3b91218c): hourly blog writer → writes to Hexo source → syncs → deploys via git commit


## GitHub
- **Username**: BiLuoNoBug (改过，原 as2251782447)
- **Token**: [REDACTED] (需确认 token 在改名后是否还有效)
- **Blog Repo**: BiLuoNoBug/biluonobug.github.io
- **Blog URL**: https://biluonobug.github.io/

## Skills Installed

- github, tencent-docs, tencentcloud-lighthouse-skill, tencent-meeting-skill, tencent-cos-skill
- find-skills, skillhub-preference, memory-hygiene, openclaw-tavily-search
- agent-browser-clawdbot, web-tools-guide, system-data-intelligence
- system-monitor-pro, command-center, auto-monitor
- hexo-blog-with-seo, website, self-improving, auto-updater, deploy-helper, multi-search-engine

## Notes

- openclaw CLI symlink is broken → use full path:
  `node /root/.local/share/pnpm/global/5/.pnpm/openclaw@2026.5.7_@types+express@5.0.6/node_modules/openclaw/openclaw.mjs`
- CLI wrapper `/root/.local/bin/openclaw` has wrong module path baked in