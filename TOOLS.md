# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup.

## What Goes Here

Things like:

- Camera names and locations
- SSH hosts and aliases
- Preferred voices for TTS
- Speaker/room names
- Device nicknames
- Anything environment-specific

## Examples

```markdown
### Cameras

- living-room → Main area, 180° wide angle
- front-door → Entrance, motion-triggered

### SSH

- home-server → 192.168.1.100, user: admin

### TTS

- Preferred voice: "Nova" (warm, slightly British)
- Default speaker: Kitchen HomePod
```

## Why Separate?

Skills are shared. Your setup is yours. Keeping them apart means you can update skills without losing your notes, and share skills without leaking your infrastructure.

---

Add whatever helps you do your job. This is your cheat sheet.

## 网络搜索

**Context7（技术文档搜索，无需 Key）**
用法：`cd /root/.openclaw/workspace/skills/context7 && npx tsx query.ts context <repo> <query>`
示例：`npx tsx query.ts context "reactjs/react.dev" "useState hook"`
支持的库：React、Next.js、Vue、Svelte 等主流技术库

**搜狗搜索（通用网页搜索，无需 Key）**
URL：`https://www.sogou.com/web?query=<关键词>&ie=utf8`
状态：✅ 可用（百度触发验证，搜狗/必应正常）

**web_search 内置工具**
当前不可用（SearXNG 被墙），需要配置 Tavily API Key 才能用。

## Context7 常用仓库 ID
- React: `reactjs/react.dev`
- Next.js: `vercel/next.js`
- Vue: `vuejs/vue`
- Svelte: `sveltejs/svelte`

## 网络访问状态
- 百度：❌ 触发安全验证
- 搜狗：✅ 正常
- 必应国际：✅ 正常
- Google/DuckDuckGo：❌ 被墙
- Tavily API：✅ 可访问（需要 Key）

---

## Related

- [Agent workspace](/concepts/agent-workspace)
