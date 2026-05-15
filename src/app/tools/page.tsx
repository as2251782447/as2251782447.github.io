export const metadata = {
  title: "Tools · biluo",
  description: "Developer cheatsheets, CLI references, and useful tools collected over the years.",
};

const tools = [
  {
    category: "CLI",
    icon: "⌨",
    items: [
      {
        name: "Git",
        desc: "Git commands I always have to look up",
        content: `# Branch
git checkout -b new-branch          # create & switch
git switch -c new-branch            # modern way
git rebase main                     # rebase onto main
git cherry-pick <commit>           # pick specific commit
git stash -u                       # include untracked files

# Undo
git commit --amend                  # change last commit msg
git reset --soft HEAD~1            # undo commit, keep staged
git reset --mixed HEAD~1           # undo commit, unstaged
git reset --hard HEAD~1            # undo commit, discard all
git revert <commit>                # create reverting commit

# Browse
git log --oneline -20
git log --graph --oneline --all    # visualize branches
git diff HEAD~3..HEAD
git diff main..feature
git blame path/to/file
git show <commit>:path/to/file     # file at commit

# Submodules
git submodule add <url> <path>
git submodule update --init --recursive`,
        tags: ["git", "version-control"],
      },
      {
        name: "Docker",
        desc: "Docker commands I actually use",
        content: `# Build & Run
docker build -t myapp .
docker run -d -p 3000:3000 --name myapp myapp
docker compose up -d
docker compose -f prod.yml up -d

# Inspect
docker ps -a                       # all containers
docker images
docker logs -f <container>
docker exec -it <container> sh
docker inspect <container>
docker stats --no-stream          # live stats once

# Clean
docker system prune -f
docker container prune -f
docker image prune -f
docker rm $(docker ps -aq)
docker rmi $(docker images -q)
docker volume prune

# Network
docker network ls
docker network create mynet
docker run --network mynet ...

# Multi-stage
FROM node:20-alpine AS builder
COPY . .
RUN npm run build
FROM nginx:alpine
COPY --from=builder/dist ./usr/share/nginx/html`,
        tags: ["docker", "containers"],
      },
      {
        name: "Linux",
        desc: "Common shell commands for daily work",
        content: `# Disk & Memory
df -h /                      # disk usage
du -sh *                    # dir sizes (human)
du -sh /* 2>/dev/null | sort -h | tail -10
free -h                      # memory

# Network
ss -tulnp                   # listening ports
curl -I <url>               # check headers
wget -qO- <url>             # fetch quietly
nc -zv host 80              # test port
mtr google.com              # traceroute

# Process
ps aux | grep <name>
top / htop                  # monitor
kill -9 <pid>
nohup cmd &                 # background
Ctrl+Z; bg                  # suspend then bg
watch -n1 'command'         # repeat every 1s

# Files
find . -name "*.log" -mtime +7
tar -czf archive.tar.gz dir/
rsync -avz src/ dest/
chmod +x script.sh

# Text
grep -r "pattern" . --include="*.ts"
awk -F',' '{print $1}' file.csv
sed -i 's/old/new/g' file
cat file | jq '.'            # pretty json`,
        tags: ["linux", "shell"],
      },
      {
        name: "npm / Node",
        desc: "npm and Node.js CLI essentials",
        content: `# Packages
npm install <pkg>
npm install -D <pkg>         # dev dependency
npm install <pkg>@<version>
npm uninstall <pkg>
npm update <pkg>

# Run
npm run dev                  # dev script
npm run build
npm run test

# Package info
npm info <pkg>
npm view <pkg> versions
npm ls                        # installed

# npx
npx create-next-app@latest
npx tsc --init
npx playwright install

# Node
node -e "console.log('hello')"
node --inspect index.js
node --version
nvm use 20                   # node version manager`,
        tags: ["node", "npm", "javascript"],
      },
    ],
  },
  {
    category: "Web Dev",
    icon: "🌐",
    items: [
      {
        name: "CSS Grid",
        desc: "CSS Grid layout patterns that actually work",
        content: `/* Holy Grail layout */
.parent {
  display: grid;
  grid-template-columns: 250px 1fr 250px;
  grid-template-rows: auto 1fr auto;
  min-height: 100vh;
}

/* Responsive grid */
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
}

/* Span tricks */
.item { grid-column: span 2; }
.item { grid-row: 2 / 4; }
.item { grid-column: 1 / -1; /* full width */ }

/* Place items center */
.wrapper {
  display: grid;
  place-items: center;
}

/* Subgrid */
.parent {
  display: grid;
  grid-template-columns: 1fr 2fr;
}
.child {
  grid-column: span 2;
  display: grid;
  grid-template-columns: subgrid;
}`,
        tags: ["css", "layout", "grid"],
      },
      {
        name: "Flexbox",
        desc: "Flexbox patterns I use constantly",
        content: `/* Center everything */
.container {
  display: flex;
  justify-content: center;
  align-items: center;
}

/* Space between columns */
.row {
  display: flex;
  gap: 1rem;
}
.row > * { flex: 1; }

/* Sticky footer */
.page {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}
.content { flex: 1; }
.footer { margin-top: auto; }

/* Wrap with gaps */
.cards {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
}

/* Vertical stack */
.vstack { display: flex; flex-direction: column; gap: 1rem; }

/* Equal height columns */
.row { display: flex; }
.row > * { flex: 1; }`,
        tags: ["css", "flexbox", "layout"],
      },
      {
        name: "Regex",
        desc: "Regex patterns for everyday use",
        content: `# Common patterns

# Email
[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}

# URL
https?://[^\s/$.?#].[^\s]*

# Chinese chars
[\u4e00-\u9fff]

# Trim whitespace
^\s+|\s+$

# Numbers only
^\d+(\.\d+)?$

# Capture groups
(\d+)-(\w+)  → match[1]=digits, match[2]=word

# Lookahead/lookbehind
foo(?=bar)    # foo followed by bar
(?<=foo)bar   # bar preceded by foo
(?!bar)foo    # foo NOT followed by bar

# Replace examples (JS)
str.replace(/(\\d+)/g, '#$1')
str.replace(/(\\w+)/g, (_, p) => p.toUpperCase())`,
        tags: ["regex", "text"],
      },
      {
        name: "HTTP Headers",
        desc: "Useful curl commands and header references",
        content: `# Common headers
curl -I https://example.com         # headers only
curl -v https://example.com          # verbose
curl -X POST https://api.com \\
  -H "Content-Type: application/json" \\
  -d '{"key":"value"}'

# Auth
curl -H "Authorization: Bearer <token>" \\
     -H "X-API-Key: <key>"

# Common response codes
200 OK | 201 Created | 204 No Content
301 Moved | 302 Found | 304 Not Modified
400 Bad Request | 401 Unauthorized
403 Forbidden | 404 Not Found
429 Too Many Requests | 500 Internal Error

# Cache headers
Cache-Control: max-age=3600
ETag: "abc123"
Last-Modified: Mon, 01 Jan 2024 00:00:00 GMT`,
        tags: ["http", "network", "curl"],
      },
      {
        name: "Chrome DevTools",
        desc: "DevTools tricks I keep forgetting",
        content: `# Console
$x('//div')           # XPath query
$$('div')             # querySelectorAll
copy(object)         # copy to clipboard
monitor(fn)           # log when function called

# Network
Copy as cURL          # right-click request
Resend from Network   # right-click → Replay

# Elements
Ctrl+Shift+P         # command palette
$0                    # current element
$_                    # last result

# Performance
Ctrl+Shift+P → Show Coverage
Ctrl+Shift+P → Rendering → FPS meter

# Application
Clear storage        # Application tab
View IndexedDB       # Application → IndexedDB

# Network throttling
Ctrl+Shift+P → Network throttling`,
        tags: ["devtools", "chrome", "debug"],
      },
    ],
  },
  {
    category: "AI & Data",
    icon: "🤖",
    items: [
      {
        name: "curl + LLM",
        desc: "Call LLM APIs from terminal",
        content: `# OpenAI-compatible API
curl https://api.openai.com/v1/chat/completions \\
  -H "Authorization: Bearer $OPENAI_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "gpt-4o",
    "messages": [{"role":"user","content":"Hello!"}]
  }'

# Ollama (local)
curl http://localhost:11434/api/chat -d '{
  "model": "llama3",
  "messages": [{"role":"user","content":"Hello!"}]
}'

# Claude (Anthropic)
curl https://api.anthropic.com/v1/messages \\
  -H "x-api-key: $ANTHROPIC_KEY" \\
  -H "anthropic-version: 2023-06-01" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [{"role":"user","content":"Hello!"}]
  }'`,
        tags: ["ai", "llm", "api"],
      },
      {
        name: "SQL",
        desc: "Common SQL patterns I keep referencing",
        content: `-- Window functions
SELECT name, salary,
  SUM(salary) OVER (ORDER BY salary DESC) as running_total,
  RANK() OVER (PARTITION BY dept ORDER BY salary DESC) as rank,
  LAG(created_at, 1) OVER (ORDER BY created_at) as prev
FROM employees;

-- CTE chain
WITH monthly_sales AS (
  SELECT DATE_TRUNC('month', created_at) as month, SUM(amount) as total
  FROM orders GROUP BY 1
),
growth AS (
  SELECT month, total,
    LAG(total) OVER (ORDER BY month) as prev
  FROM monthly_sales
)
SELECT month, total, prev,
  ROUND((total - prev) / prev * 100, 2) as growth_pct
FROM growth;

-- JSON in Postgres
SELECT data->>'name', data->'items'->0 FROM logs;
UPDATE logs SET data = jsonb_set(data, '{status}', '"active"');
SELECT jsonb_array_elements(data->'tags') FROM items;

-- Index
CREATE INDEX idx_orders_user_date ON orders(user_id, created_at DESC);
CREATE INDEX idx_users_email ON users(lower(email));`,
        tags: ["sql", "postgres", "database"],
      },
      {
        name: "Python Scripts",
        desc: "Handy Python one-liners and snippets",
        content: `# One-liners
python3 -c "print('hello')"
python3 -m http.server 8000
python3 -m json.tool data.json  # pretty print
python3 -c "import this"         # the zen

# Read/write
with open("file.txt") as f: print(f.read())
with open("out.txt", "w") as f: f.write("hi")

# HTTP server
python3 -m http.server 8080 --directory /tmp

# JSON
import json; data = json.load(open("f.json"))
print(json.dumps(data, indent=2, ensure_ascii=False))

# List tricks
[x for x in items if x['active']]
sum(1 for x in lst if condition)
list(set(items))  # dedupe

# Dict
{k: v for k, v in items.items() if v > 0}
{d.get('key', 'default') for d in list}`,
        tags: ["python", "scripting"],
      },
    ],
  },
  {
    category: "Security",
    icon: "🔒",
    items: [
      {
        name: "OpenSSL",
        desc: "SSL/TLS certificate operations",
        content: `# Check certificate
openssl s_client -connect google.com:443
openssl s_client -connect google.com:443 -servername google.com

# Generate key & CSR
openssl genrsa -out key.pem 2048
openssl req -new -key key.pem -out csr.pem

# Self-signed cert
openssl req -x509 -newkey rsa:4096 \\
  -keyout key.pem -out cert.pem \\
  -days 365 -subj "/CN=localhost"

# Convert formats
openssl x509 -in cert.pem -inform PEM -outform DER -out cert.der
openssl pkcs12 -export -in cert.pem -inkey key.pem -out bundle.p12

# Check expiry
openssl x509 -in cert.pem -noout -enddate

# Decode CSR
openssl req -in csr.pem -noout -text`,
        tags: ["ssl", "tls", "security"],
      },
      {
        name: "SSH",
        desc: "SSH config and tunnel commands",
        content: `# Quick connect
ssh user@host
ssh -i key.pem user@host
ssh -p 2222 user@host

# Tunnel
ssh -L 3000:localhost:3000 user@host
ssh -L 5432:localhost:5432 user@host
ssh -R 8080:localhost:80 user@host

# Config ~/.ssh/config
Host prod
  HostName example.com
  User admin
  Port 2222
  IdentityFile ~/.ssh/prod_key
  ProxyJump jump-host

# Copy files
scp file.txt user@host:/path/
scp -r ./dir user@host:/path/
rsync -avz -e "ssh -p 2222" ./dir user@host:/path/

# Keygen
ssh-keygen -t ed25519 -C "my key"
ssh-copy-id user@host`,
        tags: ["ssh", "remote", "security"],
      },
    ],
  },
];

function CodeBlock({ content }: { content: string }) {
  return (
    <pre className="text-xs leading-relaxed overflow-x-auto p-4 bg-[var(--color-bg)] rounded-xl border border-[var(--color-border)] font-mono text-[var(--color-text-2)]">
      {content}
    </pre>
  );
}

function ToolCard({ tool }: { tool: typeof tools[0]["items"][0] }) {
  return (
    <div className="card rounded-2xl p-6 flex flex-col gap-4">
      <div>
        <h3 className="font-bold text-base mb-1">{tool.name}</h3>
        <p className="text-xs text-[var(--color-text-2)]">{tool.desc}</p>
      </div>
      <CodeBlock content={tool.content} />
      <div className="flex flex-wrap gap-1.5">
        {tool.tags.map(t => (
          <span key={t} className="text-xs text-[var(--color-text-2)] opacity-60">#{t}</span>
        ))}
      </div>
    </div>
  );
}

export default function ToolsPage() {
  return (
    <main className="min-h-screen bg-[var(--color-bg)] text-[var(--color-text)]">
      <nav className="nav fixed top-0 left-0 right-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <a href="/" className="flex items-center gap-2">
            <span className="text-base">✦</span>
            <span className="font-semibold text-sm tracking-tight">biluo</span>
          </a>
        </div>
      </nav>

      <section className="pt-32 pb-16 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="mb-10 animate-fade-up">
            <h1 className="text-4xl font-black tracking-tight mb-3">Tools</h1>
            <p className="text-[var(--color-text-2)] text-base max-w-xl">
              Developer cheatsheets, CLI references, and useful patterns I keep coming back to.
              Copy, paste, adjust. Updated whenever I learn something new.
            </p>
          </div>

          <div className="space-y-14">
            {tools.map((section, si) => (
              <div key={section.category} className="animate-fade-up" style={{ animationDelay: `${si * 60}ms` }}>
                <div className="flex items-center gap-3 mb-5">
                  <span className="text-xl">{section.icon}</span>
                  <h2 className="text-xs uppercase tracking-widest text-[var(--color-text-2)] font-medium">{section.category}</h2>
                  <div className="h-px flex-1 bg-[var(--color-border)]" />
                  <span className="text-xs text-[var(--color-text-2)] opacity-50">{section.items.length} items</span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                  {section.items.map(tool => (
                    <ToolCard key={tool.name} tool={tool} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <footer className="footer mt-20 py-12 text-center px-6">
        <p className="text-xs text-[var(--color-text-2)] opacity-50">✦ biluo · biluonobug.github.io</p>
      </footer>
    </main>
  );
}