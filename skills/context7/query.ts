#!/usr/bin/env tsx
/**
 * Context7 Query CLI - No API Key Required
 *
 * Usage:
 *   npx tsx query.ts search <repo> <query>
 *   npx tsx query.ts context <repo_owner/repo_name> <search_query>
 */

const command = process.argv[2];
const repoName = process.argv[3];
const query = process.argv[4];

if (!command || command === "--help" || command === "-h") {
  console.log(`
Context7 Query CLI (no API key)

Usage:
  npx tsx query.ts search <repo> <query>
  npx tsx query.ts context <repo_owner/repo_name> <search_query>

Examples:
  npx tsx query.ts search "nextjs" "server components"
  npx tsx query.ts context "vercel/next.js" "app router"
  npx tsx query.ts context "reactjs/react.dev" "useState hook"
`);
  process.exit(0);
}

if (!repoName || !query) {
  console.error("Error: Missing arguments");
  process.exit(1);
}

// Library ID format
const libraryId = repoName.startsWith("/") ? repoName : `/${repoName}`;
const repoQuery = repoName.startsWith("/") ? repoName.slice(1) : repoName;

async function searchLibraries() {
  const url = new URL("https://context7.com/api/v2/libs/search");
  url.searchParams.set("libraryName", repoQuery);
  url.searchParams.set("query", query);

  const response = await fetch(url.toString());
  if (!response.ok) {
    const error = await response.text();
    console.error(`API error (${response.status}):`, error);
    process.exit(1);
  }

  const data = await response.json();
  const results = Array.isArray(data) ? data : data?.results || [];

  if (results.length === 0) {
    console.log("No results found.");
    return;
  }

  results.forEach((lib: any, i: number) => {
    console.log(`${i + 1}. ${lib.title || lib.id || lib.name}`);
    console.log(`   ID: ${lib.id}`);
    console.log(`   Trust: ${lib.trustScore || "?"} | Benchmark: ${lib.benchmarkScore || "?"}`);
    if (lib.versions?.length) {
      console.log(`   Versions: ${lib.versions.slice(0, 3).join(", ")}${lib.versions.length > 3 ? "..." : ""}`);
    }
    console.log("");
  });
}

async function getContext() {
  const url = new URL("https://context7.com/api/v2/context");
  url.searchParams.set("libraryId", libraryId);
  url.searchParams.set("query", query);
  url.searchParams.set("type", "txt");

  const response = await fetch(url.toString());
  if (!response.ok) {
    const error = await response.text();
    console.error(`API error (${response.status}):`, error);
    process.exit(1);
  }

  const text = await response.text();
  console.log(text);
}

if (command === "search" || command === "s") {
  searchLibraries();
} else if (command === "context" || command === "c") {
  getContext();
} else {
  console.error(`Unknown command: ${command}`);
  process.exit(1);
}