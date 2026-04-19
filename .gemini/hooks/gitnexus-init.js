#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

function main() {
  const cwd = process.cwd();
  const gitnexusDir = path.join(cwd, '.gitnexus');
  const metaPath = path.join(gitnexusDir, 'meta.json');

  if (!fs.existsSync(metaPath)) {
    // Silently skip if not indexed
    process.exit(0);
  }

  let meta;
  try {
    meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
  } catch (e) {
    process.exit(0);
  }

  const repoName = path.basename(cwd);
  const stats = meta.stats || {};
  const instruction = `
IMPORTANT: This project ('${repoName}') is indexed with GitNexus (${stats.nodes || 0} symbols, ${stats.processes || 0} flows).
You MUST prioritize GitNexus tools for all codebase intelligence tasks:
1. Use 'gitnexus_query' for exploration instead of generic search.
2. ALWAYS run 'gitnexus_impact' before modifying any function, class, or method.
3. Consult the 'gitnexus-guide' skill (via 'activate_skill') for tool reference and workflows.
4. If a tool warns the index is stale, run 'npx gitnexus analyze' in the terminal.
`.trim();

  const response = {
    hookSpecificOutput: {
      additionalContext: instruction
    }
  };

  console.log(JSON.stringify(response));
}

main();
