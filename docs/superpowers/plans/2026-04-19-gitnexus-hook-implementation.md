# GitNexus Hook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a `SessionStart` hook to inject GitNexus instructions at session start.

**Architecture:** A Node.js hook script registered in `.gemini/settings.json`.

**Tech Stack:** Node.js, Gemini CLI Hooks API.

---

### Task 1: Hook Script Implementation

**Files:**
- Create: `.gemini/hooks/gitnexus-init.js`

- [ ] **Step 1: Write the hook script**

```javascript
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
```

- [ ] **Step 2: Make the script executable**

Run: `chmod +x .gemini/hooks/gitnexus-init.js`

- [ ] **Step 3: Test the script manually**

Run: `./.gemini/hooks/gitnexus-init.js`
Expected: JSON output containing the instruction context.

- [ ] **Step 4: Commit**

```bash
git add .gemini/hooks/gitnexus-init.js
git commit -m "feat: add gitnexus initialization hook script"
```

### Task 2: Hook Registration

**Files:**
- Create: `.gemini/settings.json`

- [ ] **Step 1: Create or update settings.json**

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "*",
        "hooks": [
          {
            "name": "gitnexus-init",
            "type": "command",
            "command": "node .gemini/hooks/gitnexus-init.js"
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 2: Verify registration with /hooks panel (visual check)**

Run: `gemini` (in a new terminal) and type `/hooks panel`
Expected: `gitnexus-init` listed under `SessionStart`.

- [ ] **Step 3: Commit**

```bash
git add .gemini/settings.json
git commit -m "feat: register gitnexus session start hook"
```

### Task 3: Verification

- [ ] **Step 1: Start a new session and verify context**

1. Start `gemini`.
2. Check the very first message or transcript to ensure the GitNexus instructions are present.
3. Ask the agent: "What are your mandatory instructions for this project?"
Expected: The agent should mention GitNexus and the mandatory impact analysis.

- [ ] **Step 2: Commit the implementation plan**

```bash
git add docs/superpowers/plans/2026-04-19-gitnexus-hook-implementation.md
git commit -m "docs: add gitnexus hook implementation plan"
```
