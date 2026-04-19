# GitNexus Hook Design

**Goal:** Automatically steer agents to use GitNexus tools first when a session starts.

## Architecture
We use a `SessionStart` hook in Gemini CLI. This hook runs once at the beginning of every chat session. It checks if the project is indexed by GitNexus and, if so, returns a directive that is injected into the conversation history.

## Components
- `.gemini/settings.json`: Configuration to register the hook.
- `.gemini/hooks/gitnexus-init.js`: Node.js script that generates the initialization context.

## Implementation Details
The `gitnexus-init.js` script will:
1. Verify `.gitnexus/meta.json` exists.
2. Read the repo name and basic stats (symbol count, etc.).
3. Return a JSON response with `hookSpecificOutput.additionalContext`.
4. The context will include mandatory instructions for the agent to prioritize GitNexus tools.

## Success Criteria
- Starting a new session in this repository results in the agent being instructed to use GitNexus.
- The instructions are visible in the first turn of the transcript.
