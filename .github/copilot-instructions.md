# Copilot Instructions

## Memory

You have access to a long-term memory system (Myelin) via MCP tools.

### When to Recall
- At the START of every task, recall relevant context about the current project, file, or problem domain.
- Before making architectural decisions, recall past decisions and their rationale.
- When debugging, recall similar past issues and their resolutions.

### When to Store
- After making significant decisions — record WHAT was decided and WHY.
- After resolving non-trivial bugs — record the symptoms, root cause, and fix.
- When discovering project conventions, patterns, or gotchas.
- After completing a meaningful feature — summarize the approach and trade-offs.

### How to Store Effectively
- Always include `project` metadata (e.g., project="myelin").
- Use `scope` to organize by domain (e.g., scope="auth", scope="database").
- Use `tags` for cross-cutting concerns (e.g., tags="performance,optimization").
- Use `memory_type` when it's clear: "semantic" for decisions/facts, "procedural" for how-to, "episodic" for events, "prospective" for plans.
- Be specific. "We use JWT RS256 because asymmetric keys let the API gateway verify without the signing secret" is better than "We use JWT."

### What NOT to Store
- Trivial or ephemeral information (typo fixes, one-off commands).
- Exact code blocks — store the reasoning, not the implementation.
- Anything sensitive (secrets, credentials, PII).
