"""GitHub / Git integration — import commit, PR, and issue history as memories.

Reads from a local git repository (via ``git log``) and optionally from the
GitHub API (via the ``gh`` CLI) to ingest commit messages, PR descriptions,
and issue discussions as structured memories.

Each item is converted to a ``(content, metadata)`` pair:

* **Commits** → ``memory_type="episodic"``  (what happened and when)
* **Pull requests** → ``memory_type="semantic"``  (design decisions)
* **Issues** → ``memory_type="semantic"``  (problems / requirements)

Incremental imports are handled via :class:`~myelin.integrations.sync.SyncRegistry`
using commit SHAs / PR/issue numbers as stable identifiers.

CLI
---
::

    myelin github-import <repo>
        [--since DATE]          # e.g. "2025-01-01" or "6 months ago"
        [--branch BRANCH]       # default: current branch
        [--include commits,prs,issues]
        [--project NAME]
        [--scope NAME]
        [--incremental]

Local commits need no credentials.  PR/issue import requires ``gh`` CLI to be
authenticated (``gh auth login``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .base import Importer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GIT_SEP = "\x1e"  # ASCII Record Separator — commits delimited
_GIT_FS = "\x1f"  # ASCII Unit Separator — fields within a commit

# git log format: SHA FS author_name FS author_email FS iso_date FS subject FS body SEP
_GIT_FORMAT = f"%H{_GIT_FS}%an{_GIT_FS}%ae{_GIT_FS}%ai{_GIT_FS}%s{_GIT_FS}%b{_GIT_SEP}"


def _run(cmd: list[str], **kwargs: Any) -> str:
    """Run *cmd* and return stdout, raising RuntimeError on failure."""
    result: subprocess.CompletedProcess[str] = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        **kwargs,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command {cmd[0]!r} failed"
            f" (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def _gh_available() -> bool:
    return shutil.which("gh") is not None


def _files_changed(repo: Path, sha: str) -> list[str]:
    """Return file paths touched by commit *sha*."""
    out = _run(
        [
            "git",
            "-C",
            str(repo),
            "diff-tree",
            "--no-commit-id",
            "-r",
            "--name-only",
            sha,
        ]
    )
    return [line for line in out.splitlines() if line]


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------


class GitHubImporter(Importer):
    """Import git/GitHub history as myelin memories.

    Parameters passed via ``**opts`` to :meth:`import_`:

    since : str, optional
        Limit to items after this date (git ``--since`` format, e.g.
        ``"2025-01-01"`` or ``"6 months ago"``).
    branch : str, optional
        Branch to walk for commits.  Defaults to the repo's current branch.
    include : list[str], optional
        Which item types to import.  Subset of ``["commits", "prs", "issues"]``.
        Defaults to ``["commits"]``.
    project : str, optional
        Project label to attach to all imported memories.
    scope : str, optional
        Scope label to attach to all imported memories.
    only_shas : frozenset[str], optional
        If provided, only import commits whose full SHA is in this set
        (used by ``--incremental`` in the CLI).
    only_ids : frozenset[str], optional
        If provided, only import PRs/issues whose ``"<type>:<number>"`` key
        is in this set (used by ``--incremental``).
    repo_name : str, optional
        Human-readable repo name for metadata (auto-detected from remote if
        omitted).
    """

    def import_(
        self,
        src: Path,
        **opts: Any,
    ) -> list[tuple[str, dict[str, str]]]:
        repo = Path(src).resolve()
        if not (repo / ".git").exists():
            raise ValueError(
                f"{repo} is not a git repository (no .git directory found)"
            )

        include: list[str] = opts.get("include", ["commits"])
        since: str | None = opts.get("since")
        branch: str | None = opts.get("branch")
        project: str = opts.get("project", "")
        scope: str = opts.get("scope", "")
        only_shas: frozenset[str] | None = opts.get("only_shas")
        only_ids: frozenset[str] | None = opts.get("only_ids")
        repo_name: str = opts.get("repo_name", "") or self._detect_repo_name(repo)

        pairs: list[tuple[str, dict[str, str]]] = []

        if "commits" in include:
            pairs.extend(
                self._import_commits(
                    repo, since, branch, project, scope, repo_name, only_shas
                )
            )
        if "prs" in include:
            pairs.extend(
                self._import_prs(repo, since, project, scope, repo_name, only_ids)
            )
        if "issues" in include:
            pairs.extend(
                self._import_issues(repo, since, project, scope, repo_name, only_ids)
            )

        return pairs

    # ---------------------------------------------------------------- commits

    def _import_commits(
        self,
        repo: Path,
        since: str | None,
        branch: str | None,
        project: str,
        scope: str,
        repo_name: str,
        only_shas: frozenset[str] | None,
    ) -> list[tuple[str, dict[str, str]]]:
        cmd = [
            "git",
            "-C",
            str(repo),
            "log",
            f"--format=tformat:{_GIT_FORMAT}",
            "--no-merges",
        ]
        if since:
            cmd.extend(["--since", since])
        if branch:
            cmd.append(branch)

        raw = _run(cmd)
        records = [r for r in raw.split(_GIT_SEP) if r.strip()]

        pairs: list[tuple[str, dict[str, str]]] = []
        for record in records:
            fields = record.strip().split(_GIT_FS, 5)
            if len(fields) < 5:
                continue
            sha, author_name, author_email, date_iso, subject = fields[:5]
            body = fields[5].strip() if len(fields) > 5 else ""
            sha = sha.strip()

            if not sha or not subject.strip():
                continue
            if only_shas is not None and sha not in only_shas:
                continue

            files = _files_changed(repo, sha)
            file_summary = (
                ", ".join(files[:5])
                + (f" (+{len(files) - 5} more)" if len(files) > 5 else "")
                if files
                else ""
            )

            lines = [f"commit {sha[:8]} — {subject.strip()}"]
            if body:
                lines.append("")
                lines.append(body)
            lines.append("")
            lines.append(f"Author: {author_name} <{author_email}>")
            lines.append(f"Date:   {date_iso.strip()}")
            if repo_name:
                lines.append(f"Repo:   {repo_name}")
            if file_summary:
                lines.append(f"Files:  {file_summary}")
            content = "\n".join(lines)

            tags_parts = ["git", "commit"]
            if repo_name:
                tags_parts.append(repo_name.replace("/", "-"))

            pairs.append(
                (
                    content,
                    {
                        "memory_type": "episodic",
                        "source": f"git:{sha[:8]}",
                        "project": project or repo_name.split("/")[-1],
                        "scope": scope,
                        "tags": ",".join(tags_parts),
                        "created_at": date_iso.strip(),
                    },
                )
            )
        return pairs

    # ---------------------------------------------------------------- PRs

    def _import_prs(
        self,
        repo: Path,
        since: str | None,
        project: str,
        scope: str,
        repo_name: str,
        only_ids: frozenset[str] | None,
    ) -> list[tuple[str, dict[str, str]]]:
        if not _gh_available():
            raise RuntimeError(
                "gh CLI is required for PR import but was not found on PATH. "
                "Install from https://cli.github.com/ and run 'gh auth login'."
            )
        cmd = [
            "gh",
            "pr",
            "list",
            "--state",
            "all",
            "--json",
            "number,title,body,labels,author,createdAt,mergedAt,state,baseRefName",
            "--limit",
            "500",
        ]
        if repo_name:
            cmd.extend(["--repo", repo_name])

        raw = _run(cmd, cwd=str(repo))
        items = json.loads(raw)

        pairs: list[tuple[str, dict[str, str]]] = []
        for pr in items:
            key = f"pr:{pr['number']}"
            if only_ids is not None and key not in only_ids:
                continue

            created = pr.get("createdAt", "")
            if since and created and created < since:
                continue

            labels = ", ".join(lbl["name"] for lbl in pr.get("labels", []))
            lines = [f"PR #{pr['number']}: {pr['title']}"]
            lines.append(f"State:  {pr.get('state', '')}")
            lines.append(f"Author: {pr.get('author', {}).get('login', '')}")
            lines.append(f"Base:   {pr.get('baseRefName', '')}")
            if labels:
                lines.append(f"Labels: {labels}")
            lines.append(f"Opened: {created}")
            if pr.get("mergedAt"):
                lines.append(f"Merged: {pr['mergedAt']}")
            body = (pr.get("body") or "").strip()
            if body:
                lines.append("")
                lines.append(body)
            content = "\n".join(lines)

            tags_parts = ["git", "pr"]
            if repo_name:
                tags_parts.append(repo_name.replace("/", "-"))
            if labels:
                tags_parts.extend(lbl["name"] for lbl in pr.get("labels", []))

            pairs.append(
                (
                    content,
                    {
                        "memory_type": "semantic",
                        "source": f"github:pr:{pr['number']}",
                        "project": project or repo_name.split("/")[-1],
                        "scope": scope,
                        "tags": ",".join(tags_parts),
                        "created_at": created,
                    },
                )
            )
        return pairs

    # ---------------------------------------------------------------- issues

    def _import_issues(
        self,
        repo: Path,
        since: str | None,
        project: str,
        scope: str,
        repo_name: str,
        only_ids: frozenset[str] | None,
    ) -> list[tuple[str, dict[str, str]]]:
        if not _gh_available():
            raise RuntimeError(
                "gh CLI is required for issue import but was not found on PATH. "
                "Install from https://cli.github.com/ and run 'gh auth login'."
            )
        cmd = [
            "gh",
            "issue",
            "list",
            "--state",
            "all",
            "--json",
            "number,title,body,labels,author,createdAt,closedAt,state",
            "--limit",
            "500",
        ]
        if repo_name:
            cmd.extend(["--repo", repo_name])

        raw = _run(cmd, cwd=str(repo))
        items = json.loads(raw)

        pairs: list[tuple[str, dict[str, str]]] = []
        for issue in items:
            key = f"issue:{issue['number']}"
            if only_ids is not None and key not in only_ids:
                continue

            created = issue.get("createdAt", "")
            if since and created and created < since:
                continue

            labels = ", ".join(lbl["name"] for lbl in issue.get("labels", []))
            lines = [f"Issue #{issue['number']}: {issue['title']}"]
            lines.append(f"State:  {issue.get('state', '')}")
            lines.append(f"Author: {issue.get('author', {}).get('login', '')}")
            if labels:
                lines.append(f"Labels: {labels}")
            lines.append(f"Opened: {created}")
            if issue.get("closedAt"):
                lines.append(f"Closed: {issue['closedAt']}")
            body = (issue.get("body") or "").strip()
            if body:
                lines.append("")
                lines.append(body)
            content = "\n".join(lines)

            tags_parts = ["git", "issue"]
            if repo_name:
                tags_parts.append(repo_name.replace("/", "-"))
            if labels:
                tags_parts.extend(lbl["name"] for lbl in issue.get("labels", []))

            pairs.append(
                (
                    content,
                    {
                        "memory_type": "semantic",
                        "source": f"github:issue:{issue['number']}",
                        "project": project or repo_name.split("/")[-1],
                        "scope": scope,
                        "tags": ",".join(tags_parts),
                        "created_at": created,
                    },
                )
            )
        return pairs

    # ---------------------------------------------------------------- helpers

    def _detect_repo_name(self, repo: Path) -> str:
        """Infer ``owner/name`` from the git remote URL, falling back to dir name."""
        try:
            remote_url = _run(
                ["git", "-C", str(repo), "remote", "get-url", "origin"]
            ).strip()
            # github.com/owner/repo.git or git@github.com:owner/repo.git
            for prefix in ("https://github.com/", "git@github.com:"):
                if remote_url.startswith(prefix):
                    return remote_url[len(prefix) :].removesuffix(".git")
        except RuntimeError:
            pass
        return repo.name
