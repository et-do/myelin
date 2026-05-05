"""Tests for myelin.integrations.github.GitHubImporter."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from myelin.integrations.github import (
    GitHubImporter,
    _gh_available,
    _run,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _init_repo(path: Path) -> None:
    """Create a minimal local git repo with one commit."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    (path / "README.md").write_text("# test repo\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit\n\nAdded README"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def _add_commit(path: Path, filename: str, content: str, message: str) -> str:
    """Add a file, commit, and return the full SHA."""
    (path / filename).write_text(content)
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=path,
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# _run helper
# ---------------------------------------------------------------------------


def test_run_returns_stdout():
    out = _run(["echo", "hello"])
    assert "hello" in out


def test_run_raises_on_nonzero():
    with pytest.raises(RuntimeError, match="failed"):
        _run(["false"])


# ---------------------------------------------------------------------------
# _gh_available
# ---------------------------------------------------------------------------


def test_gh_available_returns_bool():
    result = _gh_available()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# GitHubImporter — not a git repo
# ---------------------------------------------------------------------------


def test_import_raises_for_non_repo(tmp_path: Path):
    importer = GitHubImporter()
    with pytest.raises(ValueError, match="not a git repository"):
        importer.import_(tmp_path)


# ---------------------------------------------------------------------------
# GitHubImporter — commits from real local repo
# ---------------------------------------------------------------------------


def test_import_commits_basic(tmp_path: Path):
    _init_repo(tmp_path)
    importer = GitHubImporter()
    pairs = importer.import_(tmp_path, include=["commits"])
    assert len(pairs) == 1
    content, meta = pairs[0]
    assert "Initial commit" in content
    assert meta["memory_type"] == "episodic"
    assert "git" in meta["tags"]
    assert "commit" in meta["tags"]


def test_import_commits_multiple(tmp_path: Path):
    _init_repo(tmp_path)
    _add_commit(tmp_path, "a.py", "x = 1", "Add a.py")
    _add_commit(tmp_path, "b.py", "y = 2", "Add b.py")
    importer = GitHubImporter()
    pairs = importer.import_(tmp_path, include=["commits"])
    assert len(pairs) == 3  # Initial + 2 more


def test_import_commits_content_includes_files(tmp_path: Path):
    _init_repo(tmp_path)
    sha = _add_commit(tmp_path, "feature.py", "pass", "Add feature module")
    importer = GitHubImporter()
    pairs = importer.import_(tmp_path, include=["commits"])
    # Find the feature commit
    feature_pair = next((p for p in pairs if "Add feature module" in p[0]), None)
    assert feature_pair is not None
    content, meta = feature_pair
    assert "feature.py" in content
    assert f"git:{sha[:8]}" == meta["source"]


def test_import_commits_since_filter(tmp_path: Path):
    _init_repo(tmp_path)
    _add_commit(tmp_path, "new.py", "pass", "New file")
    importer = GitHubImporter()
    # Using a far-future date so nothing matches
    pairs = importer.import_(tmp_path, include=["commits"], since="2099-01-01")
    assert pairs == []


def test_import_commits_only_shas_filter(tmp_path: Path):
    _init_repo(tmp_path)
    sha = _add_commit(tmp_path, "x.py", "pass", "Specific commit")
    importer = GitHubImporter()
    # Only import the specific commit by its full SHA
    pairs = importer.import_(tmp_path, include=["commits"], only_shas=frozenset([sha]))
    assert len(pairs) == 1
    assert "Specific commit" in pairs[0][0]


def test_import_commits_only_shas_empty_set(tmp_path: Path):
    _init_repo(tmp_path)
    _add_commit(tmp_path, "x.py", "pass", "A commit")
    importer = GitHubImporter()
    # Empty set — nothing should be imported
    pairs = importer.import_(tmp_path, include=["commits"], only_shas=frozenset())
    assert pairs == []


def test_import_commits_sets_project_from_arg(tmp_path: Path):
    _init_repo(tmp_path)
    importer = GitHubImporter()
    pairs = importer.import_(tmp_path, include=["commits"], project="myproject")
    assert all(meta["project"] == "myproject" for _, meta in pairs)


def test_import_commits_sets_scope(tmp_path: Path):
    _init_repo(tmp_path)
    importer = GitHubImporter()
    pairs = importer.import_(tmp_path, include=["commits"], scope="backend")
    assert all(meta["scope"] == "backend" for _, meta in pairs)


def test_import_commits_default_project_is_repo_dirname(tmp_path: Path):
    repo = tmp_path / "my-project"
    _init_repo(repo)
    importer = GitHubImporter()
    pairs = importer.import_(repo, include=["commits"])
    assert all(meta["project"] == "my-project" for _, meta in pairs)


# ---------------------------------------------------------------------------
# GitHubImporter — PRs and issues (mocked gh CLI)
# ---------------------------------------------------------------------------


_FAKE_PRS = [
    {
        "number": 42,
        "title": "Add dark mode",
        "body": "Implements dark mode toggle",
        "labels": [{"name": "enhancement"}],
        "author": {"login": "alice"},
        "createdAt": "2025-01-15T10:00:00Z",
        "mergedAt": "2025-01-20T12:00:00Z",
        "state": "MERGED",
        "baseRefName": "main",
    }
]

_FAKE_ISSUES = [
    {
        "number": 7,
        "title": "Memory leak in hippocampus",
        "body": "The hippocampus leaks memory under high load",
        "labels": [{"name": "bug"}],
        "author": {"login": "bob"},
        "createdAt": "2025-02-01T09:00:00Z",
        "closedAt": "2025-02-10T14:00:00Z",
        "state": "CLOSED",
    }
]


def test_import_prs_requires_gh(tmp_path: Path):
    _init_repo(tmp_path)
    importer = GitHubImporter()
    with patch("myelin.integrations.github._gh_available", return_value=False):
        with pytest.raises(RuntimeError, match="gh CLI is required"):
            importer.import_(tmp_path, include=["prs"])


def test_import_issues_requires_gh(tmp_path: Path):
    _init_repo(tmp_path)
    importer = GitHubImporter()
    with patch("myelin.integrations.github._gh_available", return_value=False):
        with pytest.raises(RuntimeError, match="gh CLI is required"):
            importer.import_(tmp_path, include=["issues"])


def test_import_prs_basic(tmp_path: Path):
    import json

    _init_repo(tmp_path)
    importer = GitHubImporter()
    with (
        patch("myelin.integrations.github._gh_available", return_value=True),
        patch("myelin.integrations.github._run", return_value=json.dumps(_FAKE_PRS)),
    ):
        pairs = importer.import_(tmp_path, include=["prs"])

    assert len(pairs) == 1
    content, meta = pairs[0]
    assert "PR #42" in content
    assert "Add dark mode" in content
    assert meta["memory_type"] == "semantic"
    assert meta["source"] == "github:pr:42"
    assert "pr" in meta["tags"]


def test_import_issues_basic(tmp_path: Path):
    import json

    _init_repo(tmp_path)
    importer = GitHubImporter()
    with (
        patch("myelin.integrations.github._gh_available", return_value=True),
        patch("myelin.integrations.github._run", return_value=json.dumps(_FAKE_ISSUES)),
    ):
        pairs = importer.import_(tmp_path, include=["issues"])

    assert len(pairs) == 1
    content, meta = pairs[0]
    assert "Issue #7" in content
    assert "Memory leak" in content
    assert meta["memory_type"] == "semantic"
    assert meta["source"] == "github:issue:7"
    assert "issue" in meta["tags"]


def test_import_prs_only_ids_filter(tmp_path: Path):
    import json

    _init_repo(tmp_path)
    importer = GitHubImporter()
    with (
        patch("myelin.integrations.github._gh_available", return_value=True),
        patch("myelin.integrations.github._run", return_value=json.dumps(_FAKE_PRS)),
    ):
        # Filter to a different PR number — nothing should match
        pairs = importer.import_(
            tmp_path, include=["prs"], only_ids=frozenset(["pr:99"])
        )
    assert pairs == []


def test_import_combined(tmp_path: Path):
    import json

    from myelin.integrations.github import _run as real_run

    _init_repo(tmp_path)
    importer = GitHubImporter()

    def fake_run(cmd: list[str], **kw: object) -> str:
        if cmd[0] == "gh" and "pr" in cmd:
            return json.dumps(_FAKE_PRS)
        if cmd[0] == "gh" and "issue" in cmd:
            return json.dumps(_FAKE_ISSUES)
        # Delegate real git commands to the actual implementation
        return real_run(cmd, **kw)

    with (
        patch("myelin.integrations.github._gh_available", return_value=True),
        patch("myelin.integrations.github._run", side_effect=fake_run),
    ):
        pairs = importer.import_(tmp_path, include=["commits", "prs", "issues"])

    types = {meta["memory_type"] for _, meta in pairs}
    assert "episodic" in types  # commits
    assert "semantic" in types  # prs + issues


# ---------------------------------------------------------------------------
# _detect_repo_name helper (exposed for testing)
# ---------------------------------------------------------------------------


def test_detect_repo_name_from_remote_https(tmp_path: Path):
    _init_repo(tmp_path)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/alice/my-repo.git"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    importer = GitHubImporter()
    name = importer._detect_repo_name(tmp_path)
    assert name == "alice/my-repo"


def test_detect_repo_name_from_remote_ssh(tmp_path: Path):
    _init_repo(tmp_path)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:alice/my-repo.git"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    importer = GitHubImporter()
    name = importer._detect_repo_name(tmp_path)
    assert name == "alice/my-repo"


def test_detect_repo_name_fallback_to_dirname(tmp_path: Path):
    repo = tmp_path / "cool-project"
    _init_repo(repo)
    importer = GitHubImporter()
    # No remote set — should fall back to directory name
    name = importer._detect_repo_name(repo)
    assert name == "cool-project"
