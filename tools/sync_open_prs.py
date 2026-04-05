#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_DIR = ROOT
DEFAULT_OUT_DIR = ROOT / "research" / "competition_prs" / "open_current"
API_ROOT = "https://api.github.com"
USER_AGENT = "parameter-golf-pr-sync"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "pr"


def request_text(url: str, token: str | None, accept: str = "application/vnd.github+json") -> str:
    req = Request(url, headers={"Accept": accept, "User-Agent": USER_AGENT})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urlopen(req, timeout=60) as response:
        return response.read().decode("utf-8")


def request_json(url: str, token: str | None) -> list[dict] | dict:
    return json.loads(request_text(url, token))


def ensure_base_ref(repo_dir: Path, base_ref: str) -> str:
    remote_ref = f"refs/remotes/origin/{base_ref}"
    run_git(repo_dir, "fetch", "origin", base_ref)
    return remote_ref


def local_patch_text(repo_dir: Path, base_ref: str, local_ref: str) -> str:
    remote_ref = ensure_base_ref(repo_dir, base_ref)
    merge_base = run_git(repo_dir, "merge-base", remote_ref, local_ref, capture_output=True).stdout.strip()
    return run_git(repo_dir, "diff", "--patch", merge_base, local_ref, capture_output=True).stdout


def list_open_prs(repo: str, token: str | None) -> list[dict]:
    pulls: list[dict] = []
    page = 1
    while True:
        url = f"{API_ROOT}/repos/{repo}/pulls?state=open&per_page=100&page={page}"
        data = request_json(url, token)
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected GitHub API response for {url}: {type(data)!r}")
        if not data:
            return pulls
        pulls.extend(data)
        page += 1


def run_git(repo_dir: Path, *args: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
    )


def fetch_pr_ref(repo_dir: Path, pr_number: int) -> tuple[str, str]:
    local_ref = f"refs/pr-cache/open/{pr_number}"
    run_git(repo_dir, "fetch", "origin", f"pull/{pr_number}/head:{local_ref}")
    sha = run_git(repo_dir, "rev-parse", local_ref, capture_output=True).stdout.strip()
    return local_ref, sha


def extract_ref(repo_dir: Path, refname: str, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["git", "-C", str(repo_dir), "archive", "--format=tar", refname],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
            archive.extractall(dest)
    finally:
        proc.stdout.close()
    stderr = proc.stderr.read().decode("utf-8") if proc.stderr is not None else ""
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"git archive failed for {refname}: {stderr.strip()}")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def summarize_pr(pr: dict, sha: str, local_ref: str, dirname: str) -> dict:
    head_repo = pr.get("head", {}).get("repo") or {}
    base_repo = pr.get("base", {}).get("repo") or {}
    return {
        "number": pr["number"],
        "title": pr["title"],
        "draft": pr["draft"],
        "html_url": pr["html_url"],
        "user": (pr.get("user") or {}).get("login"),
        "created_at": pr.get("created_at"),
        "updated_at": pr.get("updated_at"),
        "base_ref": pr.get("base", {}).get("ref"),
        "base_repo": base_repo.get("full_name"),
        "head_ref": pr.get("head", {}).get("ref"),
        "head_repo": head_repo.get("full_name"),
        "head_sha": sha,
        "local_git_ref": local_ref,
        "directory": dirname,
        "commits": pr.get("commits"),
        "additions": pr.get("additions"),
        "deletions": pr.get("deletions"),
        "changed_files": pr.get("changed_files"),
    }


def write_pr_readme(pr_dir: Path, meta: dict) -> None:
    body = [
        f"# PR #{meta['number']}: {meta['title']}",
        "",
        f"- Author: `{meta['user']}`",
        f"- Draft: `{meta['draft']}`",
        f"- URL: {meta['html_url']}",
        f"- Base: `{meta['base_repo']}:{meta['base_ref']}`",
        f"- Head: `{meta['head_repo']}:{meta['head_ref']}`",
        f"- Head SHA: `{meta['head_sha']}`",
        f"- Local ref: `{meta['local_git_ref']}`",
        f"- Updated at: `{meta['updated_at']}`",
        "",
        "Files:",
        f"- `metadata.json`",
        f"- `changes.patch`",
        f"- `source/`",
    ]
    write_text(pr_dir / "README.md", "\n".join(body) + "\n")


def build_index(out_dir: Path, repo: str, pr_summaries: list[dict]) -> None:
    manifest = {
        "repo": repo,
        "synced_at_utc": now_utc(),
        "open_pr_count": len(pr_summaries),
        "prs": sorted(pr_summaries, key=lambda pr: pr["number"], reverse=True),
    }
    write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    lines = [
        f"# Open PR Archive for `{repo}`",
        "",
        f"- Synced at: `{manifest['synced_at_utc']}`",
        f"- Open PR count: `{manifest['open_pr_count']}`",
        "",
        "| PR | Title | Author | Updated | Folder |",
        "|---:|---|---|---|---|",
    ]
    for pr in manifest["prs"]:
        lines.append(
            f"| {pr['number']} | {pr['title'].replace('|', '\\|')} | "
            f"{pr['user']} | {pr['updated_at']} | `{pr['directory']}` |"
        )
    write_text(out_dir / "README.md", "\n".join(lines) + "\n")


def sync_open_prs(repo_dir: Path, repo: str, out_dir: Path, token: str | None, prune: bool) -> None:
    pulls = list_open_prs(repo, token)
    out_dir.mkdir(parents=True, exist_ok=True)

    active_dirs: set[str] = set()
    pr_summaries: list[dict] = []

    for pr in pulls:
        pr_number = int(pr["number"])
        dirname = f"pr_{pr_number:03d}_{slugify(pr['title'])[:80]}"
        active_dirs.add(dirname)
        pr_dir = out_dir / dirname
        pr_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{pr_number}] syncing {pr['title']}")

        local_ref, sha = fetch_pr_ref(repo_dir, pr_number)
        patch_text = local_patch_text(repo_dir, pr["base"]["ref"], local_ref)
        meta = summarize_pr(pr, sha, local_ref, dirname)

        write_text(pr_dir / "metadata.json", json.dumps(meta, indent=2, sort_keys=True) + "\n")
        write_text(pr_dir / "changes.patch", patch_text)
        extract_ref(repo_dir, local_ref, pr_dir / "source")
        write_pr_readme(pr_dir, meta)
        pr_summaries.append(meta)

    if prune:
        for child in out_dir.iterdir():
            if child.name in {"README.md", "manifest.json"}:
                continue
            if child.is_dir() and child.name not in active_dirs:
                shutil.rmtree(child)

    build_index(out_dir, repo, pr_summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync all open PRs from a GitHub repo into a local folder.")
    parser.add_argument("--repo", default="openai/parameter-golf", help="GitHub repo in owner/name form")
    parser.add_argument("--repo-dir", default=str(DEFAULT_REPO_DIR), help="Local git clone used to fetch PR refs")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Where to write the local PR archive")
    parser.add_argument("--no-prune", action="store_true", help="Keep local folders for PRs that are no longer open")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    repo_dir = Path(args.repo_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    sync_open_prs(repo_dir=repo_dir, repo=args.repo, out_dir=out_dir, token=token, prune=not args.no_prune)
    print(f"Synced open PRs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
