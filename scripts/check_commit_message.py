from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CONVENTIONAL_PATTERN = re.compile(
    r"^(?P<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(?:\((?P<scope>[a-z0-9][a-z0-9._/-]*)\))?"
    r"(?P<breaking>!)?: (?P<description>.+)$"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Conventional Commit messages for commit-msg hooks and CI."
    )
    parser.add_argument(
        "commit_msg_file",
        nargs="?",
        help="Path to the commit message file passed by the commit-msg hook.",
    )
    parser.add_argument(
        "--message",
        help="Explicit commit subject to validate. Used by CI when checking commit subjects.",
    )
    return parser


def _read_message(args: argparse.Namespace) -> str:
    if args.message:
        return args.message.strip()
    if args.commit_msg_file:
        content = Path(args.commit_msg_file).read_text(encoding="utf-8").splitlines()
        for line in content:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
    raise ValueError("No commit message provided.")


def validate_commit_message(message: str) -> tuple[bool, str]:
    if not message:
        return False, "Commit message cannot be empty."
    if len(message) > 72:
        return False, "Commit subject must be 72 characters or fewer."
    if not CONVENTIONAL_PATTERN.match(message):
        return (
            False,
            "Commit subject must follow Conventional Commits, e.g. 'feat(ci): add quality gates'.",
        )
    return True, ""


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        message = _read_message(args)
    except Exception as exc:
        print(f"[commit-message] {exc}", file=sys.stderr)
        return 1

    ok, error = validate_commit_message(message)
    if not ok:
        print(f"[commit-message] {error}", file=sys.stderr)
        print(f"[commit-message] Got: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
