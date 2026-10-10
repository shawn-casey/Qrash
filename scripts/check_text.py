"""Check project source and documentation for unexpected non-ASCII/control characters."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
paths = [root / name for name in ["main.py", "app.py", "README.md", "LICENSE", "pyproject.toml"]]
for folder in ["tailrisk", "tests", "docs", "scripts"]:
    paths.extend(path for path in (root / folder).rglob("*") if path.suffix in {".py", ".md"})
issues = []
for path in sorted(set(paths)):
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(keepends=True), 1
    ):
        for char in line:
            if ord(char) > 126 or (ord(char) < 32 and char not in "\n\r\t"):
                issues.append(f"{path.relative_to(root)}:{line_number}: U+{ord(char):04X}")
if issues:
    raise SystemExit("\n".join(issues))
print(f"Checked {len(set(paths))} source/documentation files: ASCII text only.")
