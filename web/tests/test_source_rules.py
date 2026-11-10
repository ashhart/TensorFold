import ast
import io
import re
import tokenize
from pathlib import Path


def test_web_sources_stay_small_with_single_line_comments():
    root = Path(__file__).parents[1]
    files = [root / "pyproject.toml"]
    for folder in ("tensorfold_web", "tests", "ui/src", "ui/scripts", "ui/tests"):
        files.extend(p for p in (root / folder).rglob("*") if p.suffix in {".py", ".ts", ".tsx", ".mjs", ".css"})
    for path in files:
        text = path.read_text()
        lines = text.splitlines()
        name = str(path.relative_to(root))
        assert len(lines) <= 600, name
        assert not text.endswith("\n\n"), name
        if path.suffix == ".py":
            comments = [t for t in tokenize.generate_tokens(io.StringIO(text).readline) if t.type == tokenize.COMMENT]
            for token in comments:
                assert len(lines[token.start[0]-1][:token.end[1]].encode()) <= 120, (name, token.start)
            full = {t.start[0] for t in comments if not lines[t.start[0]-1][:t.start[1]].strip()}
            for line in full:
                assert line + 1 not in full, (name, line)
            for node in ast.walk(ast.parse(text)):
                if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or not node.body:
                    continue
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                    assert first.lineno == first.end_lineno, (name, first.lineno)
        else:
            mark = r"^\s*#" if path.suffix == ".toml" else r"^\s*//"
            full = [i for i, line in enumerate(lines) if re.match(mark, line)]
            for i in full:
                assert i + 1 not in full, (name, i + 1)
                assert len(lines[i].encode()) <= 120, (name, i + 1)
            if path.suffix == ".css":
                assert "/*" not in text, name
