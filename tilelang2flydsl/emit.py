"""Stage 3: ast.Module → Python source.

`ast.unparse` does almost everything we need. Optionally pipe through black for
predictable line lengths.
"""

from __future__ import annotations

import ast


def emit_source(mod: ast.Module, format_with_black: bool = True) -> str:
    ast.fix_missing_locations(mod)
    src = ast.unparse(mod)
    src = _restore_comment_strings(src)
    if format_with_black:
        try:
            import black

            src = black.format_str(src, mode=black.Mode(line_length=120))
        except Exception:
            # black is optional; skip silently if not installed
            pass
    if not src.endswith("\n"):
        src += "\n"
    return src


def _restore_comment_strings(src: str) -> str:
    """Convert bare string-literal lines emitted by `_comment(...)` back to real
    Python comments. Required because Python's AST has no comment node.

    Handles `'# foo'`, `"# foo"`, and `\"\"\"# foo\"\"\"` (when the string
    contains a quote that promoted the literal to triple-quoted).
    """
    out_lines: list[str] = []
    for line in src.splitlines():
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]
        for opener, closer in (("'''", "'''"), ('"""', '"""'), ("'", "'"), ('"', '"')):
            if (
                stripped.startswith(opener + "#")
                and stripped.endswith(closer)
                and len(stripped) >= len(opener) + len(closer) + 1
            ):
                inner = stripped[len(opener) : -len(closer)]
                out_lines.append(f"{indent}{inner}")
                break
        else:
            out_lines.append(line)
    return "\n".join(out_lines)
