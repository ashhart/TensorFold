"""lean_new.py BASE.txt HEAD.txt: fail when lean_check.py's output on a change has a problem its base does not have."""
import sys
from collections import Counter


def problems(path):
    """Counts of (file, message) in a lean_check.py output, so a change that only moves lines adds nothing."""
    out = Counter()
    for line in open(path, encoding="utf-8", errors="replace"):
        parts = line.rstrip("\n").split(":", 2)
        if len(parts) == 3 and parts[1].strip().isdigit():
            out[(parts[0], parts[2].strip())] += 1
        elif len(parts) == 2 and parts[1].strip().endswith("(split past 600)"):
            out[(parts[0], "over 600 lines")] += 1
    return out


def main():
    base, head = problems(sys.argv[1]), problems(sys.argv[2])
    new = {key: n - base[key] for key, n in head.items() if n > base[key]}
    for (path, message), n in sorted(new.items()):
        print(f"{path}: {message} ({n} new)")
    print(f"lean: {sum(head.values())} problems, {sum(base.values())} on the base, {sum(new.values())} new")
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())
