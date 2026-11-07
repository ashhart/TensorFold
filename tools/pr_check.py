#!/usr/bin/env python3
"""Check local contributor changes against their merge base before pushing; never fetch or publish."""
import argparse
from collections import Counter
import ipaddress
from pathlib import Path
import re
import runpy
import subprocess
import sys

SOURCE_SUFFIXES = {'.zig', '.py', '.c', '.h', '.cpp', '.hpp', '.cu', '.cuh', '.hip', '.cl', '.metal',
                   '.sh', '.js', '.jsx', '.ts', '.tsx', '.swift', '.m', '.mm', '.rs', '.go', '.java', '.kt'}
HOME_PATH = re.compile(r'(?:/(?:Users|home)/[^/\s]+|/root(?:/|$)|[A-Za-z]:[\\/]Users[\\/][^\\/\s]+)')
PATCH_PATH = re.compile(r'\.\./[^\s\"\'<>]*\.patch\b')
IPV4 = re.compile(r'(?<![\d.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![\d.])')


def git(root, *args, data=None):
    """Use local Git objects only, with neither external diff helpers nor text conversion."""
    return subprocess.run(['git', '-C', str(root), *args], input=data, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True).stdout


def merge_base(root, requested=None):
    """Pin the comparison to the common ancestor, not the other branch's latest tree."""
    if requested is None:
        for candidate in ('refs/remotes/origin/main', 'refs/heads/main'):
            try:
                git(root, 'rev-parse', '--verify', candidate)
                requested = candidate
                break
            except subprocess.CalledProcessError:
                pass
    if requested is None:
        raise ValueError('No local main ref; pass --base with the intended local base revision')
    return git(root, 'merge-base', '--', requested, 'HEAD').decode().strip()


def added_lines(diff):
    """Yield added line numbers and text from one file's zero-context Git diff."""
    number = None
    for line in diff.splitlines():
        if line.startswith('@@ '):
            match = re.search(r'\+(\d+)(?:,\d+)? @@', line)
            number = int(match[1]) if match else None
        elif number is not None:
            if line.startswith('+'):
                yield number, line[1:]
                number += 1
            elif line.startswith(' '):
                number += 1
            elif not line.startswith(('-', '\\')):
                number = None


def private_additions(path, lines):
    """Report categories and locations without echoing the private-looking value."""
    found = []
    for number, line in lines:
        for pattern, message in ((HOME_PATH, 'home-directory path'), (PATCH_PATH, 'parent-relative patch path')):
            if pattern.search(line):
                found.append(('WARN', f'{path}:{number}: added {message}'))
        for match in IPV4.finditer(line):
            try:
                address = ipaddress.IPv4Address(match[0])
            except ipaddress.AddressValueError:
                continue
            if str(address) != '127.0.0.1':
                found.append(('WARN', f'{path}:{number}: added non-loopback IPv4 address'))
    return found


def change_checks(base_files, current, changes, additions, pr=None):
    """Check touched source and advisory text/documentation findings."""
    found = []
    for name in sorted(changes):
        if name in current and Path(name).suffix in SOURCE_SUFFIXES and len(current[name].splitlines()) > 600:
            found.append(('ERROR', f'{name}: touched source exceeds 600 lines'))
        found.extend(private_additions(name, additions.get(name, [])))
        if name not in base_files and '/' not in name and Path(name).suffix.lower() == '.md':
            found.append(('WARN', f'{name}: new top-level Markdown file; prefer the existing documentation tree'))
    if any(name.startswith('zig/') for name in changes):
        changelog = current.get('CHANGELOG.md', '')
        unreleased = re.search(r'^## Unreleased[^\n]*\n(.*?)(?=^## |\Z)', changelog, re.M | re.S)
        lines = unreleased[1].splitlines() if unreleased else []
        pattern = re.compile(r'(?<!\w)#' + re.escape(str(pr)) + r'\b') if pr else None
        if pattern is None:
            found.append(('WARN', 'Zig changed: pass --pr NUMBER to check its Unreleased line'))
        elif not any(line.lstrip().startswith('- ') and pattern.search(line) for line in lines):
            found.append(('WARN', 'Zig changed: no Unreleased bullet naming this PR'))
    return found


class TextFile:
    """Give the existing lean checker file text without extracting or changing the checkout."""
    def __init__(self, name, text):
        self.path, self.text = Path(name), text
        self.name, self.suffix = self.path.name, self.path.suffix

    def relative_to(self, _):
        return self.path

    def read_text(self, **_):
        return self.text


def lean_counts(files, checker):
    """Use the same current checker on both trees, ignoring finding line-number movement."""
    result = Counter()
    for name, text in files.items():
        if not name.startswith(('zig/', 'tools/zig/')) or Path(name).suffix not in checker['COMMENT']:
            continue
        for finding in checker['problems'](TextFile(name, text)):
            parts = finding.split(':', 2)
            reason = parts[2].strip() if len(parts) == 3 and parts[1].strip().isdigit() else 'over 600 lines'
            result[(name, reason)] += 1
    return result


def new_lean(base, head):
    """A moved inherited finding stays inherited; duplicate new findings still count."""
    return [('ERROR', f'{name}: {reason} ({count} new lean finding(s))')
            for (name, reason), count in sorted((head - base).items())]


def base_text(root, revision):
    """Read local blobs in one batch, avoiding one subprocess per source file."""
    entries = git(root, 'ls-tree', '-rz', revision).split(b'\0')
    names, ids = [], []
    for entry in entries:
        if not entry:
            continue
        metadata, raw = entry.split(b'\t', 1)
        mode, kind, oid = metadata.split()
        name = raw.decode('utf-8', 'surrogateescape')
        if kind == b'blob' and (name == 'CHANGELOG.md' or Path(name).suffix in SOURCE_SUFFIXES or ('/' not in name and Path(name).suffix.lower() == '.md')):
            names.append(name)
            ids.append(oid)
    if not ids:
        return {}
    output = git(root, 'cat-file', '--batch', data=b'\n'.join(ids) + b'\n')
    files, at = {}, 0
    for name in names:
        end = output.index(b'\n', at)
        header = output[at:end].split()
        if len(header) != 3 or header[1] != b'blob':
            raise ValueError('A local baseline blob could not be read')
        size = int(header[2])
        at = end + 1
        files[name] = output[at:at + size].decode('utf-8', 'replace')
        at += size + 1
    return files


def worktree_text(root, needed=()):
    """Include staged, unstaged and untracked files; never read through a symlink."""
    names = set(git(root, 'ls-files', '-z', '--cached', '--others', '--exclude-standard').split(b'\0'))
    result = {}
    for raw in names - {b''}:
        name = raw.decode('utf-8', 'surrogateescape')
        if name not in needed and name != 'CHANGELOG.md' and Path(name).suffix not in SOURCE_SUFFIXES and not ('/' not in name and Path(name).suffix.lower() == '.md'):
            continue
        path = root / name
        if path.is_symlink():
            result[name] = str(path.readlink())
        elif path.is_file():
            result[name] = path.read_text(encoding='utf-8', errors='replace')
    return result


def check(root, requested=None, pr=None):
    """Return local findings and counts without mutating the repository or its index."""
    revision = merge_base(root, requested)
    base = base_text(root, revision)
    changes = {p.decode('utf-8', 'surrogateescape') for p in
               git(root, 'diff', '--name-only', '-z', '--no-ext-diff', revision, '--').split(b'\0') if p}
    changes.update(p.decode('utf-8', 'surrogateescape') for p in
                   git(root, 'ls-files', '-z', '--others', '--exclude-standard').split(b'\0') if p)
    current = worktree_text(root, changes)
    tracked = {p.decode('utf-8', 'surrogateescape') for p in git(root, 'ls-files', '-z').split(b'\0') if p}
    additions = {}
    for name in changes:
        if name not in tracked:
            additions[name] = list(enumerate(current.get(name, '').splitlines(), 1))
        else:
            patch = git(root, 'diff', '--no-ext-diff', '--no-textconv', '--unified=0', revision, '--', name)
            additions[name] = list(added_lines(patch.decode('utf-8', 'replace')))
    findings = change_checks(base, current, changes, additions, pr)
    checker = runpy.run_path(str(root / 'tools/zig/lean_check.py'))
    before, after = lean_counts(base, checker), lean_counts(current, checker)
    findings.extend(new_lean(before, after))
    return findings, dict(merge_base=revision,changed_files=len(changes),lean_base=sum(before.values()),
                          lean_head=sum(after.values()),new_lean=sum((after - before).values()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', help='local base ref; defaults to origin/main, then main; nothing is fetched')
    parser.add_argument('--pr', type=int, help='PR number to find in an Unreleased bullet (advisory)')
    args = parser.parse_args(argv)
    if args.pr is not None and args.pr <= 0:
        parser.error('--pr must be a positive number')
    root = Path(__file__).resolve().parents[1]
    try:
        findings, counts = check(root, args.base, args.pr)
    except ValueError as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        return 2
    except (OSError,subprocess.CalledProcessError) as error:
        print('ERROR: local check could not complete; check --base, local Git objects and file access (' +
              type(error).__name__ + ')', file=sys.stderr)
        return 2
    for level, message in findings:
        print(level + ': ' + message)
    print(f"merge base {counts['merge_base']}; {counts['changed_files']} changed files; "
          f"lean {counts['lean_head']} vs {counts['lean_base']}, {counts['new_lean']} new")
    errors = sum(level == 'ERROR' for level, _ in findings)
    warnings = sum(level == 'WARN' for level, _ in findings)
    print(f'{errors} error(s), {warnings} warning(s); warnings are advisory')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
