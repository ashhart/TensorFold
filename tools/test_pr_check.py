"""Planted contributor failures stay errors; advisories never turn a clean check red."""
from collections import Counter
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch
import pr_check as pc


class ContributorChecks(unittest.TestCase):
    def checks(self, name, text, baseline=None, pr=12, changelog='## Unreleased\n\n- Change (#12).\n'):
        current = {name: text, 'CHANGELOG.md': changelog}
        base = {} if baseline is None else {name: baseline}
        return pc.change_checks(base, current, {name}, {name: list(enumerate(text.splitlines(), 1))}, pr)

    def test_touched_source_limit_including_inherited_long_file(self):
        self.assertFalse(self.checks('zig/a.zig', 'x\n' * 600))
        found = self.checks('zig/a.zig', 'x\n' * 601, baseline='x\n' * 601)
        self.assertIn('600 lines', found[0][1])
        self.assertEqual(found[0][0], 'ERROR')

    def test_home_paths_warn_without_echoing_the_value(self):
        examples = ['/' + 'home' + '/example/a', '/' + 'Users' + '/example/a',
                    '/' + 'root' + '/a', 'C:' + '\\Users' + '\\example\\a']
        for text in examples:
            with self.subTest(style=text[0]):
                found = pc.private_additions('file', [(8, text)])
                self.assertEqual(found, [('WARN', 'file:8: added home-directory path')])
                self.assertNotIn(text, found[0][1])

    def test_parent_patch_warns_and_regular_relative_path_is_allowed(self):
        self.assertEqual(pc.private_additions('file', [(2, '..' + '/fix.patch')])[0][0], 'WARN')
        self.assertFalse(pc.private_additions('file', [(2, 'inputs/fix.patch')]))

    def test_ipv4_warns_except_the_exact_loopback(self):
        bad = '.'.join(('198', '51', '100', '7'))
        self.assertEqual(pc.private_additions('file', [(3, bad)])[0][0], 'WARN')
        self.assertFalse(pc.private_additions('file', [(3, 'http://127.0.0.1:8080')]))
        self.assertFalse(pc.private_additions('file', [(3, '.'.join(('999', '51', '100', '7')))]))

    def test_documentation_examples_only_warn_and_main_stays_successful(self):
        examples = [('docs/vision.md', '.'.join(('1','2','3','4'))),
                    ('docs/recipes/example.md', '.'.join(('198','51','100','7'))),
                    ('docs/receipts/README.md', '/' + 'home' + '/example/run'),
                    ('docs/receipts/README.md', '..' + '/example.patch')]
        found = [f for name, text in examples for f in pc.private_additions(name, [(1, text)])]
        self.assertEqual(len(found), 4)
        self.assertTrue(all(level == 'WARN' for level, _ in found))
        counts = dict(merge_base='base', changed_files=4, lean_base=0, lean_head=0, new_lean=0)
        with patch.object(pc, 'check', return_value=(found, counts)):
            self.assertEqual(pc.main([]), 0)

    def test_only_new_top_level_markdown_warns(self):
        self.assertEqual(self.checks('NEW.md', 'notes')[0][0], 'WARN')
        self.assertFalse(self.checks('README.md', 'edited', baseline='old'))
        self.assertFalse(self.checks('docs/new.md', 'notes'))

    def test_unreleased_pr_warning_never_uses_an_old_release(self):
        old = '## Unreleased\n\n- Unrelated (#120).\n\n## Released\n\n- Old (#12).\n'
        self.assertEqual(self.checks('zig/a.zig', 'x', changelog=old)[0][0], 'WARN')
        self.assertFalse(self.checks('zig/a.zig', 'x'))
        self.assertEqual(self.checks('zig/a.zig', 'x', pr=None)[0][0], 'WARN')
        self.assertFalse(self.checks('tools/a.py', 'x', pr=None))

    def test_added_diff_lines_keep_numbers_and_ignore_removed_private_text(self):
        diff = '--- a/file\n+++ b/file\n@@ -1 +4,2 @@\n-old\n+new\n+more\n@@ -9 +12 @@\n-removed\n+last\n'
        self.assertEqual(list(pc.added_lines(diff)), [(4, 'new'), (5, 'more'), (12, 'last')])

    def test_same_current_lean_checker_counts_baseline_and_planted_new_comment(self):
        checker = runpy.run_path(str(Path(__file__).parent / 'zig/lean_check.py'))
        before = pc.lean_counts({'zig/a.zig': '// inherited\n// comment\nconst x=1;\n'}, checker)
        moved = pc.lean_counts({'zig/a.zig': '\n// inherited\n// comment\nconst x=1;\n'}, checker)
        self.assertEqual(pc.new_lean(before, moved), [])
        bad = pc.lean_counts({'zig/a.zig': '// inherited\n// comment\nconst x=1; /* new */\n'}, checker)
        found = pc.new_lean(before, bad)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0][0], 'ERROR')
        self.assertIn('block comment', found[0][1])

    def test_duplicate_lean_findings_count_and_line_moves_do_not(self):
        a = Counter({('zig/a.zig', 'block comment'): 1})
        b = Counter({('zig/a.zig', 'block comment'): 2})
        self.assertEqual(len(pc.new_lean(a, b)), 1)
        self.assertIn('1 new', pc.new_lean(a, b)[0][1])

    def test_merge_base_uses_local_ref_and_common_ancestor(self):
        calls = []
        def git(root, *args, **kwargs):
            calls.append(args)
            return b'abc\n' if args[0] == 'merge-base' else b'ref\n'
        with patch.object(pc, 'git', git):
            self.assertEqual(pc.merge_base(Path('.')), 'abc')
        self.assertEqual(calls[-1], ('merge-base', '--', 'refs/remotes/origin/main', 'HEAD'))
        self.assertFalse(any(c[0] in ('fetch', 'push') for c in calls))

    def test_staged_unstaged_untracked_and_deleted_are_in_the_check(self):
        root = Path('.')
        names = {'zig/staged.zig', 'zig/unstaged.zig', 'new.md', 'zig/deleted.zig'}
        current = {'zig/staged.zig': 'x\n' * 601, 'zig/unstaged.zig': '..' + '/bad.patch', 'new.md': 'notes'}
        def git(root, *args, **kwargs):
            if args[:2] == ('diff', '--name-only'):
                return b'zig/staged.zig\0zig/unstaged.zig\0zig/deleted.zig\0'
            if args[0] == 'ls-files':
                return b'new.md\0' if '--others' in args else b'zig/staged.zig\0zig/unstaged.zig\0zig/deleted.zig\0'
            if args[0] == 'diff':
                name = args[-1]
                return ('@@ -0,0 +1 @@\n+' + current.get(name, '') + '\n').encode()
            raise AssertionError(args)
        with patch.object(pc, 'git', git), patch.object(pc, 'merge_base', return_value='base'), \
             patch.object(pc, 'base_text', return_value={'zig/deleted.zig': 'x'}), \
             patch.object(pc, 'worktree_text', return_value=current), patch.object(pc.runpy, 'run_path', return_value={}), \
             patch.object(pc, 'lean_counts', return_value=Counter()):
            found, counts = pc.check(root)
        self.assertEqual(counts['changed_files'], len(names))
        self.assertTrue(any('600 lines' in m for _, m in found))
        self.assertTrue(any('patch path' in m for _, m in found))
        self.assertTrue(any(level == 'WARN' and 'new.md' in m for level, m in found))

    def test_advisories_exit_zero_errors_exit_one_and_incomplete_exit_two(self):
        counts = dict(merge_base='base', changed_files=1, lean_base=1, lean_head=1, new_lean=0)
        for level, expected in [('WARN', 0), ('ERROR', 1)]:
            with patch.object(pc, 'check', return_value=([(level, 'planted')], counts)):
                self.assertEqual(pc.main([]), expected)
        with patch.object(pc, 'check', side_effect=ValueError('missing base')):
            self.assertEqual(pc.main([]), 2)


if __name__ == '__main__':
    unittest.main()
