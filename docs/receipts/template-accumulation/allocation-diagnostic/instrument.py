"""Instrument exact, known evaluator anchors in a disposable Git snapshot."""
from pathlib import Path


def once(text: str, old: str, new: str) -> str:
    """Replace one unique anchor or refuse the diagnostic build."""
    if text.count(old) != 1:
        raise ValueError(f'expected exactly one instrumentation anchor: {old[:80]!r}')
    return text.replace(old, new, 1)


def instrument(source: Path, side: str) -> None:
    """Wrap only string joins and list concatenation in counting allocators."""
    path = source / 'zig/src/core/template/eval.zig'
    text = path.read_text()
    text = once(text, 'const std = @import("std");',
                'const std = @import("std");\nconst diag = @import("diagnostics.zig");')
    start = text.index('fn join(c: *Ctx, l: v.Str, r: v.Str) Error!v.Str {')
    end = text.index('\nconst Flow = ', start)
    section = text[start:end]
    section = once(section, '    const a = c.rt.a;',
                   '    var probe: diag.Probe = .{ .child = c.rt.a, .counts = &diag.string_alloc };\n'
                   '    const a = probe.allocator();')
    if side == 'base':
        section = once(section, '    if (l.s.len < acc_min and !isAcc(c, l.s)) {',
                       '    const reuse = isAcc(c, l.s);\n    diag.stringJoin(l.s.len, reuse);\n'
                       '    if (l.s.len < acc_min and !reuse) {')
        section = once(section, '    if (!isAcc(c, l.s)) {', '    if (!reuse) {')
    elif side == 'candidate':
        anchor = '    const current = if (l.builder) |b| (if (isAcc(b, l)) b else null) else null;'
        section = once(section, anchor, anchor + '\n    diag.stringJoin(l.s.len, current != null);')
    else:
        raise ValueError('unknown revision side')
    section = once(section, '        try out.append(a, l);',
                   '        try out.append(a, l);\n        diag.prefix_copied += l.s.len;')
    section = once(section, '        return out.str();',
                   '        diag.string_completed += 1;\n        return out.str();')
    if side == 'base':
        section = once(section, '        try c.acc.append(a, l);',
                       '        try c.acc.append(a, l);\n        diag.prefix_copied += l.s.len;')
        section = once(section, '    return c.acc.str();',
                       '    diag.string_completed += 1;\n    return c.acc.str();')
    else:
        section = once(section, '        try b.appendRetained(a, l);',
                       '        try b.appendRetained(a, l);\n        diag.prefix_copied += l.s.len;')
        section = once(section, '    return out;', '    diag.string_completed += 1;\n    return out;')
    text = text[:start] + section + text[end:]
    old = ('    if (op == .add and l == .list and r == .list and l.list.seq == r.list.seq '
           'and (l.list.seq == .list or l.list.seq == .tuple))\n'
           '        return rt.list(try std.mem.concat(rt.a, Value, &.{ l.list.items, r.list.items }), l.list.seq);')
    new = ('    if (op == .add and l == .list and r == .list and l.list.seq == r.list.seq '
           'and (l.list.seq == .list or l.list.seq == .tuple)) {\n'
           '        diag.list_attempts += 1;\n'
           '        var probe: diag.Probe = .{ .child = rt.a, .counts = &diag.list_alloc };\n'
           '        const joined = try std.mem.concat(probe.allocator(), Value, &.{ l.list.items, r.list.items });\n'
           '        diag.list_success += 1;\n'
           '        diag.list_elements += l.list.items.len + r.list.items.len;\n'
           '        return rt.list(joined, l.list.seq);\n    }')
    text = once(text, old, new)
    path.write_text(text)
