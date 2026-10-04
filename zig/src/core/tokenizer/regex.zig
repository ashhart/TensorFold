//! Backtracking matcher for the Oniguruma (Ruby syntax) patterns in tokenizer.json; unsupported syntax is an error, never a guess.
const std = @import("std");
const unicode = @import("unicode.zig");

pub const Error = error{ UnsupportedRegex, OutOfMemory };

const Item = union(enum) {
    range: [2]u21,
    cats: u32,
    not_cats: u32,
    space: bool,
};

const Class = struct {
    items: []const Item,
    negated: bool = false,
    ascii: [2]u64 = .{ 0, 0 },

    inline fn has(c: *const Class, cp: u21) bool {
        if (cp < 128) return (c.ascii[cp >> 6] >> @intCast(cp & 63)) & 1 != 0;
        return c.slow(cp);
    }

    fn slow(c: *const Class, cp: u21) bool {
        const cat = unicode.bit(unicode.category(cp));
        for (c.items) |item| if (switch (item) {
            .range => |r| cp >= r[0] and cp <= r[1],
            .cats => |m| m & cat != 0,
            .not_cats => |m| m & cat == 0,
            .space => |positive| unicode.isSpace(cp) == positive,
        }) return !c.negated;
        return c.negated;
    }
};

const Inst = union(enum) {
    char: u21,
    class: u32,
    any,
    split: [2]u32,
    jump: u32,
    look: struct { negate: bool, next: u32 },
    done,
};

const Node = union(enum) {
    empty,
    char: u21,
    class: u32,
    any,
    concat: []const Node,
    alt: []const Node,
    repeat: struct { node: *const Node, min: u32, max: ?u32, greedy: bool },
    look: struct { node: *const Node, negate: bool },

    fn nullable(n: Node) bool {
        return switch (n) {
            .empty, .look => true,
            .char, .class, .any => false,
            .concat => |nodes| for (nodes) |x| {
                if (!x.nullable()) break false;
            } else true,
            .alt => |nodes| for (nodes) |x| {
                if (x.nullable()) break true;
            } else false,
            .repeat => |r| r.min == 0 or r.node.nullable(),
        };
    }
};

pub const Regex = struct {
    prog: []const Inst,
    classes: []const Class,

    /// Compiles `pattern`; the program lives in `a` (an arena owned by the tokenizer).
    pub fn compile(a: std.mem.Allocator, pattern: []const u8) Error!Regex {
        var scratch = std.heap.ArenaAllocator.init(a);
        defer scratch.deinit();
        var p = Parser{ .src = pattern, .a = scratch.allocator(), .keep = a };
        const root = try p.alternation(false);
        if (p.i != p.src.len) return error.UnsupportedRegex;
        var c = Compiler{ .a = a };
        try c.emit(root);
        _ = try c.add(.done);
        for (p.classes.items) |*class| for (0..128) |cp| {
            if (class.slow(@intCast(cp))) class.ascii[cp >> 6] |= @as(u64, 1) << @intCast(cp & 63);
        };
        return .{ .prog = try c.prog.toOwnedSlice(a), .classes = try p.classes.toOwnedSlice(a) };
    }

    /// Leftmost match starting at or after `from`, alternatives in priority order (Oniguruma search).
    pub fn find(re: *const Regex, m: *Matcher, s: []const u8, from: usize) Error!?[2]usize {
        var pos = from;
        while (pos <= s.len) {
            if (try m.run(re, s, 0, pos)) |end| return .{ pos, end };
            if (pos == s.len) break;
            pos += unicode.decodeAt(s, pos).len;
        }
        return null;
    }

    /// All matches as the onig crate's find_iter yields them: an empty match right after the last one is skipped.
    pub fn findAll(re: *const Regex, m: *Matcher, s: []const u8, out: *std.ArrayList([2]usize)) Error!void {
        var at: usize = 0;
        var last: ?usize = null;
        while (at <= s.len) {
            const hit = (try re.find(m, s, at)) orelse break;
            if (hit[0] == hit[1] and last == hit[1]) {
                at += if (at < s.len) unicode.decodeAt(s, at).len else 1;
                continue;
            }
            try out.append(m.a, hit);
            at = hit[1];
            last = hit[1];
        }
    }
};

/// Per-call backtracking state; a Regex is shared read-only across threads.
pub const Matcher = struct {
    a: std.mem.Allocator,
    stack: std.ArrayList(Frame) = .empty,

    const Frame = struct { pc: u32, pos: usize };

    pub fn deinit(m: *Matcher) void {
        m.stack.deinit(m.a);
    }

    fn run(m: *Matcher, re: *const Regex, s: []const u8, pc: u32, pos: usize) Error!?usize {
        const base = m.stack.items.len;
        defer m.stack.shrinkRetainingCapacity(base);
        try m.stack.append(m.a, .{ .pc = pc, .pos = pos });
        while (m.stack.items.len > base) {
            var f = m.stack.pop().?;
            while (true) switch (re.prog[f.pc]) {
                .char => |want| {
                    if (f.pos >= s.len) break;
                    const d = unicode.decodeAt(s, f.pos);
                    if (d.cp != want) break;
                    f.pos += d.len;
                    f.pc += 1;
                },
                .class => |index| {
                    if (f.pos >= s.len) break;
                    const d = unicode.decodeAt(s, f.pos);
                    if (!re.classes[index].has(d.cp)) break;
                    f.pos += d.len;
                    f.pc += 1;
                },
                .any => {
                    if (f.pos >= s.len or s[f.pos] == '\n') break;
                    f.pos += unicode.decodeAt(s, f.pos).len;
                    f.pc += 1;
                },
                .split => |to| {
                    try m.stack.append(m.a, .{ .pc = to[1], .pos = f.pos });
                    f.pc = to[0];
                },
                .jump => |to| f.pc = to,
                .look => |look| {
                    const hit = (try m.run(re, s, f.pc + 1, f.pos)) != null;
                    if (hit == look.negate) break;
                    f.pc = look.next;
                },
                .done => return f.pos,
            };
        }
        return null;
    }
};

const Compiler = struct {
    a: std.mem.Allocator,
    prog: std.ArrayList(Inst) = .empty,

    fn add(c: *Compiler, inst: Inst) Error!u32 {
        try c.prog.append(c.a, inst);
        return @intCast(c.prog.items.len - 1);
    }

    fn here(c: *Compiler) u32 {
        return @intCast(c.prog.items.len);
    }

    fn emit(c: *Compiler, node: Node) Error!void {
        switch (node) {
            .empty => {},
            .char => |cp| _ = try c.add(.{ .char = cp }),
            .class => |index| _ = try c.add(.{ .class = index }),
            .any => _ = try c.add(.any),
            .concat => |nodes| for (nodes) |n| try c.emit(n),
            .alt => |nodes| {
                var exits: std.ArrayList(u32) = .empty;
                defer exits.deinit(c.a);
                for (nodes[0 .. nodes.len - 1]) |n| {
                    const split = try c.add(.{ .split = .{ 0, 0 } });
                    try c.emit(n);
                    try exits.append(c.a, try c.add(.{ .jump = 0 }));
                    c.prog.items[split] = .{ .split = .{ split + 1, c.here() } };
                }
                try c.emit(nodes[nodes.len - 1]);
                for (exits.items) |at| c.prog.items[at] = .{ .jump = c.here() };
            },
            .repeat => |r| {
                for (0..r.min) |_| try c.emit(r.node.*);
                if (r.max) |max| {
                    var splits: std.ArrayList(u32) = .empty;
                    defer splits.deinit(c.a);
                    for (r.min..max) |_| {
                        try splits.append(c.a, try c.add(.{ .split = .{ 0, 0 } }));
                        try c.emit(r.node.*);
                    }
                    const end = c.here();
                    for (splits.items) |at| c.prog.items[at] = .{ .split = if (r.greedy) .{ at + 1, end } else .{ end, at + 1 } };
                } else {
                    const loop = try c.add(.{ .split = .{ 0, 0 } });
                    try c.emit(r.node.*);
                    _ = try c.add(.{ .jump = loop });
                    const end = c.here();
                    c.prog.items[loop] = .{ .split = if (r.greedy) .{ loop + 1, end } else .{ end, loop + 1 } };
                }
            },
            .look => |l| {
                const at = try c.add(.{ .look = .{ .negate = l.negate, .next = 0 } });
                try c.emit(l.node.*);
                _ = try c.add(.done);
                c.prog.items[at] = .{ .look = .{ .negate = l.negate, .next = c.here() } };
            },
        }
    }
};

const Parser = struct {
    src: []const u8,
    i: usize = 0,
    a: std.mem.Allocator,
    keep: std.mem.Allocator,
    classes: std.ArrayList(Class) = .empty,

    fn peek(p: *Parser) ?u8 {
        return if (p.i < p.src.len) p.src[p.i] else null;
    }

    fn eat(p: *Parser, s: []const u8) bool {
        if (!std.mem.startsWith(u8, p.src[p.i..], s)) return false;
        p.i += s.len;
        return true;
    }

    fn codepoint(p: *Parser) Error!u21 {
        const step = unicode.scan(p.src, p.i);
        const cp = step.cp orelse return error.UnsupportedRegex;
        p.i += step.len;
        return cp;
    }

    fn alternation(p: *Parser, icase: bool) Error!Node {
        var options: std.ArrayList(Node) = .empty;
        try options.append(p.a, try p.sequence(icase));
        while (p.eat("|")) try options.append(p.a, try p.sequence(icase));
        return if (options.items.len == 1) options.items[0] else .{ .alt = options.items };
    }

    fn sequence(p: *Parser, icase: bool) Error!Node {
        var nodes: std.ArrayList(Node) = .empty;
        var prev: ?u21 = null;
        while (p.peek()) |c| {
            if (c == '|' or c == ')') break;
            var node = try p.atom(icase);
            if (icase and node == .char) {
                // Oniguruma folds literal runs as strings ("ss" matches U+00DF); only single-character folds are supported.
                if (prev != null and prev.? < 128 and node.char < 128 and unicode.isMultiFoldPair(@intCast(prev.?), @intCast(node.char))) return error.UnsupportedRegex;
                prev = node.char;
                node = try p.folded(node.char);
            } else prev = null;
            try nodes.append(p.a, try p.quantified(node));
        }
        return switch (nodes.items.len) {
            0 => .empty,
            1 => nodes.items[0],
            else => .{ .concat = nodes.items },
        };
    }

    fn folded(p: *Parser, cp: u21) Error!Node {
        if (cp >= 128) return error.UnsupportedRegex;
        var buf: [4]u21 = undefined;
        const variants = unicode.asciiCaseVariants(@intCast(cp), &buf);
        if (variants.len == 1) return .{ .char = cp };
        var items: std.ArrayList(Item) = .empty;
        for (variants) |v| try items.append(p.keep, .{ .range = .{ v, v } });
        return p.class(.{ .items = items.items });
    }

    fn class(p: *Parser, c: Class) Error!Node {
        try p.classes.append(p.keep, c);
        return .{ .class = @intCast(p.classes.items.len - 1) };
    }

    fn quantified(p: *Parser, node: Node) Error!Node {
        var result = node;
        while (p.peek()) |c| {
            var min: u32 = 0;
            var max: ?u32 = null;
            switch (c) {
                '*' => p.i += 1,
                '+' => {
                    p.i += 1;
                    min = 1;
                },
                '?' => {
                    p.i += 1;
                    max = 1;
                },
                '{' => {
                    const close = std.mem.indexOfScalarPos(u8, p.src, p.i, '}') orelse return result;
                    const body = p.src[p.i + 1 .. close];
                    const comma = std.mem.indexOfScalar(u8, body, ',');
                    const lo = if (comma) |k| body[0..k] else body;
                    min = if (lo.len == 0) 0 else std.fmt.parseInt(u32, lo, 10) catch return error.UnsupportedRegex;
                    max = if (comma) |k| (if (k + 1 == body.len) null else std.fmt.parseInt(u32, body[k + 1 ..], 10) catch return error.UnsupportedRegex) else min;
                    if ((max != null and max.? < min) or (max orelse min) > 1000 or (lo.len == 0 and comma == null)) return error.UnsupportedRegex;
                    p.i = close + 1;
                },
                else => return result,
            }
            var greedy = true;
            if (p.eat("?")) greedy = false else if (p.peek() == '+' and c != '{') return error.UnsupportedRegex;
            if (max == null and result.nullable()) return error.UnsupportedRegex;
            const inner = try p.a.create(Node);
            inner.* = result;
            result = .{ .repeat = .{ .node = inner, .min = min, .max = max, .greedy = greedy } };
        }
        return result;
    }

    fn atom(p: *Parser, icase: bool) Error!Node {
        const c = p.peek().?;
        switch (c) {
            '(' => {
                p.i += 1;
                var inner_icase = icase;
                var look: ?bool = null;
                if (p.eat("?:")) {} else if (p.eat("?i:")) inner_icase = true else if (p.eat("?-i:")) inner_icase = false else if (p.eat("?=")) look = false else if (p.eat("?!")) look = true else if (p.peek() == '?') return error.UnsupportedRegex;
                const inner = try p.alternation(inner_icase);
                if (!p.eat(")")) return error.UnsupportedRegex;
                if (look) |negate| {
                    const node = try p.a.create(Node);
                    node.* = inner;
                    return .{ .look = .{ .node = node, .negate = negate } };
                }
                return inner;
            },
            '[' => {
                if (icase) return error.UnsupportedRegex;
                return p.bracket();
            },
            '.' => {
                p.i += 1;
                return .any;
            },
            '\\' => {
                p.i += 1;
                const e = try p.escape();
                if (icase and e != .char) return error.UnsupportedRegex;
                return switch (e) {
                    .char => |cp| .{ .char = cp },
                    .item => |item| p.class(.{ .items = try p.keep.dupe(Item, &.{item}) }),
                };
            },
            '^', '$', ')', '|', '*', '+', '?', '{' => return error.UnsupportedRegex,
            else => return .{ .char = try p.codepoint() },
        }
    }

    const Escape = union(enum) { char: u21, item: Item };

    fn escape(p: *Parser) Error!Escape {
        const c = p.peek() orelse return error.UnsupportedRegex;
        p.i += 1;
        return switch (c) {
            't' => .{ .char = '\t' },
            'n' => .{ .char = '\n' },
            'r' => .{ .char = '\r' },
            'f' => .{ .char = 0x0C },
            'v' => .{ .char = 0x0B },
            'a' => .{ .char = 0x07 },
            'e' => .{ .char = 0x1B },
            's' => .{ .item = .{ .space = true } },
            'S' => .{ .item = .{ .space = false } },
            'd' => .{ .item = .{ .cats = unicode.bit(.Nd) } },
            'D' => .{ .item = .{ .not_cats = unicode.bit(.Nd) } },
            'p', 'P' => {
                if (!p.eat("{")) return error.UnsupportedRegex;
                const close = std.mem.indexOfScalarPos(u8, p.src, p.i, '}') orelse return error.UnsupportedRegex;
                var name = p.src[p.i..close];
                p.i = close + 1;
                var negate = c == 'P';
                if (name.len > 0 and name[0] == '^') {
                    negate = !negate;
                    name = name[1..];
                }
                const m = unicode.propertyMask(name) orelse return error.UnsupportedRegex;
                return .{ .item = if (negate) .{ .not_cats = m } else .{ .cats = m } };
            },
            'x' => {
                const braced = p.eat("{");
                const end = if (braced) std.mem.indexOfScalarPos(u8, p.src, p.i, '}') orelse return error.UnsupportedRegex else @min(p.i + 2, p.src.len);
                const cp = std.fmt.parseInt(u21, p.src[p.i..end], 16) catch return error.UnsupportedRegex;
                p.i = end + @intFromBool(braced);
                return .{ .char = cp };
            },
            'u' => {
                if (p.i + 4 > p.src.len) return error.UnsupportedRegex;
                const cp = std.fmt.parseInt(u21, p.src[p.i .. p.i + 4], 16) catch return error.UnsupportedRegex;
                p.i += 4;
                return .{ .char = cp };
            },
            else => if (c < 128 and !std.ascii.isAlphanumeric(c)) .{ .char = c } else error.UnsupportedRegex,
        };
    }

    fn bracket(p: *Parser) Error!Node {
        p.i += 1;
        var c = Class{ .items = &.{} };
        if (p.eat("^")) c.negated = true;
        var items: std.ArrayList(Item) = .empty;
        var first = true;
        while (true) : (first = false) {
            const b = p.peek() orelse return error.UnsupportedRegex;
            if (b == ']' and !first) break;
            if (b == '[' or b == ']' or std.mem.startsWith(u8, p.src[p.i..], "&&")) return error.UnsupportedRegex;
            const lo = try p.member();
            const lo_cp = switch (lo) {
                .item => |item| {
                    try items.append(p.keep, item);
                    continue;
                },
                .char => |cp| cp,
            };
            if (p.peek() == '-' and p.i + 1 < p.src.len and p.src[p.i + 1] != ']') {
                p.i += 1;
                const hi = try p.member();
                if (hi != .char or hi.char < lo_cp) return error.UnsupportedRegex;
                try items.append(p.keep, .{ .range = .{ lo_cp, hi.char } });
            } else try items.append(p.keep, .{ .range = .{ lo_cp, lo_cp } });
        }
        p.i += 1;
        c.items = items.items;
        return p.class(c);
    }

    fn member(p: *Parser) Error!Escape {
        if (p.eat("\\")) return p.escape();
        return .{ .char = try p.codepoint() };
    }
};

fn splitAll(re: *const Regex, s: []const u8) ![]const []const u8 {
    const a = std.testing.allocator;
    var m = Matcher{ .a = a };
    defer m.deinit();
    var spans: std.ArrayList([2]usize) = .empty;
    defer spans.deinit(a);
    try re.findAll(&m, s, &spans);
    var out: std.ArrayList([]const u8) = .empty;
    for (spans.items) |span| try out.append(a, s[span[0]..span[1]]);
    return out.toOwnedSlice(a);
}

test "qwen pre-tokenizer pattern splits like Oniguruma" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const re = try Regex.compile(arena.allocator(), "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
    const got = try splitAll(&re, "It'S x'\u{17F} 12 caf\u{E9}  \n\n  hi!!\r\n  ");
    defer std.testing.allocator.free(got);
    const want = [_][]const u8{ "It", "'S", " x", "'\u{17F}", " ", "1", "2", " caf\u{E9}", "  \n\n", " ", " hi", "!!\r\n", "  " };
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try std.testing.expectEqualStrings(w, g);
}

test "counted repetition, ranges and refusals" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const digits = try Regex.compile(a, "\\p{N}{1,3}");
    const got = try splitAll(&digits, "a12345b6");
    defer std.testing.allocator.free(got);
    try std.testing.expectEqual(@as(usize, 3), got.len);
    try std.testing.expectEqualStrings("45", got[1]);
    const kana = try Regex.compile(a, "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}]+|[!\"#\\-./\\[\\\\\\]^_`{|}~][A-Za-z]+");
    const k = try splitAll(&kana, "x\u{3042}\u{4e00}y-abc");
    defer std.testing.allocator.free(k);
    try std.testing.expectEqualStrings("\u{3042}\u{4e00}", k[0]);
    try std.testing.expectEqualStrings("-abc", k[1]);
    for ([_][]const u8{ "\\w+", "(?i:ss)", "a(?<=b)", "\\bx", "(?i)a", "[a[b]]", "(a*)*", "\\p{Latin}", "a++" }) |bad|
        try std.testing.expectError(error.UnsupportedRegex, Regex.compile(a, bad));
}
