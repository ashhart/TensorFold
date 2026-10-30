//! grammar_budget against brute force: the real libtfgrammar, a toy tokenizer, small grammars and random walks.
const std = @import("std");
const sys = @import("sys");
const lanes = @import("lanes");
const built = @import("grammar_budget_library");
const grammar = lanes.grammar;

/// The toy vocabulary: id 0 is the stop token; several texts have shorter spellings in longer tokens.
const vocab = [_][]const u8{
    "<eos>", "a", "b",  "c",   "d",  "x",   "{",  "}",    "[",   "]",   "\"", ":", ",", " ",     "\n",  "0",
    "1",     "-", "ab", "abc", "aa", "\"}", "}}", "{\"",  "\":", "\",", "]}", "q", "n", "\"q\"", "<t>", "</t>",
    "<",     "t", ">",  "/",   "e",  "ee",  "ex", "\"\"", "1}",  "12",
};
const words = (vocab.len + 31) / 32;

const Lib = struct {
    f: grammar.Library,
    h: *anyopaque,

    fn open() !Lib {
        var dl = try sys.DynLib.open(built.library);
        var f: grammar.Library = undefined;
        const info = @typeInfo(grammar.Library).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(f, name) = if (@typeInfo(T) == .optional) dl.lookup(@typeInfo(T).optional.child, "tfg_" ++ name) else dl.lookup(T, "tfg_" ++ name) orelse return error.MissingSymbol;
        }
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(std.testing.allocator);
        try text.appendSlice(std.testing.allocator, "{\"model\":{\"type\":\"BPE\",\"vocab\":{");
        for (vocab, 0..) |v, i| {
            if (i > 0) try text.append(std.testing.allocator, ',');
            try text.print(std.testing.allocator, "{f}:{d}", .{ std.json.fmt(v, .{}), i });
        }
        try text.appendSlice(std.testing.allocator, "}},\"decoder\":{\"type\":\"Fuse\"}}");
        var err: [512]u8 = undefined;
        const stops = [_]i32{0};
        const h = f.open(text.items.ptr, text.items.len, vocab.len, &stops, 1, &err, err.len) orelse return error.GrammarSetup;
        return .{ .f = f, .h = h };
    }

    fn compile(l: Lib, kind: c_int, text: []const u8) !*anyopaque {
        var err: [512]u8 = undefined;
        err[0] = 0;
        return l.f.compile(l.h, kind, text.ptr, text.len, &err, err.len) orelse {
            std.debug.print("compile: {s}\n", .{std.mem.sliceTo(&err, 0)});
            return error.Grammar;
        };
    }
};

/// One matcher driven straight through the C ABI.
const Walk = struct {
    f: grammar.Library,
    m: *anyopaque,

    /// xgrammar's bits for the toy vocabulary's ids (the last word's spare bits cleared).
    fn mask(w: Walk) ![words]u32 {
        var bits: [words]u32 = @splat(0);
        if (w.f.fill(w.m, @ptrCast(&bits), words) != 0) return error.Fill;
        bits[words - 1] &= (@as(u32, 1) << @intCast(vocab.len % 32)) - 1;
        return bits;
    }

    fn take(w: Walk, t: usize) !void {
        if (w.f.accept(w.m, @intCast(t)) != 1) return error.Refused;
    }

    fn back(w: Walk, n: usize) void {
        _ = w.f.rollback(w.m, @intCast(n));
    }

    /// The reply can stop here: the stop token is allowed.
    fn complete(w: Walk) !bool {
        return grammar.allows(&try w.mask(), 0);
    }

    /// Brute force: some `k` tokens or fewer reach a place where the reply can stop.
    fn reaches(w: Walk, k: usize) !bool {
        if (try w.complete()) return true;
        if (k == 0) return false;
        const bits = try w.mask();
        for (1..vocab.len) |t| {
            if (!grammar.allows(&bits, t)) continue;
            try w.take(t);
            const ok = try w.reaches(k - 1);
            w.back(1);
            if (ok) return true;
        }
        return false;
    }

    /// The budget mask with `left` draws, and the brute-force set it should be (xgrammar's when nothing completes).
    fn compare(w: Walk, left: usize, exact: bool) !bool {
        const plain = try w.mask();
        var got = plain;
        if (w.f.budget.?(w.m, @ptrCast(&got), words, @intCast(left)) != 0) return error.NotCovered;
        var want: [words]u32 = @splat(0);
        for (0..vocab.len) |t| {
            if (!grammar.allows(&plain, t)) continue;
            var ok = t == 0;
            if (!ok) {
                try w.take(t);
                ok = try w.reaches(left - 1);
                w.back(1);
            }
            if (ok) want[t / 32] |= @as(u32, 1) << @intCast(t % 32);
        }
        if (std.mem.allEqual(u32, &want, 0)) want = plain;
        for (got, want, plain) |g, x, p| {
            try std.testing.expectEqual(@as(u32, 0), g & ~p); // never a token xgrammar refuses
            try std.testing.expectEqual(@as(u32, 0), x & ~g); // never masks a token that can still complete
            if (exact) try std.testing.expectEqual(x, g);
        }
        return !std.mem.eql(u32, &got, &plain);
    }
};

fn pick(r: std.Random, bits: [words]u32) ?usize {
    var n: usize = 0;
    for (1..vocab.len) |t| n += @intFromBool(grammar.allows(&bits, t));
    if (n == 0) return null;
    var k = r.uintLessThan(usize, n);
    for (1..vocab.len) |t| if (grammar.allows(&bits, t)) {
        if (k == 0) return t;
        k -= 1;
    };
    unreachable;
}

const schema = "{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"},\"n\":{\"type\":\"integer\"}},\"required\":[\"q\",\"n\"],\"additionalProperties\":false}";
const call = "{\"type\":\"structural_tag\",\"format\":{\"type\":\"sequence\",\"elements\":[{\"type\":\"const_string\",\"value\":\"<t>\"}," ++
    "{\"type\":\"json_schema\",\"json_schema\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}},\"required\":[\"q\"],\"additionalProperties\":false}," ++
    "\"style\":\"json\",\"max_whitespace_cnt\":32},{\"type\":\"const_string\",\"value\":\"</t>\"}]}}";

const Case = struct { kind: c_int, text: []const u8, exact: bool, most: usize };

const cases = [_]Case{
    .{ .kind = 4, .text = "root ::= \"abc\" \"abc\" \"d\" | \"a\" \"b\"* \"x\" | \"ee\" (\"e\" | \"x\")*", .exact = true, .most = 6 },
    .{ .kind = 4, .text = "root ::= \"[\" root \"]\" | \"x\"", .exact = false, .most = 6 },
    .{ .kind = 1, .text = schema, .exact = true, .most = 4 },
    .{ .kind = 5, .text = call, .exact = true, .most = 4 },
};

test "grammar budget: the mask keeps exactly the tokens a brute-force search can still complete within the draws left" {
    const l = try Lib.open();
    defer l.f.close(l.h);
    var prng: std.Random.DefaultPrng = .init(1473);
    const r = prng.random();
    for (cases) |cs| {
        const g = try l.compile(cs.kind, cs.text);
        defer l.f.free(g);
        try std.testing.expectEqual(@as(c_int, 1), l.f.budget_ready.?(g));
        var narrowed: usize = 0;
        for (0..12) |_| {
            const w: Walk = .{ .f = l.f, .m = l.f.matcher(g).? };
            defer l.f.matcher_free(w.m);
            try std.testing.expectEqual(@as(c_int, 0), l.f.budget.?(w.m, undefined, 0, 0));
            for (0..14) |_| {
                if (l.f.terminated(w.m) == 1) break;
                narrowed += @intFromBool(try w.compare(1 + r.uintLessThan(usize, cs.most), cs.exact));
                try w.take(pick(r, try w.mask()) orelse break);
            }
        }
        try std.testing.expect(narrowed > 0); // the budget bound somewhere, so equality is not vacuous
    }
}

test "grammar budget: a grammar the byte-level tables do not follow is refused, and its masks are left alone" {
    const l = try Lib.open();
    defer l.f.close(l.h);
    const g = try l.compile(4, "root ::= body\nbody[max_tokens=3] ::= \"a\"* \"x\"");
    defer l.f.free(g);
    try std.testing.expectEqual(@as(c_int, 0), l.f.budget_ready.?(g));
    const w: Walk = .{ .f = l.f, .m = l.f.matcher(g).? };
    defer l.f.matcher_free(w.m);
    try std.testing.expectEqual(@as(c_int, 0), l.f.budget.?(w.m, undefined, 0, 0));
    var bits = try w.mask();
    try std.testing.expectEqual(@as(c_int, 1), l.f.budget.?(w.m, @ptrCast(&bits), words, 1));
    try std.testing.expectEqualSlices(u32, &try w.mask(), &bits);
}

test "grammar budget: a forced tool call cut by a tiny budget always closes, and without the mask it does not" {
    const l = try Lib.open();
    defer l.f.close(l.h);
    const g = try l.compile(5, call);
    defer l.f.free(g);
    var cut: usize = 0;
    for ([_]bool{ true, false }) |budgeted| for ([_]usize{ 7, 8, 10, 14 }) |limit| for (0..40) |seed| {
        var prng: std.Random.DefaultPrng = .init(seed);
        const w: Walk = .{ .f = l.f, .m = l.f.matcher(g).? };
        defer l.f.matcher_free(w.m);
        if (budgeted) try std.testing.expectEqual(@as(c_int, 0), l.f.budget.?(w.m, undefined, 0, 0));
        for (0..limit) |i| {
            var bits = try w.mask();
            if (budgeted) _ = l.f.budget.?(w.m, @ptrCast(&bits), words, @intCast(limit - i));
            const t = pick(prng.random(), bits) orelse break; // the stop token is never drawn: the walk only adds text
            try w.take(t);
        }
        const closed = try w.complete();
        if (budgeted) try std.testing.expect(closed) else cut += @intFromBool(!closed);
    };
    try std.testing.expect(cut > 0); // the budget, not the walk, closes the calls
}

test "grammar budget: a constraint's window rows carry the masks its plain steps get, and off leaves xgrammar's" {
    const gpa = std.testing.allocator;
    const l = try Lib.open();
    defer l.f.close(l.h);
    const old = grammar.swapLibrary(l.f);
    defer _ = grammar.swapLibrary(old);
    var compiler: grammar.Compiler = .{ .f = l.f, .h = l.h, .words = words };
    var err: [512]u8 = undefined;
    const g = try compiler.compile(std.testing.io, .structural_tag, call, &err);
    defer g.free();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const path = [_]u32{ 30, 23, 27, 24, 10, 1, 21, 31 }; // <t> {" q ": " a "} </t>
    for ([_]u32{ 8, 9, 12 }) |limit| {
        const drafted = try compiler.constrained(gpa, g, null);
        defer drafted.deinit(gpa);
        try drafted.arm();
        drafted.budget.?.max_new = limit;
        const plain = try compiler.constrained(gpa, g, null);
        defer plain.deinit(gpa);
        try plain.arm();
        plain.budget.?.max_new = limit;
        const off = try compiler.constrained(gpa, g, null);
        defer off.deinit(gpa);
        try std.testing.expect(try drafted.follow(path[0]) and try plain.follow(path[0]) and try off.follow(path[0]));
        const rows = path.len - 1;
        var parents: [rows]i32 = undefined;
        for (&parents, 0..) |*p, i| p.* = @as(i32, @intCast(i)) - 1;
        const w = try drafted.window(a, path[0..rows], &parents, false, rows);
        for (0..w.rows.len) |i| {
            const one = (try plain.next(a)).?;
            try std.testing.expectEqualSlices(u32, one, w.masks[i * words ..][0..words]);
            const raw = (try off.next(a)).?;
            var direct: [words]u32 = undefined;
            try std.testing.expectEqual(@as(c_int, 0), l.f.fill(off.rules.ptr, @ptrCast(&direct), words));
            try std.testing.expectEqualSlices(u32, &direct, raw);
            if (i + 1 < rows) try std.testing.expect(try plain.follow(path[i + 1]) and try off.follow(path[i + 1]));
        }
    }
}
