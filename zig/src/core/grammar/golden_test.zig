//! `test-grammar`: gen_goldens.py's cuts, mask digests, structural tags and refusals replayed through `Grammars`.

const std = @import("std");
const testing = std.testing;
const grammars_mod = @import("grammars.zig");
const vocab_mod = @import("vocab.zig");
const tags = @import("tags.zig");
const xgr = @import("xgr.zig");
const Constraint = @import("constraint.zig").Constraint;

const fixtures = "zig/tests/grammar/fixtures/";

const Step = struct {
    window: ?[]const u32 = null,
    keep: u32 = 0,
    first: u32 = 0,
    stop: bool = false,
    rows: []const [2]std.json.Value = &.{},
    advance: []const u32,
};
const Case = struct { name: []const u8, kind: []const u8, text: []const u8, tag: ?[]const u8, active: bool, steps: []const Step, finished: bool };
const Bad = struct { name: []const u8, kind: []const u8, text: []const u8, field: []const u8, message: ?[]const u8 };
const View = struct { vocab_type: i32, add_prefix_space: bool, vocab: []const u8, never_digest: []const u8, never_count: u32 };
const Golden = struct {
    vocab_size: u32,
    eos: u32,
    think_open: ?u32,
    think_end: ?u32,
    words: u32,
    views: struct { text: View, tools: View },
    cases: []const Case,
    bad: []const Bad,
};

fn hex16(bytes: []const u8) [16]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    var out: [16]u8 = undefined;
    _ = std.fmt.bufPrint(&out, "{x}", .{d[0..8]}) catch unreachable;
    return out;
}

fn vocabDigest(tokens: []const []const u8) [16]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (tokens) |s| {
        var n: [4]u8 = undefined;
        std.mem.writeInt(u32, &n, @intCast(s.len), .little);
        h.update(&n);
        h.update(s);
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    var out: [16]u8 = undefined;
    _ = std.fmt.bufPrint(&out, "{x}", .{d[0..8]}) catch unreachable;
    return out;
}

fn read(a: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(testing.io, path, a, .limited(1 << 30));
}

fn run(golden_path: []const u8, tokenizer_path: []const u8) !void {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const g = try std.json.parseFromSliceLeaky(Golden, a, try read(a, golden_path), .{ .ignore_unknown_fields = true });
    const tok_json = try read(a, tokenizer_path);

    // the views: the vocabulary each gives xgrammar, its metadata, the tokens it never takes
    const meta = try xgr.detectMetadata(gpa, tok_json);
    const dsml = [_][]const u8{"｜DSML｜"};
    for ([_]View{ g.views.text, g.views.tools }, [_][]const []const u8{ &.{}, &dsml }) |want, keep| {
        try testing.expectEqual(want.vocab_type, meta.vocab_type);
        try testing.expectEqual(want.add_prefix_space, meta.add_prefix_space);
        var v = try vocab_mod.build(gpa, tok_json, g.vocab_size, keep);
        defer v.deinit();
        try testing.expectEqualStrings(want.vocab, &vocabDigest(v.tokens));
        try testing.expectEqual(g.think_open, v.think_open);
        try testing.expectEqual(g.think_end, v.think_end);
    }
    const gs = try grammars_mod.Grammars.init(gpa, tok_json, .{ .vocab_size = g.vocab_size, .stops = &.{g.eos}, .tool_tokens = &dsml, .tag_model = .deepseek_v4_1 });
    defer gs.deinit();
    try testing.expectEqual(g.words, gs.words);
    for ([_]View{ g.views.text, g.views.tools }, [_]grammars_mod.Kind{ .json, .tools }) |want, kind| {
        const never = &gs.views[if (kind == .tools) 1 else 0].never;
        try testing.expectEqual(want.never_count, @as(u32, @intCast(never.count())));
        var list: std.Io.Writer.Allocating = .init(gpa);
        defer list.deinit();
        try list.writer.writeByte('[');
        var it = never.iterator(.{});
        var i: usize = 0;
        while (it.next()) |t| : (i += 1) try list.writer.print("{s}{d}", .{ if (i > 0) ", " else "", t });
        try list.writer.writeByte(']');
        try testing.expectEqualStrings(want.never_digest, &hex16(list.written()));
    }

    const bits = try gpa.alloc(u32, 16 * gs.words);
    defer gpa.free(bits);
    var rows_checked: usize = 0;
    for (g.cases) |c| {
        const kind = std.meta.stringToEnum(grammars_mod.Kind, c.kind).?;
        if (c.tag) |want| {
            var problem: tags.Problem = .{};
            const got = try tags.build(gpa, .deepseek_v4_1, c.text, &problem);
            defer gpa.free(got);
            testing.expectEqualStrings(want, got) catch |e| {
                std.debug.print("case {s}: the structural tag differs\n", .{c.name});
                return e;
            };
        }
        var why: grammars_mod.Grammars.Why = .{};
        const compiled = gs.compile(.{ .kind = kind, .text = c.text }, &why) catch |e| {
            std.debug.print("case {s}: {s}\n", .{ c.name, why.text() });
            return e;
        };
        defer compiled.deinit();
        var con = try gs.bound(kind, compiled, c.active);
        defer con.deinit();
        for (c.steps, 0..) |s, si| {
            if (s.window) |w| {
                const k = try con.cut(w);
                const bad = k.keep != s.keep or (k.rows() > 0 and k.first != s.first) or k.stop != s.stop;
                if (bad) {
                    std.debug.print("case {s} step {d}: cut keep {d} first {d} stop {} vs reference {d} {d} {}\n", .{ c.name, si, k.keep, k.first, k.stop, s.keep, s.first, s.stop });
                    return error.TestUnexpectedResult;
                }
                try testing.expectEqual(s.rows.len, k.rows());
                try con.fill(w, k, bits);
                for (s.rows, 0..) |want, j| {
                    const row = bits[j * gs.words ..][0..gs.words];
                    var allowed: usize = 0;
                    for (row, 0..) |word, wi| {
                        const valid: u32 = @min(32, g.vocab_size - @as(u32, @intCast(32 * wi)));
                        const m: u32 = if (valid == 32) 0xffff_ffff else (@as(u32, 1) << @intCast(valid)) - 1;
                        allowed += @popCount(word & m);
                    }
                    const digest = hex16(std.mem.sliceAsBytes(row));
                    if (!std.mem.eql(u8, &digest, want[0].string) or allowed != @as(usize, @intCast(want[1].integer))) {
                        std.debug.print("case {s} step {d} row {d}: mask {s} ({d} allowed) vs reference {s} ({d})\n", .{ c.name, si, j, &digest, allowed, want[0].string, want[1].integer });
                        return error.TestUnexpectedResult;
                    }
                    rows_checked += 1;
                }
            }
            try con.advance(s.advance);
        }
        try testing.expectEqual(c.finished, con.finished());
    }
    for (g.bad) |b| {
        const kind = std.meta.stringToEnum(grammars_mod.Kind, b.kind).?;
        var why: grammars_mod.Grammars.Why = .{};
        if (gs.compile(.{ .kind = kind, .text = b.text }, &why)) |compiled| {
            compiled.deinit();
            try testing.expect(b.message == null);
        } else |e| {
            try testing.expectEqual(error.Invalid, e);
            const got = try std.fmt.allocPrint(gpa, "{s}: the grammar cannot be enforced: {s}", .{ b.field, why.text() });
            defer gpa.free(got);
            try testing.expectEqualStrings(b.message.?, got);
        }
    }
    // every case has masked rows (a walk that never masked would prove nothing)
    try testing.expect(rows_checked >= 10 * g.cases.len);
}

test "grammar goldens: synthetic tokenizer" {
    try run(fixtures ++ "synthetic.golden.json", fixtures ++ "synthetic/tokenizer.json");
}
