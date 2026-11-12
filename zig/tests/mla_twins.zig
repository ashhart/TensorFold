//! tf_mla's absorb_q, expand_v and merge_part against the GLM kernels they copy: GLM's tokens at GLM's widths.
const std = @import("std");

const mla_path = "zig/kernels/cuda/common/mla.cuh";

/// A copy's body tokens from `from` to `to` (inclusive; null: the end), `subs` naming its width parameters as GLM's.
const Twin = struct {
    mla: []const u8,
    glm_file: []const u8,
    glm: []const u8,
    from: []const u8,
    to: ?[]const u8 = null,
    subs: []const [2][]const u8 = &.{},
};

const twins = [_]Twin{
    // absorb_q<D> at GLM's D = 256: R = D / 32 is 8, q rows D apart, qa rows LW apart
    .{ .mla = "absorb_q", .glm_file = "zig/kernels/cuda/glm_glue_latent.cuh", .glm = "tf_glm_absorb_q", .from = "__shared__", .subs = &.{ .{ "R", "8" }, .{ "qs", "D" }, .{ "qaw", "LW" } } },
    .{ .mla = "expand_v", .glm_file = "zig/kernels/cuda/glm_glue_latent.cuh", .glm = "tf_glm_expand_v", .from = "__shared__" },
    // tf_glm_merge's chunk loop; the copy keeps o, m and l in fp32 where GLM writes bf16(o / l)
    .{ .mla = "merge_part", .glm_file = "zig/kernels/cuda/glm_glue_attn.cuh", .glm = "tf_glm_merge", .from = "const int k0", .to = "m = nm ; }" },
};

fn identifier(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or ch == '_';
}

/// C++ tokens with comments dropped: identifiers and numbers whole, every other character alone.
fn tokens(a: std.mem.Allocator, text: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var i: usize = 0;
    while (i < text.len) {
        const ch = text[i];
        if (std.ascii.isWhitespace(ch)) {
            i += 1;
        } else if (std.mem.startsWith(u8, text[i..], "//")) {
            i = std.mem.indexOfScalarPos(u8, text, i, '\n') orelse text.len;
        } else if (identifier(ch)) {
            const start = i;
            while (i < text.len and (identifier(text[i]) or (std.ascii.isDigit(text[start]) and text[i] == '.'))) i += 1;
            try out.append(a, text[start..i]);
        } else {
            try out.append(a, text[i .. i + 1]);
            i += 1;
        }
    }
    return out.items;
}

fn find(ts: []const []const u8, want: []const []const u8, from: usize) ?usize {
    var i = from;
    while (i + want.len <= ts.len) : (i += 1) {
        for (want, ts[i..][0..want.len]) |x, y| {
            if (!std.mem.eql(u8, x, y)) break;
        } else return i;
    }
    return null;
}

/// The tokens of function `name`'s body, between its braces.
fn body(ts: []const []const u8, name: []const u8) ![]const []const u8 {
    const at = find(ts, &.{ name, "(" }, 0) orelse return error.NoSuchFunction;
    const open = find(ts, &.{"{"}, at) orelse return error.NoBody;
    var depth: usize = 0;
    for (ts[open..], open..) |t, i| {
        if (std.mem.eql(u8, t, "{")) depth += 1;
        if (std.mem.eql(u8, t, "}")) depth -= 1;
        if (depth == 0) return ts[open + 1 .. i];
    }
    return error.UnclosedBody;
}

/// The span from `tw.from` to `tw.to` of one side's body.
fn span(a: std.mem.Allocator, b: []const []const u8, tw: Twin) ![]const []const u8 {
    const from = find(b, try tokens(a, tw.from), 0) orelse return error.NoStart;
    const to_ts = if (tw.to) |t| try tokens(a, t) else &[_][]const u8{};
    const end = if (tw.to != null) (find(b, to_ts, from) orelse return error.NoEnd) + to_ts.len else b.len;
    return b[from..end];
}

fn read(a: std.mem.Allocator, path: []const u8) ![]const u8 {
    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(4 << 20));
}

test "mla twins: absorb_q, expand_v and merge_part are GLM's kernels token for token at GLM's widths" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const mla = try tokens(a, try read(a, mla_path));
    for (twins) |tw| {
        const ours = try span(a, try body(mla, tw.mla), tw);
        const glm = try span(a, try body(try tokens(a, try read(a, tw.glm_file)), tw.glm), tw);
        const renamed = try a.dupe([]const u8, ours);
        for (renamed) |*t| for (tw.subs) |s| {
            if (std.mem.eql(u8, t.*, s[0])) t.* = s[1];
        };
        const same = renamed.len == glm.len and for (renamed, glm) |x, y| {
            if (!std.mem.eql(u8, x, y)) break false;
        } else true;
        if (!same) std.debug.print("tf_mla::{s} no longer matches {s} in {s}\n", .{ tw.mla, tw.glm, tw.glm_file });
        try std.testing.expect(same);
        try std.testing.expect(glm.len > 100);
    }
}
