//! The widths of the tensors the engine reads straight from the checkpoint: experts, embedding, n-gram tables.
const std = @import("std");
const config = @import("config.zig");
const widths = @import("widths.zig");
const variants = @import("variants.zig");
const replay = @import("replay.zig");

/// Null wherever a tensor is the recorded 6/32: those keep their recorded kernels.
pub const Set = struct {
    layers: [config.max_layers]?variants.Experts = @splat(null),
    mtp: ?variants.Experts = null,
    embed: ?[2]usize = null,
    ple: ?[2]usize = null,
    /// Every quantized matrix 6/32: the uniform format, its packs pinned and its fast paths on.
    uniform: bool = true,
};

fn width(w: *const widths.Widths, path: []const u8) ![2]usize {
    const got = (try w.of(path)) orelse return error.UnquantizedAffineTensor;
    return .{ got.bits, got.group };
}

fn same(x: [2]usize, y: [2]usize) bool {
    return x[0] == y[0] and x[1] == y[1];
}

/// A layer's experts: gate and up share one width (one kernel reads both), routed and shared each their own.
pub fn experts(w: *const widths.Widths, stem: []const u8) !?variants.Experts {
    var buf: [192]u8 = undefined;
    var v: [6][2]usize = undefined;
    const projs = [_][]const u8{ "switch_mlp.gate_proj", "switch_mlp.up_proj", "shared_expert.gate_proj", "shared_expert.up_proj", "switch_mlp.down_proj", "shared_expert.down_proj" };
    for (projs, &v) |p, *out| out.* = try width(w, try std.fmt.bufPrint(&buf, "{s}.mlp.{s}", .{ stem, p }));
    if (!same(v[0], v[1]) or !same(v[2], v[3])) {
        std.log.warn("{s}: gate and up of one expert differ in width; one kernel reads both", .{stem});
        return error.ExpertGateUpWidths;
    }
    const all_six = for (v) |x| {
        if (!same(x, .{ 6, 32 })) break false;
    } else true;
    if (all_six) return null;
    return .{ .gu = .{ v[0][0], v[0][1], v[2][0], v[2][1] }, .down = .{ v[4][0], v[4][1], v[5][0], v[5][1] } };
}

fn orNull(x: [2]usize) ?[2]usize {
    return if (same(x, .{ 6, 32 })) null else x;
}

fn sixUniform(w: *const widths.Widths) bool {
    return w.uniform() and w.q.global.bits == 6 and w.q.global.group == 32;
}

/// Whether a checkpoint folder is the uniform 6/32 format, which check programs that read only it require.
pub fn uniformDir(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !bool {
    var c = try config.Config.read(gpa, io, dir);
    defer c.deinit();
    var w = try widths.Widths.init(gpa, &c);
    defer w.deinit();
    return sixUniform(&w);
}

/// Every directly read tensor's width from the checkpoint's config; `spelling` names the n-gram shards.
pub fn read(c: *const config.Config, w: *const widths.Widths, spelling: []const u8) !Set {
    var s: Set = .{};
    var buf: [192]u8 = undefined;
    for (0..c.layers) |i| s.layers[i] = try experts(w, try std.fmt.bufPrint(&buf, "language_model.model.layers.{d}", .{i}));
    if (c.mtp.layers > 0) s.mtp = try experts(w, "language_model.mtp.layers.0");
    s.embed = orNull(try width(w, "language_model.model.embed_tokens"));
    for (0..c.layers) |i| if (c.ple[i]) {
        const first = try width(w, try std.fmt.bufPrint(&buf, "language_model.model.layers.{d}.ple.ple_embedding.ngram_embedding.{s}0", .{ i, spelling }));
        for (1..c.ngram_shards) |k| {
            const other = try width(w, try std.fmt.bufPrint(&buf, "language_model.model.layers.{d}.ple.ple_embedding.ngram_embedding.{s}{d}", .{ i, spelling, k }));
            if (!same(first, other)) return error.NgramTableWidths; // one lookup kernel reads every shard
        }
        s.ple = orNull(first);
    };
    s.uniform = sixUniform(w);
    return s;
}

test {
    _ = @import("engine_widths_test.zig");
}

/// Widths from the engine's own config and index into `m` and the prompt, kernels compiled; true when all 6/32.
pub fn apply(a: std.mem.Allocator, c: *const config.Config, map: std.json.ObjectMap, p: *replay.Prompt, m: *replay.Model) !bool {
    var w = try widths.Widths.init(a, c);
    defer w.deinit();
    const ws = try read(c, &w, @import("index.zig").ngramSpelling(map));
    for (&m.layers, 0..) |*L, i| L.xw = ws.layers[i];
    m.mtp.xw = ws.mtp;
    m.embed_w = ws.embed;
    m.ple.w = ws.ple;
    try replay.prompt_w.attach(p, m);
    try @import("engine_warm.zig").all(p.r, m);
    return ws.uniform;
}
