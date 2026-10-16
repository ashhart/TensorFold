//! A block-drafter release's tensors: the names and shapes its config.json implies, and a release folder read whole.

const std = @import("std");
const Io = std.Io;
const safetensors = @import("safetensors.zig");
const release = @import("drafter_release.zig");
const Config = release.Config;
const Why = release.Why;

pub const max_rank = 2;

/// One tensor the release must hold, with its shape (rows, columns as a linear layer's weight stores them).
pub const Want = struct { name: []const u8, rank: u8, shape: [max_rank]usize };

/// Every tensor `c` implies, in a fixed order; names live in `a`.
pub fn expected(a: std.mem.Allocator, c: *const Config) ![]Want {
    var out: std.ArrayList(Want) = .empty;
    const s = c.drafter;
    const t = c.target;
    const d: usize = s.hidden;
    const att: usize = @as(usize, s.heads) * s.head_dim;
    const v: usize = c.draft_vocab.size;
    try out.append(a, two("target.embed", t.vocab, t.hidden)); // the whole embedding: context rows read any token
    if (!t.tied) try out.append(a, two("target.head", v, t.hidden)); // the head's draft-vocab rows, in vocab order
    if (t.head_bias) try out.append(a, one("target.head_bias", v));
    try out.append(a, two("fuse.weight", d, c.taps.ids.len * @as(usize, t.hidden))); // the taps, concatenated
    try out.append(a, two("fc.weight", d, @as(usize, t.hidden) + d)); // cat(embed(token after the row), fused row)
    try out.append(a, one("mask", d)); // block rows 1.. start from it
    for (0..s.layers) |i| {
        for ([_][]const u8{ "q", "k", "v", "ck", "cv" }) |p| try out.append(a, two(try layer(a, i, p, "weight"), att, d));
        try out.append(a, two(try layer(a, i, "o", "weight"), d, att));
        try out.append(a, two(try layer(a, i, "gate", "weight"), s.ffn, d));
        try out.append(a, two(try layer(a, i, "up", "weight"), s.ffn, d));
        try out.append(a, two(try layer(a, i, "down", "weight"), d, s.ffn));
        for ([_][]const u8{ "n1", "n2", "nc" }) |p| try out.append(a, one(try layer(a, i, p, "w"), d));
    }
    try out.append(a, one("out_norm.w", d));
    if (s.pred_rank > 0) {
        try out.append(a, two("pred_h.weight", s.pred_rank, d));
        try out.append(a, two("pred_e.weight", s.pred_rank, t.hidden));
        try out.append(a, two("pred_out.weight", d, s.pred_rank));
    }
    if (s.spine_rank > 0) for (0..s.layers - 1) |i| {
        try out.append(a, one(try std.fmt.allocPrint(a, "spine_n.{d}.w", .{i}), d));
        try out.append(a, two(try std.fmt.allocPrint(a, "spine_a.{d}.weight", .{i}), s.spine_rank, d));
        try out.append(a, two(try std.fmt.allocPrint(a, "spine_u.{d}.weight", .{i}), d, s.spine_rank));
    };
    return out.items;
}

/// The weights' header holds exactly the tensors `c` implies, each float (bf16, f16 or f32) at its shape.
pub fn checkWeights(a: std.mem.Allocator, c: *const Config, header: safetensors.Header, why: *Why) !void {
    const want = try expected(a, c);
    for (want) |w| {
        const e = header.get(w.name) orelse return refuse(why, "{s} has no {s}", .{ c.weights, w.name });
        if (e.dtype != .bf16 and e.dtype != .f16 and e.dtype != .f32) return refuse(why, "{s} is {s}; the format stores bf16, f16 or f32", .{ w.name, @tagName(e.dtype) });
        if (e.rank != w.rank or !std.mem.eql(usize, e.shape[0..e.rank], w.shape[0..w.rank])) return refuse(why, "{s} is {any}; config.json implies {any}", .{ w.name, e.shape[0..e.rank], w.shape[0..w.rank] });
    }
    if (header.count() != want.len) for (header.keys()) |name| {
        for (want) |w| {
            if (std.mem.eql(u8, w.name, name)) break;
        } else return refuse(why, "{s} holds {s}, which the format does not name", .{ c.weights, name });
    };
}

/// A release folder read and checked whole: config.json, the draft vocabulary and the weights' header.
pub const Release = struct {
    manifest: release.Manifest,
    vocab: []u32,

    pub fn open(gpa: std.mem.Allocator, io: Io, dir: Io.Dir, why: *Why) !Release {
        const text = try dir.readFileAlloc(io, "config.json", gpa, .limited(1 << 20));
        defer gpa.free(text);
        var m = try release.Manifest.parse(gpa, text, why);
        errdefer m.deinit();
        const c = &m.config;
        const vt = try dir.readFileAlloc(io, c.draft_vocab.file, gpa, .limited(64 << 20));
        defer gpa.free(vt);
        const vocab = try release.parseVocab(c, gpa, vt, why);
        errdefer gpa.free(vocab);
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        try checkWeights(arena.allocator(), c, try readHeader(arena.allocator(), io, dir, c.weights, why), why);
        return .{ .manifest = m, .vocab = vocab };
    }

    pub fn deinit(r: *Release, gpa: std.mem.Allocator) void {
        gpa.free(r.vocab);
        r.manifest.deinit();
        r.* = undefined;
    }
};

/// A safetensors file's header, read without mapping its data.
fn readHeader(a: std.mem.Allocator, io: Io, dir: Io.Dir, name: []const u8, why: *Why) !safetensors.Header {
    var f = try dir.openFile(io, name, .{});
    defer f.close(io);
    const len = try f.length(io);
    var head: [8]u8 = undefined;
    if (len < 8 or try f.readPositionalAll(io, &head, 0) != 8) return refuse(why, "{s} is too short for safetensors", .{name});
    const n = std.mem.readInt(u64, &head, .little);
    if (n > len - 8 or n > 64 << 20) return refuse(why, "{s} has a header of {d} bytes in {d}", .{ name, n, len });
    const json = try a.alloc(u8, @intCast(n));
    if (try f.readPositionalAll(io, json, 8) != n) return refuse(why, "{s} ended inside its header", .{name});
    return safetensors.parseHeader(a, json, @intCast(len - 8 - n)) catch |e| return refuse(why, "{s}'s header is refused ({s})", .{ name, @errorName(e) });
}

fn layer(a: std.mem.Allocator, i: usize, part: []const u8, kind: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "layers.{d}.{s}.{s}", .{ i, part, kind });
}

fn one(name: []const u8, n: usize) Want {
    return .{ .name = name, .rank = 1, .shape = .{ n, 0 } };
}

fn two(name: []const u8, rows: usize, cols: usize) Want {
    return .{ .name = name, .rank = 2, .shape = .{ rows, cols } };
}

fn refuse(why: *Why, comptime fmt: []const u8, args: anytype) error{BadDrafterRelease} {
    why.set(fmt, args);
    return error.BadDrafterRelease;
}
