//! A drafter release: config.json names the target, its adapter and taps, and the drafter's shape; files beside it.

const std = @import("std");
const Io = std.Io;
const safetensors = @import("safetensors.zig");

pub const format = "tensorfold-drafter";
pub const version: u32 = 1; // a release with a newer version is refused, never read in part

/// Why a check refused the release, kept for the caller's log line (tests read it instead).
pub const Why = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    pub fn set(self: *Why, comptime fmt: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = w.end;
    }

    pub fn text(self: *const Why) []const u8 {
        return self.buf[0..self.len];
    }
};

/// A tapped row's element type, as `lanes.Features` gives it and the recorder writes it.
pub const Dtype = enum {
    bf16,
    f16,
    f32,

    pub fn size(d: Dtype) u32 {
        return if (d == .f32) 4 else 2;
    }
};

/// Which residual stream a tap reads; a family with several (hyper-connection lanes) adds its merge here.
pub const Stream = enum { residual };

/// The rows the drafter reads, as the target's `features` gives them and the recorder writes them to `.st`.
pub const Taps = struct {
    ids: []const u32, // tap k: the stream after k of the target's layers (k = layers: past the last layer)
    stream: Stream,
    normed: bool, // by the norm that reads the row next: layer k's input norm, or the final norm at k = layers
    dtype: Dtype,
    row_bytes: u32, // the taps in `ids` order, each `target.hidden` wide
};

/// How the target turns tokens into rows and rows into logits; a drafter using its embedding and head does the same.
pub const Adapter = struct {
    layers: u32,
    hidden: u32,
    vocab: u32,
    embed_scale: f64, // rows = embed[token] x embed_scale
    tied: bool, // the head is the embedding: the release ships no `target.head`
    softcap: ?f64, // logits = softcap x tanh(logits / softcap); null: none
    head_bias: bool, // logits += a bias, shipped as `target.head_bias` for the draft vocabulary's rows
};

pub const Target = struct {
    repo: []const u8,
    revision: []const u8,
    layers: u32,
    hidden: u32,
    vocab: u32,
    embed_scale: f64,
    tied: bool,
    softcap: ?f64,
    head_bias: bool,

    pub fn adapter(t: Target) Adapter {
        return .{ .layers = t.layers, .hidden = t.hidden, .vocab = t.vocab, .embed_scale = t.embed_scale, .tied = t.tied, .softcap = t.softcap, .head_bias = t.head_bias };
    }
};

/// The block drafter's shape: row 0 a chain head on the anchor's own row, rows 1.. drafted in the same pass.
pub const Shape = struct {
    hidden: u32, // equals target.hidden: the target's head reads the drafter's rows
    heads: u32,
    head_dim: u32,
    ffn: u32,
    layers: u32,
    block: u32, // positions one pass drafts
    window: u32, // context rows an anchor attends to
    eps: f64,
    theta: f64, // RoPE base
    pred_rank: u32, // 0: no predecessor head
    spine_rank: u32, // 0: no spine
};

pub const Vocab = struct { file: []const u8, size: u32 }; // the target's ids the drafter proposes, in head-slice order
pub const Weight = enum { bf16, fp8, nvfp4 };
pub const Serving = struct { depth: u32, body: Weight, head: Weight }; // defaults a server may override

pub const Architecture = enum { @"block-drafter" };

/// config.json, field for field.
pub const Config = struct {
    format: []const u8,
    version: u32,
    architecture: Architecture,
    target: Target,
    taps: Taps,
    drafter: Shape,
    draft_vocab: Vocab,
    weights: []const u8,
    serving: Serving,
};

pub const Manifest = struct {
    arena: std.heap.ArenaAllocator,
    config: Config,

    /// config.json's text, read and checked; anything outside the format is refused with `why` naming it.
    pub fn parse(gpa: std.mem.Allocator, text: []const u8, why: *Why) !Manifest {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const c = std.json.parseFromSliceLeaky(Config, arena.allocator(), text, .{ .allocate = .alloc_always }) catch |e| {
            why.set("config.json does not parse as a drafter release ({s})", .{@errorName(e)});
            return error.BadDrafterRelease;
        };
        try validate(&c, why);
        return .{ .arena = arena, .config = c };
    }

    pub fn deinit(m: *Manifest) void {
        m.arena.deinit();
        m.* = undefined;
    }
};

/// config.json's text for `c`, as `parse` reads it back.
pub fn write(c: *const Config, w: *std.Io.Writer) !void {
    var j: std.json.Stringify = .{ .writer = w, .options = .{ .whitespace = .indent_1 } };
    try j.write(c.*);
}

/// The format's own rules, before any file or target is looked at.
pub fn validate(c: *const Config, why: *Why) !void {
    const t = c.target;
    const s = c.drafter;
    if (!std.mem.eql(u8, c.format, format)) return refuse(why, "format \"{s}\", not \"{s}\"", .{ c.format, format });
    if (c.version != version) return refuse(why, "version {d}; this build reads version {d}", .{ c.version, version });
    if (t.layers == 0 or t.hidden == 0 or t.vocab == 0) return refuse(why, "target layers, hidden and vocab must be positive", .{});
    if (!(t.embed_scale > 0) or !std.math.isFinite(t.embed_scale)) return refuse(why, "target embed_scale {d} is not a positive number", .{t.embed_scale});
    if (t.softcap) |cap| if (!(cap > 0) or !std.math.isFinite(cap)) return refuse(why, "target softcap {d} is not a positive number", .{cap});
    if (c.taps.ids.len == 0) return refuse(why, "no taps", .{});
    for (c.taps.ids, 0..) |id, i| {
        if (id > t.layers) return refuse(why, "tap {d} is past the target's {d} layers", .{ id, t.layers });
        if (i > 0 and id <= c.taps.ids[i - 1]) return refuse(why, "taps must rise: {d} after {d}", .{ id, c.taps.ids[i - 1] });
    }
    const row = @as(u64, c.taps.ids.len) * t.hidden * c.taps.dtype.size();
    if (c.taps.row_bytes != row) return refuse(why, "taps row_bytes {d}; {d} taps of {d} {s} are {d}", .{ c.taps.row_bytes, c.taps.ids.len, t.hidden, @tagName(c.taps.dtype), row });
    if (s.hidden != t.hidden) return refuse(why, "drafter hidden {d} differs from the target's {d}, whose head reads it", .{ s.hidden, t.hidden });
    if (s.heads == 0 or s.head_dim == 0 or s.head_dim % 2 != 0) return refuse(why, "drafter heads {d} x head_dim {d}: both positive, head_dim even", .{ s.heads, s.head_dim });
    if (s.ffn == 0 or s.layers == 0 or s.window == 0) return refuse(why, "drafter ffn, layers and window must be positive", .{});
    if (s.block < 2) return refuse(why, "drafter block {d}: a block drafts at least 2 positions", .{s.block});
    if (!(s.eps > 0) or !std.math.isFinite(s.eps)) return refuse(why, "drafter eps {d} is not a positive number", .{s.eps});
    if (!(s.theta > 0) or !std.math.isFinite(s.theta)) return refuse(why, "drafter theta {d} is not a positive number", .{s.theta});
    if (c.serving.depth == 0 or c.serving.depth > s.block) return refuse(why, "serving depth {d} outside 1..{d}", .{ c.serving.depth, s.block });
    if (c.draft_vocab.size == 0 or c.draft_vocab.size > t.vocab) return refuse(why, "draft vocab of {d} outside 1..{d}", .{ c.draft_vocab.size, t.vocab });
    for ([_][]const u8{ c.draft_vocab.file, c.weights }) |name| if (!plainName(name)) return refuse(why, "\"{s}\" is not a file name in the release folder", .{name});
}

/// The release fits the target the server loaded: every adapter field equal, else the first that differs is named.
pub fn check(c: *const Config, have: Adapter, why: *Why) !void {
    const want = c.target.adapter();
    inline for (@typeInfo(Adapter).@"struct".field_names) |f| if (!std.meta.eql(@field(want, f), @field(have, f))) {
        why.set("the release was trained against target {s} = {any}; the loaded target has {any}", .{ f, @field(want, f), @field(have, f) });
        return error.AdapterMismatch;
    };
}

/// A recording's `.json` has this release's taps, row bytes and dtype; one without rows has no layout to compare.
pub fn checkRecording(c: *const Config, gpa: std.mem.Allocator, text: []const u8, why: *Why) !void {
    const Rec = struct { rows: u64, taps: []const u32, row_bytes: u32, dtype: []const u8 };
    const r = std.json.parseFromSlice(Rec, gpa, text, .{ .ignore_unknown_fields = true }) catch |e| {
        why.set("the recording's .json does not parse ({s})", .{@errorName(e)});
        return error.BadRecording;
    };
    defer r.deinit();
    if (!std.mem.eql(u32, r.value.taps, c.taps.ids)) return refuseAs(error.RecordingMismatch, why, "recorded taps {any}, release taps {any}", .{ r.value.taps, c.taps.ids });
    if (r.value.rows == 0) return;
    if (r.value.row_bytes != c.taps.row_bytes) return refuseAs(error.RecordingMismatch, why, "recorded row_bytes {d}, release {d}", .{ r.value.row_bytes, c.taps.row_bytes });
    if (!std.mem.eql(u8, r.value.dtype, @tagName(c.taps.dtype))) return refuseAs(error.RecordingMismatch, why, "recorded dtype \"{s}\", release {s}", .{ r.value.dtype, @tagName(c.taps.dtype) });
}

/// draft_vocab.json: a JSON array of `size` distinct target ids, each below the target's vocab (caller frees).
pub fn parseVocab(c: *const Config, gpa: std.mem.Allocator, text: []const u8, why: *Why) ![]u32 {
    const parsed = std.json.parseFromSlice([]u32, gpa, text, .{}) catch |e| {
        why.set("{s} is not a JSON array of token ids ({s})", .{ c.draft_vocab.file, @errorName(e) });
        return error.BadDrafterRelease;
    };
    defer parsed.deinit();
    const ids = try gpa.dupe(u32, parsed.value);
    errdefer gpa.free(ids);
    if (ids.len != c.draft_vocab.size) return refuse(why, "{s} has {d} ids; config.json says {d}", .{ c.draft_vocab.file, ids.len, c.draft_vocab.size });
    var seen = try std.DynamicBitSetUnmanaged.initEmpty(gpa, c.target.vocab);
    defer seen.deinit(gpa);
    for (ids) |id| {
        if (id >= c.target.vocab) return refuse(why, "draft id {d} is past the target's vocab of {d}", .{ id, c.target.vocab });
        if (seen.isSet(id)) return refuse(why, "draft id {d} appears twice", .{id});
        seen.set(id);
    }
    return ids;
}

fn plainName(name: []const u8) bool {
    if (name.len == 0 or std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) return false;
    return std.mem.indexOfAny(u8, name, "/\\") == null;
}

fn refuse(why: *Why, comptime fmt: []const u8, args: anytype) error{BadDrafterRelease} {
    return refuseAs(error.BadDrafterRelease, why, fmt, args);
}

fn refuseAs(err: anytype, why: *Why, comptime fmt: []const u8, args: anytype) @TypeOf(err) {
    why.set(fmt, args);
    return err;
}
