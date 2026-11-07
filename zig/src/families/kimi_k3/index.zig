//! K3.1's experimental indexer launch contract; the native 2047-position ceiling remains unchanged.
const std = @import("std");
const mtl = @import("metal");
const Kn = @import("kernels.zig");
const Config = @import("config.zig").Config;
const Indexer = @import("weights.zig").Indexer;
const Ref = @import("kernels.zig").Ref;
const MlaArgs = @import("kernels.zig").MlaArgs;

pub const source = @import("kimi_index_sources").index;
pub const attention_source = @import("kimi_index_sources").attention;
pub const dense_source = @import("kimi_index_sources").dense;
pub const common_source = @import("kimi_index_sources").common;

pub const heads = 32;
pub const dim = 128;
pub const topk = 2048;
pub const Row = extern struct { slot: u32, pos: u32, list: u32, nkeys: u32 = 0 };
pub const Args = extern struct {
    rows: u32,
    slot_keys: u32,
    q_stride: u32 = heads * dim,
    k_stride: u32 = dim,
    w_stride: u32 = heads,
    score_stride: u32,
    eps: f32 = 1e-5,
    key_scale: f32 = 1.0 / @sqrt(@as(f32, dim)),
    head_scale: f32 = 1.0 / @sqrt(@as(f32, heads)),
};

pub const Pipelines = struct {
    key: mtl.Pipeline,
    scores: mtl.Pipeline,
    select: mtl.Pipeline,
    attend: mtl.Pipeline,
    merge: mtl.Pipeline,

    pub fn load(device: mtl.Device, enabled: bool) !?Pipelines {
        if (!enabled) return null;
        const library = try mtl.Library.fromSource(device, source ++ "\n" ++ attention_source, mtl.CompileOptions.mlx());
        defer library.deinit();
        return try Pipelines.init(device, library);
    }

    pub fn init(device: mtl.Device, library: mtl.Library) !Pipelines {
        const key = try mtl.Pipeline.init(device, library, "k3_index_key", false);
        errdefer key.deinit();
        const scores = try mtl.Pipeline.init(device, library, "k3_index_scores", false);
        errdefer scores.deinit();
        const select = try mtl.Pipeline.init(device, library, "k3_index_topk", false);
        errdefer select.deinit();
        const attend = try mtl.Pipeline.init(device, library, "k3_index_attend", false);
        errdefer attend.deinit();
        return .{ .key = key, .scores = scores, .select = select, .attend = attend, .merge = try mtl.Pipeline.init(device, library, "k3_index_merge", false) };
    }

    pub fn deinit(p: *Pipelines) void {
        p.merge.deinit();
        p.attend.deinit();
        p.select.deinit();
        p.scores.deinit();
        p.key.deinit();
    }

    pub fn store(p: *const Pipelines, e: mtl.ComputeEncoder, key_proj: Ref, norm: Ref, bias: Ref, rows: Ref, cache: Ref, a: Args) void {
        e.setPipeline(p.key);
        bind(e, .{ key_proj, norm, bias, rows });
        e.setValue(a, 4);
        e.setBuffer(cache.buf, cache.off, 5);
        e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(32, 1, 1));
    }

    pub fn score(p: *const Pipelines, e: mtl.ComputeEncoder, q: Ref, head_weights: Ref, cache: Ref, rows: Ref, scores: Ref, a: Args, visible: u32) void {
        e.setPipeline(p.scores);
        bind(e, .{ q, head_weights, cache, rows });
        e.setValue(a, 4);
        e.setBuffer(scores.buf, scores.off, 5);
        e.dispatchGroups(mtl.Size.of((visible + 127) / 128, a.rows, 1), mtl.Size.of(128, 1, 1));
    }

    pub fn top(p: *const Pipelines, e: mtl.ComputeEncoder, scores: Ref, rows: Ref, list: Ref, a: Args) void {
        e.setPipeline(p.select);
        bind(e, .{ scores, rows });
        e.setValue(a, 2);
        e.setBuffer(list.buf, list.off, 3);
        e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(1024, 1, 1));
    }

    pub fn attention(p: *const Pipelines, e: mtl.ComputeEncoder, q: Ref, qlat: Ref, rows: Ref, list: Ref, partial: Ref, olat: Ref, a: MlaArgs) void {
        e.setPipeline(p.attend);
        bind(e, .{ q, qlat, rows, partial });
        e.setValue(a, 4);
        e.setBuffer(list.buf, list.off, 5);
        const fit: u32 = @intCast(@min(32, p.attend.maxThreads() / 32));
        const groups = (a.heads + fit - 1) / fit;
        const per = (a.heads + groups - 1) / groups;
        e.dispatchGroups(mtl.Size.of(groups, 16, a.rows), mtl.Size.of(32 * per, 1, 1));
        e.barrier();
        e.setPipeline(p.merge);
        bind(e, .{ partial, rows, olat });
        e.setValue(a, 3);
        e.dispatchGroups(mtl.Size.of(a.heads, a.rows, 1), mtl.Size.of(256, 1, 1));
    }
};

fn bind(e: mtl.ComputeEncoder, refs: anytype) void {
    inline for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
}

/// Independent index projections and lists never alias MLA or residual buffers.
pub const Work = struct {
    capacity: u32,
    most: u32,
    count: u32 = 0,
    visible: u32 = 0,
    q: mtl.Buffer,
    k: mtl.Buffer,
    weights: mtl.Buffer,
    scores: mtl.Buffer,
    rows: mtl.Buffer,
    list: mtl.Buffer,

    pub fn init(device: mtl.Device, most: u32, capacity: u32) !Work {
        if (most == 0 or most > 256 or capacity == 0 or capacity > 65536) return error.KimiIndexCapacity;
        var w: Work = undefined;
        w.capacity = capacity;
        w.most = most;
        w.count = 0;
        w.visible = 0;
        const sizes = .{
            .{ "q", @as(usize, most) * heads * dim * 2 }, .{ "k", @as(usize, most) * dim * 2 },
            .{ "weights", @as(usize, most) * heads * 2 }, .{ "scores", @as(usize, most) * capacity * 4 },
            .{ "rows", @as(usize, most) * @sizeOf(Row) }, .{ "list", @as(usize, most) * topk * 4 },
        };
        var made: usize = 0;
        errdefer inline for (sizes, 0..) |entry, i| {
            if (i < made) @field(w, entry[0]).deinit();
        };
        inline for (sizes) |entry| {
            @field(w, entry[0]) = try device.buffer(entry[1], mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            made += 1;
        }
        return w;
    }

    pub fn deinit(w: *Work) void {
        inline for (@typeInfo(Work).@"struct".field_names, @typeInfo(Work).@"struct".field_types) |name, T| {
            if (T == mtl.Buffer) @field(w, name).deinit();
        }
    }

    pub fn buffers(w: *const Work, out: *std.ArrayList(mtl.Buffer), a: std.mem.Allocator) !void {
        inline for (@typeInfo(Work).@"struct".field_names, @typeInfo(Work).@"struct".field_types) |name, T| {
            if (T == mtl.Buffer) try out.append(a, @field(w.*, name));
        }
    }

    pub fn ref(w: *const Work, comptime name: []const u8) Ref {
        return .{ .buf = @field(w.*, name) };
    }

    pub fn load(w: *Work, entries: []const Kn.MlaRow) !void {
        if (entries.len == 0 or entries.len > w.most) return error.KimiIndexCapacity;
        const rows = w.rows.slice(Row, entries.len);
        w.visible = 0;
        for (entries, rows, 0..) |s, *row, i| {
            if (s.pos >= w.capacity) return error.KimiIndexCapacity;
            row.* = .{ .slot = s.slot, .pos = s.pos, .list = @intCast(i * topk) };
            w.visible = @max(w.visible, s.pos + 1);
        }
        w.count = @intCast(entries.len);
    }

    /// These query/key/head-weight taps and BF16 roundings are model assumptions, not release-reference facts.
    pub fn project(w: *Work, k: *const Kn.Kernels, e: mtl.ComputeEncoder, c: *const Config, iw: Indexer, qan: Ref, xin: Ref, part: Ref, cache: Ref) void {
        k.rows(e, qan, iw.q.ref, w.ref("q"), part, false, .{ .K = c.q_lora, .N = heads * dim, .rows = w.count, .x_stride = c.q_lora, .y_stride = heads * dim });
        k.rows(e, xin, iw.k.ref, w.ref("k"), part, false, .{ .K = c.hidden, .N = dim, .rows = w.count, .x_stride = c.hidden, .y_stride = dim });
        k.rows(e, xin, iw.weights.ref, w.ref("weights"), part, false, .{ .K = c.hidden, .N = heads, .rows = w.count, .x_stride = c.hidden, .y_stride = heads });
        const a: Args = .{ .rows = w.count, .slot_keys = w.capacity, .score_stride = w.capacity, .eps = c.eps };
        const p = k.indexView();
        p.store(e, w.ref("k"), iw.norm.ref, iw.bias.ref, w.ref("rows"), cache, a);
        e.barrier();
        if (w.visible > topk) {
            p.score(e, w.ref("q"), w.ref("weights"), cache, w.ref("rows"), w.ref("scores"), a, w.visible);
            e.barrier();
        }
        p.top(e, w.ref("scores"), w.ref("rows"), w.ref("list"), a);
        e.barrier();
    }
};

pub const Span = struct { count: u32, indexed: bool };

/// Each row retains its solo attention path even when a peer has a longer history.
pub fn span(entries: []const Kn.MlaRow, first: usize, enabled: bool) Span {
    const indexed = enabled and entries[first].pos >= topk;
    var end = first + 1;
    while (end < entries.len and (enabled and entries[end].pos >= topk) == indexed) : (end += 1) {}
    return .{ .count = @intCast(end - first), .indexed = indexed };
}

test "Kimi index row path is independent of longer peers at the top-k boundary" {
    const entries = [_]Kn.MlaRow{
        .{ .slot = 0, .pos = 16 },   .{ .slot = 1, .pos = topk - 1 },
        .{ .slot = 1, .pos = topk }, .{ .slot = 1, .pos = topk + 1 },
        .{ .slot = 2, .pos = 31 },
    };
    try std.testing.expectEqual(Span{ .count = 2, .indexed = false }, span(&entries, 0, true));
    try std.testing.expectEqual(Span{ .count = 2, .indexed = true }, span(&entries, 2, true));
    try std.testing.expectEqual(Span{ .count = 1, .indexed = false }, span(&entries, 4, true));
    for (entries) |row| {
        const alone = span(&.{row}, 0, true).indexed;
        const before = [_]Kn.MlaRow{ row, entries[3] };
        const after = [_]Kn.MlaRow{ entries[3], row };
        try std.testing.expectEqual(alone, span(&before, 0, true).indexed);
        try std.testing.expectEqual(alone, span(&after, 1, true).indexed);
    }
    try std.testing.expectEqual(Span{ .count = 5, .indexed = false }, span(&entries, 0, false));
}

test "Kimi index pipelines stay absent without DSA" {
    try std.testing.expect((try Pipelines.load(undefined, false)) == null);
}
