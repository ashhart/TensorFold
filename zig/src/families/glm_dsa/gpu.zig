//! A packed GLM-5.3 layer on the GPU (bytes, tables, jobs, dense groups) and the window scratch the layers share.
const std = @import("std");
const mtl = @import("metal");
const ex = @import("../../core/exlr8.zig");
const op = @import("../../core/exlr8_projection.zig");
const dg = @import("../../core/exlr8_dense_gpu.zig");
const cfg = @import("config.zig");
const kn = @import("kernels.zig");
const pack = ex.pack;
const Ref = kn.Ref;

fn upload(device: mtl.Device, bytes: []const u8) !mtl.Buffer {
    const b = try device.buffer(@max(bytes.len, 16), mtl.ResourceOptions.shared);
    @memcpy(b.contents()[0..bytes.len], bytes);
    return b;
}

/// A run of dense groups in the layer's group buffers: 32-row blocks, then 8-row blocks for 1-4 rows.
pub const Groups = struct { off: usize, count: usize, off8: usize = 0, count8: usize = 0 };

pub const Packed = struct {
    bytes: mtl.Buffer,
    halves: mtl.Buffer,
    signs: mtl.Buffer,
    bases: mtl.Buffer,
    dense_jobs: mtl.Buffer,
    jobs: mtl.Buffer,
    groups: mtl.Buffer,
    groups8: mtl.Buffer,
    group_list: std.ArrayList([2]u32) = .empty,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, p: *const pack.Pack, dispatches: []const pack.Range, out: []Groups) !Packed {
        var k: Packed = undefined;
        k.group_list = .empty;
        var list8: std.ArrayList([2]u32) = .empty;
        defer list8.deinit(gpa);
        for (dispatches, out) |r, *g| {
            g.off = k.group_list.items.len * 8;
            g.off8 = list8.items.len * 8;
            for (r.lo..r.hi) |j| {
                for (0..(p.dense_jobs.items[j].outs + 31) / 32) |b| try k.group_list.append(gpa, .{ @intCast(j), @intCast(b) });
                for (0..(p.dense_jobs.items[j].outs + 7) / 8) |b| try list8.append(gpa, .{ @intCast(j), @intCast(b) });
            }
            g.count = k.group_list.items.len - g.off / 8;
            g.count8 = list8.items.len - g.off8 / 8;
        }
        k.bytes = try upload(device, p.bytes.items);
        k.halves = try upload(device, std.mem.sliceAsBytes(p.halves.items));
        k.signs = try upload(device, std.mem.sliceAsBytes(p.signs.items));
        k.bases = try upload(device, std.mem.sliceAsBytes(p.bases.items));
        k.dense_jobs = try upload(device, std.mem.sliceAsBytes(p.dense_jobs.items));
        k.jobs = try upload(device, std.mem.sliceAsBytes(p.jobs.items));
        k.groups = try upload(device, std.mem.sliceAsBytes(k.group_list.items));
        k.groups8 = try upload(device, std.mem.sliceAsBytes(list8.items));
        return k;
    }

    pub fn deinit(k: *Packed, gpa: std.mem.Allocator) void {
        inline for (.{ k.bytes, k.halves, k.signs, k.bases, k.dense_jobs, k.jobs, k.groups, k.groups8 }) |b| b.deinit();
        k.group_list.deinit(gpa);
    }

    pub fn dense(k: *const Packed, g: Groups) dg.Tables {
        return .{ .weights = k.bytes, .jobs = k.dense_jobs, .groups = k.groups, .count = g.count, .groups_off = g.off };
    }

    /// `dense` for a window of `rows`: a row a simdgroup for 1-4 rows (the same sums, four times the simdgroups).
    pub fn denseFor(k: *const Packed, g: Groups, rows: u32) dg.Tables {
        if (rows > dg.narrow_rows) return k.dense(g);
        return .{ .weights = k.bytes, .jobs = k.dense_jobs, .groups = k.groups8, .count = g.count8, .groups_off = g.off8, .narrow = true };
    }

    pub fn trellis(k: *const Packed, t: pack.Trellis) op.Tables {
        return .{ .weights = k.bytes, .bases = k.bases, .halves = k.halves, .signs = k.signs, .jobs = k.jobs, .count = t.jobs.hi - t.jobs.lo, .jobs_off = t.jobs.lo * @sizeOf(ex.job.Job), .blocks = t.in / 128 };
    }

    pub fn at(k: *const Packed, off: u64) Ref {
        return .{ .buf = k.bytes, .off = off };
    }
};

/// The most rows a window takes, and the picks a row routes (eight experts and the shared one).
pub const max_rows = 16;
pub const slots = 9;
/// A window's distinct experts at most (every pick its own, and the shared one), and the jobs before the down ones.
pub const max_experts = max_rows * 8 + 1;
pub const gu_cap = max_experts * 2 * 16;

/// The rows a prompt window takes at most: attention runs them 16 at a time, the experts all at once.
pub const prompt_rows = 512;

/// A prompt window's buffers beside the 16-row scratch: its rows' hidden states and the experts' work for all of them.
pub const Prompt = struct {
    h: mtl.Buffer,
    x: mtl.Buffer,
    rot: mtl.Buffer,
    proj: mtl.Buffer,
    q_resid: mtl.Buffer,
    qq: mtl.Buffer,
    qidx: mtl.Buffer,
    olat: mtl.Buffer,
    o: mtl.Buffer,
    keys: mtl.Buffer,
    score: mtl.Buffer,
    logits: mtl.Buffer,
    rows: mtl.Buffer,
    ids: mtl.Buffer,
    /// The MTP head's rows and tokens over the same window (one command buffer runs both).
    rows_mtp: mtl.Buffer,
    ids_mtp: mtl.Buffer,
    picks: mtl.Buffer,
    /// 0, 1, 2, ...: a gather's indices for rows in order.
    idx: mtl.Buffer,
    pick: mtl.Buffer,
    wts8: mtl.Buffer,
    gu: mtl.Buffer,
    act: mtl.Buffer,
    arot: mtl.Buffer,
    z: mtl.Buffer,
    wts: mtl.Buffer,
    sv: mtl.Buffer,
    suh: mtl.Buffer,
    jobs: mtl.Buffer,
    members: mtl.Buffer,
    dispatch: mtl.Buffer,
    /// Expert tiles of 32 rows a window makes at most, and the gate/up jobs before the down ones.
    tiles: u32,
    gu_cap: u32,

    pub fn init(device: mtl.Device, c: *const cfg.Config, context: u32) !Prompt {
        const P: usize = prompt_rows;
        const nh: usize = c.localHeads();
        const S = P * slots;
        const loc: usize = c.localHalves() * 128;
        // each picked expert's rows in tiles of 32: at most a tile an expert plus one a 32 picks (the shared one's too)
        const tiles: u32 = @intCast((P * (c.topk + 1) + 31) / 32 + c.experts + 2);
        const cap = tiles * 2 * 16;
        var q: Prompt = undefined;
        q.tiles = tiles;
        q.gu_cap = cap;
        const sizes = .{
            .{ &q.h, P * c.hidden * 4 },                 .{ &q.x, P * c.hidden * 4 },                                                    .{ &q.rot, P * 16384 * 4 },
            .{ &q.proj, P * 2784 * 4 },                  .{ &q.q_resid, P * c.q_lora * 4 },                                              .{ &q.qq, P * nh * c.qHead() * 4 },
            .{ &q.qidx, P * c.i_heads * c.i_dim * 4 },   .{ &q.olat, P * nh * c.kv_lora * 4 },                                           .{ &q.o, P * nh * c.v_dim * 4 },
            .{ &q.keys, P * c.i_topk * 4 },              .{ &q.score, P * @as(usize, context) * 4 },                                     .{ &q.logits, P * c.experts * 4 },
            .{ &q.rows, P * @sizeOf(kn.Row) },           .{ &q.ids, P * 4 },                                                             .{ &q.pick, P * c.topk * 4 },
            .{ &q.rows_mtp, P * @sizeOf(kn.Row) },       .{ &q.ids_mtp, P * 4 },                                                         .{ &q.picks, P * 4 },
            .{ &q.idx, P * 4 },                          .{ &q.wts8, P * c.topk * 4 },                                                   .{ &q.gu, S * 2 * loc * 4 },
            .{ &q.act, S * loc * 4 },                    .{ &q.arot, S * loc * 4 },                                                      .{ &q.z, S * c.hidden * 4 },
            .{ &q.wts, S * 4 },                          .{ &q.sv, S * 4 },                                                              .{ &q.suh, S * 4 },
            .{ &q.members, @as(usize, tiles) * 64 * 4 }, .{ &q.jobs, (@as(usize, cap) + @as(usize, tiles) * 48) * @sizeOf(ex.job.Job) }, .{ &q.dispatch, 32 },
        };
        inline for (sizes) |e| e[0].* = try device.buffer(@max(e[1], 16), mtl.ResourceOptions.shared);
        for (q.idx.slice(u32, P), 0..) |*v, i| v.* = @intCast(i);
        return q;
    }

    pub fn deinit(q: *Prompt) void {
        const info = @typeInfo(Prompt).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            if (T == mtl.Buffer) @field(q, name).deinit();
        }
    }
};

/// Activations for one window of up to 16 rows, sized for one Mac's share of the model.
pub const Scratch = struct {
    h: mtl.Buffer,
    x: mtl.Buffer,
    proj: mtl.Buffer,
    q_resid: mtl.Buffer,
    rot: mtl.Buffer,
    q: mtl.Buffer,
    qidx: mtl.Buffer,
    qlat: mtl.Buffer,
    qrope: mtl.Buffer,
    score: mtl.Buffer,
    keys: mtl.Buffer,
    rows: mtl.Buffer,
    partial: mtl.Buffer,
    olat: mtl.Buffer,
    o: mtl.Buffer,
    y: mtl.Buffer,
    logits: mtl.Buffer,
    route: [9]mtl.Buffer,
    gu: mtl.Buffer,
    act: mtl.Buffer,
    arot: mtl.Buffer,
    z: mtl.Buffer,
    wts: mtl.Buffer,
    sv: mtl.Buffer,
    suh: mtl.Buffer,
    moe_jobs: mtl.Buffer,
    members: mtl.Buffer,
    dispatch: mtl.Buffer,
    context: u32,
    /// The rows table's first entry for this window (a prompt window's 16-row part).
    rows_off: usize = 0,

    pub fn init(device: mtl.Device, c: *const cfg.Config, context: u32) !Scratch {
        const R: usize = max_rows;
        const nh: usize = c.localHeads();
        const P = R * slots;
        const f = 4;
        var s: Scratch = undefined;
        s.context = context;
        s.rows_off = 0;
        const sizes = .{
            .{ &s.h, R * c.hidden * f },                                         .{ &s.x, R * c.hidden * f },              .{ &s.proj, R * 2784 * f },
            .{ &s.q_resid, R * c.q_lora * f },                                   .{ &s.rot, R * 16384 * f },               .{ &s.q, R * nh * c.qHead() * f },
            .{ &s.qidx, R * c.i_heads * c.i_dim * f },                           .{ &s.qlat, R * nh * c.kv_lora * f },     .{ &s.qrope, R * nh * c.rope * f },
            .{ &s.score, R * @as(usize, context) * f },                          .{ &s.keys, R * c.i_topk * 4 },           .{ &s.rows, R * @sizeOf(kn.Row) },
            .{ &s.partial, R * nh * kn.splits * 514 * f },                       .{ &s.olat, R * nh * c.kv_lora * f },     .{ &s.o, R * nh * c.v_dim * f },
            .{ &s.y, 2 * R * c.hidden * f },                                     .{ &s.logits, R * c.experts * f },        .{ &s.gu, P * 2 * c.moe_inter * f },
            .{ &s.act, P * c.moe_inter * f },                                    .{ &s.arot, P * c.moe_inter * f },        .{ &s.z, P * c.hidden * f },
            .{ &s.wts, P * f },                                                  .{ &s.sv, P * 4 },                        .{ &s.suh, P * 4 },
            .{ &s.moe_jobs, (gu_cap + max_experts * 48) * @sizeOf(ex.job.Job) }, .{ &s.members, max_experts * 2 * R * 4 }, .{ &s.dispatch, 32 },
        };
        inline for (sizes) |e| e[0].* = try device.buffer(@max(e[1], 16), mtl.ResourceOptions.shared);
        for (&s.route) |*b| b.* = try device.buffer(64 << 10, mtl.ResourceOptions.shared);
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        const info = @typeInfo(Scratch).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            if (T == mtl.Buffer) @field(s, name).deinit();
        }
        for (s.route) |b| b.deinit();
    }

    pub fn ref(_: *const Scratch, b: mtl.Buffer) Ref {
        return .{ .buf = b };
    }
};
