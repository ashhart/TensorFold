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

/// A run of dense groups in the layer's group buffer.
pub const Groups = struct { off: usize, count: usize };

pub const Packed = struct {
    bytes: mtl.Buffer,
    halves: mtl.Buffer,
    signs: mtl.Buffer,
    bases: mtl.Buffer,
    dense_jobs: mtl.Buffer,
    jobs: mtl.Buffer,
    groups: mtl.Buffer,
    group_list: std.ArrayList([2]u32) = .empty,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, p: *const pack.Pack, dispatches: []const pack.Range, out: []Groups) !Packed {
        var k: Packed = undefined;
        k.group_list = .empty;
        for (dispatches, out) |r, *g| {
            g.off = k.group_list.items.len * 8;
            for (r.lo..r.hi) |j| for (0..(p.dense_jobs.items[j].outs + 31) / 32) |b| try k.group_list.append(gpa, .{ @intCast(j), @intCast(b) });
            g.count = k.group_list.items.len - g.off / 8;
        }
        k.bytes = try upload(device, p.bytes.items);
        k.halves = try upload(device, std.mem.sliceAsBytes(p.halves.items));
        k.signs = try upload(device, std.mem.sliceAsBytes(p.signs.items));
        k.bases = try upload(device, std.mem.sliceAsBytes(p.bases.items));
        k.dense_jobs = try upload(device, std.mem.sliceAsBytes(p.dense_jobs.items));
        k.jobs = try upload(device, std.mem.sliceAsBytes(p.jobs.items));
        k.groups = try upload(device, std.mem.sliceAsBytes(k.group_list.items));
        return k;
    }

    pub fn deinit(k: *Packed, gpa: std.mem.Allocator) void {
        inline for (.{ k.bytes, k.halves, k.signs, k.bases, k.dense_jobs, k.jobs, k.groups }) |b| b.deinit();
        k.group_list.deinit(gpa);
    }

    pub fn dense(k: *const Packed, g: Groups) dg.Tables {
        return .{ .weights = k.bytes, .jobs = k.dense_jobs, .groups = k.groups, .count = g.count, .groups_off = g.off };
    }

    pub fn trellis(k: *const Packed, t: pack.Trellis) op.Tables {
        return .{ .weights = k.bytes, .bases = k.bases, .halves = k.halves, .signs = k.signs, .jobs = k.jobs, .count = t.jobs.hi - t.jobs.lo, .jobs_off = t.jobs.lo * @sizeOf(ex.job.Job) };
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

    pub fn init(device: mtl.Device, c: *const cfg.Config, context: u32) !Scratch {
        const R: usize = max_rows;
        const nh: usize = c.localHeads();
        const P = R * slots;
        const f = 4;
        var s: Scratch = undefined;
        s.context = context;
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
