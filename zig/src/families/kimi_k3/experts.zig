//! Routed experts behind one interface: this node runs the experts it owns; the cluster layer adds the rest.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("config.zig").Config;
const kernels = @import("kernels.zig");
const Scratch = @import("round.zig").Scratch;

/// The MoE layer's call: from sc.lat (latent rows), sc.tids and sc.tw (routes), write sc.ysum (bf16 latent sum).
pub const Experts = struct {
    ptr: *anyopaque,
    runFn: *const fn (ptr: *anyopaque, e: mtl.ComputeEncoder, table: mtl.Buffer, sc: *Scratch) anyerror!void,

    pub fn run(x: Experts, e: mtl.ComputeEncoder, table: mtl.Buffer, sc: *Scratch) !void {
        return x.runFn(x.ptr, e, table, sc);
    }
};

/// Experts [first, last) on this GPU (whole canonical groups of experts/8): plan by expert, gate|up, down, combine.
pub const Local = struct {
    k: *const kernels.Kernels,
    c: *const Config,
    first: u32 = 0,
    last: u32,
    /// fp32 partial sums of a node's expert groups (the cluster adds them in the groups' tree); bf16 when alone.
    partial: ?mtl.Buffer = null,

    pub fn all(k: *const kernels.Kernels, c: *const Config) Local {
        return .{ .k = k, .c = c, .last = c.experts };
    }

    pub fn experts(l: *Local) Experts {
        return .{ .ptr = l, .runFn = runFn };
    }

    fn runFn(ptr: *anyopaque, e: mtl.ComputeEncoder, table: mtl.Buffer, sc: *Scratch) anyerror!void {
        const l: *Local = @ptrCast(@alignCast(ptr));
        const c = l.c;
        const k = l.k;
        const pairs = sc.rows * c.topk;
        k.plan(e, sc.ref("tids"), sc.ref("slots"), sc.ref("nslots"), sc.ref("pairs"), sc.ref("where"), .{ .pairs = pairs, .experts = c.experts, .first = l.first, .last = l.last });
        const max_slots = @min(l.last - l.first, pairs);
        const many = sc.rows > 1;
        const up = kernels.XpArgs{ .K = c.latent, .N = c.moe_inter, .topk = c.topk, .rows_max = sc.rows, .beta = c.situ_beta, .lin = c.situ_linear };
        k.experts(e, false, sc.ref("lat"), .{ .buf = table }, sc.ref("slots"), sc.ref("nslots"), sc.ref("pairs"), sc.ref("act"), many, max_slots, up);
        var down = up;
        down.K = c.moe_inter;
        down.N = c.latent;
        k.experts(e, true, sc.ref("act"), .{ .buf = table }, sc.ref("slots"), sc.ref("nslots"), sc.ref("pairs"), sc.ref("xout"), many, max_slots, down);
        const per = c.experts / 8;
        k.combine(e, sc.ref("xout"), sc.ref("where"), sc.ref("tw"), if (l.partial) |p| .{ .buf = p } else sc.ref("ysum"), sc.ref("tids"), sc.rows, l.partial != null, per, l.first / per, (l.last - l.first) / per, down);
    }
};
