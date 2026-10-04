//! Projections copied once into the slice-interleaved layout the row kernels read; the checkpoint's bytes stay as they are.
const std = @import("std");
const mtl = @import("metal");
const kernels = @import("kernels.zig");
const weights = @import("weights.zig");
const Tensor = @import("store.zig").Tensor;

/// Buffers a prepared layer or head owns, freed together.
pub const Owned = struct {
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(mtl.Buffer) = .empty,
    bytes: usize = 0,

    pub fn deinit(o: *Owned) void {
        for (o.bufs.items) |b| b.deinit();
        o.bufs.deinit(o.gpa);
    }
};

const Job = struct { k: *const kernels.Kernels, device: mtl.Device, e: mtl.ComputeEncoder, owned: *Owned };

fn one(j: Job, t: *Tensor) !void {
    if (t.rank != 2 or t.dtype != .bf16 or t.shape[1] % 64 != 0) return error.NotAProjection;
    const buf = try j.device.buffer(t.bytes(), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    try j.owned.bufs.append(j.owned.gpa, buf);
    j.owned.bytes += t.bytes();
    j.k.interleaved(j.e, t.ref, .{ .buf = buf }, t.shape[0], t.shape[1]);
    t.ref = .{ .buf = buf };
}

fn run(k: *const kernels.Kernels, device: mtl.Device, queue: mtl.Queue, owned: *Owned, tensors: []const *Tensor) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    for (tensors) |t| try one(.{ .k = k, .device = device, .e = e, .owned = owned }, t);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |text| {
        std.log.err("interleave failed: {s}", .{text});
        return error.GpuFailed;
    }
}

/// Every projection of layer `l` (not kv_b, which the MLA kernels read as stored) into its interleaved copy.
pub fn layer(k: *const kernels.Kernels, device: mtl.Device, queue: mtl.Queue, l: *weights.Layer, owned: *Owned) !void {
    var list: std.ArrayList(*Tensor) = .empty;
    defer list.deinit(owned.gpa);
    switch (l.attn) {
        .kda => |*a| try list.appendSlice(owned.gpa, &.{ &a.q, &a.k, &a.v, &a.g, &a.o, &a.fa, &a.fb, &a.b }),
        .mla => |*a| try list.appendSlice(owned.gpa, &.{ &a.qa, &a.qb, &a.kva, &a.g, &a.o }),
    }
    switch (l.mlp) {
        .dense => |*d| try list.appendSlice(owned.gpa, &.{ &d.gate, &d.up, &d.down }),
        .moe => |*m| try list.appendSlice(owned.gpa, &.{ &m.router, &m.down, &m.up, &m.sh_gate, &m.sh_up, &m.sh_down }),
    }
    try run(k, device, queue, owned, list.items);
}

/// The LM head into its interleaved copy (the embedding is read by rows, as stored).
pub fn head(k: *const kernels.Kernels, device: mtl.Device, queue: mtl.Queue, h: *weights.Head, owned: *Owned) !void {
    try run(k, device, queue, owned, &.{&h.lm_head});
}
