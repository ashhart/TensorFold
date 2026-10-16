//! The CUDA H3 family as a C library for a host that shares its CUDA context: tf_h3_open, tf_h3_forward, tf_h3_close.
const std = @import("std");
const h3 = @import("cuda.zig");
const weights = @import("cuda_weights.zig");
const tp = h3.tp;

const gpa = std.heap.c_allocator;

/// The transformer at `path` (one safetensors file) on the GPU in its bf16; null on failure.
export fn tf_h3_open(path: [*:0]const u8) ?*h3.Model {
    return h3.Model.load(gpa, path) catch |e| {
        std.debug.print("tf_h3_open: {t}\n", .{e});
        return null;
    };
}

export fn tf_h3_close(model: *h3.Model) void {
    model.deinit();
}

/// hidden, inner, heads, mlp, blocks, the load's seconds in thousandths, and the weights' bits (16).
export fn tf_h3_shape(model: *const h3.Model, out: *[7]u32) void {
    out.* = .{ @intCast(model.hidden), @intCast(model.inner), @intCast(model.heads), @intCast(model.mlp), @intCast(model.blocks.len), @intFromFloat(model.loaded_seconds * 1000), 16 };
}

/// Every block over the packed stream, in place. Returns the seconds taken, or a negative number on failure.
export fn tf_h3_forward(model: *h3.Model, in: *const h3.Model.Inputs) f64 {
    return model.forward(in) catch |e| {
        std.debug.print("tf_h3_forward: {t}\n", .{e});
        return -1;
    };
}

/// One block projection (weights.Projection) from bf16 (rows, cols) on the device; 0, or -1 on a wrong block or shape.
export fn tf_h3_set_weight(model: *h3.Model, block: u32, which: c_int, w: u64, rows: u32, cols: u32) c_int {
    const projection = std.enums.fromInt(weights.Projection, which) orelse return -1;
    weights.set(model, block, projection, w, rows, cols) catch |e| {
        std.debug.print("tf_h3_set_weight: {t}\n", .{e});
        return -1;
    };
    return 0;
}

/// With `on`, a forward waits for the GPU after every stage and adds up where the time went.
export fn tf_h3_profile(model: *h3.Model, on: c_int) void {
    model.profile = on != 0;
    model.spent = @splat(0);
}

/// The stages' seconds since profiling was switched on; returns how many there are.
export fn tf_h3_spent(model: *const h3.Model, out: [*]f64, names: [*][*:0]const u8) c_int {
    inline for (h3.stages, 0..) |s, i| {
        out[i] = model.spent[i];
        names[i] = s ++ "";
    }
    return h3.stages.len;
}

/// Joins a sequence-parallel world (tp.init): `device` is one RDMA device, or two (rails) by a comma; 0 when it worked.
export fn tf_h3_tp_init(model: *h3.Model, device: [*:0]const u8, gid_index: c_int, world: c_int, rank: c_int, region_bytes: u64) c_int {
    tp.init(model, device, gid_index, @intCast(world), @intCast(rank), region_bytes) catch |e| {
        std.debug.print("tf_h3_tp_init: {t}\n", .{e});
        return -1;
    };
    return 0;
}

export fn tf_h3_tp_close(model: *h3.Model) void {
    tp.close(model);
}

/// This rank's connection record for `peer` (72 bytes), to be handed to that rank's tf_h3_tp_connect.
export fn tf_h3_tp_info(model: *h3.Model, peer: c_int, out: *tp.XchgInfo) c_int {
    tp.info(model, @intCast(peer), out) catch return -1;
    return 0;
}

export fn tf_h3_tp_connect(model: *h3.Model, peer: c_int, in: *const tp.XchgInfo) c_int {
    tp.connect(model, @intCast(peer), in) catch return -1;
    return 0;
}

/// The last forward's exchanges: seconds in them, how many, and the seconds of those spent packing this rank's tiles.
export fn tf_h3_tp_stats(model: *const h3.Model, out: *[3]f64) void {
    out.* = if (model.tp) |t| .{ t.waited, @floatFromInt(t.swaps), t.packed_seconds } else .{ 0, 0, 0 };
}

/// tf_h3_forward for inputs in host memory (every address of `in` a host address; pinned memory copies fastest).
export fn tf_h3_forward_host(model: *h3.Model, in: *const h3.Model.Inputs) f64 {
    return tp.forwardHost(model, in) catch |e| {
        std.debug.print("tf_h3_forward_host: {t}\n", .{e});
        return -1;
    };
}

/// Pinned host memory the GPU maps, for a host without a framework: returns its host address, stores its device one.
export fn tf_h3_host_alloc(model: *h3.Model, bytes: u64, device: *u64) ?*anyopaque {
    return tp.hostAlloc(model, bytes, device) catch |e| {
        std.debug.print("tf_h3_host_alloc: {t}\n", .{e});
        return null;
    };
}

export fn tf_h3_host_free(model: *h3.Model, ptr: *anyopaque) void {
    tp.hostFree(model, ptr);
}
