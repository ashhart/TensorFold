//! The CUDA H3 family as a C library for a host that shares its CUDA context: tf_h3_open, tf_h3_forward, tf_h3_close.
const std = @import("std");
const h3 = @import("cuda.zig");

const gpa = std.heap.c_allocator;

/// The transformer at `path` (one safetensors file), quantized onto the GPU; null on failure.
export fn tf_h3_open(path: [*:0]const u8) ?*h3.Model {
    return h3.Model.load(gpa, path) catch |e| {
        std.debug.print("tf_h3_open: {t}\n", .{e});
        return null;
    };
}

export fn tf_h3_close(model: *h3.Model) void {
    model.deinit();
}

/// hidden, inner, heads, mlp, blocks, the load's seconds in thousandths, and whether it read a saved int8 copy.
export fn tf_h3_shape(model: *const h3.Model, out: *[7]u32) void {
    out.* = .{ @intCast(model.hidden), @intCast(model.inner), @intCast(model.heads), @intCast(model.mlp), @intCast(model.blocks.len), @intFromFloat(model.loaded_seconds * 1000), @intFromBool(model.from_cache) };
}

/// Every block over the packed stream, in place. Returns the seconds taken, or a negative number on failure.
export fn tf_h3_forward(model: *h3.Model, in: *const h3.Model.Inputs) f64 {
    return model.forward(in) catch |e| {
        std.debug.print("tf_h3_forward: {t}\n", .{e});
        return -1;
    };
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
