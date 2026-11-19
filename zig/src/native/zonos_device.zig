//! The ZONOS3 host's device, CUDA's or Metal's: what the host asks of it beside the family's parts.
const std = @import("std");
const api = @import("engine_api");
const z3 = @import("zonos3");

/// The family this build has: Metal's carries its device type.
pub const on_metal = @hasDecl(z3, "gpu");
const cuda = if (on_metal) void else @import("cuda");

/// What an admission is asked on: the CUDA context, or the Metal device.
pub const Context = if (on_metal) z3.gpu.Gpu else cuda.Context;

pub fn context(e: *z3.Engine) *const Context {
    return if (comptime on_metal) e.g else &e.ctx;
}

/// Device bytes the process holds: CUDA's usage, or the Metal buffers made.
pub fn held(e: *z3.Engine) u64 {
    return if (comptime on_metal) e.g.held else cuda.usage(false).device;
}

pub fn memory(e: *z3.Engine, reset_peak: bool) api.Memory {
    if (comptime on_metal) {
        const g = e.g;
        const peak = g.peak;
        if (reset_peak) g.peak = g.held;
        return .{ .active = g.held, .pinned = 0, .peak = peak };
    }
    const u = cuda.usage(reset_peak);
    return .{ .active = u.device, .pinned = u.host, .peak = u.peak };
}

/// The engine's context current on this thread (Metal has none to make).
pub fn makeCurrent(e: *z3.Engine) bool {
    if (comptime on_metal) return true;
    e.ctx.makeCurrent() catch return false;
    return true;
}

/// Waits for the work queued so far, after a failure.
pub fn settle(e: *z3.Engine) void {
    if (comptime on_metal) e.g.flush() catch {} else e.stream.synchronize() catch {};
}

/// A codec on the engine's device.
pub fn codec(gpa: std.mem.Allocator, io: std.Io, e: *z3.Engine, generator: []const u8, head: []const u8, o: z3.codec.Options) !*z3.Codec {
    if (comptime on_metal) return z3.Codec.create(gpa, io, e.g, generator, head, o);
    return z3.Codec.create(gpa, io, &e.driver, e.stream, generator, head, o);
}

/// The clip encoder on the engine's device, sharing the codec's tables.
pub fn clip(gpa: std.mem.Allocator, io: std.Io, e: *z3.Engine, dir: []const u8, generator: []const u8, c: *z3.Codec) !*z3.clip.Clip {
    if (comptime on_metal) return z3.clip.Clip.create(gpa, io, e.g, dir, generator, c.w);
    return z3.clip.Clip.create(gpa, io, &e.driver, e.stream, dir, generator, c.w);
}

/// The forced aligner on the engine's device (on CUDA through the codec's cuBLAS handle).
pub fn aligner(gpa: std.mem.Allocator, io: std.Io, e: *z3.Engine, dir: []const u8, c: *z3.Codec) !*z3.aligner.Aligner {
    if (comptime on_metal) return z3.aligner.Aligner.create(gpa, io, e.g, dir);
    return z3.aligner.Aligner.create(gpa, io, &e.driver, e.stream, dir, &c.blas);
}
