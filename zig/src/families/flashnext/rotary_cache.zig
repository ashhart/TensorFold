//! A bounded GPU-produced position cache behind the existing rotary constant-buffer argument.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
pub const host_source = @embedFile("rotary_cache.zig");
pub const prefix = @import("rotary_config.zig").words;
pub const slots = 64;
pub const max_positions = 32;
pub const keys = prefix + 1;
pub const data = (keys + slots + 3) / 4 * 4;
pub const words = data + slots * 32 * 4;
pub const bytes = words * 4;
pub const growth = bytes - prefix * 4;
pub fn fits(enabled: bool, steady: u64, cap: u64) bool {
    return enabled and steady <= cap and growth <= cap - steady;
}
pub const header = std.fmt.comptimePrint(
    "#define TF_ROPE_ENABLE {d}\n#define TF_ROPE_KEYS {d}\n#define TF_ROPE_DATA {d}\n#define TF_ROPE_SLOTS {d}\n",
    .{ prefix, keys, data, slots },
);

pub fn source(a: std.mem.Allocator, text: []const u8, enabled: bool) ![]const u8 {
    if (!enabled or std.mem.indexOf(u8, text, "tf_rotary_scalar") == null) return text;
    const original = ks.flashnext_rotary_scalar;
    if (std.mem.count(u8, text, original) != 1) return error.RotarySourceMismatch;
    const body = std.mem.indexOf(u8, original, ") {") orelse return error.RotarySourceMismatch;
    const patched = try std.mem.concat(a, u8, &.{ original[0 .. body + 3], ks.flashnext_rotary_lookup, original[body + 3 ..] });
    defer a.free(patched);
    const changed = try std.mem.replaceOwned(u8, a, text, original, patched);
    defer a.free(changed);
    return std.mem.concat(a, u8, &.{ header, changed });
}

pub fn producer(a: std.mem.Allocator) ![]const u8 {
    return std.mem.concat(a, u8, &.{ "#include <metal_stdlib>\nusing namespace metal;\n", header, ks.flashnext_rotary_scalar, ks.flashnext_rotary_prepare });
}

pub fn compile(r: anytype) !void {
    if (!r.rotary_enabled) return;
    const library = try r.ownResource(try mtl.Library.fromSource(r.device, try producer(r.arena), mtl.CompileOptions.mlx()));
    const pipe = try r.ownResource(try mtl.Pipeline.init(r.device, library, "tf_rotary_prepare", false));
    if (comptime @hasDecl(mtl.Pipeline, "maxThreads")) if (pipe.maxThreads() < 128 or pipe.simdWidth() != 32) return;
    r.rotary_pipe = pipe;
}

pub fn reserve(r: anytype) void {
    if (!r.rotary_enabled) return;
    r.rotary_buffer = r.buffer(bytes) catch |err| {
        r.rotary_enabled = false;
        std.log.warn("flash next: rotary cache table unavailable: {s}", .{@errorName(err)});
        return;
    };
}

pub fn start(r: anytype) void {
    reserve(r);
    compile(r) catch |err| {
        r.rotary_pipe = null;
        std.log.warn("flash next: rotary cache disabled: {s}", .{@errorName(err)});
    };
}

pub fn forRun(r: anytype, text: []const u8) []const u8 {
    return source(r.arena, text, r.rotary_pipe != null) catch |err| {
        r.rotary_pipe = null;
        std.log.warn("flash next: rotary cache source refused: {s}", .{@errorName(err)});
        return text;
    };
}

pub fn encode(enc: anytype, pipe: anytype, table: anytype, positions: anytype, count: usize, reset: bool) void {
    std.debug.assert(count > 0 and count <= max_positions);
    enc.barrier();
    enc.setPipeline(pipe);
    enc.setBuffer(table.b, table.off, 0);
    enc.setBuffer(table.b, table.off, 1);
    enc.setBuffer(positions.b, positions.off, 2);
    const range = [2]u32{ @intCast(count), @intFromBool(reset) };
    enc.setBytes(std.mem.asBytes(&range), 3);
    enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(128, 1, 1));
    enc.barrier();
}

pub fn prepare(r: anytype, table: anytype, positions: anytype, count: usize) !void {
    const pipe = r.rotary_pipe orelse return;
    if (count == 0 or count > 16) return error.WindowOutOfStep;
    encode(r.enc, pipe, table, positions, count, true);
}

pub fn append(r: anytype, table: anytype, positions: anytype, count: usize) !void {
    const pipe = r.rotary_pipe orelse return;
    if (count == 0 or count > 16) return error.WindowOutOfStep;
    encode(r.enc, pipe, table, positions, count, false);
}
