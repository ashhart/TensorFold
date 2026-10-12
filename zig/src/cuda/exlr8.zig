//! exlr8.cu's EXLR8 lane projections (the Metal kernels' arithmetic, row for row): resolve once, launch on a stream.
const std = @import("std");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Function = module.Function;
const Stream = stream_.Stream;
/// The row capacities exlr8.cu instantiates (core/exlr8_job.zig's list).
const capacities = [_]usize{ 1, 2, 4, 8, 16 };

/// Argument structs come from core/exlr8_job.zig (Rotate, Rows, Combine), passed by value as the kernels take them.
pub const Kernels = struct {
    rotate_fn: Function,
    project_fns: [capacities.len]Function,
    combine_fn: Function,
    codebook_fn: Function,

    pub fn resolve(mod: module.Module) !Kernels {
        var k: Kernels = .{ .rotate_fn = try mod.function("tf_exlr8_rotate"), .project_fns = undefined, .combine_fn = try mod.function("tf_exlr8_combine"), .codebook_fn = try mod.function("tf_exlr8_weights") };
        for (capacities, &k.project_fns) |c, *f| {
            var name: [32]u8 = undefined;
            f.* = try mod.function(try std.fmt.bufPrintSentinel(&name, "tf_exlr8_project_{d}", .{c}, 0));
        }
        return k;
    }

    /// One warp per (row, 128-block, slot); `slot_scales` (device u32 per slot) when a.scales is set, else 0.
    pub fn rotate(k: *const Kernels, s: Stream, x: u64, signs: u64, halves: u64, r: u64, slot_scales: u64, a: anytype, blocks: usize, slots: usize) !void {
        if (a.rows == 0 or blocks == 0 or slots == 0 or (a.scales != 0) != (slot_scales != 0)) return error.InvalidExlr8Rows;
        var args: launch_.Args = .{};
        args.add(x);
        args.add(signs);
        args.add(halves);
        args.add(a);
        args.add(r);
        args.add(slot_scales);
        try launch_.launch(k.rotate_fn, .{ .grid = .{ .x = @intCast(blocks), .y = a.rows, .z = @intCast(slots) }, .block = .{ .x = 32 } }, s, &args);
    }

    /// One block of 256 threads per job over 1-16 rows (`rows` holds the count) of rotated fp32 input.
    pub fn project(k: *const Kernels, s: Stream, x: u64, weights: u64, bases: u64, halves: u64, signs: u64, jobs: u64, count: usize, rows: anytype, y: u64) !void {
        if (rows.rows == 0 or rows.rows > capacities[capacities.len - 1] or count == 0) return error.InvalidExlr8Rows;
        var args: launch_.Args = .{};
        args.add(x);
        args.add(weights);
        args.add(bases);
        args.add(halves);
        args.add(signs);
        args.add(jobs);
        args.add(rows);
        args.add(y);
        const at = for (capacities, 0..) |c, i| {
            if (rows.rows <= c) break i;
        } else unreachable;
        try launch_.launch(k.project_fns[at], .{ .grid = .{ .x = @intCast(count), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &args);
    }

    /// One warp per (row, 128-block) over `blocks` output blocks; sv holds each slot's block-scale offset.
    pub fn combine(k: *const Kernels, s: Stream, z: u64, weights: u64, sv: u64, halves: u64, signs: u64, a: anytype, blocks: usize, y: u64) !void {
        if (a.rows == 0 or a.slots == 0 or blocks == 0) return error.InvalidExlr8Rows;
        var args: launch_.Args = .{};
        args.add(z);
        args.add(weights);
        args.add(sv);
        args.add(halves);
        args.add(signs);
        args.add(a);
        args.add(y);
        try launch_.launch(k.combine_fn, .{ .grid = .{ .x = @intCast(blocks), .y = a.rows, .z = 1 }, .block = .{ .x = 32 } }, s, &args);
    }

    /// Every codebook state's weight (65536 fp32) into `out`.
    pub fn codebook(k: *const Kernels, s: Stream, out: u64) !void {
        var args: launch_.Args = .{};
        args.add(out);
        try launch_.launch(k.codebook_fn, .{ .grid = .{ .x = 256, .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &args);
    }
};
