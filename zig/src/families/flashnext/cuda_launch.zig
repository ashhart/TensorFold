//! Flash Next's Triton launches from the captured cubins: each grid and constexpr matches the Python wrapper.

const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;

pub const embed = @import("cuda_embed.zig");

const p = aot.ptr;

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

fn u(x: usize) u32 {
    return @intCast(x);
}

/// One program per token row and per group of 32 hidden features.
pub fn embedGrid(rows: usize, dims: usize) [3]u32 {
    return .{ u(rows), u(dims / 32), 1 };
}

pub const Tri = struct {
    set: *const aot.Set,
    s: cuda.Stream,

    fn run(t: Tri, name: []const u8, grid: [3]u32, args: []const aot.Arg, consts: []const aot.Const) !void {
        try t.set.run(t.s, name, grid, args, consts);
    }

    /// glue.embed: ids (R,) int32 into (R, copies * dims) bf16, MLX 4-bit group 32.
    pub fn embed(t: Tri, ids: u64, weight: u64, scales: u64, biases: u64, out: u64, dims: usize, copies: usize, rows: usize) !void {
        try t.run("_embed", embedGrid(rows, dims), &.{
            p("IDS", "*i32", ids),
            p("W", "*i32", weight),
            p("S", "*bf16", scales),
            p("B", "*bf16", biases),
            p("OUT", "*bf16", out),
        }, &.{ ci("D", dims), ci("S_COPIES", copies) });
    }
};

test {
    _ = embed;
}

test "embed grid is one program per row and per group of 32" {
    try std.testing.expectEqual([3]u32{ 4, 80, 1 }, embedGrid(4, 2560));
}
