//! Packed projection fixtures own their contexts until every contract gate completes.
const std = @import("std");
const cuda = @import("cuda");
const g = @import("core").contract.gates;
const wide = @import("contract_projection_f32.zig");
const bf16 = @import("contract_prompt_bf16.zig");

pub const Set = struct {
    products: [wide.modes.len * 2]wide.Product,
    narrow: [wide.bf16_modes.len * 2 + 1]wide.Product,
    tiles: [6]bf16.Tile,
    prompt: cuda.nvfp4.Prompt,

    pub fn init(gpu: @import("check.zig").Gpu, s: cuda.Stream, lane: *const cuda.qmmf.Lane, reduce: *const cuda.qmmf.Lane, prompt: *const cuda.nvfp4.Prompt) !Set {
        var out: Set = undefined;
        out.prompt = prompt.*;
        for (wide.modes, 0..) |mode, i| {
            out.products[i * 2] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = lane, .mode = mode, .clusters = true };
            out.products[i * 2 + 1] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = reduce, .mode = mode, .clusters = false };
        }
        for (wide.bf16_modes, 0..) |mode, i| {
            out.narrow[i * 2] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = lane, .mode = mode, .clusters = true, .out_kind = .bf16 };
            out.narrow[i * 2 + 1] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = reduce, .mode = mode, .clusters = false, .out_kind = .bf16 };
        }
        out.narrow[out.narrow.len - 1] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = lane, .mode = .fp8, .clusters = true, .out_kind = .bf16, .prompt = prompt };
        for (&out.tiles, 0..) |*t, i| t.* = .{
            .product = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = reduce, .mode = .bf16, .clusters = false, .out_kind = if (i % 2 == 0) .bf16 else .f32 },
            .prompt = undefined,
            .replacement = i >= 2,
            .control = i >= 4,
        };
        return out;
    }

    pub fn deinit(set: *Set) void {
        for (&set.products) |*p| p.deinit();
        for (&set.narrow) |*p| p.deinit();
        for (&set.tiles) |*t| t.deinit();
    }

    pub fn append(set: *Set, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !void {
        for (&set.products) |*p| try kernels.append(a, p.kernel());
        for (&set.narrow) |*p| try kernels.append(a, p.kernel());
        for (&set.tiles) |*t| {
            t.prompt = &set.prompt;
            try kernels.append(a, t.kernel());
        }
    }
};
