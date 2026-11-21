//! Greedy selection with the host sampler's ordering of the stored BF16 scores.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const draw = @import("../lfm2_moe/backend.zig").draw;
const Buffer = mtl.Buffer;

const tile_values = 1024;
const tiles = (c.vocab + tile_values - 1) / tile_values;
pub const bytes_per_row = tiles * 8;
pub const nonfinite = std.math.maxInt(u32);

pub const Kernels = struct {
    tiles_pipe: mtl.Pipeline,
    finish_pipe: mtl.Pipeline,
    audit: bool,

    pub fn load(device: mtl.Device) !?Kernels {
        const lib = try mtl.Library.fromSource(device, @embedFile("gpu_greedy.metal"), mtl.CompileOptions.mlx());
        defer lib.deinit();
        const tile = try mtl.Pipeline.init(device, lib, "tf_greedy_tiles", false);
        errdefer tile.deinit();
        return .{ .tiles_pipe = tile, .finish_pipe = try mtl.Pipeline.init(device, lib, "tf_greedy_finish", false), .audit = false };
    }

    pub fn deinit(p: Kernels) void {
        p.tiles_pipe.deinit();
        p.finish_pipe.deinit();
    }

    pub fn encodeTiles(p: Kernels, enc: mtl.ComputeEncoder, logits: Buffer, part: Buffer, rows: usize) void {
        const dims = [2]u32{ c.vocab, tiles };
        enc.setPipeline(p.tiles_pipe);
        enc.setBuffer(logits, 0, 0);
        enc.setBuffer(part, 0, 1);
        enc.setBytes(std.mem.asBytes(&dims), 2);
        enc.dispatchGroups(mtl.Size.of(tiles, rows, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn encodeFinish(p: Kernels, enc: mtl.ComputeEncoder, part: Buffer, tokens: Buffer, rows: usize) void {
        comptime std.debug.assert(tiles <= 128);
        const count: u32 = tiles;
        enc.setPipeline(p.finish_pipe);
        enc.setBuffer(part, 0, 0);
        enc.setBuffer(tokens, 0, 1);
        enc.setBytes(std.mem.asBytes(&count), 2);
        enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(128, 1, 1));
    }

    pub fn check(p: Kernels, gpa: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !void {
        const rows = 7;
        const options = mtl.ResourceOptions.shared;
        const logits = try device.buffer(rows * c.vocab * 2, options);
        defer logits.deinit();
        const part = try device.buffer(rows * bytes_per_row, options);
        defer part.deinit();
        const tokens = try device.buffer(rows * 4, options);
        defer tokens.deinit();
        const input = logits.slice(u16, rows * c.vocab);
        @memset(input, 0xbf80); // -1
        input[1023] = 0x3f80;
        input[1024] = 0x3f80; // a tie across tile boundaries
        input[c.vocab + 87] = 0x8000;
        input[c.vocab + 88] = 0; // signed zero tie
        input[2 * c.vocab + 127] = 0;
        input[2 * c.vocab + 128] = 1; // positive subnormal
        @memset(input[3 * c.vocab ..][0..c.vocab], 0x8002);
        input[4 * c.vocab - 1] = 0x8001; // negative subnormal in the last tile
        input[5 * c.vocab - 1] = 0xff80; // nonfinite in the last tile
        input[5 * c.vocab + 1] = 0x7fc1; // NaN

        const cb = queue.commandBuffer();
        const enc = cb.compute(.concurrent);
        p.encodeTiles(enc, logits, part, rows);
        enc.barrier();
        p.encodeFinish(enc, part, tokens, rows);
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |message| {
            std.log.err("DeepSeek GPU greedy check: {s}", .{message});
            return error.DeepseekGreedyGpuFailed;
        }
        for (tokens.slice(u32, rows), 0..) |picked, row| {
            const gold = draw(gpa, input[row * c.vocab ..][0..c.vocab], null, 0, &.{}) catch |err| switch (err) {
                error.NonfiniteLfmLogits => nonfinite,
                else => return err,
            };
            if (gold != picked) return error.DeepseekGreedyGpuDiffers;
        }
        std.log.info("deepseek_v41 bench: exact GPU greedy, {d} BF16 fixtures passed, audit={}", .{ rows, p.audit });
    }
};
