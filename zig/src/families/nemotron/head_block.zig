//! A one-pass draft block's learned inputs: each placeholder lane's residual row (mask embedding + lane position).
const std = @import("std");
const mtl = @import("metal");

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// bf16 [lanes, D]: lane j's residual input in the block (row j + 1, after the stock first level's row 0).
pub const Weights = struct {
    rows: mtl.Buffer,
    lanes: usize,

    /// Random rows (contract tests: a drafter's guesses change rounds, never verified tokens).
    pub fn random(device: mtl.Device, d: usize, lanes: usize, seed: u64) !Weights {
        const rows = try device.buffer(lanes * d * 2, opts);
        var prng: std.Random.DefaultPrng = .init(seed);
        const rng = prng.random();
        for (rows.slice(u16, lanes * d)) |*v| v.* = @truncate(@as(u32, @bitCast(rng.floatNorm(f32))) >> 16);
        return .{ .rows = rows, .lanes = lanes };
    }

    /// From a raw file: "TFBLOCK1", u32 lanes, u32 D, then bf16 [lanes, D] (the trainer's export, mask + position summed).
    pub fn load(device: mtl.Device, io: std.Io, gpa: std.mem.Allocator, path: []const u8, d: usize) !Weights {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
        defer gpa.free(bytes);
        if (bytes.len < 16 or !std.mem.eql(u8, bytes[0..8], "TFBLOCK1")) return error.BadBlockWeights;
        const lanes = std.mem.readInt(u32, bytes[8..12], .little);
        const width = std.mem.readInt(u32, bytes[12..16], .little);
        if (width != d or lanes == 0 or bytes.len != 16 + @as(usize, lanes) * d * 2) return error.BadBlockWeights;
        const rows = try device.buffer(lanes * d * 2, opts);
        @memcpy(rows.slice(u8, lanes * d * 2), bytes[16..]);
        return .{ .rows = rows, .lanes = lanes };
    }

    pub fn deinit(self: *Weights) void {
        self.rows.deinit();
    }
};
