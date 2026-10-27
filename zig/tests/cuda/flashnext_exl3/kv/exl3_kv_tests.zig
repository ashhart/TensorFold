//! exl3-kv: the KV-memory levers for 200k contexts (docs/plans/exl3-kv-memory.md), on one card.

const std = @import("std");
const check = @import("../../check.zig");
const kvcodes = @import("exl3_kv_codes.zig");
const kernels = @import("exl3_kv_kernel.zig");
const benches = @import("exl3_kv_bench.zig");

const kernelMode = kernels.kernelMode;
const benchMode = benches.benchMode;

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    const mode = rest[0];
    if (std.mem.eql(u8, mode, "kernel")) return kernelMode(gpu);
    if (std.mem.eql(u8, mode, "full")) return @import("exl3_kv_full.zig").run(gpu, rest[1..]);
    if (std.mem.eql(u8, mode, "real")) return @import("exl3_kv_real.zig").run(gpu, rest[1..]);
    if (std.mem.eql(u8, mode, "bench")) return benchMode(gpu, if (rest.len > 1) try std.fmt.parseInt(usize, rest[1], 10) else 200_016);
    return error.UnknownMode;
}

pub const wht32 = kvcodes.wht32;
pub const quant8 = kvcodes.quant8;
pub const dequant8 = kvcodes.dequant8;
pub const roundTrip4 = kvcodes.roundTrip4;
pub const packRow = kvcodes.packRow;
pub const tileRow = kvcodes.tileRow;
pub const valueRow = kvcodes.valueRow;
pub const Err = kernels.Err;
pub const rel = kernels.rel;
pub const rms = kernels.rms;
