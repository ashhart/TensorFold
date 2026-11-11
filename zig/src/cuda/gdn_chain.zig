//! Qwen prompt recurrence: the old half-head chain, or its whole-head chain and staged gated norm.
const std = @import("std");
const module = @import("module.zig");
const launch = @import("launch.zig");
const Stream = @import("stream.zig").Stream;

pub const old_symbol = "_ZN14tf_gdn_prefill12chain_kernelI13__nv_bfloat16S1_Li64ELi32EEEvPKT_S4_PKS1_PKfPKT0_S8_PfPS1_iii";
pub const symbol = "tf_gdn_chain_norm";

pub const Buffers = struct {
    q: u64,
    k: u64,
    v: u64,
    g: u64,
    beta: u64,
    state: u64,
    last: u64,
    y: u64,
    z: u64,
    norm: u64,
    mix: u64,
    sums: u64 = 0,
};

pub const Chain = struct {
    old: module.Function,
    fused: ?module.Function = null,

    pub fn resolve(mod: module.Module) !Chain {
        return resolveMode(mod, try enabled());
    }

    pub fn enabled() !bool {
        return @import("sys").environment.read(bool, "TF_CUDA_GDN_CHAIN_NORM", true);
    }
    pub fn enabledValue(value: ?[]const u8) !bool {
        return (try @import("sys").environment.parse("TF_CUDA_GDN_CHAIN_NORM", value) orelse 1) != 0;
    }

    pub fn resolveMode(mod: module.Module, on: bool) !Chain {
        return .{ .old = try mod.function(old_symbol), .fused = if (on) try mod.function(symbol) else null };
    }

    /// Returns whether the norm was written; false requires the caller's original post launch.
    pub fn run(f: Chain, s: Stream, b: Buffers, rows: usize, hk: usize, hv: usize, eps: f32) !bool {
        if (rows == 0 or hk == 0 or hv == 0 or hv % hk != 0) return error.BadGdnChainShape;
        var a: launch.Args = .{};
        for ([_]u64{ b.q, b.k, b.v, b.g, b.beta, b.state, b.last, b.y }) |v| a.add(v);
        for ([_]usize{ rows, hk, hv }) |v| a.add(@as(c_int, @intCast(v)));
        if (f.fused) |kernel| {
            for ([_]u64{ b.z, b.norm, b.mix, b.sums }) |v| a.add(v);
            a.add(eps);
            try launch.launch(kernel, .{ .grid = .{ .x = @intCast(hv) }, .block = .{ .x = 256 } }, s, &a);
            return true;
        }
        try launch.launch(f.old, .{ .grid = .{ .x = @intCast(hv), .y = 2 }, .block = .{ .x = 128 } }, s, &a);
        return false;
    }
};
