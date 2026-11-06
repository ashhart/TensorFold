//! Main's prompt sources and launch layouts for isolated masked-value cases; no arithmetic source edits.
const std = @import("std");
const metal = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("tensorfold").frags;
const data = @import("masked_value_data.zig");
const rig = @import("rig.zig");

pub const Kind = enum { gemma31_global, gemma31_sliding, gemma12_global, gemma12_sliding, spark_full, spark_sliding, minimax, llama33, llama3, nemotron, lfm2, qwen35, gemmae4_global, gemmae4_sliding, gemma31_wrap, spark_wrap };
pub fn available(_: Kind) bool {
    return true;
}
pub fn output(_: Kind) @import("contract").spec.Elem {
    return .bf16;
}
pub fn known(k: Kind, _: data.Poison) bool {
    return k != .qwen35 and k != .nemotron;
}
pub fn finding(_: Kind) []const u8 {
    return "#1009: a later causal key's non-finite V reaches the preceding row as NaN through zero P x V";
}
pub fn geometry(k: Kind) data.Geometry {
    return switch (k) {
        .gemmae4_global => .{ .heads = 8, .kv = 2, .dim = 512, .cache = .head },
        .gemmae4_sliding => .{ .heads = 8, .kv = 2, .dim = 256, .cap = 1536, .window = 1024, .ring = 1536, .cache = .head },
        .gemma31_wrap => .{ .heads = 32, .kv = 16, .dim = 256, .cap = 1536, .first = 1535, .window = 1024, .ring = 1536, .cache = .head },
        .spark_wrap => .{ .heads = 16, .kv = 4, .dim = 256, .cap = 2560, .first = 2559, .window = 512, .ring = 2560 },
        .gemma31_global => .{ .heads = 32, .kv = 4, .dim = 512, .cache = .head },
        .gemma31_sliding => .{ .heads = 32, .kv = 16, .dim = 256, .cap = 1536, .window = 1024, .ring = 1536, .cache = .head },
        .gemma12_global => .{ .heads = 16, .kv = 1, .dim = 512, .cache = .head },
        .gemma12_sliding => .{ .heads = 16, .kv = 8, .dim = 256, .cap = 1536, .window = 1024, .ring = 1536, .cache = .head },
        .spark_full => .{ .heads = 16, .kv = 4, .dim = 256 },
        .spark_sliding => .{ .heads = 16, .kv = 4, .dim = 256, .cap = 2560, .window = 512, .ring = 2560 },
        .minimax => .{ .heads = 64, .kv = 4, .dim = 128 },
        .llama33 => .{ .heads = 56, .kv = 8, .dim = 128 },
        .llama3 => .{ .heads = 20, .kv = 4, .dim = 128 },
        .nemotron => .{ .heads = 32, .kv = 2, .dim = 128, .cache = .head },
        .lfm2 => .{ .heads = 32, .kv = 8, .dim = 64, .cache = .head },
        .qwen35 => .{ .heads = 8, .kv = 2, .dim = 256, .cache = .head, .query = .head },
    };
}

pub fn label(k: Kind) []const u8 {
    return switch (k) {
        .gemmae4_global => "Gemma4 E4B global",
        .gemmae4_sliding => "Gemma4 E4B sliding",
        .gemma31_wrap => "Gemma4 31B sliding wrap",
        .spark_wrap => "Spark2.5 sliding wrap",
        .gemma31_global => "Gemma4 31B global",
        .gemma31_sliding => "Gemma4 31B sliding",
        .gemma12_global => "Gemma4 12B global",
        .gemma12_sliding => "Gemma4 12B sliding",
        .spark_full => "Spark2.5 full",
        .spark_sliding => "Spark2.5 sliding",
        .minimax => "MiniMaxM3 prompt",
        .llama33 => "Llama Nanbeige 33B prompt",
        .llama3 => "Llama Nanbeige 3B prompt",
        .nemotron => "Nemotron prompt",
        .lfm2 => "LFM2 prompt nax",
        .qwen35 => "Qwen3.5 small dense contiguous",
    };
}

fn prefill() ![]const u8 {
    for (sources.prefill) |f| if (std.mem.eql(u8, f.name, "attention_nax")) return f.text;
    return error.MissingAttentionSource;
}

pub fn load(p: anytype) !void {
    var arena = std.heap.ArenaAllocator.init(p.a);
    defer arena.deinit();
    const a = arena.allocator();
    const g = p.geo;
    var header: []const u8 = "";
    var text: []const u8 = "";
    var name: []const u8 = "";
    var merge: ?[]const u8 = null;
    switch (p.kind) {
        .gemma31_global, .gemma31_sliding, .gemma12_global, .gemma12_sliding, .gemmae4_global, .gemmae4_sliding, .gemma31_wrap => {
            header = try std.fmt.allocPrint(a, "#define DH {d}\n#define NQ {d}\n#define G {d}\n#define SL {d}\n#define WINDOW {d}\n#define RING {d}\n", .{ g.dim, g.heads, g.heads / g.kv, g.dim / 128, g.window, g.ring });
            text = sources.gemma4_prompt_attention;
            name = "gemma_prompt_attention";
        },
        .spark_full, .spark_sliding, .spark_wrap, .llama33, .llama3, .minimax => {
            header = try std.fmt.allocPrint(a, "#define DH {d}\n#define NQ {d}\n#define G {d}\n#define KVH {d}\n#define WINDOW {d}\n#define RING {d}\n#define WORDS 256\n#define SCALE {e}f\n", .{ g.dim, g.heads, g.heads / g.kv, g.kv, g.window, g.ring, 1 / @sqrt(@as(f64, @floatFromInt(g.dim))) });
            text = switch (p.kind) {
                .minimax => sources.minimax_m3_attention_nax,
                .llama33, .llama3 => sources.llama_attention_nax,
                else => sources.spark25_attention_nax,
            };
            name = switch (p.kind) {
                .minimax => "mm_prompt_attention",
                .llama33, .llama3 => "ll_prompt_attention",
                else => "sp_prompt_attention",
            };
        },
        .nemotron => {
            text = sources.nemotron_prompt_masked.contiguous;
            name = sources.nemotron_prompt_masked.name(false, 0);
        },
        .lfm2 => {
            text = try std.mem.concat(a, u8, &.{ sources.lfm2, sources.lfm2_attn_nax });
            name = "lfm2_attn_nax";
            merge = "lfm2_attn_merge";
        },
        .qwen35 => {
            for (sources.qwen35.all) |s| {
                if (std.mem.eql(u8, s.key, "attn_partial")) {
                    text = s.source;
                    name = s.function;
                }
                if (std.mem.eql(u8, s.key, "attn_merge")) {
                    p.pipes[1] = try rig.pipeline(p.rig.device, s.source, s.function);
                }
            }
        },
    }
    const body = try frags.source(p.rig.device, a, text);
    const full = try std.mem.concat(a, u8, &.{ header, body });
    p.pipes[0] = try rig.pipeline(p.rig.device, full, name);
    if (merge) |m| p.pipes[1] = try rig.pipeline(p.rig.device, full, m);
}

pub fn prepare(p: anytype) !void {
    const d = p.fixture;
    const g = d.geo;
    p.q = try p.rig.input(std.mem.sliceAsBytes(d.q));
    p.k = try p.rig.input(std.mem.sliceAsBytes(d.k));
    p.v = try p.rig.input(std.mem.sliceAsBytes(d.v));
    p.out = try p.rig.output(d.count() * 2);
    const a = p.arena.?.allocator();
    switch (p.kind) {
        .gemma31_global, .gemma31_sliding, .gemma12_global, .gemma12_sliding, .gemmae4_global, .gemmae4_sliding, .gemma31_wrap => {
            const hi = try a.alloc(i32, @max(32, d.rows));
            for (hi, 0..) |*v, i| v.* = @intCast(g.first + i);
            p.aux[0] = try p.rig.input(std.mem.sliceAsBytes(hi));
        },
        .minimax => {
            const sel = try a.alloc(u32, @max(32, d.rows) * g.kv * 256);
            @memset(sel, 0xffffffff);
            p.aux[0] = try p.rig.input(std.mem.sliceAsBytes(sel));
        },
        .lfm2 => {
            const n = d.rows * g.heads * 32;
            p.aux[0] = try p.rig.scratch(n * 4);
            p.aux[1] = try p.rig.scratch(n * 4);
            p.aux[2] = try p.rig.scratch(n * g.dim * 4);
        },
        .qwen35 => {
            const dims = [_]i32{ @intCast(g.first), @intCast(d.rows), @intCast(g.cap), 1, 16, @intCast(d.rows), 0 };
            p.aux[0] = try p.rig.input(std.mem.sliceAsBytes(&dims));
            const n = d.rows * g.heads;
            for (0..3) |i| p.aux[1 + i] = try p.rig.scratch(n * (if (i == 2) g.dim else 1) * 4);
        },
        else => {},
    }
}

fn bind(e: metal.ComputeEncoder, ref: rig.Ref, at: usize) void {
    e.setBuffer(ref.buf, ref.off, at);
}
pub fn encode(p: anytype, e: metal.ComputeEncoder) void {
    const d = p.fixture;
    const g = d.geo;
    const rows = d.rows;
    e.setPipeline(p.pipes[0].?);
    for ([_]rig.Ref{ p.q, p.k, p.v }, 0..) |r, i| bind(e, r, i);
    switch (p.kind) {
        .gemma31_global, .gemma31_sliding, .gemma12_global, .gemma12_sliding, .gemmae4_global, .gemmae4_sliding, .gemma31_wrap => {
            bind(e, p.out, 3);
            e.setValue([3]i32{ @intCast(rows), @intCast(g.first), @intCast(g.cap) }, 4);
            bind(e, p.aux[0], 5);
            e.dispatchGroups(.{ .width = (rows + 31) / 32, .height = g.heads }, .{ .width = 32 * g.dim / 64 });
        },
        .spark_full, .spark_sliding, .spark_wrap, .llama33, .llama3, .minimax => {
            bind(e, p.out, 3);
            e.setValue([2]i32{ @intCast(rows), @intCast(g.first) }, 4);
            if (p.kind == .minimax) bind(e, p.aux[0], 5);
            e.dispatchGroups(.{ .width = (rows + (if (p.kind == .minimax) @as(usize, 1) else 31)) / (if (p.kind == .minimax) @as(usize, 2) else 32), .height = if (p.kind == .minimax) g.kv else g.heads }, .{ .width = 32 * g.dim / 64 });
        },
        .nemotron => {
            const kl = g.first + rows;
            const row = g.heads * g.dim;
            const kvs = g.kv * g.cap * g.dim;
            var pr: [32]i32 = @splat(0);
            const head = [_]usize{ 1, g.heads, rows, kl, g.heads / g.kv, (rows + 63) / 64, (kl + 31) / 32, rows / 64, kl / 32, rows % 64, kl % 32, g.first };
            const st = [12]usize{ rows * row, g.dim, row, kvs, g.cap * g.dim, g.dim, kvs, g.cap * g.dim, g.dim, rows * row, g.dim, row };
            for (head ++ st, 0..) |x, i| pr[i] = @intCast(x);
            var fl: [16]f32 = @splat(0);
            fl[0] = 1 / @sqrt(@as(f32, @floatFromInt(g.dim)));
            e.setValue(pr, 3);
            e.setValue(fl, 4);
            bind(e, p.out, 5);
            e.dispatchThreads(.{ .width = (rows + 63) / 64 * 32, .height = g.heads * 4 }, .{ .width = 32, .height = 4 });
        },
        .lfm2 => {
            const Args = extern struct { len: u32, cap: u32, scale: f32, subs: u32 };
            const args = Args{ .len = @intCast(g.first), .cap = @intCast(g.cap), .scale = 0.125, .subs = 1 };
            e.setValue(args, 3);
            for (p.aux[0..3], 4..) |r, i| bind(e, r, i);
            e.setValue(@as(u32, @intCast(rows)), 7);
            e.dispatchGroups(.{ .width = (rows + 15) / 16, .height = g.kv }, .{ .width = 128 });
            e.barrier();
            e.setPipeline(p.pipes[1].?);
            for (p.aux[0..3], 0..) |r, i| bind(e, r, i);
            e.setValue(args, 3);
            bind(e, p.out, 4);
            e.dispatchGroups(.{ .width = g.heads, .height = rows }, .{ .width = 32 });
        },
        .qwen35 => {
            e.setValue(@as(f32, 0.0625), 3);
            bind(e, p.aux[0], 4);
            for (p.aux[1..4], 5..) |r, i| bind(e, r, i);
            e.dispatchGroups(.{ .width = 1, .height = 1, .depth = g.kv * rows }, .{ .width = 512 });
            e.barrier();
            e.setPipeline(p.pipes[1].?);
            for (p.aux[1..4], 0..) |r, i| bind(e, r, i);
            bind(e, p.aux[0], 3);
            bind(e, p.out, 4);
            e.dispatchGroups(.{ .width = 1, .height = g.heads, .depth = rows }, .{ .width = 32 });
        },
    }
}
