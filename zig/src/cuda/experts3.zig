//! MLX affine-3 experts (experts3.cu, experts3_prefill.cu): affine-4's steps on bit planes, a 4-bit shared one beside.
const std = @import("std");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const grouped = @import("grouped.zig");
const experts = @import("experts.zig");

const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;
const Plan = grouped.Plan;
const Affine4Experts = experts.Affine4Experts;

fn int(x: usize) c_int {
    return @intCast(x);
}

pub const Affine3Experts = struct {
    pair_fn: Function,
    split_fn: Function, // the pair form with a column block a CTA and its groups shared by the CTA's warps
    down_fn: Function,
    pre_pair_fn: Function,
    pre_down_fn: Function,
    widen_fn: Function,
    planes_fn: Function,
    resident: [2]usize, // blocks the decode pair and down kernels fill: per SM times SMs
    split_resident: usize, // blocks split_fn fills
    split_below: usize, // pair column blocks below which split_fn runs: 7/8 of the warps pair_fn keeps resident

    pub const words = 224; // uint32 a 3-bit block: a 2-bit plane (128), a 1-bit plane (64), then scales and biases (32)
    /// The prompt kernels' dynamic shared bytes: the larger of the 3-bit and 4-bit (shared expert) stage sets.
    pub const prompt_smem: u32 = 47104; // M 1: four stages of 736 uint4 (3-bit) over three of 800 (4-bit)
    pub const pair_prompt_smem: u32 = 46080; // M 2: three stages of 960 uint4 (3-bit) over two of 1088 (4-bit)

    pub const symbols = struct {
        pub const pair = "_ZN11tf_experts313expert_kernelILi2ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_PvifS6_i";
        pub const split = "_ZN11tf_experts312split_kernelILi2ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_PvifS6_i";
        pub const down = "_ZN11tf_experts313expert_kernelILi1ELi0ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_PvifS6_i";
        pub const pre_pair = "_ZN19tf_experts3_prefill14prefill_kernelILi2ELi2ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_PvifS6_i";
        pub const pre_down = "_ZN19tf_experts3_prefill14prefill_kernelILi1ELi3ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_PvifS6_i";
        pub const widen = "tf_experts3_widen";
        pub const planes = "tf_experts3_planes";
    };

    /// From loaded experts3 and experts3_prefill modules.
    pub fn resolve(decode_mod: Module, prompt_mod: Module, sms: usize) !Affine3Experts {
        var a: Affine3Experts = undefined;
        a.pair_fn = try decode_mod.function(symbols.pair);
        a.split_fn = try decode_mod.function(symbols.split);
        a.down_fn = try decode_mod.function(symbols.down);
        a.widen_fn = try decode_mod.function(symbols.widen);
        a.planes_fn = try decode_mod.function(symbols.planes);
        a.pre_pair_fn = try prompt_mod.function(symbols.pre_pair);
        a.pre_down_fn = try prompt_mod.function(symbols.pre_down);
        try a.pre_pair_fn.allowDynamicShared(pair_prompt_smem);
        try a.pre_down_fn.allowDynamicShared(prompt_smem);
        a.resident = .{ @max(1, try a.pair_fn.occupancy(128, 0)) * sms, @max(1, try a.down_fn.occupancy(128, 0)) * sms };
        a.split_resident = @max(1, try a.split_fn.occupancy(128, 0)) * sms;
        a.split_below = 4 * a.resident[0] * 7 / 8;
        return a;
    }

    fn args(x: u64, x_stride: usize, slots: usize, w: u64, kg: usize, nb: usize, p: Plan, out: u64, n: usize, w4: u64, e4: i32) launch_.Args {
        var a: launch_.Args = .{};
        a.add(x);
        a.add(int(x_stride));
        a.add(int(slots));
        a.add(w);
        a.add(int(kg));
        a.add(int(nb));
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        a.add(int(n));
        a.add(@as(f32, 0.0));
        a.add(w4);
        a.add(e4);
        return a;
    }

    /// Decode rows: `pair` takes token rows through SwiGLU over the pair tables, else pair rows down to fp32 sums.
    pub fn run(e: *const Affine3Experts, s: Stream, pair: bool, in: experts.Rows, l: experts.Layer, p: Plan, out: u64, items: usize) !void {
        const v = view(l, pair);
        const units = items * v.nb;
        const split = pair and units < e.split_below;
        const grid = if (split) @min(units, e.split_resident) else @min((units + 3) / 4, e.resident[@intFromBool(!pair)]);
        if (grid < 1) return;
        var a = args(in.x, in.stride, if (pair) in.slots else 0, v.w, v.kg, v.nb, p, out, v.n, v.w4, l.shared);
        const cfg: launch_.Config = .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 128 }, .shared = 0 };
        try launch_.launch(if (split) e.split_fn else if (pair) e.pair_fn else e.down_fn, cfg, s, &a);
    }

    /// Prompt rows, 64 pairs x 128 columns a CTA: `pair` SwiGLU over the pair tables, else down to bf16 sums.
    pub fn runPrompt(e: *const Affine3Experts, s: Stream, pair: bool, in: experts.Rows, l: experts.Layer, p: Plan, out: u64, items: usize) !void {
        const v = view(l, pair);
        const grid = items * ((v.nb + 3) / 4);
        if (grid < 1) return;
        var a = args(in.x, in.stride, if (pair) in.slots else 0, v.w, v.kg, v.nb, p, out, v.n, v.w4, l.shared);
        const smem = if (pair) pair_prompt_smem else prompt_smem;
        const cfg: launch_.Config = .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 256 }, .shared = smem };
        try launch_.launch(if (pair) e.pre_pair_fn else e.pre_down_fn, cfg, s, &a);
    }

    /// The tables a call reads: the pair form takes D inputs to NI outputs, down NI to D; K in groups of 64.
    fn view(l: experts.Layer, pair: bool) struct { w: u64, w4: u64, kg: usize, nb: usize, n: usize } {
        return if (pair)
            .{ .w = l.up, .w4 = l.shared_up, .kg = l.dims / 64, .nb = l.width / 32, .n = l.width }
        else
            .{ .w = l.down, .w4 = l.shared_down, .kg = l.width / 64, .nb = l.dims / 32, .n = l.dims };
    }

    /// MLX 3-bit words (n, k * 3 / 32) of `count` experts at `words3` into 4-bit words at `words4`.
    pub fn widen(e: *const Affine3Experts, s: Stream, words3: u64, words4: u64, count: usize, n: usize, k: usize) !void {
        const total = count * n * k / 8;
        var a: launch_.Args = .{};
        a.add(words3);
        a.add(words4);
        a.add(@as(c_long, @intCast(total)));
        try launch_.launch(e.widen_fn, .{ .grid = .{ .x = @intCast(@min((total + 255) / 256, 4096)) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// 4-bit blocks at `blocks4` (one table) as 3-bit blocks into the `mi`-th of `m` interleaved tables at `out`.
    pub fn planes(e: *const Affine3Experts, s: Stream, blocks4: u64, out: u64, count: usize, n: usize, k: usize, m: usize, mi: usize) !void {
        const blocks = count * (n / 32) * (k / 64);
        var a: launch_.Args = .{};
        a.add(blocks4);
        a.add(out);
        a.add(@as(c_long, @intCast(blocks)));
        a.add(int(m * words));
        a.add(int(mi * words));
        try launch_.launch(e.planes_fn, .{ .grid = .{ .x = @intCast(@min((blocks * 32 + 255) / 256, 4096)) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// experts.make for 3-bit codes: widen, pack as 4-bit blocks into `scratch`, then planes into the table.
    pub fn packInto(e: *const Affine3Experts, a4: *const Affine4Experts, s: Stream, words3: u64, scales: u64, biases: u64, scratch: [2]u64, out: u64, count: usize, n: usize, k: usize, m: usize, mi: usize) !void {
        try e.widen(s, words3, scratch[0], count, n, k);
        try a4.packInto(s, scratch[0], scales, biases, scratch[1], count, n, k, 1, 0);
        try e.planes(s, scratch[1], out, count, n, k, m, mi);
    }

    /// Scratch bytes packInto needs for `count` experts of n by k: the widened words, then their 4-bit blocks.
    pub fn scratchBytes(count: usize, n: usize, k: usize) [2]usize {
        return .{ count * n * k / 2, Affine4Experts.bytes(count, n, k) };
    }

    /// Bytes of e experts' 3-bit blocks for n outputs from k inputs ([E][n/32][k/64][224] int32).
    pub fn bytes(count: usize, n: usize, k: usize) usize {
        return count * (n / 32) * (k / 64) * words * 4;
    }

    /// Bytes of e experts' 3-bit gate-and-up pair tables ([E][n/32][k/64][2][224] int32).
    pub fn pairBytes(count: usize, n: usize, k: usize) usize {
        return 2 * bytes(count, n, k);
    }
};

test "3-bit tables take 224 words a block and the prompt stages fit the 48 KiB of dynamic shared memory" {
    try std.testing.expectEqual(@as(usize, 2 * 16 * 32 * 224 * 4), Affine3Experts.bytes(2, 512, 2048));
    try std.testing.expectEqual(Affine4Experts.bytes(3, 2048, 512) * 224, Affine3Experts.bytes(3, 2048, 512) * 288);
    try std.testing.expect(Affine3Experts.prompt_smem <= 49152 and Affine3Experts.pair_prompt_smem <= 49152);
    try std.testing.expect(Affine3Experts.prompt_smem >= Affine4Experts.prompt_smem and Affine3Experts.pair_prompt_smem >= Affine4Experts.pair_prompt_smem);
}

test {
    _ = @import("experts3_host_test.zig");
}
