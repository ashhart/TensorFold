//! Stored-format experts for Inkling 2 Mini: MXFP8 (fp8_experts.cu) and the soft-capped SwiGLU on NVFP4's kernels.
const std = @import("std");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const driver = @import("driver.zig");
const experts = @import("experts.zig");

const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;
const Plan = @import("grouped.zig").Plan;
const Layer = experts.Layer;
const Epi = experts.Epi;
const Rows = experts.Rows;

/// NVFP4's per-expert factor: amax / denominator, each rounded to bf16 as the reference loads it (open question F1).
pub fn nvfp4Factor(amax: f32, denominator: f32) f32 {
    return bf16(amax) / bf16(denominator);
}

/// x rounded to bf16 (nearest, ties to even), as an f32; the checkpoint's amax values are finite.
fn bf16(x: f32) f32 {
    const b: u32 = @bitCast(x);
    return @bitCast((b +% 0x7FFF +% ((b >> 16) & 1)) & 0xFFFF0000);
}

/// A layer's caps: both 0 (plain), or both positive and finite (soft-capped SwiGLU); refused where no kernel has them.
pub fn capsBuilt(l: Layer) !void {
    if (l.caps[0] == 0 and l.caps[1] == 0) {
        if (l.format == .mxfp8) return error.OptionNotBuilt; // MXFP8's gate-up is built soft-capped only
        return;
    }
    for (l.caps) |c| if (!(c > 0 and c < std.math.inf(f32))) return error.BadSoftcap;
    if (l.format != .nvfp4 and l.format != .mxfp8) return error.OptionNotBuilt;
    if (l.relu2 or l.limit != 0 or l.swiglu or l.after) return error.OptionNotBuilt;
    if (l.width % 32 != 0 or l.dims % 32 != 0) return error.UnexpectedShape; // whole blocks of 32 both ways
}

/// Inkling's w13 interleaves rows (2j gate j, 2j + 1 up j): one expert's rows of `row_bytes` split into gate, up.
pub fn splitGateUp(rows: []const u8, row_bytes: usize, gate: []u8, up: []u8) void {
    const pairs = rows.len / (2 * row_bytes);
    std.debug.assert(rows.len == 2 * pairs * row_bytes and gate.len == pairs * row_bytes and up.len == gate.len);
    for (0..pairs) |j| {
        @memcpy(gate[j * row_bytes ..][0..row_bytes], rows[2 * j * row_bytes ..][0..row_bytes]);
        @memcpy(up[j * row_bytes ..][0..row_bytes], rows[(2 * j + 1) * row_bytes ..][0..row_bytes]);
    }
}

/// The caps as the kernels take them: a negative up cap asks for gate and up rounded to bf16 first (round_caps).
pub fn kernelCaps(l: Layer) [2]f32 {
    return .{ l.caps[0], if (l.round_caps) -l.caps[1] else l.caps[1] };
}

/// An expert call's arguments as the soft-capped and MXFP8 entries take them; `scale` null for MXFP8 (in its blocks).
pub fn argsOf(in: Rows, gate: bool, w: u64, scale: ?u64, kg: usize, nb: usize, p: Plan, out: u64, n: usize, caps: [2]f32, skip: i32) launch_.Args {
    var a: launch_.Args = .{};
    a.add(in.x);
    a.add(@as(i32, @intCast(in.stride)));
    a.add(@as(i32, @intCast(if (gate) in.slots else 0)));
    a.add(w);
    if (scale) |v| a.add(v);
    a.add(@as(i32, @intCast(kg)));
    a.add(@as(i32, @intCast(nb)));
    for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
    a.add(@as(i32, @intCast(n)));
    a.add(caps[0]);
    a.add(caps[1]);
    a.add(skip);
    return a;
}

/// nvfp4_experts.cu's soft-capped SwiGLU gate-up: decode units of 4 warps, and the prompt tiles of 4 warps.
pub const Softcap = struct {
    fns: [2]Function = undefined, // decode, prompt
    resident: usize = 0,

    pub const symbols = [2][:0]const u8{ "tf_nvfp4_softcap", "tf_nvfp4_prompt_softcap" };
    pub const prompt_wn = 4;

    pub fn resolve(mod: Module, sms: usize) !Softcap {
        var c: Softcap = .{};
        for (symbols, &c.fns) |name, *f| f.* = try mod.function(name);
        try c.fns[1].allowDynamicShared(experts.Nvfp4Experts.promptSmem(2, prompt_wn));
        c.resident = @max(1, try c.fns[0].occupancy(128, 0)) * sms;
        return c;
    }

    /// Gate-up on a plan of at most `items` items (16 pairs decode, 64 prompt); capsBuilt checked the layer.
    pub fn run(c: Softcap, s: Stream, prompt: bool, in: Rows, l: Layer, p: Plan, out: u64, items: usize, skip: i32) !void {
        const nb = l.width / 32;
        var a = argsOf(in, true, l.up, l.up_scale, l.dims / 32, nb, p, out, l.width, kernelCaps(l), skip);
        const grid = if (prompt) items * ((nb + prompt_wn - 1) / prompt_wn) else @min((items * nb + 3) / 4, c.resident);
        if (grid < 1) return;
        const smem: u32 = if (prompt) experts.Nvfp4Experts.promptSmem(2, prompt_wn) else 0;
        const cfg: launch_.Config = .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 128 }, .shared = smem };
        try launch_.launch(c.fns[@intFromBool(prompt)], cfg, s, &a);
    }
};

/// MXFP8 experts: e4m3 codes with one E8M0 scale per output row per 32 inputs, blocks of 32 columns by 32 inputs.
pub const Mxfp8Experts = struct {
    mod: Module,
    fns: [5]Function, // decode: soft-capped gate-up, fp32 down, bf16 down; prompt: soft-capped gate-up, bf16 down
    resident: [3]usize, // the decode entries' blocks a grid may hold

    pub const words = 264; // int32 a block: fp8g's 256 code words ([32 lanes][2][4]), then 32 E8M0 bytes
    pub const prompt_cw = [2]usize{ 2, 4 }; // column warps a prompt CTA: gate-up, down (64 pairs an item)
    pub const symbols = [5][:0]const u8{ "tf_mxfp8_softcap", "tf_mxfp8_down_f32", "tf_mxfp8_down_bf16", "tf_mxfp8_prompt_softcap", "tf_mxfp8_prompt_down" };

    pub fn load(d: *const driver.Driver, sms: usize) !Mxfp8Experts {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.fp8_experts);
        errdefer mod.unload();
        var e: Mxfp8Experts = .{ .mod = mod, .fns = undefined, .resident = undefined };
        for (symbols, &e.fns) |name, *f| f.* = try mod.function(name);
        for (&e.resident, 0..) |*r, i| r.* = @max(1, try e.fns[i].occupancy(128, 0)) * sms;
        return e;
    }

    pub fn unload(e: *Mxfp8Experts) void {
        e.mod.unload();
    }

    /// The entry a call takes, or the refusal: prompt rows have no fp32 down.
    pub fn entry(prompt: bool, epi: Epi) !usize {
        return switch (epi) {
            .up => if (prompt) 3 else 0,
            .down_f32 => if (prompt) error.EpilogueNotBuilt else 1,
            .down_bf16 => if (prompt) 4 else 2,
        };
    }

    /// One call on a plan of at most `items` items (16 pairs decode, 64 prompt); the layer's caps already checked.
    pub fn run(e: *const Mxfp8Experts, s: Stream, prompt: bool, epi: Epi, in: Rows, l: Layer, p: Plan, out: u64, items: usize, skip: i32) !void {
        const fi = try entry(prompt, epi);
        const gate = epi == .up;
        const n = if (gate) l.width else l.dims;
        const k = if (gate) l.dims else l.width;
        if (n % 32 != 0 or k % 32 != 0) return error.UnexpectedShape;
        const nb = n / 32;
        const cw = prompt_cw[@intFromBool(!gate)];
        if (prompt and nb % cw != 0) return error.UnexpectedShape;
        const grid = if (prompt) items * (nb / cw) else @min((items * nb + 3) / 4, e.resident[fi]);
        if (grid < 1) return;
        var a = argsOf(in, gate, if (gate) l.up else l.down, null, k / 32, nb, p, out, n, kernelCaps(l), skip);
        try launch_.launch(e.fns[fi], .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 128 } }, s, &a);
    }

    /// Bytes of `count` experts' blocks for n outputs from k inputs: [E][n/32][k/32][parts][264] int32.
    pub fn bytes(count: usize, n: usize, k: usize, parts: usize) usize {
        return count * (n / 32) * (k / 32) * parts * words * 4;
    }

    /// One expert's e4m3 codes [n, k] and E8M0 scales [n, k/32] (row-major, as stored) as blocks [n/32][k/32][264].
    pub fn packOne(out: []u32, codes: []const u8, scales: []const u8, n: usize, k: usize) void {
        packPart(out, 1, 0, codes, scales, n, k);
    }

    /// Gate and up interleaved per (column block, k32 group): [n/32][k/32][2][264], each [codes, scales] as stored.
    pub fn packGateUp(out: []u32, gate: [2][]const u8, up: [2][]const u8, n: usize, k: usize) void {
        packPart(out, 2, 0, gate[0], gate[1], n, k);
        packPart(out, 2, 1, up[0], up[1], n, k);
    }

    /// Lane (gq, t)'s words [h][j]: row j * 8 + gq, inputs 16h + 4t .. + 3 (fp8g's order); scale byte 4 gq + j.
    fn packPart(out: []u32, parts: usize, part: usize, codes: []const u8, scales: []const u8, n: usize, k: usize) void {
        const kg = k / 32;
        std.debug.assert(n % 32 == 0 and k % 32 == 0 and out.len == n / 32 * kg * parts * words and part < parts);
        std.debug.assert(codes.len == n * k and scales.len == n * kg);
        for (0..n / 32) |cb| for (0..kg) |g| {
            const block = out[((cb * kg + g) * parts + part) * words ..][0..words];
            for (0..8) |gq| for (0..4) |t| for (0..2) |h| for (0..4) |j| {
                const row = cb * 32 + j * 8 + gq;
                const at = row * k + g * 32 + h * 16 + t * 4;
                block[((gq * 4 + t) * 2 + h) * 4 + j] = std.mem.readInt(u32, codes[at..][0..4], .little);
            };
            const sc = std.mem.sliceAsBytes(block[256..]);
            for (0..8) |gq| for (0..4) |j| {
                sc[gq * 4 + j] = scales[(cb * 32 + j * 8 + gq) * kg + g];
            };
        };
    }
};

test "the NVFP4 factor rounds amax and the denominator to bf16 before it divides" {
    const amax: f32 = 0.3310546875 + 0x1p-12; // not a bf16
    const want = @as(f32, 0.33203125) / 2688.0; // bf16(amax) is 0.33203125; 2688 = 6 x 448 is a bf16
    try std.testing.expectEqual(want, nvfp4Factor(amax, 2688));
    try std.testing.expect(nvfp4Factor(amax, 2688) != amax / 2688);
}

test "w13's row 1 is up row 0 and row 2 gate row 1: a code planted in row 1 lands in the up part, never the gate" {
    const n = 32; // gate and up rows each
    const k = 32;
    var rows: [2 * n * k]u8 = @splat(0x00);
    var sc: [2 * n]u8 = @splat(127);
    rows[1 * k + 5] = 0x38; // stored row 1 (up row 0), input 5
    rows[2 * k + 7] = 0x40; // stored row 2 (gate row 1), input 7
    var g: [n * k]u8 = undefined;
    var u: [n * k]u8 = undefined;
    var gs: [n]u8 = undefined;
    var us: [n]u8 = undefined;
    splitGateUp(&rows, k, &g, &u);
    splitGateUp(&sc, 1, &gs, &us);
    try std.testing.expectEqual(@as(u8, 0x38), u[5]);
    try std.testing.expectEqual(@as(u8, 0x40), g[k + 7]);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, &u, &.{0x38}) + std.mem.count(u8, &u, &.{0x40}));
    var out: [2 * Mxfp8Experts.words]u32 = undefined;
    Mxfp8Experts.packGateUp(&out, .{ &g, &gs }, .{ &u, &us }, n, k);
    // row 0, input 5 = lane (gq 0, t 1), half 0, tile j 0, byte 1 of the word; part 1 (up) holds it, part 0 not
    const at = ((0 * 4 + 1) * 2 + 0) * 4 + 0;
    try std.testing.expectEqual(@as(u32, 0x38), (out[Mxfp8Experts.words + at] >> 8) & 0xFF);
    try std.testing.expectEqual(@as(u32, 0), (out[at] >> 8) & 0xFF);
}

test "a packed MXFP8 block holds fp8g's code words and each row's scale at byte 4 gq + j" {
    const gpa = std.testing.allocator;
    const n = 64;
    const k = 96;
    const codes = try gpa.alloc(u8, n * k);
    defer gpa.free(codes);
    for (codes, 0..) |*c, i| c.* = @truncate(i * 31 + 7);
    var scales: [n * k / 32]u8 = undefined;
    for (&scales, 0..) |*c, i| c.* = @truncate(i * 13 + 5);
    const out = try gpa.alloc(u32, Mxfp8Experts.bytes(1, n, k, 1) / 4);
    defer gpa.free(out);
    Mxfp8Experts.packOne(out, codes, &scales, n, k);
    // the code words of every block are Fp8Experts.packOne's order (fp8_experts.cu's lane loads)
    for (0..n / 32) |cb| for (0..k / 32) |g| for (0..8) |gq| for (0..4) |t| for (0..2) |h| for (0..4) |j| {
        const row = cb * 32 + j * 8 + gq;
        const w = std.mem.readInt(u32, codes[row * k + g * 32 + h * 16 + t * 4 ..][0..4], .little);
        try std.testing.expectEqual(w, out[(cb * (k / 32) + g) * Mxfp8Experts.words + ((gq * 4 + t) * 2 + h) * 4 + j]);
    };
    // block (cb 1, g 2): row 32 + 8 j + gq's scale for group 2 at byte 4 gq + j of the block's last 32
    const sc = std.mem.sliceAsBytes(out[(1 * 3 + 2) * Mxfp8Experts.words + 256 ..][0..8]);
    for (0..8) |gq| for (0..4) |j| try std.testing.expectEqual(scales[(32 + 8 * j + gq) * 3 + 2], sc[gq * 4 + j]);
}

test "packGateUp stacks gate then up per (column block, k32 group)" {
    const n = 32;
    const k = 64;
    var codes: [2][n * k]u8 = undefined;
    var scales: [2][n * k / 32]u8 = undefined;
    for (0..2) |m| {
        for (&codes[m], 0..) |*c, i| c.* = @truncate(i * 29 + 3 + m * 101);
        for (&scales[m], 0..) |*c, i| c.* = @truncate(i * 7 + 1 + m * 13);
    }
    var one: [2][Mxfp8Experts.bytes(1, n, k, 1) / 4]u32 = undefined;
    for (0..2) |m| Mxfp8Experts.packOne(&one[m], &codes[m], &scales[m], n, k);
    var both: [2 * one[0].len]u32 = undefined;
    Mxfp8Experts.packGateUp(&both, .{ &codes[0], &scales[0] }, .{ &codes[1], &scales[1] }, n, k);
    const w = Mxfp8Experts.words;
    for (0..one[0].len / w) |b| for (0..2) |m| {
        try std.testing.expectEqualSlices(u32, one[m][b * w ..][0..w], both[(2 * b + m) * w ..][0..w]);
    };
}

test {
    _ = @import("experts_mx_test.zig"); // dispatch, arguments and refusals on a mock launcher
}
