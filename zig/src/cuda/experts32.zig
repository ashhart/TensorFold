//! GLM's routed NVFP4 experts at block 32 with csf0 scales: nvfp4_experts32.cu's kernels, packed on the host.
const std = @import("std");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const grouped = @import("grouped.zig");
const driver = @import("driver.zig");
const experts = @import("experts.zig");

/// A matrix of n outputs from k inputs (k a multiple of 64) as its checkpoint holds it.
pub const Matrix = struct {
    codes: []const u8, // [n, k/2], NVFP4's
    bases: []const u8, // [n] e4m3: a scale every 32 inputs is its row's base plus its group's offset
    offsets: []const u8, // [n, k/64] 4 bits: group g in byte g/2 of its row, an even group's in the low nibble

    /// The first row whose base plus an offset passes 0x7e, e4m3's largest finite: its scale would be NaN or negative.
    pub fn pastFinite(m: Matrix, k: usize) ?usize {
        const pairs = k / 64;
        for (m.bases, 0..) |b, r| {
            var top: u8 = 0;
            for (m.offsets[r * pairs ..][0..pairs]) |o| top = @max(top, o & 0xF, o >> 4);
            if (@as(u16, b) + top > 0x7e) return r;
        }
        return null;
    }
};

pub const Nvfp4b32Experts = struct {
    mod: module.Module,
    fns: [3]module.Function, // by Epi
    resident: [3]usize, // blocks the grid may hold: per SM times SMs
    prompt_fns: [2]module.Function, // SwiGLU gate-up, bf16 sums down

    pub const cols = 32; // output columns a block
    pub const pair_words = 264; // int32 of two k32 groups of one matrix: 256 code words, then 4 of offsets each
    pub const pair4 = 66; // the same in uint4
    pub const base_words = 8; // a matrix's 32 row bases, opening a column block
    pub const warps = 4; // a decode CTA's

    pub const symbols = struct {
        pub const decode = [3][:0]const u8{ "tf_nvfp4_decode32_swiglu", "tf_nvfp4_decode32_down_f32", "tf_nvfp4_decode32_down_bf16" };
        pub const prompt = [2][:0]const u8{ "tf_nvfp4_prompt32_swiglu", "tf_nvfp4_prompt32_down" };
    };

    /// The prompt entries' warps (32-column blocks) a CTA and matrices a block: SwiGLU gate-up, down.
    pub const prompt_wn = [2]usize{ 4, 8 };
    pub const prompt_parts = [2]usize{ 2, 1 };

    /// Prompt4<M, 4, 1, WN>'s dynamic shared bytes: stages of 64 rows' 64 inputs, then WN pairs of `parts`.
    pub fn promptSmem(parts: usize, wn: usize) u32 {
        const stage: u32 = @intCast((64 * 8 + wn * parts * pair4) * 16);
        const stages: u32 = if (stage * 4 <= 49152) 4 else if (stage * 3 <= 49152) 3 else 2;
        return stages * stage;
    }

    pub fn load(d: *const driver.Driver, sms: usize) !Nvfp4b32Experts {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try module.Module.load(d, kernels.nvfp4_experts32);
        errdefer mod.unload();
        var e: Nvfp4b32Experts = .{ .mod = mod, .fns = undefined, .resident = undefined, .prompt_fns = undefined };
        for (symbols.decode, &e.fns, &e.resident) |name, *f, *r| {
            f.* = try mod.function(name);
            r.* = @max(1, try f.occupancy(warps * 32, 0)) * sms;
        }
        for (symbols.prompt, &e.prompt_fns, prompt_parts, prompt_wn) |name, *f, parts, wn| {
            f.* = try mod.function(name);
            try f.allowDynamicShared(promptSmem(parts, wn));
        }
        return e;
    }

    pub fn unload(e: *Nvfp4b32Experts) void {
        e.mod.unload();
    }

    /// Nvfp4Experts.run's outputs on the same scales, on a plan of at most `items` items.
    pub fn run(e: *const Nvfp4b32Experts, s: stream_.Stream, epi: experts.Epi, in: experts.Rows, l: experts.Layer, p: grouped.Plan, out: u64, items: usize, skip: i32) !void {
        if (l.relu2) return error.OptionNotBuilt;
        const nb = (if (epi == .up) l.width else l.dims) / cols;
        const ei: usize = @backingInt(epi);
        const grid = @min((items * nb + warps - 1) / warps, e.resident[ei]);
        if (grid < 1) return;
        var a = args(epi == .up, in, l, p, out, skip);
        try launch_.launch(e.fns[ei], .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = warps * 32, .y = 1, .z = 1 } }, s, &a);
    }

    /// Nvfp4Experts.runPrompt's outputs on the same scales, on a plan of at most `items` items of 64 pairs.
    pub fn runPrompt(e: *const Nvfp4b32Experts, s: stream_.Stream, epi: experts.Epi, in: experts.Rows, l: experts.Layer, p: grouped.Plan, out: u64, items: usize, skip: i32) !void {
        if (epi == .down_f32) return error.EpilogueNotBuilt;
        if (l.relu2) return error.OptionNotBuilt;
        if (items < 1) return;
        const gate = epi == .up;
        const nb = (if (gate) l.width else l.dims) / cols;
        const fi: usize = if (gate) 0 else 1;
        const wn = prompt_wn[fi];
        var a = args(gate, in, l, p, out, skip);
        const cfg: launch_.Config = .{ .grid = .{ .x = @intCast(items * ((nb + wn - 1) / wn)), .y = 1, .z = 1 }, .block = .{ .x = @intCast(wn * 32), .y = 1, .z = 1 }, .shared = promptSmem(prompt_parts[fi], wn) };
        try launch_.launch(e.prompt_fns[fi], cfg, s, &a);
    }

    fn args(gate: bool, in: experts.Rows, l: experts.Layer, p: grouped.Plan, out: u64, skip: i32) launch_.Args {
        var a: launch_.Args = .{};
        a.add(in.x);
        a.add(@as(i32, @intCast(in.stride)));
        a.add(@as(i32, @intCast(if (gate) in.slots else 0)));
        a.add(if (gate) l.up else l.down);
        a.add(if (gate) l.up_scale else l.down_scale);
        a.add(@as(i32, @intCast((if (gate) l.dims else l.width) / 32)));
        a.add(@as(i32, @intCast((if (gate) l.width else l.dims) / cols)));
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        a.add(@as(i32, @intCast(if (gate) l.width else l.dims)));
        a.add(if (gate) l.limit else @as(f32, 0.0));
        a.add(skip);
        return a;
    }

    /// Bytes of `count` experts, n outputs from k inputs: [E][n/32] column blocks of cbWords.
    pub fn bytes(count: usize, n: usize, k: usize, parts: usize) usize {
        return count * (n / cols) * cbWords(k, parts) * 4;
    }

    /// int32 of a column block: each matrix's row bases, then [k/64][parts][264].
    pub fn cbWords(k: usize, parts: usize) usize {
        return parts * (base_words + k / 64 * pair_words);
    }

    /// One expert's matrix as column blocks of `cbWords(k, 1)`.
    pub fn packOne(out: []u32, m: Matrix, n: usize, k: usize) void {
        packPart(out, 1, 0, m, n, k);
    }

    /// Gate and up interleaved, as column blocks of `cbWords(k, 2)`, packed in place.
    pub fn packGateUp(out: []u32, gate: Matrix, up: Matrix, n: usize, k: usize) void {
        packPart(out, 2, 0, gate, n, k);
        packPart(out, 2, 1, up, n, k);
    }

    /// Part `part` of each block: bases at quad t's [j][c] for column j * 8 + 2t + c, then the pairs' words.
    fn packPart(out: []u32, parts: usize, part: usize, m: Matrix, n: usize, k: usize) void {
        const pairs = k / 64;
        const per = cbWords(k, parts);
        std.debug.assert(n % cols == 0 and k % 64 == 0 and out.len == n / cols * per and part < parts);
        std.debug.assert(m.codes.len == n * k / 2 and m.bases.len == n and m.offsets.len == n * pairs);
        for (0..n / cols) |cb| {
            const block = out[cb * per ..][0..per];
            const bases = std.mem.sliceAsBytes(block[part * base_words ..][0..base_words]);
            for (0..4) |t| for (0..4) |j| for (0..2) |c| {
                bases[(t * 4 + j) * 2 + c] = m.bases[cb * cols + j * 8 + t * 2 + c];
            };
            for (0..pairs) |p| {
                const pair = block[parts * base_words + (p * parts + part) * pair_words ..][0..pair_words];
                for (0..2) |odd| {
                    const words = pair[odd * 128 ..][0..128];
                    for (0..8) |gq| for (0..4) |t| for (0..4) |j| {
                        const row = cb * cols + j * 8 + gq;
                        var w: u32 = 0;
                        for (0..2) |h| for (0..4) |q| {
                            const input = (p * 2 + odd) * 32 + h * 16 + t * 4 + q;
                            const byte = m.codes[row * (k / 2) + input / 2];
                            const nib: u32 = if (input & 1 == 0) byte & 0xF else byte >> 4;
                            w |= nib << experts.Nvfp4Experts.slot(h, q);
                        };
                        words[(gq * 4 + t) * 4 + j] = w;
                    };
                    const shift: u3 = @intCast(odd * 4);
                    const offsets = std.mem.sliceAsBytes(pair[256 + odd * 4 ..][0..4]); // quad t's 4 bytes, nibble 2j + c
                    for (0..4) |t| for (0..4) |j| {
                        const row = cb * cols + j * 8 + t * 2;
                        const c0 = (m.offsets[row * pairs + p] >> shift) & 0xF;
                        const c1 = (m.offsets[(row + 1) * pairs + p] >> shift) & 0xF;
                        offsets[t * 4 + j] = c0 | c1 << 4;
                    };
                }
            }
        }
    }
};

const testing = std.testing;
const Y = Nvfp4b32Experts;
const X = experts.Nvfp4Experts;

/// A random matrix of n outputs from k inputs, its offsets also returned as bytes [n, k/32].
fn matrix(a: std.mem.Allocator, rng: std.Random, n: usize, k: usize) !struct { m: Matrix, each: []u8 } {
    const codes = try a.alloc(u8, n * k / 2);
    rng.bytes(codes);
    const bases = try a.alloc(u8, n);
    rng.bytes(bases);
    const each = try a.alloc(u8, n * (k / 32));
    for (each) |*o| o.* = rng.uintLessThan(u8, 16);
    const offsets = try a.alloc(u8, n * (k / 64));
    for (0..n) |r| for (0..k / 64) |p| {
        offsets[r * (k / 64) + p] = each[r * (k / 32) + 2 * p] | each[r * (k / 32) + 2 * p + 1] << 4;
    };
    return .{ .m = .{ .codes = codes, .bases = bases, .offsets = offsets }, .each = each };
}

/// The byte the kernel reads as column `col`'s row base: bq[m], quad t, byte 2j + c.
fn kernelBase(w: []const u32, per: usize, cb: usize, part: usize, col: usize) u8 {
    const t = (col & 7) / 2;
    const j = col / 8;
    const c = col & 1;
    const quad = @as(u64, w[cb * per + part * Y.base_words + t * 2]) | @as(u64, w[cb * per + part * Y.base_words + t * 2 + 1]) << 32;
    return @truncate(quad >> @intCast((j * 2 + c) * 8));
}

/// The offset the kernel adds there for group g: st.s[m], then nibble c of byte j.
fn kernelOffset(w: []const u32, per: usize, parts: usize, cb: usize, part: usize, g: usize, col: usize) u8 {
    const t = (col & 7) / 2;
    const j = col / 8;
    const c = col & 1;
    const o = w[cb * per + parts * Y.base_words + (g / 2 * parts + part) * Y.pair_words + 256 + (g & 1) * 4 + t];
    return @truncate((o >> @intCast(j * 8 + c * 4)) & 15);
}

fn expectLaid(want: anytype, each: []const u8, got: []const u32, block16: []const u32, parts: usize, part: usize, n: usize, k: usize) !void {
    const per = Y.cbWords(k, parts);
    for (0..n / 32) |cb| {
        for (0..32) |col| {
            const row = cb * 32 + col;
            try testing.expectEqual(want.m.bases[row], kernelBase(got, per, cb, part, col));
            for (0..k / 32) |g| try testing.expectEqual(each[row * (k / 32) + g], kernelOffset(got, per, parts, cb, part, g, col));
        }
        for (0..k / 32) |g| {
            const words = got[cb * per + parts * Y.base_words + (g / 2 * parts + part) * Y.pair_words + (g & 1) * 128 ..][0..128];
            const same = block16[((cb * (k / 32) + g) * parts + part) * X.words ..][0..128];
            try testing.expectEqualSlices(u32, same, words);
        }
    }
}

test "packed block 32 keeps block 16's code words, and every base and offset where the kernel reads it" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng: std.Random.DefaultPrng = .init(11);
    const rng = prng.random();
    const n = 64;
    const k = 192;
    const plain = try a.alloc(u8, n * k / 16); // block 16's scales play no part in its code words
    @memset(plain, 0x30);
    const one = try matrix(a, rng, n, k);
    const got = try a.alloc(u32, Y.bytes(1, n, k, 1) / 4);
    @memset(got, 0xdeadbeef);
    Y.packOne(got, one.m, n, k);
    const block16 = try a.alloc(u32, X.bytes(1, n, k, 1) / 4);
    X.packOne(block16, one.m.codes, plain, n, k);
    try expectLaid(one, one.each, got, block16, 1, 0, n, k);

    const gate = try matrix(a, rng, n, k);
    const up = try matrix(a, rng, n, k);
    const both = try a.alloc(u32, Y.bytes(1, n, k, 2) / 4);
    @memset(both, 0xdeadbeef);
    Y.packGateUp(both, gate.m, up.m, n, k);
    const both16 = try a.alloc(u32, X.bytes(1, n, k, 2) / 4);
    X.packGateUp(both16, .{ gate.m.codes, plain }, .{ up.m.codes, plain }, n, k);
    try expectLaid(gate, gate.each, both, both16, 2, 0, n, k);
    try expectLaid(up, up.each, both, both16, 2, 1, n, k);
}

test "a column block's words: 2,048 outputs from 4,096 inputs lose the scale bytes block 16 stored" {
    const n = 2048;
    const k = 4096;
    try testing.expectEqual(@as(usize, 2 * (8 + 64 * 264)), Y.cbWords(k, 2));
    const block16 = X.bytes(1, n, k, 1);
    const scales16 = n * k / 16; // [n, k/16] e4m3
    const scales32 = n + n * k / 64; // a base a row, 4 bits every 32 inputs
    try testing.expectEqual(block16 - scales16 + scales32, Y.bytes(1, n, k, 1));
}
