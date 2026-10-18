//! Bolt's ggml blocks (ggml.cu): IQ2_XXS and Q8_0 repacked into 16-column tiles, for grouped experts and dense rows.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const grouped = @import("grouped.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// The checkpoint's block formats: IQ2_XXS (66 bytes a 256 inputs), Q8_0 (34 bytes a 32).
pub const Kind = enum { iq2xxs, q8_0 };

/// Inputs and bytes of one checkpoint block.
pub fn block(kind: Kind) struct { k: usize, bytes: usize } {
    return switch (kind) {
        .iq2xxs => .{ .k = 256, .bytes = 66 },
        .q8_0 => .{ .k = 32, .bytes = 34 },
    };
}

/// Bytes of 16 columns' blocks side by side (a packed tile's step along K).
pub fn tileBytes(kind: Kind) usize {
    return 16 * block(kind).bytes;
}

/// Bytes of an (n, k) matrix, the checkpoint's and the packed layout's alike.
pub fn bytes(kind: Kind, n: usize, k: usize) usize {
    const b = block(kind);
    return n * (k / b.k) * b.bytes;
}

/// One packed projection on the device: n outputs from k inputs.
pub const Weight = struct { kind: Kind, w: u64, n: usize, k: usize };

pub fn q8Weight(w: u64, n: usize, k: usize) Weight {
    return .{ .kind = .q8_0, .w = w, .n = n, .k = k };
}

pub fn iq2Weight(w: u64, n: usize, k: usize) Weight {
    return .{ .kind = .iq2xxs, .w = w, .n = n, .k = k };
}

/// What a call writes: fp32, bf16 of relu(x)^2 (Nemotron's expert activation), or bf16.
pub const Out = enum { f32, relu2, bf16 };

pub const prompt_rows = 64; // rows a prompt item holds (the plan's prompt tile)
pub const decode_rows = 16; // rows a decode item holds at most (the plan's decode tile)
const prompt_warps = 8; // 16-column tiles a prompt CTA
/// Decode CTAs: 4 warps split a tile's K, or (short K: fewer than `split_loads` loads a tile) a warp a tile, 4 tiles.
const split_loads = 16;
const prompt_smem: u32 = 3 * 64 * 128 * 2; // three stages of 64 rows' 128 inputs

/// Where byte `o` of a packed matrix comes from in the checkpoint's [n][k / KB][B] (tf_ggml_pack's map).
pub fn source(kind: Kind, n: usize, k: usize, o: usize) usize {
    _ = n;
    const b = block(kind);
    const kb = k / b.k;
    const tb_bytes = tileBytes(kind);
    const tb = o / tb_bytes;
    const r = o % tb_bytes;
    const tile = tb / kb;
    const blk = tb % kb;
    var col: usize = undefined;
    var at: usize = undefined;
    if (r >= tb_bytes - 32) {
        const q = r - (tb_bytes - 32);
        col = tile * 16 + (q >> 2) + 8 * ((q >> 1) & 1);
        at = q & 1;
    } else if (kind == .iq2xxs) {
        col = tile * 16 + ((r >> 4) & 7) + 8 * ((r >> 3) & 1);
        at = 2 + (r >> 7) * 8 + (r & 7);
    } else {
        const ln = r >> 4;
        const q = (r >> 2) & 3;
        col = tile * 16 + (ln >> 2) + 8 * (q & 1);
        at = 2 + 8 * (ln & 3) + 4 * (q >> 1) + (r & 3);
    }
    return (col * kb + blk) * b.bytes + at;
}

/// The host repack of one (n, k) matrix, as tf_ggml_pack does it on the device.
pub fn packHost(kind: Kind, out: []u8, raw: []const u8, n: usize, k: usize) void {
    std.debug.assert(out.len == bytes(kind, n, k) and raw.len == out.len and n % 16 == 0);
    for (out, 0..) |*o, i| o.* = raw[source(kind, n, k, i)];
}

fn int(x: usize) c_int {
    return @intCast(x);
}

/// The ggml module's kernels: decode (GEMV) and prompt entries per format and output, and the repack.
pub const Kernels = struct {
    mod: Module,
    gemv: [2][2][3]Function, // by Kind, K split or 4 tiles a CTA, then Out
    prompt: [2][3]Function,
    pack_fn: Function,

    /// Entry names: `form` is "gemv" (K split), "gemv4" (4 tiles a CTA) or "prompt".
    pub fn symbol(comptime kind: Kind, comptime form: []const u8, comptime out: Out) [:0]const u8 {
        return "tf_ggml_" ++ (if (kind == .iq2xxs) "iq2" else "q8") ++ "_" ++ form ++ "_" ++ @tagName(out);
    }

    pub fn load(d: *const Driver) !Kernels {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.ggml);
        errdefer mod.unload();
        var g: Kernels = .{ .mod = mod, .gemv = undefined, .prompt = undefined, .pack_fn = try mod.function("tf_ggml_pack") };
        inline for (.{ Kind.iq2xxs, Kind.q8_0 }, 0..) |kind, ki| inline for (.{ Out.f32, Out.relu2, Out.bf16 }, 0..) |out, oi| {
            g.gemv[ki][0][oi] = try mod.function(symbol(kind, "gemv", out));
            g.gemv[ki][1][oi] = try mod.function(symbol(kind, "gemv4", out));
            g.prompt[ki][oi] = try mod.function(symbol(kind, "prompt", out));
            try g.prompt[ki][oi].allowDynamicShared(prompt_smem);
        };
        return g;
    }

    pub fn unload(g: *Kernels) void {
        g.mod.unload();
    }

    /// `count` checkpoint matrices [n][k / KB][B] back to back at `raw` into packed tiles at `out` (as many bytes).
    pub fn pack(g: *const Kernels, s: Stream, kind: Kind, raw: u64, out: u64, n: usize, k: usize, count: usize) !void {
        if (n % 16 != 0 or k % block(kind).k != 0) return error.ShapeNotTiled;
        const total = count * bytes(kind, n, k);
        if (total == 0) return;
        var a: launch_.Args = .{};
        a.add(raw);
        a.add(out);
        a.add(@as(c_int, @intFromBool(kind == .iq2xxs)));
        a.add(int(n));
        a.add(int(k / block(kind).k));
        a.add(@as(i64, @intCast(total)));
        try launch_.launch(g.pack_fn, .{ .grid = .{ .x = @intCast(@min((total + 255) / 256, 1 << 20)), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }

    fn check(kind: Kind, n: usize, k: usize, prompt: bool) !void {
        if (n % 16 != 0 or k % block(kind).k != 0 or (prompt and k % 128 != 0)) return error.ShapeNotTiled;
    }

    /// The common launch: `items`, `counts` and `members` 0 for dense rows (`rows` of them), else a plan's.
    fn go(g: *const Kernels, s: Stream, prompt: bool, out_kind: Out, kind: Kind, x: u64, stride: usize, slots: usize, w: u64, n: usize, k: usize, p: ?grouped.Plan, out: u64, units: usize, rows: usize, skip: i32) !void {
        try check(kind, n, k, prompt);
        const ki = @backingInt(kind);
        const oi = @backingInt(out_kind);
        const nt16 = n / 16;
        if (units == 0) return;
        var a: launch_.Args = .{};
        a.add(x);
        a.add(int(stride));
        a.add(int(slots));
        a.add(w);
        a.add(int(k / 32));
        a.add(int(nt16));
        for ([_]u64{ if (p) |q| q.items else 0, if (p) |q| q.counts else 0, if (p) |q| q.members else 0, out }) |v| a.add(v);
        a.add(int(n));
        a.add(int(rows));
        a.add(skip);
        if (prompt) {
            const grid = units * ((nt16 + prompt_warps - 1) / prompt_warps);
            const cfg: launch_.Config = .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 32 * prompt_warps }, .shared = prompt_smem };
            return launch_.launch(g.prompt[ki][oi], cfg, s, &a);
        }
        const loads = k / 32 / @as(usize, if (kind == .iq2xxs) 4 else 1);
        const tiles: usize = if (loads < split_loads) 4 else 1;
        const grid = (units * nt16 + tiles - 1) / tiles; // items past the plan's count return at once
        try launch_.launch(g.gemv[ki][@intFromBool(tiles > 1)][oi], .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 128 } }, s, &a);
    }

    /// Dense decode rows (at most 16): out (rows, n) = x (rows, k; rows `ldx` apart, 0 packed) times w.
    pub fn decode(g: *const Kernels, s: Stream, w: Weight, x: u64, ldx: usize, rows: usize, out: u64, out_kind: Out) !void {
        if (rows > decode_rows) return error.TooManyRows;
        return g.go(s, false, out_kind, w.kind, x, if (ldx == 0) w.k else ldx, 0, w.w, w.n, w.k, null, out, @intFromBool(rows > 0), rows, -1);
    }

    /// Dense prompt rows (any count): 64 rows by 128 columns a CTA.
    pub fn promptRows(g: *const Kernels, s: Stream, w: Weight, x: u64, ldx: usize, rows: usize, out: u64, out_kind: Out) !void {
        return g.go(s, true, out_kind, w.kind, x, if (ldx == 0) w.k else ldx, 0, w.w, w.n, w.k, null, out, (rows + prompt_rows - 1) / prompt_rows, rows, -1);
    }

    /// Either form by row count: decode at 16 rows or fewer, else prompt.
    pub fn matmul(g: *const Kernels, s: Stream, w: Weight, x: u64, ldx: usize, rows: usize, out: u64, out_kind: Out) !void {
        return if (rows <= decode_rows) g.decode(s, w, x, ldx, rows, out, out_kind) else g.promptRows(s, w, x, ldx, rows, out, out_kind);
    }

    /// Grouped experts, at most `items` items: [E] packed (n, k) at `w`; x token rows if `slots` > 0, else pair rows.
    pub fn experts(g: *const Kernels, s: Stream, prompt: bool, kind: Kind, out_kind: Out, x: u64, stride: usize, slots: usize, w: u64, n: usize, k: usize, p: grouped.Plan, out: u64, items: usize, skip: i32) !void {
        return g.go(s, prompt, out_kind, kind, x, stride, slots, w, n, k, p, out, items, 0, skip);
    }
};

test "the repack is a permutation of each matrix's bytes" {
    const gpa = std.testing.allocator;
    inline for (.{ Kind.iq2xxs, Kind.q8_0 }) |kind| {
        const n = 32;
        const k = 512;
        const len = bytes(kind, n, k);
        const seen = try gpa.alloc(bool, len);
        defer gpa.free(seen);
        @memset(seen, false);
        for (0..len) |o| {
            const src = source(kind, n, k, o);
            try std.testing.expect(!seen[src]);
            seen[src] = true;
        }
    }
    // q8_0: tile 1, block 3, lane 13 (gq 3, t 1), step 1, column gq + 8, byte 2: column 27, input 8 + 4 + 2
    const o = (1 * 16 + 3) * 544 + 13 * 16 + 3 * 4 + 2;
    try std.testing.expectEqual((27 * 16 + 3) * 34 + 2 + 8 + 4 + 2, source(.q8_0, 32, 512, o));
    // iq2xxs: tile 0, block 1, group 5, gq 6, column gq + 8, byte 7; then column 6's d byte 1
    try std.testing.expectEqual((14 * 2 + 1) * 66 + 2 + 40 + 7, source(.iq2xxs, 32, 512, 1056 + 5 * 128 + 6 * 16 + 8 + 7));
    try std.testing.expectEqual((6 * 2 + 1) * 66 + 1, source(.iq2xxs, 32, 512, 1056 + 1024 + 6 * 4 + 1));
}
