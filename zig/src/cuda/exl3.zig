//! The EXL3 decode-row linear (exl3/linear.cu's device code): the plan, a layer's tensors, and the two kernels.
const std = @import("std");
const candidate = @import("exl3_candidate.zig");
const platform = @import("sys");
const driver = @import("driver.zig");
const abi = @import("abi.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const memory = @import("memory.zig");
const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// x's and y's element types (linear.cu DType).
pub const DType = enum(i32) { f16 = 0, bf16 = 1, f32 = 2 };

/// The three codebooks (exl3_format.zig's, numbered as the kernel's).
pub const Codebook = enum(u3) { inst3 = 0, mcg = 1, mul1 = 2 };

/// One compiled kernel: 2 * bits halves, a codebook, and the warps one block runs.
pub const Variant = struct { halves: u32, codebook: Codebook, warps: u32 };

/// Every instantiation of exl3_linear.cu: every width (whole bits, the mul1 half steps), 2, 4 and 8 warps.
pub const variants = blk: {
    var vs: [90]Variant = undefined;
    var i: usize = 0;
    const counts = [_]u32{ 4, 8, 2 }; // 2 last: the 4- and 8-warp indices stay as they were
    for (counts) |warps| {
        for ([_]u32{ 2, 1, 0 }) |cbi| {
            const cb: Codebook = @fromBackingInt(@intCast(cbi));
            var halves: u32 = 2;
            while (halves <= 16) : (halves += 2) {
                vs[i] = .{ .halves = halves, .codebook = cb, .warps = warps };
                i += 1;
            }
            if (cb != .mul1) continue; // the half steps exist only with mul1
            halves = 3;
            while (halves <= 13) : (halves += 2) { // the mul1 half steps, 1.5 to 6.5 bits (format.py's BITS)
                vs[i] = .{ .halves = halves, .codebook = cb, .warps = warps };
                i += 1;
            }
        }
    }
    break :blk vs;
};

/// The kernel's extern "C" name for an instantiation (linear.cu's TF_EXL3_LINEAR macro).
pub fn symbol(comptime halves: u32, comptime codebook: Codebook, comptime warps: u32) [:0]const u8 {
    return std.fmt.comptimePrint("tf_exl3_linear_{d}_{d}_{d}", .{ halves, @backingInt(codebook), warps });
}

pub const rot_in_symbol = "tf_exl3_rot_in";

/// The mid-M kernels (17-128 rows, the same bits as the linear's): mul1 at 3 to 6 bits, 2, 4 and 8 warps, P 1 and 2.
pub const Mpg = struct { halves: u32, warps: u32, p: u32 };
pub const mpg_variants = blk: {
    var vs: [24]Mpg = undefined;
    var i: usize = 0;
    for ([_]u32{ 6, 8, 10, 12 }) |h| for ([_]u32{ 1, 2 }) |p| for ([_]u32{ 2, 4, 8 }) |w| {
        vs[i] = .{ .halves = h, .warps = w, .p = p };
        i += 1;
    };
    break :blk vs;
};

/// linear_wc (17-128 rows, 4- and 6-bit mul1): 16 mt rows a program, ks k steps a stage, fold: every split in one.
pub const Wc = struct { halves: u32, mt: u32, ks: u32, fold: bool };
pub const wc_variants = blk: {
    var vs: [16]Wc = undefined;
    var i: usize = 0;
    for ([_]u32{ 8, 12 }) |h| for ([_]bool{ false, true }) |f| for ([_]u32{ 2, 4 }) |mt| for ([_]u32{ 2, 4 }) |ks| {
        vs[i] = .{ .halves = h, .mt = mt, .ks = ks, .fold = f };
        i += 1;
    };
    break :blk vs;
};

fn wcShared(v: Wc) u32 {
    const raw = 8 * 4 * v.halves;
    const stage = v.ks * raw * 4 + v.ks * 16 * v.mt * 32;
    return @max(4 * stage, 16 * v.mt * 128 * 4);
}

/// linear.cu dispatch_wc: which linear_wc takes m rows of this weight, or null for the mid-M kernels.
pub fn wcPick(w: Weight, m: usize, sk: u32, wk: u32, min_rows: usize) ?usize {
    if (w.codebook != .mul1 or (w.halves != 8 and w.halves != 12)) return null;
    const per_warp = w.k / 16 / sk / wk;
    if (per_warp % 2 != 0 or m < min_rows or m > 128) return null;
    const nb = w.n / 128;
    const fold = sk > 1 and nb >= 32;
    if (!fold and w.halves == 8 and m > 32 and m <= 40) return null;
    const four = per_warp % 4 == 0 and !(w.halves == 12 and m > 96);
    const mt2 = m <= 32 or (m > 64 and m <= 96) or (fold and m > 96 and nb >= 64);
    const want: Wc = .{ .halves = w.halves, .mt = if (mt2) 2 else 4, .ks = if (four) 4 else 2, .fold = fold };
    for (wc_variants, 0..) |v, i| if (std.meta.eql(v, want)) return i;
    return null;
}

fn mpgIndex(halves: u32, codebook: Codebook, warps: u32, p: u32) ?usize {
    if (codebook != .mul1) return null;
    for (mpg_variants, 0..) |v, i| if (v.halves == halves and v.warps == warps and v.p == p) return i;
    return null;
}

pub fn variantIndex(halves: u32, codebook: Codebook, warps: u32) ?usize {
    for (variants, 0..) |v, i| {
        if (v.halves == halves and v.codebook == codebook and v.warps == warps) return i;
    }
    return null;
}

/// From this many 128-column blocks a shape plans one K split, so a view that wide keeps the whole matrix's bits.
pub const wide_blocks = 64;

/// What one shape launches, from the shape alone so a row's bits never depend on the row count (linear.py plan).
pub fn plan(k: usize, n: usize, blocks: usize, min_tiles: usize) struct { sk: u32, wk: u32 } {
    const kt = k / 16;
    const nb = n / 128;
    var sk: usize = 1;
    var wk: usize = if (nb >= wide_blocks) 8 else 4;
    if (nb < wide_blocks) {
        while (nb * sk < blocks and sk < 64) sk *= 2;
    }
    while ((wk > 4 or sk > 1) and (kt % (sk * wk) != 0 or kt / (sk * wk) < min_tiles)) {
        if (wk > 4) wk = 4 else if (sk > 1) sk /= 2 else break;
    }
    return .{ .sk = @intCast(sk), .wk = @intCast(wk) };
}

/// Words between k tiles and 128-column blocks of one layout: strips [N/128, K/16, 8, W] or stored [K/16, N/16, W].
pub fn strides(k: usize, n: usize, halves: usize, strips: bool) struct { k: usize, nb: usize } {
    const words = halves * 4;
    return if (strips)
        .{ .k = 8 * words, .nb = (k / 16) * 8 * words }
    else
        .{ .k = (n / 16) * words, .nb = 8 * words };
}

/// One layer's tensors on the device: the trellis words, the Hadamard scales, the optional bias, shape and layout.
pub const Weight = struct {
    trellis: u64, // u32 words, `stride_k` and `stride_nb` apart
    suh: u64, // fp16 [k]
    svh: u64, // fp16 [n]
    bias: u64 = 0, // fp16 [n], 0 when the layer has none
    k: u32,
    n: u32,
    halves: u32, // 2 * bits, odd for the mul1 half steps
    codebook: Codebook,
    stride_k: u64,
    stride_nb: u64,
    sk: u32 = 0, // the K splits and warps a program the family fixed for this shape; 0: plan()'s
    wk: u32 = 0,

    /// Columns [from, from + n) as a view of the stored matrix: whole 128-column blocks, the parent's strides.
    pub fn columns(w: Weight, from: u32, n: u32) !Weight {
        if (from % 128 != 0 or n % 128 != 0 or n == 0 or @as(u64, from) + n > w.n) return error.Exl3Shape;
        var v = w;
        v.n = n;
        v.sk = 0; // a narrower view plans as its own shape
        v.wk = 0;
        v.trellis = w.trellis + @as(u64, from / 128) * w.stride_nb * 4;
        v.svh = w.svh + @as(u64, from) * 2;
        if (w.bias != 0) v.bias = w.bias + @as(u64, from) * 2;
        return v;
    }
};

/// A weight's launch: the family's fixed (sk, wk) when given, else plan()'s; either is the shape's alone.
pub fn planOf(w: Weight) struct { sk: u32, wk: u32 } {
    if (w.sk != 0) return .{ .sk = w.sk, .wk = w.wk };
    const p = plan(w.k, w.n, 192, 8);
    return .{ .sk = p.sk, .wk = p.wk };
}

/// Whether the current context's device takes programmatic dependent launches (compute capability 9.0 or newer).
fn dependentLaunches(d: *const Driver) bool {
    var dev: abi.Device = 0;
    var major: c_int = 0;
    if (d.api.cuCtxGetDevice(&dev) != abi.success) return false;
    return d.api.cuDeviceGetAttribute(&major, .compute_capability_major, dev) == abi.success and major >= 9;
}

/// The module and its kernels.
pub const Linear = struct {
    mod: Module,
    fns: [variants.len]Function,
    rot_in: Function,
    mpg: [mpg_variants.len]Function = undefined,
    group: [2]Function = undefined, // tf_exl3_group_8_2_{4,8}: projection groups of 4-bit mul1 layers
    rot_group: Function = undefined,
    wc: [wc_variants.len]Function = undefined,
    /// 17-128 rows of 4- and 6-bit mul1 layers on linear_wc (false: the mid-M kernels; the same bits either way).
    use_wc: bool = true,
    /// Programmatic dependent launch for the rotations and the decode-row linears (TF_EXL3_PDL=0: plain launches).
    pdl: bool = true,
    /// 17-128 rows on the mid-M kernels where a width has them (false: the linear at every row count, same bits).
    midm: bool = true,
    /// The least rows the mid-M kernels and linear_wc take (the kernel contract: 1, proving their bits at every count).
    min_rows: usize = 17,

    pub fn load(d: *const Driver) !Linear {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var l: Linear = .{ .mod = try Module.load(d, kernels.exl3_linear), .fns = undefined, .rot_in = undefined };
        errdefer l.mod.unload();
        l.rot_in = try l.mod.function(rot_in_symbol);
        inline for (0..variants.len) |i| {
            l.fns[i] = try l.mod.function(symbol(variants[i].halves, variants[i].codebook, variants[i].warps));
        }
        inline for (mpg_variants, 0..) |v, i| {
            l.mpg[i] = try l.mod.function(std.fmt.comptimePrint("tf_exl3_mpg_{d}_{d}_{d}", .{ v.halves, v.warps, v.p }));
        }
        inline for (wc_variants, 0..) |v, i| {
            l.wc[i] = try l.mod.function(std.fmt.comptimePrint("tf_exl3_wc_{d}_{d}_{d}_{d}", .{ v.halves, v.mt, v.ks, @intFromBool(v.fold) }));
            try l.wc[i].allowDynamicShared(wcShared(v));
        }
        l.group = .{ try l.mod.function("tf_exl3_group_8_2_4"), try l.mod.function("tf_exl3_group_8_2_8") };
        l.rot_group = try l.mod.function("tf_exl3_rot_group");
        l.pdl = dependentLaunches(d); // sm_90 and newer; the kernels' waits compile to nothing below
        if (platform.c.getenv("TF_EXL3_PDL")) |v| l.pdl = !std.mem.eql(u8, std.mem.span(v), "0");
        return l;
    }

    pub fn unload(l: *Linear) void {
        l.mod.unload();
    }

    /// Shared bytes one call needs: WK * min(m, 8) * 128 floats of warp sums, at most 32 KB.
    pub fn sharedBytes(w: Weight, m: usize) u32 {
        const p = planOf(w);
        return @intCast(p.wk * @min(m, 8) * 128 * 4);
    }

    /// Split partials one call needs (fp32 [sk, m, n]) when the plan splits K, else 0.
    pub fn zBytes(w: Weight, m: usize) usize {
        const p = planOf(w);
        return if (p.sk > 1) @as(usize, p.sk) * m * w.n * 4 else 0;
    }

    /// Counter bytes one call needs: per pass and block, zeroed (fill8(0), never zeros()).
    pub fn counterBytes(w: Weight) usize {
        return 8 * (w.n / 128) * 4;
    }

    /// y [m, n] = x [m, k] through the weight; `xh` is fp16 [m, k] scratch, `z` the split partials, `counters` zeroed.
    pub fn matmul(l: *const Linear, s: Stream, w: Weight, x: u64, x_dtype: DType, m: usize, y: u64, y_dtype: DType, xh: u64, z: u64, counters: u64) !void {
        // Shapes the loader admits (linear.py's k % 128 and n % 128, 1..128 rows) are returned errors here too.
        if (m < 1 or m > 128) return error.Exl3Shape;
        if (w.k % 128 != 0 or w.n % 128 != 0) return error.Exl3Shape;
        const p = planOf(w);
        if ((w.k / 16) % (p.sk * p.wk) != 0) return error.Exl3Shape;
        const idx = variantIndex(w.halves, w.codebook, p.wk) orelse return error.UnsupportedExl3Width;
        if (p.sk > 1 and z == 0) return error.Exl3Scratch;

        var r: launch_.Args = .{};
        r.add(x);
        r.add(w.suh);
        r.add(xh);
        r.add(@as(i32, @backingInt(x_dtype)));
        r.add(@as(i32, @intCast(m)));
        r.add(@as(i32, @intCast(w.k)));
        try launch_.launch(l.rot_in, .{
            .grid = .{ .x = @intCast((w.k / 128 + 3) / 4), .y = @intCast(m), .z = 1 },
            .block = .{ .x = 128, .y = 1, .z = 1 },
            .shared = 0,
            .pdl = l.pdl, // tf_exl3_rot_in waits for the previous kernel's rows itself
        }, s, &r);

        var args: launch_.Args = .{};
        args.add(xh);
        args.add(w.trellis);
        args.add(@as(i64, @intCast(w.stride_k)));
        args.add(@as(i64, @intCast(w.stride_nb)));
        args.add(w.svh);
        args.add(w.bias);
        args.add(y);
        args.add(if (p.sk > 1) z else @as(u64, 0));
        args.add(counters);
        args.add(@as(i32, @backingInt(y_dtype)));
        args.add(@as(i32, @intCast(w.n)));
        args.add(@as(i32, @intCast(p.sk)));
        args.add(@as(i32, @intCast(m)));
        args.add(@as(i32, @intCast(w.k)));
        // 17-48 rows: 16-row groups; 49-128: 32-row groups, each group's tiles decoded once (linear.cu's dispatch)
        if (l.midm and l.use_wc) if (wcPick(w, m, p.sk, p.wk, l.min_rows)) |wi| {
            const v = wc_variants[wi];
            args.add(@as(i32, @intCast(p.wk))); // WcArgs: LinearArgs, then the plan's warps
            const rg = 16 * v.mt;
            return launch_.launch(l.wc[wi], .{
                .grid = .{ .x = @intCast(w.n / 128 * ((m + rg - 1) / rg)), .y = if (v.fold) 1 else p.sk, .z = 1 },
                .block = .{ .x = 256, .y = 1, .z = 1 },
                .shared = wcShared(v),
            }, s, &args);
        };
        const pg: u32 = if (m > 48) 2 else 1;
        if (l.midm and m >= l.min_rows) if (mpgIndex(w.halves, w.codebook, p.wk, pg)) |mi| {
            const groups = (m + 16 * pg - 1) / (16 * pg);
            return launch_.launch(l.mpg[mi], .{
                .grid = .{ .x = @intCast(w.n / 128 * groups), .y = p.sk, .z = 1 },
                .block = .{ .x = @intCast(p.wk * 32), .y = 1, .z = 1 },
                .shared = sharedBytes(w, m),
            }, s, &args);
        };
        try launch_.launch(l.fns[idx], .{
            .grid = .{ .x = @intCast(w.n / 128 * p.sk), .y = 1, .z = 1 },
            .block = .{ .x = @intCast(p.wk * 32), .y = 1, .z = 1 },
            .shared = sharedBytes(w, m),
            .pdl = l.pdl, // prefetches its weights, then waits for the rotation
        }, s, &args);
    }

    /// y_i = x through each of `ws` in one rotation and one launch (a member's own bits); scratch back to back, zeroed.
    pub fn matmulGroup(l: *const Linear, s: Stream, ws: []const Weight, x: u64, x_dtype: DType, m: usize, ys: []const u64, y_dtype: DType, xh: u64, z: u64, counters: u64) !void {
        if (!groupable(ws, m) or ys.len != ws.len) return error.Exl3Group;
        const k = ws[0].k;
        var rot: RotGroupC = .{ .x = x, .suh = @splat(0), .xh = @splat(0), .x_dtype = @backingInt(x_dtype), .m = @intCast(m), .k = @intCast(k) };
        var g: GroupArgsC = .{ .l = undefined, .ends = @splat(0) };
        var zat = z;
        var cat = counters;
        var end: i32 = 0;
        for (ws, ys, 0..) |w, y, i| {
            const p = planOf(w);
            if (p.sk > 1 and z == 0) return error.Exl3Scratch;
            rot.suh[i] = w.suh;
            rot.xh[i] = xh + i * m * k * 2;
            g.l[i] = .{ .xh = rot.xh[i], .t = w.trellis, .stride_k = @intCast(w.stride_k), .stride_nb = @intCast(w.stride_nb), .svh = w.svh, .bias = w.bias, .y = y, .z = if (p.sk > 1) zat else 0, .counters = cat, .y_dtype = @backingInt(y_dtype), .n = @intCast(w.n), .sk = @intCast(p.sk), .m = @intCast(m), .k = @intCast(k) };
            zat += zBytes(w, m);
            cat += counterBytes(w);
            end += @intCast(w.n / 128 * p.sk);
            g.ends[i] = end;
        }
        for (ws.len..3) |i| g.ends[i] = end;
        var ra: launch_.Args = .{};
        ra.add(rot);
        try launch_.launch(l.rot_group, .{ .grid = .{ .x = @intCast((k / 128 + 3) / 4), .y = @intCast(m), .z = @intCast(ws.len) }, .block = .{ .x = 128, .y = 1, .z = 1 }, .pdl = l.pdl }, s, &ra);
        var ga: launch_.Args = .{};
        ga.add(g);
        const wk = planOf(ws[0]).wk;
        try launch_.launch(l.group[if (wk == 4) 0 else 1], .{ .grid = .{ .x = @intCast(end), .y = 1, .z = 1 }, .block = .{ .x = wk * 32, .y = 1, .z = 1 }, .shared = sharedBytes(ws[0], m), .pdl = l.pdl }, s, &ga);
    }
};

/// Members one group launch takes: 2 or 3 4-bit mul1 layers of one K and one warp count, at 1 to 16 rows.
pub fn groupable(ws: []const Weight, m: usize) bool {
    if (ws.len < 2 or ws.len > 3 or m < 1 or m > 16) return false;
    const wk = planOf(ws[0]).wk;
    if (wk != 4 and wk != 8) return false;
    for (ws) |w| {
        if (w.halves != 8 or w.codebook != .mul1 or w.k != ws[0].k or planOf(w).wk != wk) return false;
        if (w.k % 128 != 0 or w.n % 128 != 0 or (w.k / 16) % (planOf(w).sk * wk) != 0) return false;
    }
    return true;
}

/// linear.cu's LinearArgs, GroupArgs and RotGroupArgs as the kernels read them.
const LinearArgsC = extern struct { xh: u64, t: u64, stride_k: i64, stride_nb: i64, svh: u64, bias: u64, y: u64, z: u64, counters: u64, y_dtype: i32, n: i32, sk: i32, m: i32, k: i32 };
const GroupArgsC = extern struct { l: [3]LinearArgsC, ends: [3]i32 };
const RotGroupC = extern struct { x: u64, suh: [3]u64, xh: [3]u64, x_dtype: i32, m: i32, k: i32 };

/// The loaded kernels plus the scratch one stream's dense calls share; prompt rows run in 128-row blocks (forward.py).
pub const Dense = struct {
    d: *const Driver,
    lin: Linear,
    xh: ?memory.DeviceBuffer = null, // the rotated inputs, 128 rows of the widest layer at most
    z: ?memory.DeviceBuffer = null, // the split partials, 128 rows at most
    candidate_mod: ?Module = null,
    candidate_k: ?candidate.Kernels = null,
    implementation: enum { reference, tiles } = .reference,
    counters: ?memory.DeviceBuffer = null, // per pass and block: zeroed on the calls' stream, reset by the kernel

    pub fn load(d: *const Driver) !Dense {
        return .{ .d = d, .lin = try Linear.load(d) };
    }

    pub fn unload(de: *Dense) void {
        if (de.candidate_k) |*k| k.deinit();
        if (de.candidate_mod) |*m| m.unload();
        de.lin.unload();
        const slots = [_]*?memory.DeviceBuffer{ &de.xh, &de.z, &de.counters };
        for (slots) |slot| {
            if (slot.* != null) slot.*.?.free();
            slot.* = null;
        }
    }

    /// A scratch buffer's address, grown when a layer needs more than it holds.
    fn scratch(de: *Dense, slot: *?memory.DeviceBuffer, bytes: usize) !u64 {
        if (slot.* != null and slot.*.?.len >= bytes) return slot.*.?.ptr;
        const replacement = try memory.DeviceBuffer.alloc(de.d, bytes);
        if (slot.* != null) slot.*.?.free();
        slot.* = replacement;
        return slot.*.?.ptr;
    }

    /// Grow every scratch buffer to what 128 rows of `w` need on `s`, so a call in a graph capture allocates nothing.
    pub fn reserve(de: *Dense, s: Stream, w: Weight) !void {
        if (planOf(w).sk > 1) _ = try de.scratch(&de.z, Linear.zBytes(w, 128));
        _ = try de.grownCounters(s, Linear.counterBytes(w));
        _ = try de.scratch(&de.xh, 128 * w.k * 2);
    }

    /// The counters, zeroed on `s` when they grow: each block's last arriving program resets its own.
    fn grownCounters(de: *Dense, s: Stream, bytes: usize) !u64 {
        if (de.counters != null and de.counters.?.len >= bytes) return de.counters.?.ptr;
        const ptr = try de.scratch(&de.counters, bytes);
        try de.counters.?.fill8(0, s.handle);
        return ptr;
    }

    /// `reserve` for a group `decodeGroup` launches at once: its members' scratch side by side, at 16 rows.
    pub fn reserveGroup(de: *Dense, s: Stream, ws: []const Weight) !void {
        if (groupable(ws, 16)) _ = try de.groupScratch(s, ws, 16);
    }

    /// A group's scratch at `rows`: its members' split partials, counters and rotated inputs side by side.
    pub fn groupScratch(de: *Dense, s: Stream, ws: []const Weight, rows: usize) !struct { z: u64, counters: u64, xh: u64 } {
        var zb: usize = 0;
        var cb: usize = 0;
        for (ws) |w| {
            zb += Linear.zBytes(w, 128);
            cb += Linear.counterBytes(w);
        }
        return .{
            .z = if (zb > 0) try de.scratch(&de.z, zb) else 0,
            .counters = try de.grownCounters(s, cb),
            .xh = try de.scratch(&de.xh, ws.len * rows * ws[0].k * 2),
        };
    }

    /// Decode rows: one call (the kernel takes 1..128), any count above in 128-row blocks.
    pub fn decode(de: *Dense, s: Stream, w: Weight, x: u64, x_dtype: DType, rows: usize, y: u64, y_dtype: DType) !void {
        if (de.implementation == .tiles) return de.tiles(s, w, x, x_dtype, rows, y, y_dtype, false);
        return de.blocks(s, w, x, x_dtype, rows, y, y_dtype);
    }

    /// Prompt rows: 128-row blocks of `matmul`, so a row's bits never depend on where its chunk starts.
    pub fn prompt(de: *Dense, s: Stream, w: Weight, x: u64, x_dtype: DType, rows: usize, y: u64, y_dtype: DType) !void {
        if (de.implementation == .tiles) return de.tiles(s, w, x, x_dtype, rows, y, y_dtype, true);
        return de.blocks(s, w, x, x_dtype, rows, y, y_dtype);
    }
    pub fn candidateWeight(w: Weight) !candidate.Matrix {
        const st = strides(w.k, w.n, w.halves, false);
        if (w.halves % 2 != 0 or w.bias != 0 or w.stride_k != st.k or w.stride_nb != st.nb)
            return error.CandidateFormat;
        const m: candidate.Matrix = .{ .trellis = w.trellis, .suh = w.suh, .svh = w.svh, .k = w.k, .n = w.n, .bits = w.halves / 2, .codebook = @fromBackingInt(@intCast(@backingInt(w.codebook))) };
        try m.check();
        return m;
    }

    fn tiles(de: *Dense, s: Stream, w: Weight, x: u64, xd: DType, rows: usize, y: u64, yd: DType, prompt_path: bool) !void {
        const m = try candidateWeight(w);
        if (xd != .bf16 or yd == .f16) return error.CandidateFormat;
        if (de.candidate_k == null or de.candidate_k.?.k_max < w.k) {
            if (!kernels.available) return error.BuiltWithoutKernels;
            var mod = try Module.load(de.d, kernels.exl3_candidate);
            errdefer mod.unload();
            const replacement = try candidate.Kernels.resolve(de.d, mod, w.k);
            if (de.candidate_k) |*k| k.deinit();
            if (de.candidate_mod) |*old| old.unload();
            de.candidate_mod = mod;
            de.candidate_k = replacement;
        }
        const k = &de.candidate_k.?;
        if (prompt_path) return k.prompt(s, x, w.k, m, y, rows, yd == .f32);
        return k.rows(s, x, w.k, m, y, rows, yd == .f32);
    }

    /// Decode rows of a projection group in one launch where `groupable`, else member by member (the same bits).
    pub fn decodeGroup(de: *Dense, s: Stream, ws: []const Weight, x: u64, x_dtype: DType, rows: usize, ys: []const u64, y_dtype: DType) !void {
        if (!groupable(ws, rows)) {
            for (ws, ys) |w, y| try de.decode(s, w, x, x_dtype, rows, y, y_dtype);
            return;
        }
        const sc = try de.groupScratch(s, ws, rows);
        try de.lin.matmulGroup(s, ws, x, x_dtype, rows, ys, y_dtype, sc.xh, sc.z, sc.counters);
    }

    fn blocks(de: *Dense, s: Stream, w: Weight, x: u64, x_dtype: DType, rows: usize, y: u64, y_dtype: DType) !void {
        const p = planOf(w);
        const z: u64 = if (p.sk > 1) try de.scratch(&de.z, Linear.zBytes(w, 128)) else 0;
        const counters = try de.grownCounters(s, Linear.counterBytes(w));
        const xe: u64 = if (x_dtype == .f32) 4 else 2;
        const ye: u64 = if (y_dtype == .f32) 4 else 2;
        var r: usize = 0;
        while (r < rows) : (r += 128) {
            const m = @min(rows - r, 128);
            const xh = try de.scratch(&de.xh, m * w.k * 2);
            try de.lin.matmul(s, w, x + @as(u64, r) * w.k * xe, x_dtype, m, y + @as(u64, r) * w.n * ye, y_dtype, xh, z, counters);
        }
    }
};

test "the plan matches linear.py's for known shapes" {
    try std.testing.expectEqual(@as(@TypeOf(plan(0, 0, 192, 8)), .{ .sk = 8, .wk = 4 }), plan(4096, 1024, 192, 8));
    try std.testing.expectEqual(@as(@TypeOf(plan(0, 0, 192, 8)), .{ .sk = 2, .wk = 4 }), plan(1024, 4096, 192, 8));
    try std.testing.expectEqual(@as(@TypeOf(plan(0, 0, 192, 8)), .{ .sk = 4, .wk = 4 }), plan(2048, 512, 192, 8));
    try std.testing.expectEqual(@as(@TypeOf(plan(0, 0, 192, 8)), .{ .sk = 1, .wk = 4 }), plan(512, 1024, 192, 8));
    try std.testing.expectEqual(@as(@TypeOf(plan(0, 0, 192, 8)), .{ .sk = 1, .wk = 4 }), plan(512, 512, 192, 8));
    try std.testing.expectEqual(@as(@TypeOf(plan(0, 0, 192, 8)), .{ .sk = 4, .wk = 4 }), plan(2048, 4096, 192, 8));
}

test "the stored and strips strides of one layout" {
    try std.testing.expectEqual(@as(@TypeOf(strides(0, 0, 0, false)), .{ .k = 4096, .nb = 256 }), strides(4096, 2048, 8, false));
    try std.testing.expectEqual(@as(@TypeOf(strides(0, 0, 0, true)), .{ .k = 256, .nb = 65536 }), strides(4096, 2048, 8, true));
}

test "every variant's symbol is its instantiation's name" {
    try std.testing.expectEqualStrings("tf_exl3_linear_8_2_4", symbol(8, .mul1, 4));
    try std.testing.expectEqualStrings("tf_exl3_linear_7_2_8", symbol(7, .mul1, 8));
    try std.testing.expectEqualStrings("tf_exl3_linear_8_0_4", symbol(8, .inst3, 4));
    try std.testing.expectEqual(@as(?usize, 5), variantIndex(12, .mul1, 4));
    try std.testing.expectEqual(@as(?usize, 2), variantIndex(6, .mul1, 4));
    try std.testing.expectEqual(@as(?usize, null), variantIndex(3, .mcg, 4)); // the half steps are mul1's alone
}

test "matmul refuses shapes the loader would refuse, before any launch" {
    const l: Linear = .{ .mod = undefined, .fns = undefined, .rot_in = undefined };
    const s: Stream = undefined; // never reached: the refusal comes first
    const st = strides(4096, 1024, 8, false);
    const w: Weight = .{ .trellis = 0, .suh = 0, .svh = 0, .bias = 0, .k = 4096, .n = 1024, .halves = 8, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
    try std.testing.expectError(error.Exl3Shape, l.matmul(s, w, 0, .bf16, 0, 0, .bf16, 0, 0, 0));
    try std.testing.expectError(error.Exl3Shape, l.matmul(s, w, 0, .bf16, 129, 0, .bf16, 0, 0, 0));
    var kbad = w;
    kbad.k = 144; // k % 16 == 0 but not a multiple of 128
    try std.testing.expectError(error.Exl3Shape, l.matmul(s, kbad, 0, .bf16, 8, 0, .bf16, 0, 0, 0));
    var nbad = w;
    nbad.n = 100;
    try std.testing.expectError(error.Exl3Shape, l.matmul(s, nbad, 0, .bf16, 8, 0, .bf16, 0, 0, 0));
    try std.testing.expectError(error.Exl3Scratch, l.matmul(s, w, 0, .bf16, 8, 0, .bf16, 0, 0, 0)); // plan(4096, 1024) splits K
}

test "a failed scratch allocation keeps the old scratch owned" {
    const mock = @import("exl3_mock.zig");
    var d = mock.init();
    var de: Dense = .{ .d = &d, .lin = undefined };
    var slot: ?memory.DeviceBuffer = try memory.DeviceBuffer.alloc(&d, 16);
    defer {
        if (mock.frees == 0) slot.?.free();
        slot = null;
    }
    mock.fail = true;
    try std.testing.expectError(error.NotReady, de.scratch(&slot, 32));
    try std.testing.expectEqual(@as(usize, 0), mock.frees);
    try std.testing.expectEqual(@as(usize, 16), slot.?.len);
}

test "a column view keeps the parent's strides and moves to its first 128-column block" {
    for ([_]bool{ false, true }) |strips| {
        const st = strides(256, 1024, 6, strips);
        const w: Weight = .{ .trellis = 0x1000, .suh = 0x2000, .svh = 0x3000, .bias = 0x4000, .k = 256, .n = 1024, .halves = 6, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
        const v = try w.columns(384, 512);
        try std.testing.expectEqual(@as(u32, 512), v.n);
        try std.testing.expectEqual(w.trellis + 3 * st.nb * 4, v.trellis);
        try std.testing.expectEqual(w.svh + 384 * 2, v.svh);
        try std.testing.expectEqual(w.bias + 384 * 2, v.bias);
        try std.testing.expectEqual(w.suh, v.suh);
        try std.testing.expectEqual(w.stride_k, v.stride_k);
        try std.testing.expectEqual(w.stride_nb, v.stride_nb);
        try std.testing.expectEqual(w, try w.columns(0, 1024));
        try std.testing.expectError(error.Exl3Shape, w.columns(64, 128));
        try std.testing.expectError(error.Exl3Shape, w.columns(0, 192));
        try std.testing.expectError(error.Exl3Shape, w.columns(896, 256));
    }
}

test "the counters are zeroed on the calls' stream only as they grow" {
    const mock = @import("exl3_mock.zig");
    var d = mock.init();
    var de: Dense = .{ .d = &d, .lin = undefined };
    defer if (de.counters) |*c| c.free();
    const s: Stream = .{ .d = &d, .handle = @ptrFromInt(0x5100) };
    _ = try de.grownCounters(s, 64);
    try std.testing.expectEqual(@as(usize, 1), mock.memsets);
    try std.testing.expectEqual(s.handle, mock.memset_stream);
    _ = try de.grownCounters(s, 64);
    _ = try de.grownCounters(s, 32);
    try std.testing.expectEqual(@as(usize, 1), mock.memsets);
    _ = try de.grownCounters(s, 128);
    try std.testing.expectEqual(@as(usize, 2), mock.memsets);
    try std.testing.expectEqual(@as(usize, 0), mock.legacy_memsets);
}

test "candidate selection refuses layouts it cannot represent" {
    const st = strides(128, 256, 8, false);
    var w: Weight = .{ .trellis = 1, .suh = 2, .svh = 3, .k = 128, .n = 256, .halves = 8, .codebook = .mcg, .stride_k = st.k, .stride_nb = st.nb };
    try std.testing.expectEqual(@as(u32, 4), (try Dense.candidateWeight(w)).bits);
    w.halves = 7;
    try std.testing.expectError(error.CandidateFormat, Dense.candidateWeight(w));
    w.halves = 8;
    w.bias = 4;
    try std.testing.expectError(error.CandidateFormat, Dense.candidateWeight(w));
    w.bias = 0;
    w.stride_k = 0;
    try std.testing.expectError(error.CandidateFormat, Dense.candidateWeight(w));
}

pub const block = @import("exl3_block.zig");

test {
    _ = @import("exl3_dense_test.zig");
}
