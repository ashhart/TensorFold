//! Gate 5: every registered launch against each chip's limits, from declarations alone, so it runs without the chip.
const std = @import("std");

/// An Apple GPU family we serve; on M1 and M2 a pipeline's thread limit falls as its registers rise.
pub const Apple = struct { name: []const u8, gen: u32, threads: u32 = 1024, tg_memory: u32 = 32768, by_registers: bool };

pub const apple = [_]Apple{
    .{ .name = "M1", .gen = 13, .by_registers = true },
    .{ .name = "M2", .gen = 14, .by_registers = true },
    .{ .name = "M3", .gen = 15, .by_registers = false },
    .{ .name = "M4", .gen = 16, .by_registers = false },
    .{ .name = "M5", .gen = 17, .by_registers = false },
};

/// A Metal pipeline as launched: most threads a group, threadgroup bytes, a source reservation, M1/M2 compiled limits.
pub const MetalLaunch = struct {
    name: []const u8,
    threads: u32,
    tg_memory: u32,
    /// Threadgroup bytes the encoder adds to the source's own (setThreadgroupMemory).
    dynamic: u32 = 0,
    reserved: u32 = 0,
    m1: u32 = 0,
    m2: u32 = 0,
    /// The launcher reads the pipeline's own thread limit and sizes its groups to it.
    adapts: bool = false,
    /// False for Metal 4 tensor-op kernels: Xcode's translator cannot build them for M1/M2 ahead of time.
    offline: bool = true,
};

/// An NVIDIA target we build for: its per-block and per-SM limits (CUDA's compute capability tables).
pub const Sm = struct {
    sm: u32,
    threads: u32 = 1024,
    regs_thread: u32 = 255,
    regs_block: u32 = 65536,
    regs_sm: u32 = 65536,
    threads_sm: u32,
    smem_default: u32 = 48 * 1024,
    smem_optin: u32,
    clusters: bool,
};

pub const sms = [_]Sm{
    .{ .sm = 80, .threads_sm = 2048, .smem_optin = 163 * 1024, .clusters = false },
    .{ .sm = 86, .threads_sm = 1536, .smem_optin = 99 * 1024, .clusters = false },
    .{ .sm = 89, .threads_sm = 1536, .smem_optin = 99 * 1024, .clusters = false },
    .{ .sm = 120, .threads_sm = 1536, .smem_optin = 99 * 1024, .clusters = true },
    .{ .sm = 121, .threads_sm = 1536, .smem_optin = 99 * 1024, .clusters = true },
};

/// A CUDA kernel as launched: most threads a block, dynamic shared bytes, cluster width, raised limit, least SM.
pub const CudaLaunch = struct {
    name: []const u8,
    symbol: []const u8,
    image: []const u8,
    threads: u32,
    dynamic: u32 = 0,
    cluster: u32 = 1,
    opt_in: bool = false,
    min_sm: u32 = 0,
};

/// What the compiler made of one kernel for one SM (registers, static shared bytes, __launch_bounds__ threads).
pub const Usage = struct { regs: u32, static: u32, bound: u32 = 0, local: u32 = 0 };

/// An nvcc release, as a cubin's .note.nv.tkinfo names it ("release 13.2").
pub const Nvcc = struct { major: u32, minor: u32 };

/// Every problem one Metal launch has on one family, written to `out`; returns how many.
pub fn metal(l: MetalLaunch, f: Apple, out: *std.ArrayList(u8), a: std.mem.Allocator) !usize {
    var n: usize = 0;
    if (l.threads > f.threads) n += try note(out, a, "{s} on {s}: {d} threads a group, the family's most is {d}", .{ l.name, f.name, l.threads, f.threads });
    const tg = l.tg_memory + l.dynamic;
    if (tg > f.tg_memory) n += try note(out, a, "{s} on {s}: {d} threadgroup bytes, the family's most is {d}", .{ l.name, f.name, tg, f.tg_memory });
    if (f.by_registers and l.reserved < l.threads and !l.adapts) {
        const lim = if (f.gen == 13) l.m1 else l.m2;
        if (!l.offline and l.threads > 256) n += try note(out, a, "{s} on {s}: {d} threads past 256 and no M1/M2 compile to show they fit (reserve them)", .{ l.name, f.name, l.threads });
        if (l.offline and lim == 0) n += try note(out, a, "{s} on {s}: no compiled thread limit recorded", .{ l.name, f.name });
        if (lim != 0 and lim < l.threads) n += try note(out, a, "{s} on {s}: {d} threads, the pipeline compiles for {d} (reserve the threads)", .{ l.name, f.name, l.threads, lim });
    }
    return n;
}

/// The most threads (a multiple of 32, 1024 down) whose build `probe.plainAt` finds equal to the plain build; 0: none.
pub fn plainLimit(probe: anytype) !u32 {
    var threads: u32 = 1024;
    while (threads > 0) : (threads -= 32) if (try probe.plainAt(threads)) return threads;
    return 0;
}

/// Builds that equal the plain one exactly at the listed thread counts, counting how many the search asks for.
const FakeBuilds = struct {
    equal: []const u32,
    asked: *u32,

    pub fn plainAt(f: FakeBuilds, threads: u32) !bool {
        f.asked.* += 1;
        return std.mem.indexOfScalar(u32, f.equal, threads) != null;
    }

    fn upTo(comptime limit: u32) [limit / 32]u32 {
        var out: [limit / 32]u32 = undefined;
        for (&out, 1..) |*t, i| t.* = @intCast(32 * i);
        return out;
    }
};

test "the offline search: the largest count equal to plain when every count up to it is" {
    var asked: u32 = 0;
    inline for (.{ 1024, 640, 448, 256, 32 }) |limit| {
        const below = FakeBuilds.upTo(limit);
        try std.testing.expectEqual(@as(u32, limit), try plainLimit(FakeBuilds{ .equal = &below, .asked = &asked }));
    }
    try std.testing.expectEqual(@as(u32, 0), try plainLimit(FakeBuilds{ .equal = &.{}, .asked = &asked }));
}

test "the offline search: the largest count equal to plain past counts that differ" {
    var asked: u32 = 0;
    // Nemotron's expert rows on M1/M2: plain at 224 and 320-448, not at 32-192, 256, 288 or 480 and up.
    const gaps = [_]u32{ 224, 320, 352, 384, 416, 448 };
    try std.testing.expectEqual(@as(u32, 448), try plainLimit(FakeBuilds{ .equal = &gaps, .asked = &asked }));
    const lone = [_]u32{ 64, 512 };
    try std.testing.expectEqual(@as(u32, 512), try plainLimit(FakeBuilds{ .equal = &lone, .asked = &asked }));
    const odd = [_]u32{ 32, 96, 160 };
    try std.testing.expectEqual(@as(u32, 160), try plainLimit(FakeBuilds{ .equal = &odd, .asked = &asked }));
}

test "the offline search: one build when 1024 equals plain, every count when none does" {
    var asked: u32 = 0;
    const all = FakeBuilds.upTo(1024);
    try std.testing.expectEqual(@as(u32, 1024), try plainLimit(FakeBuilds{ .equal = &all, .asked = &asked }));
    try std.testing.expectEqual(@as(u32, 1), asked);
    asked = 0;
    try std.testing.expectEqual(@as(u32, 0), try plainLimit(FakeBuilds{ .equal = &.{}, .asked = &asked }));
    try std.testing.expectEqual(@as(u32, 32), asked);
}

/// Registers a block takes on one SM: each warp's registers rounded up to 256, times its warps.
pub fn blockRegs(regs: u32, threads: u32) u32 {
    const warp = (regs * 32 + 255) / 256 * 256;
    return warp * ((threads + 31) / 32);
}

/// Every problem one CUDA launch has on one SM given what the compiler made of it; returns how many.
pub fn cuda(l: CudaLaunch, u: Usage, t: Sm, out: *std.ArrayList(u8), a: std.mem.Allocator) !usize {
    var n: usize = 0;
    const name = l.name;
    if (t.sm < l.min_sm) return 0;
    if (l.threads > t.threads) n += try note(out, a, "{s} on sm_{d}: {d} threads a block, the most is {d}", .{ name, t.sm, l.threads, t.threads });
    if (u.bound != 0 and l.threads > u.bound) n += try note(out, a, "{s} on sm_{d}: launched at {d} threads past its __launch_bounds__ {d}", .{ name, t.sm, l.threads, u.bound });
    if (u.regs > t.regs_thread) n += try note(out, a, "{s} on sm_{d}: {d} registers a thread", .{ name, t.sm, u.regs });
    if (blockRegs(u.regs, l.threads) > t.regs_block) n += try note(out, a, "{s} on sm_{d}: {d} registers x {d} threads need {d}, a block has {d}", .{ name, t.sm, u.regs, l.threads, blockRegs(u.regs, l.threads), t.regs_block });
    const smem = u.static + l.dynamic;
    if (smem > t.smem_optin) n += try note(out, a, "{s} on sm_{d}: {d} shared bytes, the most a block takes is {d}", .{ name, t.sm, smem, t.smem_optin });
    if (smem > t.smem_default and !l.opt_in) n += try note(out, a, "{s} on sm_{d}: {d} shared bytes past 48 KiB without raising the limit", .{ name, t.sm, smem });
    if (l.cluster > 1 and !t.clusters) n += try note(out, a, "{s} on sm_{d}: a cluster of {d} where the SM has no clusters", .{ name, t.sm, l.cluster });
    return n;
}

/// An RDNA chip we serve, one a generation: a workgroup's limits and a SIMD's wave32 register file (WGP mode).
pub const Rdna = struct {
    name: []const u8,
    threads: u32 = 1024,
    lds: u32 = 64 * 1024,
    wave: u32 = 32,
    /// VGPRs one wave32 may address.
    vgprs_wave: u32 = 256,
    /// VGPRs a SIMD holds for each lane, and the step a wave32 allocates them in.
    vgprs_simd: u32,
    granule: u32,
    simds: u32 = 4,
};

pub const rdna = [_]Rdna{
    .{ .name = "gfx1030", .vgprs_simd = 1024, .granule = 16 },
    .{ .name = "gfx1100", .vgprs_simd = 1536, .granule = 24 },
    .{ .name = "gfx1151", .vgprs_simd = 1536, .granule = 24 },
    .{ .name = "gfx1201", .vgprs_simd = 1536, .granule = 24 },
};

/// A HIP kernel as launched: its symbol, its code object, most threads a workgroup and dynamic LDS bytes.
pub const HipLaunch = struct {
    name: []const u8,
    symbol: []const u8,
    image: []const u8,
    threads: u32,
    dynamic: u32 = 0,
    /// The symbol on each chip (rdna's order) where it differs, as a stream tile's dot2 type does.
    symbols: ?[rdna.len][]const u8 = null,

    /// The symbol the launch resolves on chip `i` of `rdna`.
    pub fn symbolOn(l: HipLaunch, i: usize) []const u8 {
        return if (l.symbols) |s| s[i] else l.symbol;
    }
};

/// What hipcc made of one kernel for one chip, from the code object's metadata (vgprs 0: none recorded).
pub const HipUsage = struct {
    vgprs: u32,
    sgprs: u32 = 0,
    lds: u32 = 0,
    private: u32 = 0,
    bound: u32 = 1024,
    wave: u32 = 32,
};

/// VGPRs a workgroup of `threads` takes from each SIMD: its waves spread over the WGP's SIMDs, each rounded up.
pub fn simdVgprs(vgprs: u32, threads: u32, t: Rdna) u32 {
    const alloc = (vgprs + t.granule - 1) / t.granule * t.granule;
    const waves = (threads + t.wave - 1) / t.wave;
    return alloc * ((waves + t.simds - 1) / t.simds);
}

/// Every problem one HIP launch has on one chip given what hipcc made of it; returns how many.
pub fn hip(l: HipLaunch, u: HipUsage, t: Rdna, out: *std.ArrayList(u8), a: std.mem.Allocator) !usize {
    var n: usize = 0;
    const at = .{ l.name, t.name };
    const wave = "{s} on {s}: built for wave{d}, the kernels assume wave{d}";
    if (u.wave != t.wave) n += try note(out, a, wave, at ++ .{ u.wave, t.wave });
    const most = "{s} on {s}: {d} threads a workgroup, the most is {d}";
    if (l.threads > t.threads) n += try note(out, a, most, at ++ .{ l.threads, t.threads });
    const past = "{s} on {s}: launched at {d} threads past its bound {d}";
    if (l.threads > u.bound) n += try note(out, a, past, at ++ .{ l.threads, u.bound });
    if (u.vgprs > t.vgprs_wave) n += try note(out, a, "{s} on {s}: {d} VGPRs a wave", at ++ .{u.vgprs});
    const simd = simdVgprs(u.vgprs, l.threads, t);
    const file = "{s} on {s}: {d} VGPRs x {d} threads need {d} a SIMD, it has {d}";
    if (simd > t.vgprs_simd) n += try note(out, a, file, at ++ .{ u.vgprs, l.threads, simd, t.vgprs_simd });
    const lds = u.lds + l.dynamic;
    const shared = "{s} on {s}: {d} LDS bytes, the most a workgroup takes is {d}";
    if (lds > t.lds) n += try note(out, a, shared, at ++ .{ lds, t.lds });
    return n;
}

fn note(out: *std.ArrayList(u8), a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !usize {
    try out.print(a, fmt ++ "\n", args);
    return 1;
}

test "a launch inside every family passes; each planted excess is named" {
    const a = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    const ok = MetalLaunch{ .name = "ok", .threads = 1024, .tg_memory = 256, .m1 = 1024, .m2 = 1024 };
    for (apple) |f| try std.testing.expectEqual(@as(usize, 0), try metal(ok, f, &out, a));
    const big = MetalLaunch{ .name = "big", .threads = 256, .tg_memory = 36 * 1024, .m1 = 1024, .m2 = 1024 };
    for (apple) |f| try std.testing.expectEqual(@as(usize, 1), try metal(big, f, &out, a));
    const set = MetalLaunch{ .name = "set", .threads = 64, .tg_memory = 24 * 1024, .dynamic = 10 * 1024, .m1 = 1024, .m2 = 1024 };
    for (apple) |f| try std.testing.expectEqual(@as(usize, 1), try metal(set, f, &out, a));
    const heavy = MetalLaunch{ .name = "heavy", .threads = 1024, .tg_memory = 0, .m1 = 448, .m2 = 640 };
    try std.testing.expectEqual(@as(usize, 1), try metal(heavy, apple[0], &out, a));
    try std.testing.expectEqual(@as(usize, 0), try metal(heavy, apple[2], &out, a));
    var kept = heavy;
    kept.reserved = 1024;
    try std.testing.expectEqual(@as(usize, 0), try metal(kept, apple[0], &out, a));
    const unknown = MetalLaunch{ .name = "unknown", .threads = 64, .tg_memory = 0 };
    try std.testing.expectEqual(@as(usize, 1), try metal(unknown, apple[1], &out, a));
    const tensor_op = MetalLaunch{ .name = "tensor op", .threads = 128, .tg_memory = 0, .offline = false };
    try std.testing.expectEqual(@as(usize, 0), try metal(tensor_op, apple[0], &out, a));
    var wide_op = tensor_op;
    wide_op.threads = 512;
    try std.testing.expectEqual(@as(usize, 1), try metal(wide_op, apple[0], &out, a));
}

/// The table's row for `sm` (a target the records cover).
pub fn target(sm: u32) ?Sm {
    for (sms) |t| if (t.sm == sm) return t;
    return null;
}

test "CUDA: registers by warp, launch bounds, shared bytes and clusters per SM" {
    const a = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    try std.testing.expectEqual(@as(u32, 65536), blockRegs(64, 1024));
    try std.testing.expectEqual(@as(u32, 512 * 4), blockRegs(13, 128)); // 13 registers round up to 16 a thread
    const l = CudaLaunch{ .name = "k", .symbol = "k", .image = "k", .threads = 256, .dynamic = 40 * 1024 };
    const sm80 = target(80).?;
    const sm86 = target(86).?;
    try std.testing.expectEqual(@as(usize, 0), try cuda(l, .{ .regs = 64, .static = 0, .bound = 256 }, sm86, &out, a));
    try std.testing.expectEqual(@as(usize, 1), try cuda(l, .{ .regs = 64, .static = 16 * 1024, .bound = 256 }, sm86, &out, a));
    var big = l;
    big.dynamic = 120 * 1024;
    big.opt_in = true;
    try std.testing.expectEqual(@as(usize, 0), try cuda(big, .{ .regs = 64, .static = 0, .bound = 256 }, sm80, &out, a));
    try std.testing.expectEqual(@as(usize, 1), try cuda(big, .{ .regs = 64, .static = 0, .bound = 256 }, sm86, &out, a));
    var wide = l;
    wide.threads = 1024;
    try std.testing.expectEqual(@as(usize, 2), try cuda(wide, .{ .regs = 72, .static = 0, .bound = 256 }, target(121).?, &out, a));
    var cl = l;
    cl.cluster = 4;
    try std.testing.expectEqual(@as(usize, 1), try cuda(cl, .{ .regs = 32, .static = 0 }, target(89).?, &out, a));
    try std.testing.expectEqual(@as(usize, 1), try cuda(cl, .{ .regs = 32, .static = 0 }, sm80, &out, a));
    try std.testing.expectEqual(@as(usize, 0), try cuda(cl, .{ .regs = 32, .static = 0 }, target(120).?, &out, a));
    cl.min_sm = 90;
    try std.testing.expectEqual(@as(usize, 0), try cuda(cl, .{ .regs = 32, .static = 0 }, sm80, &out, a));
    try std.testing.expectEqual(@as(?Sm, null), target(75));
}

test "HIP: bounds, wave size, VGPRs a SIMD and LDS per chip" {
    const a = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    const l = HipLaunch{ .name = "k", .symbol = "k", .image = "ops", .threads = 1024 };
    try std.testing.expectEqual(@as(u32, 1024), simdVgprs(128, 1024, rdna[0])); // 32 waves, 8 a SIMD
    try std.testing.expectEqual(@as(u32, 1536), simdVgprs(183, 1024, rdna[1])); // 183 rounds up to 192
    for (rdna) |t| try std.testing.expectEqual(@as(usize, 0), try hip(l, .{ .vgprs = 128 }, t, &out, a));
    try std.testing.expectEqual(@as(usize, 1), try hip(l, .{ .vgprs = 183 }, rdna[0], &out, a));
    try std.testing.expectEqual(@as(usize, 0), try hip(l, .{ .vgprs = 183 }, rdna[3], &out, a));
    try std.testing.expectEqual(@as(usize, 1), try hip(l, .{ .vgprs = 64, .bound = 256 }, rdna[1], &out, a));
    try std.testing.expectEqual(@as(usize, 1), try hip(l, .{ .vgprs = 64, .wave = 64 }, rdna[2], &out, a));
    var big = l;
    big.threads = 256;
    big.dynamic = 32 * 1024;
    try std.testing.expectEqual(@as(usize, 1), try hip(big, .{ .vgprs = 32, .lds = 40 * 1024 }, rdna[0], &out, a));
    try std.testing.expectEqual(@as(usize, 0), try hip(big, .{ .vgprs = 32, .lds = 32 * 1024 }, rdna[0], &out, a));
}
