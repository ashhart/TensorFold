//! The KV side of the priced memory floor: boot pool size, prompt and NVMe resume prices, and admission fit.

const std = @import("std");
const sessions = @import("sessions");

pub const GiB: f64 = @floatFromInt(@as(u64, 1) << 30);

/// A slot's bounded state outside the pool: 43 SWA rings x 256 x 584, 3 ratio-2 carries, Engram lookback, DSpark taps.
pub const slot_bytes: u64 = 43 * 256 * 584 + 3 * 1024 * 4 + 3 * 4 + 3 * 5120 * 2 * 128;

pub const Knobs = struct {
    /// TF_DSV41_PREFILL_ROWS and TF_DSV41_INDEX_BUDGET_MIB
    prefill_rows: u32 = 2048,
    index_budget_mib: u32 = 256,
    /// fitted prefill price terms
    pf_chunk_gib: f64 = 0.533,
    pf_host_gib: f64 = 0.6,
    pf_other_gib: f64 = 0.0,
    pf_index_gib: f64 = 1.067,
    index_positions: u64 = 32768,
    /// split KV's union receive cap (TF_DSV41_KV_SPLIT_UNION_MIB); 0 when unsplit
    union_cap: u64 = 0,
    /// the NVMe tier's staging set: a streamed park's or restore's whole host transient
    staging_bytes: u64 = 128 << 20,
    hard_gib: f64 = 4.0,
    target_gib: f64 = 5.0,
};

/// A rank's boot budget in GiB: what is allocated before the pool, and what must stay free.
pub const Boot = struct {
    available: f64 = 112.0,
    weights: f64,
    slots: u32 = 4,
    /// the planned workspace (graphs 1.2, prefill 1.3, select 0.25, drafting 0.35, slack 0.9)
    workspace: f64 = 4.0,
    runtime: f64 = 1.0,
    /// the RAM tier's bounded state (TF_DSV41_SESSION_RAM_MIB)
    session_ram: f64 = 0.25,

    pub fn fixed(b: Boot, k: Knobs) f64 {
        const rings = @as(f64, @floatFromInt(b.slots * slot_bytes)) / GiB;
        const staging = @as(f64, @floatFromInt(k.staging_bytes)) / GiB;
        const unions = 1.5 * @as(f64, @floatFromInt(k.union_cap)) / GiB;
        return b.weights + rings + b.workspace + b.runtime + b.session_ram + staging + unions;
    }

    /// The floor left with a pool of pool_bytes: under the hard floor the boot refuses.
    pub fn floor(b: Boot, k: Knobs, pool_bytes: u64) f64 {
        return b.available - b.fixed(k) - @as(f64, @floatFromInt(pool_bytes)) / GiB;
    }

    /// The most pool tokens keeping the hard floor, in whole pages, a multiple of world pages (split residues).
    pub fn poolTokens(b: Boot, k: Knobs, layout: sessions.Layout, world: u32) u64 {
        const spare = b.available - b.fixed(k) - k.hard_gib;
        if (spare <= 0) return 0;
        const per_page: f64 = @floatFromInt(layout.page);
        const page_bytes = layout.bytesPerToken(world) * per_page;
        const pages: u64 = @intFromFloat(@floor(spare * GiB / page_bytes));
        return pages / world * world * layout.page;
    }
};

/// A prompt's prefill transient in bytes: once (the window's buffers) and index (grows to index_positions).
pub const Price = struct {
    once: u64 = 0,
    index: u64 = 0,
    start: u64 = 0,
    end: u64 = 0,

    pub fn total(p: Price) u64 {
        return p.once + p.index;
    }

    /// What has not materialised at `done` tokens (the floor's reservation for a prompt still prefilling).
    pub fn outstanding(p: Price, done: u64) u64 {
        if (p.total() == 0) return 0;
        const once = if (done <= p.start) p.once else 0;
        const span = p.end - p.start;
        const left = p.end -| @max(done, p.start);
        return once + if (span > 0) @as(u64, @intFromFloat(@as(f64, @floatFromInt(p.index)) * @as(f64, @floatFromInt(left)) / @as(f64, @floatFromInt(span)))) else 0;
    }
};

/// Prefilling prompt tokens [cached, n - 1), plus split KV's union buffers (receive cap + half for the send).
pub fn prefillPrice(k: Knobs, n: u64, cached: u64) Price {
    const todo = (n -| 1) -| cached;
    if (todo == 0) return .{};
    const w = @as(f64, @floatFromInt(@min(k.prefill_rows, todo))) / 2048.0;
    var once = (k.pf_chunk_gib + k.pf_host_gib) * w + k.pf_other_gib;
    once += 1.5 * @as(f64, @floatFromInt(k.union_cap)) / GiB;
    const end = @min(n - 1, k.index_positions);
    const share = @as(f64, @floatFromInt(end -| cached)) / @as(f64, @floatFromInt(k.index_positions));
    const index = k.pf_index_gib * @as(f64, @floatFromInt(k.index_budget_mib)) / 256.0 * share;
    return .{ .once = @intFromFloat(once * GiB), .index = @intFromFloat(index * GiB), .start = cached, .end = @max(cached, end) };
}

/// An NVMe resume's host transient: the staging set (streamed), or nothing when the bytes are already in host memory.
pub fn restorePrice(k: Knobs, from_disk: bool) u64 {
    return if (from_disk) k.staging_bytes else 0;
}

pub const Verdict = enum { admit, wait_pages, wait_memory };

pub const Ask = struct {
    /// pages the request needs reserved (prompt + reply + slack) less those a RAM resume shares
    pages: u32,
    /// the prompt's price, an NVMe resume's price, and what admitted prompts have not materialised yet
    price: u64,
    restore: u64 = 0,
    outstanding: u64 = 0,
};

/// May a request start: pages from any residue, and every rank's `usable` less the transient at the hard floor.
pub fn admit(k: Knobs, pool: *const sessions.Pool, ask: Ask, usable: []const u64) struct { verdict: Verdict, rank: u32 } {
    if (pool.available() < ask.pages) return .{ .verdict = .wait_pages, .rank = 0 };
    const want = ask.price + ask.restore + ask.outstanding + @as(u64, @intFromFloat(k.hard_gib * GiB));
    var tight: u32 = 0;
    for (usable, 0..) |u, r| if (u < usable[tight]) {
        tight = @intCast(r);
    };
    if (usable[tight] < want) return .{ .verdict = .wait_memory, .rank = tight };
    return .{ .verdict = .admit, .rank = tight };
}

// -- the measured boot check ----------------------------------------------------------------------------------------

/// A rank's measured GiB terms: boot growth past the check less the pool, and the worst serving dip below idle.
pub const Serve = struct {
    growth: f64,
    dip: f64,

    /// Measured: rank 0 (head) 1.9 / 4.9, every other rank 1.7 / 4.2.
    pub fn measured(rank: u32) Serve {
        return if (rank == 0) .{ .growth = 1.9, .dip = 4.9 } else .{ .growth = 1.7, .dip = 4.2 };
    }

    /// The rank's terms overridden by TF_DSV41_BOOT_GROWTH_GIB / TF_DSV41_SERVE_DIP_GIB (null or blank: measured).
    pub fn withOverrides(rank: u32, growth: ?[]const u8, dip: ?[]const u8) !Serve {
        const m = measured(rank);
        return .{ .growth = try gibEnv(growth, m.growth), .dip = try gibEnv(dip, m.dip) };
    }

    pub fn fromEnv(rank: u32) !Serve {
        return withOverrides(rank, env(growth_env), env(dip_env));
    }
};

/// Zig's dip below idle (TF_DSV41_SERVE_DIP_GIB unset): 2 x 1M then 4 x 740K measured 2.48 (rank 1), 1.1 (rank 0).
pub const zig_dip_gib: f64 = 3.0;

/// Zig's terms before the pool: TF_DSV41_BOOT_GROWTH_GIB (unset 0: measured at boot's end), TF_DSV41_SERVE_DIP_GIB.
pub fn zigFromEnv() !Serve {
    return .{ .growth = try gibEnv(env(growth_env), 0), .dip = try gibEnv(env(dip_env), zig_dip_gib) };
}

/// The measured check at the end of boot (GiB): whether the worst point keeps the floor, and what the boot took.
pub const AfterBoot = struct {
    ok: bool,
    /// what the boot took after the check besides the pool (plan, forms, rows, drafter, sampler)
    growth: f64,
    /// the worst serving point: idle less the dip
    worst: f64,
    /// the pool's tokens that would leave the worst point at the hard floor (0: none)
    largest: u64,
};

/// The measured check: idle - post-boot reserve - dip >= the hard floor (growth = at the check - pool - now, logged).
pub fn afterBoot(at_check: f64, pool_gib: f64, idle: f64, reserve: f64, dip: f64, hard: f64, pool_tokens: u64, bytes_a_token: f64) AfterBoot {
    const worst = idle - reserve - dip;
    const room_tokens = (worst - hard) * GiB / @max(1.0, bytes_a_token);
    const largest = @as(f64, @floatFromInt(pool_tokens)) + room_tokens;
    return .{ .ok = worst >= hard, .growth = at_check - pool_gib - idle, .worst = worst, .largest = if (largest > 0) @intFromFloat(@floor(largest)) else 0 };
}

/// TF_DSV41_POST_BOOT_GIB (unset 0): post-boot one-time cost (graphs, held sessions; ~1.5 GiB measured), not dip.
pub const reserve_env = "TF_DSV41_POST_BOOT_GIB";

pub fn reserveFromEnv() !f64 {
    return gibEnv(env(reserve_env), 0);
}

pub const growth_env = "TF_DSV41_BOOT_GROWTH_GIB";
pub const dip_env = "TF_DSV41_SERVE_DIP_GIB";
pub const measured_env = "TF_DSV41_BOOT_MEASURED";

fn env(name: [*:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

/// A GiB knob: blank is the default, a negative figure an error.
fn gibEnv(raw: ?[]const u8, default: f64) !f64 {
    const s = std.mem.trim(u8, raw orelse "", " \t\r\n");
    if (s.len == 0) return default;
    const v = std.fmt.parseFloat(f64, s) catch return error.BadGiBEnv;
    if (v < 0) return error.BadGiBEnv;
    return v;
}

/// TF_DSV41_BOOT_MEASURED: 0 / off / no / false, 1 / on / yes / true, anything else auto.
pub const Mode = enum { auto, on, off };

pub fn parseMode(raw: ?[]const u8) Mode {
    var buf: [8]u8 = undefined;
    const s = std.mem.trim(u8, raw orelse "", " \t\r\n");
    if (s.len > buf.len) return .auto;
    const l = std.ascii.lowerString(&buf, s);
    const offs = [_][]const u8{ "0", "off", "no", "false" };
    const ons = [_][]const u8{ "1", "on", "yes", "true" };
    for (offs) |o| if (std.mem.eql(u8, l, o)) return .off;
    for (ons) |o| if (std.mem.eql(u8, l, o)) return .on;
    return .auto;
}

pub fn modeFromEnv() Mode {
    return parseMode(env(measured_env));
}

/// The measured check runs if forced, else where the GPU is integrated or its memory within 15 % of host MemTotal.
pub fn measuredApplies(mode: Mode, integrated: bool, device_total: u64, host_total: u64) bool {
    if (mode == .off) return false;
    const d: f64 = @floatFromInt(device_total);
    const h: f64 = @floatFromInt(host_total);
    const shared = integrated or (h > 0 and @abs(d - h) < 0.15 * h);
    return shared or mode == .on;
}

/// /proc/meminfo's MemTotal and MemAvailable in bytes.
pub const MemInfo = struct { total: u64, available: u64 };

pub fn parseMeminfo(text: []const u8) !MemInfo {
    var m: MemInfo = .{ .total = 0, .available = 0 };
    var seen: u2 = 0;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        const dst = if (std.mem.startsWith(u8, line, "MemTotal:")) &m.total else if (std.mem.startsWith(u8, line, "MemAvailable:")) &m.available else continue;
        var f = std.mem.tokenizeScalar(u8, line[std.mem.indexOfScalar(u8, line, ':').? + 1 ..], ' ');
        dst.* = 1024 * (std.fmt.parseInt(u64, f.next() orelse return error.BadMeminfo, 10) catch return error.BadMeminfo);
        seen += 1;
    }
    if (seen < 2) return error.BadMeminfo;
    return m;
}

pub fn meminfo() !MemInfo {
    var buf: [8192]u8 = undefined;
    const fd = std.c.open("/proc/meminfo", .{ .ACCMODE = .RDONLY });
    if (fd < 0) return error.NoMeminfo;
    defer _ = std.c.close(fd);
    const n = std.c.read(fd, &buf, buf.len);
    if (n <= 0) return error.NoMeminfo;
    return parseMeminfo(buf[0..@intCast(n)]);
}

/// MemAvailable now in bytes: what the engine measures at the check (after the weights, before the pool).
pub fn memAvailable() !u64 {
    return (try meminfo()).available;
}

/// A rank's pool bytes for `tokens` (whole rows a family, the split families' rows / W).
pub fn poolBytes(layout: sessions.Layout, world: u32, tokens: u64) u64 {
    var b: u64 = 0;
    for (layout.families) |f| b += @as(u64, f.row_bytes) * (tokens / f.ratio) / (if (f.split and world > 1) world else 1);
    return b;
}

/// Bytes a token: a 1M-token pool's bytes / 2^20 (1,790 replicated, 1,060 split over two ranks).
pub fn tokenBytes(layout: sessions.Layout, world: u32) f64 {
    return @as(f64, @floatFromInt(poolBytes(layout, world, 1 << 20))) / @as(f64, 1 << 20);
}

/// What the check measured: MemAvailable at the check (GiB) and the rank's terms.
pub const Measured = struct {
    rank: u32,
    avail_gib: f64,
    terms: Serve,
};

pub const Result = struct {
    rank: u32,
    ok: bool,
    /// the predicted worst MemAvailable while serving (GiB)
    worst: f64,
    avail_gib: f64,
    pool_tokens: u64,
    pool_gib: f64,
    bytes_a_token: f64,
    split: bool,
    growth: f64,
    dip: f64,
    hard: f64,
    /// the largest pool that fits (room / bytes a token, floored), and the same in whole pages of every residue
    largest: u64,
    largest_paged: u64,

    /// The refusal (or the report on a pass).
    pub fn message(r: Result, buf: []u8) []const u8 {
        var a: [32]u8 = undefined;
        var b: [32]u8 = undefined;
        var c: [32]u8 = undefined;
        if (r.ok) return std.fmt.bufPrint(buf, "rank {d} pool {s} tokens = {d:.2} GiB ({d:.0} B a token{s}); predicted worst MemAvailable {d:.2} GiB (hard floor {s})", .{
            r.rank, grouped(&a, r.pool_tokens), r.pool_gib, r.bytes_a_token, if (r.split) ", split KV" else "", r.worst, pyFloat(&b, r.hard),
        }) catch buf;
        return std.fmt.bufPrint(buf, "rank {d}: a {d:.2} GiB pool leaves {d:.2} GiB at the worst measured serving point (MemAvailable {d:.2} now - boot growth {d:.2} - serving dip {d:.2}), under the {s} GiB hard floor (TF_DSV41_FLOOR_HARD_GIB); the largest pool that fits: {s} tokens ({s} in whole pages; TF_DSV41_POOL_TOKENS; " ++ growth_env ++ " / " ++ dip_env ++ " override the measured terms)", .{
            r.rank, r.pool_gib, r.worst, r.avail_gib, r.growth, r.dip, pyFloat(&a, r.hard), grouped(&b, r.largest), grouped(&c, r.largest_paged),
        }) catch buf;
    }
};

/// The largest pool the terms leave room for: max(0, avail - growth - dip - hard) GiB / bytes a token, floored.
pub fn largestPool(avail_gib: f64, terms: Serve, hard_gib: f64, bytes_a_token: f64) u64 {
    const room = @max(0.0, avail_gib - terms.growth - terms.dip - hard_gib);
    return @intFromFloat(@floor(room * GiB / bytes_a_token));
}

/// The predicted worst MemAvailable (GiB) with a pool of pool_gib.
pub fn worstAt(avail_gib: f64, pool_gib: f64, terms: Serve) f64 {
    return avail_gib - pool_gib - terms.growth - terms.dip;
}

/// Load-time check: MemAvailable - pool (split-aware) - growth - worst dip >= hard floor; after Boot, if it applies.
pub fn serveCheck(k: Knobs, m: Measured, layout: sessions.Layout, world: u32, pool_tokens: u64) Result {
    const per = tokenBytes(layout, world);
    const pool_gib = @as(f64, @floatFromInt(poolBytes(layout, world, pool_tokens))) / GiB;
    const worst = worstAt(m.avail_gib, pool_gib, m.terms);
    var split = false;
    for (layout.families) |f| split = split or (f.split and world > 1);
    const largest = largestPool(m.avail_gib, m.terms, k.hard_gib, per);
    const step = @as(u64, layout.page) * (if (split) world else 1);
    return .{
        .rank = m.rank,
        .ok = worst >= k.hard_gib,
        .worst = worst,
        .avail_gib = m.avail_gib,
        .pool_tokens = pool_tokens,
        .pool_gib = pool_gib,
        .bytes_a_token = per,
        .split = split,
        .growth = m.terms.growth,
        .dip = m.terms.dip,
        .hard = k.hard_gib,
        .largest = largest,
        .largest_paged = largest / step * step,
    };
}

/// n with thousands separators.
fn grouped(buf: *[32]u8, n: u64) []const u8 {
    var tmp: [24]u8 = undefined;
    const d = std.fmt.bufPrint(&tmp, "{d}", .{n}) catch unreachable;
    var o: usize = 0;
    for (d, 0..) |ch, i| {
        if (i > 0 and (d.len - i) % 3 == 0) {
            buf[o] = ',';
            o += 1;
        }
        buf[o] = ch;
        o += 1;
    }
    return buf[0..o];
}

/// A float as the refusal message prints the hard floor (4.0, 4.5).
fn pyFloat(buf: *[32]u8, v: f64) []const u8 {
    const s = std.fmt.bufPrint(buf, "{d}", .{v}) catch unreachable;
    if (std.mem.indexOfAny(u8, s, ".en") != null) return s;
    buf[s.len] = '.';
    buf[s.len + 1] = '0';
    return buf[0 .. s.len + 2];
}

const layout_mod = @import("layout.zig");

test "the boot budget: the reference's pool cap, its second form's, and split KV's pool from the same memory" {
    var l = try layout_mod.Layout.fromConfig(layout_mod.Release{}, .{});
    const k: Knobs = .{ .staging_bytes = 0 };
    // a 99.48 GiB-weights boot refuses pools above 3.23 GiB = 1.94M tokens (111.75 - 99.48 - 4 slots - 4.0 - 1.0 - 4)
    const boot: Boot = .{ .weights = 99.48 };
    const t = boot.poolTokens(k, l.pool(), 1);
    try std.testing.expect(t > 1_930_000 and t < 1_945_000);
    try std.testing.expect(boot.floor(k, t * 1790) >= 4.0);
    // split KV over two ranks holds 1,790 / 1,060 times the tokens (less the union buffers it prices)
    var s = try layout_mod.Layout.fromConfig(layout_mod.Release{}, .{ .split = true });
    const ks: Knobs = .{ .staging_bytes = 0, .union_cap = 128 << 20 };
    const ts = boot.poolTokens(ks, s.pool(), 2);
    const ratio = @as(f64, @floatFromInt(ts)) / @as(f64, @floatFromInt(t));
    const spare = boot.available - boot.fixed(k) - k.hard_gib;
    const want = (spare - 0.1875) / spare * 1790.0 / 1060.0; // 1.59: the union's 192 MiB comes off first
    try std.testing.expectApproxEqAbs(want, ratio, 1e-3);
    try std.testing.expectEqual(@as(u64, 0), ts % (2 * 256));
}

test "prefill and restore prices, admission against both ranks and residues" {
    const k: Knobs = .{};
    const p = prefillPrice(k, 1_000_000, 0); // a 1M prompt is priced as a 32K one: 1.133 + 1.067 GiB at 2,048 rows
    const p32 = prefillPrice(k, 32_769, 0);
    try std.testing.expectEqual(p.total(), p32.total());
    try std.testing.expectApproxEqAbs(@as(f64, 1.133 + 1.067), @as(f64, @floatFromInt(p.total())) / GiB, 1e-6);
    try std.testing.expectEqual(@as(u64, 0), prefillPrice(k, 100, 99).total());
    try std.testing.expectEqual(p.index / 2, p.outstanding(16384) - 0);
    const ks: Knobs = .{ .union_cap = 128 << 20 };
    try std.testing.expectEqual(@as(u64, 192 << 20), prefillPrice(ks, 5000, 0).once - prefillPrice(k, 5000, 0).once);
    const fams = [_]sessions.Family{.{ .name = "comp", .ratio = 1, .row_bytes = 584, .split = true }};
    var pool = try sessions.Pool.init(std.testing.allocator, .{ .families = &fams }, 16 * 256, 2, 0);
    defer pool.deinit();
    const g: u64 = 1 << 30;
    const ask: Ask = .{ .pages = 10, .price = p.total(), .restore = restorePrice(k, true) };
    try std.testing.expectEqual(Verdict.admit, admit(k, &pool, ask, &.{ 9 * g, 8 * g }).verdict);
    const r = admit(k, &pool, ask, &.{ 9 * g, 6 * g }); // rank 1 binds: 6 - 2.2 - 0.125 < 4
    try std.testing.expect(r.verdict == .wait_memory and r.rank == 1);
    try std.testing.expectEqual(Verdict.wait_pages, admit(k, &pool, .{ .pages = 17, .price = 0 }, &.{ 9 * g, 9 * g }).verdict);
}

test "the measured boot check: the reference's served and middle-profile goldens" {
    var rl = try layout_mod.Layout.fromConfig(layout_mod.Release{}, .{});
    var sl = try layout_mod.Layout.fromConfig(layout_mod.Release{}, .{ .split = true });
    const rep = rl.pool();
    const spl = sl.pool();
    try std.testing.expectEqual(@as(f64, 1790), tokenBytes(rep, 2));
    try std.testing.expectEqual(@as(f64, 1060), tokenBytes(spl, 2));
    const k: Knobs = .{};
    const at_check = [2]f64{ 14.0, 12.9 }; // MemAvailable at load_check, head / worker
    var buf: [512]u8 = undefined;
    for (0..2) |r| {
        const m: Measured = .{ .rank = @intCast(r), .avail_gib = at_check[r], .terms = Serve.measured(@intCast(r)) };
        const served = serveCheck(k, m, rep, 2, 1_400_000); // replicated 1.4M
        const mid = serveCheck(k, m, spl, 2, 3_000_000); // the middle profile: split 3.0M
        try std.testing.expect(served.ok and mid.ok);
        try std.testing.expectApproxEqAbs(@as(f64, if (r == 0) 4.86610562801361 else 4.666105628013612), served.worst, 1e-12);
        try std.testing.expectApproxEqAbs(@as(f64, if (r == 0) 4.238394212722778 else 4.038394212722779), mid.worst, 1e-12);
        try std.testing.expect(!serveCheck(k, m, rep, 2, 2_200_000).ok); // replicated 2.2M
        try std.testing.expect(!serveCheck(k, m, spl, 2, 3_500_000).ok); // split 3.5M, which the planned budget accepts
    }
    const w: Measured = .{ .rank = 1, .avail_gib = 12.9, .terms = Serve.measured(1) };
    const refused = serveCheck(k, w, spl, 2, 3_500_000);
    try std.testing.expectApproxEqAbs(@as(f64, 3.5447932481765747), refused.worst, 1e-12);
    try std.testing.expectEqual(@as(u64, 3_038_891), refused.largest); // int(3.0 GiB / 1,060)
    try std.testing.expectEqual(@as(u64, 3_038_720), refused.largest_paged); // 5,935 x 512
    try std.testing.expect(serveCheck(k, w, spl, 2, refused.largest_paged).ok);
    try std.testing.expect(!serveCheck(k, w, spl, 2, refused.largest_paged + 512).ok);
    const msg = refused.message(&buf);
    try std.testing.expect(std.mem.indexOf(u8, msg, "rank 1: a 3.46 GiB pool leaves 3.54 GiB at the worst measured serving point (MemAvailable 12.90 now - boot growth 1.70 - serving dip 4.20), under the 4.0 GiB hard floor") != null);
    try std.testing.expect(std.mem.indexOf(u8, msg, "the largest pool that fits: 3,038,891 tokens") != null);
    const rr = serveCheck(k, w, rep, 2, 2_200_000);
    try std.testing.expectEqual(@as(u64, 1_799_567), rr.largest);
    try std.testing.expectEqual(@as(u64, 1_799_424), rr.largest_paged);
    try std.testing.expect(std.mem.indexOf(u8, rr.message(&buf), "hard floor") != null);
    const ok = serveCheck(k, w, spl, 2, 3_000_000).message(&buf);
    try std.testing.expectEqualStrings("rank 1 pool 3,000,000 tokens = 2.96 GiB (1060 B a token, split KV); predicted worst MemAvailable 4.04 GiB (hard floor 4.0)", ok);
    // a nothing-fits rank: no room, no pool
    try std.testing.expectEqual(@as(u64, 0), largestPool(8.0, Serve.measured(1), 4.0, 1060));
}

test "the measured check's knobs: growth / dip overrides, TF_DSV41_BOOT_MEASURED, MemAvailable" {
    var sl = try layout_mod.Layout.fromConfig(layout_mod.Release{}, .{ .split = true });
    const spl = sl.pool();
    const k: Knobs = .{};
    // TF_DSV41_SERVE_DIP_GIB=3.0 (a profile measured with a smaller dip): split 3.5M passes on the worker
    const t = try Serve.withOverrides(1, null, "3.0");
    try std.testing.expectEqual(Serve{ .growth = 1.7, .dip = 3.0 }, t);
    try std.testing.expect(serveCheck(k, .{ .rank = 1, .avail_gib = 12.9, .terms = t }, spl, 2, 3_500_000).ok);
    // worst = 12.9 - 2.0 - 1.7 - 3.0 (rank 1, 1060 bytes a token, hard 4)
    try std.testing.expectApproxEqAbs(@as(f64, 12.9 - 2.0 - 1.7 - 3.0), worstAt(12.9, 2.0, t), 1e-12);
    try std.testing.expectEqual(Serve{ .growth = 2.5, .dip = 4.9 }, try Serve.withOverrides(0, " 2.5 ", ""));
    try std.testing.expectEqual(Serve.measured(7), try Serve.withOverrides(7, null, null)); // any rank past 0: the worker's
    try std.testing.expectError(error.BadGiBEnv, Serve.withOverrides(1, "-1", null));
    try std.testing.expectError(error.BadGiBEnv, Serve.withOverrides(1, null, "lots"));
    // when it applies: forced, integrated, or device memory within 15 % of the host's
    try std.testing.expectEqual(Mode.off, parseMode(" OFF"));
    try std.testing.expectEqual(Mode.on, parseMode("True"));
    try std.testing.expectEqual(Mode.auto, parseMode(null));
    try std.testing.expectEqual(Mode.auto, parseMode("maybe"));
    const g: u64 = 1 << 30;
    try std.testing.expect(measuredApplies(.auto, true, 0, 121 * g));
    try std.testing.expect(measuredApplies(.auto, false, 119 * g, 121 * g)); // GB10 without the integrated flag
    try std.testing.expect(!measuredApplies(.auto, false, 96 * g, 1024 * g)); // a discrete card
    try std.testing.expect(measuredApplies(.on, false, 96 * g, 1024 * g));
    try std.testing.expect(!measuredApplies(.off, true, 121 * g, 121 * g));
    const mi = try parseMeminfo("MemTotal:       126877032 kB\nMemFree:         1000 kB\nMemAvailable:   13526528 kB\n");
    try std.testing.expectEqual(MemInfo{ .total = 126877032 * 1024, .available = 13526528 * 1024 }, mi);
    try std.testing.expectError(error.BadMeminfo, parseMeminfo("MemTotal: 1 kB\n"));
    if (@import("builtin").os.tag == .linux) try std.testing.expect(try memAvailable() > 0);
}

test "the Zig check: measured at the end of boot, a measured rank 1 passes, a pool 1 GiB past the largest refuses" {
    var sl = try layout_mod.Layout.fromConfig(layout_mod.Release{}, .{ .split = true });
    const spl = sl.pool();
    const k: Knobs = .{};
    const per = tokenBytes(spl, 2);
    const pool = @as(f64, @floatFromInt(poolBytes(spl, 2, 3_000_000))) / GiB;
    // a rank 1 before the pool: the reference's terms refused it (6.07 - 2.10 workspace < 4); Zig's pass it
    const terms: Serve = .{ .growth = 0, .dip = zig_dip_gib };
    const pre = serveCheck(k, .{ .rank = 1, .avail_gib = 14.94, .terms = terms }, spl, 2, 3_000_000);
    try std.testing.expect(pre.ok);
    // after the boot: a measured rank 1 idle 8.70 - 3.0 dip = 5.70, over the 4.0 floor
    const ok = afterBoot(14.94, pool, 8.70, 0, zig_dip_gib, k.hard_gib, 3_000_000, per);
    try std.testing.expect(ok.ok);
    try std.testing.expectApproxEqAbs(@as(f64, 14.94 - pool - 8.70), ok.growth, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 5.70), ok.worst, 1e-9);
    // the largest pool lands the worst point on the floor; 1 GiB more pool (1 GiB less idle) refuses
    const more = @as(u64, @intFromFloat(GiB / per));
    try std.testing.expect(ok.largest > 3_000_000);
    const edge = ok.largest - 3_000_000;
    const over = afterBoot(14.94, pool + (@as(f64, @floatFromInt(edge)) * per / GiB) + 1.0, 8.70 - (@as(f64, @floatFromInt(edge)) * per / GiB) - 1.0, 0, zig_dip_gib, k.hard_gib, ok.largest + more, per);
    try std.testing.expect(!over.ok);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), over.worst, 1e-6); // the floor less the GiB
    try std.testing.expect(over.largest < ok.largest + more);
    // an earlier boot's idle (6.39, before the trims) refuses at the measured dip: 2.48 would have left 3.91
    try std.testing.expect(!afterBoot(14.94, pool, 6.39, 0, zig_dip_gib, k.hard_gib, 3_000_000, per).ok);
    try std.testing.expectEqual(Serve{ .growth = 0, .dip = zig_dip_gib }, try zigFromEnv());
    // the reserve apart: a measured rank 1 keeps the floor (8.76 - 1.5 - 3.0 = 4.26); 7.33 idle (4.2M untrimmed) not
    const r = afterBoot(14.94, pool, 8.76, 1.5, zig_dip_gib, k.hard_gib, 3_000_000, per);
    try std.testing.expect(r.ok);
    try std.testing.expectApproxEqAbs(@as(f64, 4.26), r.worst, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 14.94 - pool - 8.76), r.growth, 1e-9); // growth from the true idle
    try std.testing.expect(!afterBoot(14.94, pool, 7.33, 1.5, zig_dip_gib, k.hard_gib, 3_000_000, per).ok);
    try std.testing.expectEqual(@as(f64, 0), try reserveFromEnv());
}
