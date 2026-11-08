//! This machine's window and draft prices, timed at load as real rounds over real text; rank 0 keeps them per build.
const std = @import("std");
const sys = @import("sys");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const fw = @import("cuda_forward.zig");
const dflash = @import("dflash.zig");
const rule = lanes.cost_rule;
const cache = lanes.cost_cache;

/// The widths timed, straight lines between: all 32 would add most of a minute to a new build's first load on GB10s.
pub const widths = [_]u32{ 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 };
/// Entries of `widths` the expert forms race over.
const race = [_]usize{ 1, 3, 5 };
/// The factored form must win by this fraction: a tie keeps the classic kernel.
const margin = 0.01;
/// The text's tokens before the timed windows.
const prefix = 16;
/// Widest windows first: the GPU's clocks ramp up under load.
const warm = 8;
/// Rows a scratch sequence runs at most before it begins again, within one KV page or two.
const room = 512;
/// Head chains of 1 and 8 levels: their difference over 7 is a level's ms.
const levels = [2]usize{ 1, 8 };
const max_held = dflash.BLOCK - 1;

comptime {
    std.debug.assert(widths[0] == 1 and widths[widths.len - 1] == fw.window_rows);
}

/// What the lane core prices by: a window's ms at every width and one draft's.
pub const Table = struct { window: [fw.window_rows]lanes.config.Cost, step_ms: f64 };

/// What a load times and a later load of this build reads back.
pub const Kept = extern struct { ms: [widths.len]f64, step_ms: f64, fact: u64 };

/// Every width's ms, straight between the timed ones and never falling as rows grow.
pub fn table(k: Kept) Table {
    var t: Table = .{ .window = undefined, .step_ms = k.step_ms };
    var seg: usize = 0;
    var last: f64 = 0;
    for (&t.window, 1..) |*c, w| {
        while (widths[seg + 1] < w) seg += 1;
        const lo: f64 = @floatFromInt(widths[seg]);
        const hi: f64 = @floatFromInt(widths[seg + 1]);
        const at = (@as(f64, @floatFromInt(w)) - lo) / (hi - lo);
        last = @max(last, k.ms[seg] + at * (k.ms[seg + 1] - k.ms[seg]));
        c.* = .{ .width = @intCast(w), .ms = last };
    }
    return t;
}

/// Whether a table read back holds prices at all.
pub fn usable(k: Kept) bool {
    for (k.ms) |m| if (!(m > 0) or !std.math.isFinite(m)) return false;
    return k.step_ms >= 0 and std.math.isFinite(k.step_ms);
}

/// Rank 0's prices, with every rank on the faster expert form; null: this engine has no drafts or too few rows to time.
pub fn price(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, x: follow.Leader) !?Table {
    const e = x.e;
    if (!e.mtp or e.capacity < prefix + 2 * fw.window_rows + 2) return null;
    var name: [256]u8 = undefined;
    var shape: [512]u8 = undefined;
    const parts = try keyParts(e, ctx, &name, &shape);
    const key = cache.key(gpa, io, &parts) catch null;
    if (key) |k| if (cache.load(Kept, gpa, io, k)) |kept| if (usable(kept)) {
        try x.expertForm(kept.fact != 0);
        try x.idle();
        report(kept, "kept");
        return table(kept);
    };
    const path = try std.fs.path.join(gpa, &.{ dir, "tokenizer.json" });
    defer gpa.free(path);
    var tok = try core.tokenizer.loadTokenizer(io, gpa, path);
    defer tok.deinit();
    const ids = try tok.encode(gpa, rule.text);
    defer gpa.free(ids);
    if (ids.len < prefix + fw.window_rows + 1) return error.CostTextTooShort;
    var s: Scratch = .{ .x = x, .io = io, .ids = ids, .most = @min(room, e.capacity - fw.window_rows - prefix - 2) };
    try s.begin();
    var kept = try timeAll(&s);
    const reference = cache.referenceKey(&parts);
    if (cache.load(Kept, gpa, io, reference)) |r| if (rule.drifted(&kept.ms, &r.ms)) try rule.again(Windows{ .s = &s }, &kept.ms);
    try x.release(0);
    try x.idle();
    if (key) |k| cache.save(Kept, gpa, io, k, kept);
    cache.save(Kept, gpa, io, reference, kept);
    report(kept, "timed");
    return table(kept);
}

fn report(k: Kept, how: []const u8) void {
    std.log.info("glm: windows ({s}): {d:.2} ms at 1 row, {d:.2} at 4, {d:.2} at 32, a draft {d:.2} ms; experts {s}", .{ how, k.ms[0], k.ms[3], k.ms[widths.len - 1], k.step_ms, if (k.fact != 0) "factored" else "classic" });
}

/// Cache identity: the GPU and driver, this checkpoint and drafter, and the settings a window's time follows.
fn keyParts(e: *const engine.Engine, ctx: *const cuda.Context, name: []u8, shape: []u8) ![3][]const u8 {
    const at = try keyShape(e, try ctx.capability(), try ctx.d.version(), shape);
    return .{ "glm-cuda", ctx.name(name) catch "gpu", at };
}

fn keyShape(e: *const engine.Engine, sm: usize, driver: c_int, shape: []u8) ![]const u8 {
    const set = e.settings;
    const hex = std.fmt.bytesToHex(e.checkpoint_digest, .lower);
    const draft_hex = std.fmt.bytesToHex(if (e.drafter) |dr| dr.checkpoint_digest else @as([32]u8, @splat(0)), .lower);
    return std.fmt.bufPrint(shape, "sm{d} cuda{d} ranks{d} {s} drafter{d}/{d}/{s} layers{d} l2{d}/{d}/{d}/{d}/{d} warps{d} kda{d} mm{d} bits{d}/{d} kv8{d} pages{d} host{d} roce{d} p2p{s}/{s}/{s}", .{
        sm,                                          driver,                       e.share.world,                          hex[0..],                                   @intFromBool(e.drafter != null),
        if (e.drafter) |dr| dr.bits else @as(u8, 0), draft_hex[0..],               set.layers,                             @intFromBool(set.l2pf.on),                  @intFromBool(set.l2pf.named),
        set.l2pf.mb,                                 set.l2pf.ctas,                set.l2pf.pace_gbps,                     e.m.experts.nvfp4.warps,                    @intFromBool(set.kda_wide),
        @intFromBool(set.tiled),                     set.dense_bits,               set.head_bits,                          @intFromBool(set.kv8),                      @intFromBool(set.pages),
        set.kv_host,                                 @intFromBool(set.roce_multi), sys.getenv("NCCL_P2P_LEVEL") orelse "", sys.getenv("NCCL_MIN_NCHANNELS") orelse "", sys.getenv("NCCL_MAX_NCHANNELS") orelse "",
    });
}

/// Warm the clocks, race the expert forms on every rank, then time each width and a draft on the winner.
fn timeAll(s: *Scratch) !Kept {
    const w: Windows = .{ .s = s };
    for (0..warm) |_| _ = try s.round(fw.window_rows, 0);
    var forms: [2]f64 = .{ 0, 0 };
    const d: Drafts = .{ .s = s };
    for (&forms, [_]bool{ false, true }) |*sum, fact| {
        try s.x.expertForm(fact);
        for (race) |i| sum.* += try rule.fastest(w, i);
        if (s.x.e.drafter == null) sum.* += try rule.fastest(d, levels[0]); // the head's own experts, a row at a time
    }
    const fact = forms[1] < forms[0] * (1 - margin);
    try s.x.expertForm(fact);
    var k: Kept = .{ .ms = undefined, .step_ms = 0, .fact = @intFromBool(fact) };
    for (&k.ms, 0..) |*m, i| m.* = try rule.fastest(w, i);
    try rule.smooth(w, &k.ms);
    if (s.x.e.drafter != null) {
        k.step_ms = try rule.fastest(d, max_held) / @as(f64, @floatFromInt(max_held));
    } else {
        const chain = [2]f64{ try rule.fastest(d, levels[0]), try rule.fastest(d, levels[1]) };
        k.step_ms = @max(0, (chain[1] - chain[0]) / @as(f64, @floatFromInt(levels[1] - levels[0])));
    }
    return k;
}

const Windows = struct {
    s: *Scratch,
    pub fn time(w: Windows, i: usize) !f64 {
        return (try w.s.round(widths[i], 0)).window;
    }
};

const Drafts = struct {
    s: *Scratch,
    pub fn time(d: Drafts, depth: usize) !f64 {
        return (try d.s.round(1, depth)).draft;
    }
};

/// Slot 0's scratch stream: rounds as a served stream runs them, begun again before it passes `most` rows.
const Scratch = struct {
    x: follow.Leader,
    io: std.Io,
    ids: []const u32,
    most: usize,
    at: usize = prefix, // the next window's pending token in the text, which repeats
    out: [fw.window_rows]u32 = undefined,
    held: [@max(max_held, engine.max_drafts)]u32 = undefined,
    chances: [@max(max_held, engine.max_drafts)]f64 = undefined,

    fn token(s: *const Scratch, r: usize) u32 {
        return s.ids[(s.at + r) % s.ids.len];
    }

    fn begin(s: *Scratch) !void {
        const e = s.x.e;
        try s.x.bind(0);
        const pages: []const u32 = if (e.pages) |p| try p.open(0, prefix, s.most) else &.{};
        try s.x.beginOn(null, pages, 0);
        _ = try s.x.prefill(s.ids[0..prefix], null, null, null);
        s.at = prefix;
        try s.absorb(true, s.token(0), 0);
    }

    /// A window of `rows` from its step to its picks, then one row kept and `depth` drafts from its next token.
    fn round(s: *Scratch, rows: usize, depth: usize) !struct { window: f64, draft: f64 } {
        const e = s.x.e;
        if (e.pos() + fw.window_rows + 2 >= prefix + s.most) {
            try s.x.release(0);
            try s.begin();
        }
        var ids: [fw.window_rows]u32 = undefined;
        for (ids[0..rows], 0..) |*t, r| t.* = s.token(r);
        try e.stream.synchronize(); // the last round's work is not this window's
        var clock: lanes.backend.WallClock = .{ .io = s.io };
        clock.clock().start();
        try s.x.verify(ids[0..rows], s.out[0..rows], 0);
        const window = clock.clock().elapsedMs(.cost);
        try s.x.keep(1);
        s.at += 1;
        try e.stream.synchronize();
        clock.clock().start();
        try s.absorb(false, s.token(0), depth);
        return .{ .window = window, .draft = clock.clock().elapsedMs(.cost) };
    }

    /// The head, or the drafter's block, takes the kept row and proposes `depth` drafts after `next`.
    fn absorb(s: *Scratch, from_prompt: bool, next: u32, depth: usize) !void {
        if (s.x.e.drafter != null) {
            var reqs = [_]dflash.Req{.{ .slot = 0, .pos = 0, .n = @intFromBool(!from_prompt), .anchor = next, .propose = depth > 0, .out = &s.held, .probs = &s.chances }};
            return s.x.block(&reqs);
        }
        try s.x.draft(from_prompt, &.{next}, depth, &s.held, null);
    }
};

test "GLM cost identity: the consumed key includes drafter precision and all timing knobs" {
    var e: engine.Engine = undefined;
    e.settings = .{};
    e.checkpoint_digest = @splat(1);
    e.share.world = 2;
    e.m.experts.nvfp4.warps = 4;
    var dr: dflash.Drafter = undefined;
    dr.bits = 4;
    dr.checkpoint_digest = @splat(2);
    e.drafter = &dr;
    var base: [512]u8 = undefined;
    const before = try keyShape(&e, 121, 13000, &base);
    for (0..11) |which| {
        const original = e.settings;
        switch (which) {
            0 => dr.bits = 0,
            1 => dr.checkpoint_digest[31] ^= 1,
            2 => e.settings.layers = 1,
            3 => e.settings.l2pf.on = false,
            4 => e.settings.l2pf.named = true,
            5 => e.settings.l2pf.mb += 1,
            6 => e.settings.l2pf.ctas += 1,
            7 => e.settings.l2pf.pace_gbps += 1,
            8 => e.m.experts.nvfp4.warps = 1,
            9 => e.drafter = null,
            10 => e.checkpoint_digest[31] ^= 1,
            else => unreachable,
        }
        var changed: [512]u8 = undefined;
        try std.testing.expect(!std.mem.eql(u8, before, try keyShape(&e, 121, 13000, &changed)));
        e.settings = original;
        e.drafter = &dr;
        dr.bits = 4;
        dr.checkpoint_digest = @splat(2);
        e.checkpoint_digest = @splat(1);
        e.m.experts.nvfp4.warps = 4;
    }
}
