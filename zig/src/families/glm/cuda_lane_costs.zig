//! Rank 0 keeps MTP prices and each drafter's expert-form choice per build and timing identity.
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

pub const measurements = @import("cuda_lane_cost_measure.zig");
/// Timed widths, with straight-line prices between adjacent measurements.
pub const widths = measurements.widths;
/// The text's tokens before the timed windows.
const prefix = 16;
/// Rows a scratch sequence runs at most before it begins again, within one KV page or two.
const room = 512;
const max_held = dflash.BLOCK - 1;

comptime {
    std.debug.assert(widths[0] == 1 and widths[widths.len - 1] == fw.window_rows);
}

/// What the lane core prices by: a window's ms at every width and one draft's.
pub const Table = struct { window: [fw.window_rows]lanes.config.Cost, step_ms: f64 };

/// What a load times and a later load of this build reads back.
pub const Kept = measurements.Kept;

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
    return measurements.reusable(k, false);
}

/// DFlash2 retains its seed prices; only MTP publishes a measured table to the lane core.
pub fn tableFor(k: Kept, block: bool) ?Table {
    return if (block) null else table(k);
}

/// Rank 0 chooses expert forms; prices are null for DFlash2, no drafts, or too little scratch context.
pub fn price(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, x: follow.Leader) !?Table {
    const e = x.e;
    const pin = try pinnedForm();
    if (!e.mtp or e.capacity < prefix + 2 * fw.window_rows + 2) return untimed(x, pin);
    var name: [256]u8 = undefined;
    var shape: [512]u8 = undefined;
    const block = e.drafter != null;
    const parts = try keyParts(e, ctx, &name, &shape, pin);
    const key = if (parts) |v| cache.key(gpa, io, &v) catch null else null;
    if (key) |k| if (cache.load(Kept, gpa, io, k)) |kept| if (reuses(kept, block, pin)) {
        try x.expertForm(kept.fact != 0);
        try x.idle();
        report(kept, "kept", block, pin != null);
        return tableFor(kept, block);
    };
    const path = try std.fs.path.join(gpa, &.{ dir, "tokenizer.json" });
    defer gpa.free(path);
    var tok = try core.tokenizer.loadTokenizer(io, gpa, path);
    defer tok.deinit();
    const ids = try tok.encode(gpa, rule.text);
    defer gpa.free(ids);
    if (ids.len < prefix + fw.window_rows + 1) return error.CostTextTooShort;
    var s: Scratch = .{ .x = x, .io = io, .ids = ids, .most = @min(room, e.capacity - fw.window_rows - prefix - 2), .drafting = !block };
    try s.begin();
    var kept = try measurements.measure(&s, block, pin);
    const reference = if (!block and parts != null) cache.referenceKey(&parts.?) else null;
    if (reference) |key_ref| if (cache.load(Kept, gpa, io, key_ref)) |r| if (rule.drifted(&kept.ms, &r.ms)) try rule.again(Windows{ .s = &s }, &kept.ms);
    try x.release(0);
    try x.idle();
    if (key) |k| cache.save(Kept, gpa, io, k, kept);
    if (reference) |key_ref| cache.save(Kept, gpa, io, key_ref, kept);
    report(kept, "timed", block, pin != null);
    return tableFor(kept, block);
}

/// A kept record stands for this load when it holds prices and, under a named form, was kept at that form.
fn reuses(kept: Kept, block: bool, pin: ?bool) bool {
    return measurements.reusable(kept, block) and (pin == null or pin.? == (kept.fact != 0));
}

/// TF_GLM_EXPERT_FORM on rank 0, which chooses every rank's form: null races classic and factored, else the one named.
fn pinnedForm() !?bool {
    const i = (try sys.environment.read(?u64, "TF_GLM_EXPERT_FORM", null)) orelse return null;
    return i == 1;
}

/// A load that times nothing still sends a named form to every rank; unnamed, the forms stay classic.
fn untimed(x: anytype, pin: ?bool) !?Table {
    const f = pin orelse return null;
    try x.expertForm(f);
    try x.idle();
    std.log.info(untimed_line, .{if (f) "factored" else "classic"});
    return null;
}

/// The startup lines; a named form puts ` (TF_GLM_EXPERT_FORM)` right after the form's word.
const window_line = "glm: windows ({s}): {d:.2} ms at 1 row, {d:.2} at 4, {d:.2} at 32, a draft {d:.2} ms; experts {s}{s}";
const form_line = "glm: expert form ({s}): {s}{s}; DFlash2 keeps its seed window and step prices";
const untimed_line = "glm: expert form (untimed): {s} (TF_GLM_EXPERT_FORM)";

fn report(k: Kept, how: []const u8, block: bool, pinned: bool) void {
    const form = if (k.fact != 0) "factored" else "classic";
    if (block) {
        std.log.info(form_line, .{ how, form, named(pinned) });
        return;
    }
    std.log.info(window_line, .{ how, k.ms[0], k.ms[3], k.ms[widths.len - 1], k.step_ms, form, named(pinned) });
}

fn named(pinned: bool) []const u8 {
    return if (pinned) " (TF_GLM_EXPERT_FORM)" else "";
}

/// Cache identity: the GPU and driver, this checkpoint and drafter, the settings a window's time follows, a named form.
pub fn keyParts(e: *const engine.Engine, ctx: *const cuda.Context, name: []u8, shape: []u8, pin: ?bool) !?[3][]const u8 {
    const at = cacheShape(e, try ctx.capability(), try ctx.d.version(), shape, pin) orelse return null;
    return .{ if (e.drafter != null) "glm-cuda-form" else "glm-cuda", ctx.name(name) catch "gpu", at };
}

/// An oversized identity disables caching; it does not reject a valid model or truncate a key.
fn cacheShape(e: *const engine.Engine, sm: usize, driver: c_int, shape: []u8, pin: ?bool) ?[]const u8 {
    return keyShape(e, sm, driver, shape, pin) catch {
        std.log.warn("glm: lane cost identity exceeds its buffer; skipping the lane cost cache", .{});
        return null;
    };
}

/// A named form ends the identity, so a race's kept prices and a named form's never stand in for each other.
fn keyShape(e: *const engine.Engine, sm: usize, driver: c_int, shape: []u8, pin: ?bool) ![]const u8 {
    const set = e.settings;
    const hex = std.fmt.bytesToHex(e.checkpoint_digest, .lower);
    const draft_hex = std.fmt.bytesToHex(if (e.drafter) |dr| dr.checkpoint_digest else @as([32]u8, @splat(0)), .lower);
    const tail: []const u8 = if (pin) |f| (if (f) " form factored" else " form classic") else "";
    return std.fmt.bufPrint(shape, "sm{d} cuda{d} ranks{d} {s} drafter{d}/{d}/{s} layers{d} l2{d}/{d}/{d}/{d}/{d} warps{d} kda{d} mm{d} bits{d}/{d} kv8{d} pages{d} host{d} roce{d} p2p{s}/{s}/{s}{s}", .{
        sm,                                          driver,                       e.share.world,                          hex[0..],                                   @intFromBool(e.drafter != null),
        if (e.drafter) |dr| dr.bits else @as(u8, 0), draft_hex[0..],               set.layers,                             @intFromBool(set.l2pf.on),                  @intFromBool(set.l2pf.named),
        set.l2pf.mb,                                 set.l2pf.ctas,                set.l2pf.pace_gbps,                     e.m.experts.nvfp4.warps,                    @intFromBool(set.kda_wide),
        @intFromBool(set.tiled),                     set.dense_bits,               set.head_bits,                          @intFromBool(set.kv8),                      @intFromBool(set.pages),
        set.kv_host,                                 @intFromBool(set.roce_multi), sys.getenv("NCCL_P2P_LEVEL") orelse "", sys.getenv("NCCL_MIN_NCHANNELS") orelse "", sys.getenv("NCCL_MAX_NCHANNELS") orelse "",
        tail,
    });
}

const Windows = struct {
    s: *Scratch,
    pub fn time(w: Windows, i: usize) !f64 {
        return (try w.s.round(widths[i], 0)).window;
    }
};

/// Slot 0's scratch stream: rounds as a served stream runs them, begun again before it passes `most` rows.
const Scratch = struct {
    x: follow.Leader,
    io: std.Io,
    ids: []const u32,
    most: usize,
    drafting: bool = true,
    at: usize = prefix, // the next window's pending token in the text, which repeats
    out: [fw.window_rows]u32 = undefined,
    held: [@max(max_held, engine.max_drafts)]u32 = undefined,
    chances: [@max(max_held, engine.max_drafts)]f64 = undefined,

    pub fn form(s: *Scratch, fact: bool) !void {
        try s.x.expertForm(fact);
    }

    pub fn window(s: *Scratch, i: usize) !f64 {
        return (try s.round(widths[i], 0)).window;
    }

    pub fn draft(s: *Scratch, depth: usize) !f64 {
        return (try s.round(1, depth)).draft;
    }

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
        if (s.drafting) try s.absorb(true, s.token(0), 0);
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
        const window_ms = clock.clock().elapsedMs(.cost);
        try s.x.keep(1);
        s.at += 1;
        try e.stream.synchronize();
        clock.clock().start();
        if (s.drafting) try s.absorb(false, s.token(0), depth);
        return .{ .window = window_ms, .draft = clock.clock().elapsedMs(.cost) };
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
    const before = try keyShape(&e, 121, 13000, &base, null);
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
        try std.testing.expect(!std.mem.eql(u8, before, try keyShape(&e, 121, 13000, &changed, null)));
        e.settings = original;
        e.drafter = &dr;
        dr.bits = 4;
        dr.checkpoint_digest = @splat(2);
        e.checkpoint_digest = @splat(1);
        e.m.experts.nvfp4.warps = 4;
    }
    // a named form only ends the identity, so the race's key stays as it was
    var classic: [512]u8 = undefined;
    var factored: [512]u8 = undefined;
    const pinned = try keyShape(&e, 121, 13000, &classic, false);
    try std.testing.expect(std.mem.startsWith(u8, pinned, before) and pinned.len > before.len);
    try std.testing.expect(!std.mem.eql(u8, pinned, try keyShape(&e, 121, 13000, &factored, true)));
}

test "GLM cost identity: a named form's startup line ends at (TF_GLM_EXPERT_FORM), and a raced one as it did" {
    var buf: [256]u8 = undefined;
    const raced = try std.fmt.bufPrint(&buf, window_line, .{ "timed", 29.73, 44.02, 130.65, 1.53, "classic", named(false) });
    try std.testing.expect(std.mem.endsWith(u8, raced, "130.65 at 32, a draft 1.53 ms; experts classic"));
    const kept = try std.fmt.bufPrint(&buf, window_line, .{ "kept", 29.73, 44.02, 130.65, 1.53, "factored", named(true) });
    try std.testing.expect(std.mem.endsWith(u8, kept, "a draft 1.53 ms; experts factored (TF_GLM_EXPERT_FORM)"));
    const form = try std.fmt.bufPrint(&buf, form_line, .{ "timed", "classic", named(true) });
    try std.testing.expect(std.mem.startsWith(u8, form, "glm: expert form (timed): classic (TF_GLM_EXPERT_FORM); DFlash2"));
    try std.testing.expectEqualStrings("glm: expert form (untimed): factored (TF_GLM_EXPERT_FORM)", try std.fmt.bufPrint(&buf, untimed_line, .{"factored"}));
}

test "GLM cost identity: a load that times nothing sends a named form to every rank once, and nothing unnamed" {
    const Leader = struct {
        form: ?bool = null,
        sent: usize = 0,
        idles: usize = 0,
        pub fn expertForm(l: *@This(), fact: bool) !void {
            l.form = fact;
            l.sent += 1;
        }
        pub fn idle(l: *@This()) !void {
            l.idles += 1;
        }
    };
    for ([_]?bool{ null, false, true }) |pin| {
        var l: Leader = .{};
        try std.testing.expect((try untimed(&l, pin)) == null);
        try std.testing.expectEqual(pin, l.form);
        try std.testing.expectEqual(@as(usize, @intFromBool(pin != null)), l.sent);
        try std.testing.expectEqual(l.sent, l.idles);
    }
}

test "GLM cost identity: a record kept at one named form is not reused at the other" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    var e: engine.Engine = undefined;
    e.settings = .{};
    e.checkpoint_digest = @splat(1);
    e.share.world = 2;
    e.m.experts.nvfp4.warps = 4;
    e.drafter = null;
    var shapes: [2][512]u8 = undefined;
    var keys: [2][32]u8 = undefined;
    for ([_]bool{ false, true }, 0..) |fact, i| {
        const parts = [_][]const u8{ "glm-cuda", "gpu", try keyShape(&e, 121, 13000, &shapes[i], fact) };
        keys[i] = cache.buildKey("engine", &parts);
    }
    var ms: [widths.len]f64 = undefined;
    for (&ms, widths) |*m, w| m.* = 20 + @as(f64, @floatFromInt(w));
    cache.saveAt(Kept, a, io, keys[0], .{ .ms = ms, .step_ms = 1.5, .fact = 0 }, root);
    // a load that names factored looks under its own key and finds nothing to reuse
    try std.testing.expect(cache.loadAt(Kept, a, io, keys[1], root) == null);
    // the classic load reads the record back and reuses it, as the race does; factored would refuse even this record
    const classic = cache.loadAt(Kept, a, io, keys[0], root).?;
    try std.testing.expect(reuses(classic, false, false) and reuses(classic, false, null));
    try std.testing.expect(!reuses(classic, false, true));
    var factored = classic;
    factored.fact = 1;
    try std.testing.expect(reuses(factored, false, true) and !reuses(factored, false, false));
}
