//! LACI (zonos3.tts.laci): a request's reading position watched as it decodes, and rolled back when it skips or stalls.
const std = @import("std");
const cfg = @import("config.zig");
const short_text = @import("short_text.zig");
const Allocator = std.mem.Allocator;

const C = cfg.codebooks;
const W = cfg.frame_width;

const detect = @import("laci_detect.zig");
pub const Config = detect.Config;
pub const Kind = detect.Kind;
pub const Failure = detect.Failure;
pub const Detector = detect.Detector;

/// The ACI attention centre (arXiv 2404.19723): the cheapest monotone path's index, never back and at most one on.
pub const AciCenter = struct {
    a: Allocator,
    text_len: usize,
    c0: usize,
    d: std.ArrayList(f64) = .empty,
    center: usize,

    pub fn init(a: Allocator, c0: usize, text_len: usize) !AciCenter {
        const start = @min(c0, text_len - 1);
        var aci: AciCenter = .{ .a = a, .text_len = text_len, .c0 = start, .center = start };
        try aci.d.append(a, 0.0);
        return aci;
    }

    pub fn deinit(aci: *AciCenter) void {
        aci.d.deinit(aci.a);
    }

    pub fn update(aci: *AciCenter, m: f64) !usize {
        if (aci.c0 + aci.d.items.len < aci.text_len) try aci.d.append(aci.a, std.math.inf(f64));
        const d = aci.d.items;
        var i = d.len - 1;
        while (i > 0) : (i -= 1) {
            const off = m - @as(f64, @floatFromInt(aci.c0 + i));
            d[i] = @min(d[i], d[i - 1]) + off * off;
        }
        const off0 = m - @as(f64, @floatFromInt(aci.c0));
        d[0] += off0 * off0;
        var best: usize = 0;
        for (d, 0..) |x, j| {
            if (x < d[best]) best = j;
        }
        aci.center = @min(@max(aci.c0 + best, aci.center), aci.center + 1);
        return aci.center;
    }
};

/// An alignment head (an attention layer's query head) and the half-width of its ACI window.
pub const Head = struct { layer: u32, head: u32, radius: i64 };

/// A masked head's inclusive window of text positions, before it is clipped to the text.
pub const Window = struct { layer: u32, head: u32, lo: i64, hi: i64 };

/// The alignment heads of one model and language (a JSON file of the reference's assets/laci_heads).
pub const Heads = struct {
    heads: []const Head,
    /// The pace prior until the reading has covered warmup_bins bins.
    frames_per_token: f64,

    pub fn parse(a: Allocator, text: []const u8) !Heads {
        const root = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        if (root != .object) return error.BadLaciHeads;
        const list = root.object.get("heads") orelse return error.BadLaciHeads;
        if (list != .array or list.array.items.len == 0) return error.BadLaciHeads;
        const heads = try a.alloc(Head, list.array.items.len);
        for (heads, list.array.items) |*h, v| {
            if (v != .object) return error.BadLaciHeads;
            h.* = .{ .layer = try int(u32, v.object.get("layer")), .head = try int(u32, v.object.get("head")), .radius = try int(i64, v.object.get("radius")) };
            if (h.head >= cfg.heads or h.layer >= cfg.layers or cfg.usesGdn(h.layer)) return error.LaciHeadsOfAnotherModel;
        }
        const pace = root.object.get("frames_per_token") orelse return error.BadLaciHeads;
        const fpt: f64 = switch (pace) {
            .float => |f| f,
            .integer => |n| @floatFromInt(n),
            else => return error.BadLaciHeads,
        };
        if (!(fpt > 0) or !std.math.isFinite(fpt)) return error.BadLaciHeads;
        return .{ .heads = heads, .frames_per_token = fpt };
    }

    fn int(comptime T: type, v: ?std.json.Value) !T {
        const x = v orelse return error.BadLaciHeads;
        return if (x == .integer) std.math.cast(T, x.integer) orelse error.BadLaciHeads else error.BadLaciHeads;
    }

    /// The heads' windows around `center`: (center - radius + 1, center + radius - 1).
    pub fn windows(h: Heads, center: usize, out: []Window) []const Window {
        const c: i64 = @intCast(center);
        for (h.heads, out[0..h.heads.len]) |head, *w| w.* = .{ .layer = head.layer, .head = head.head, .lo = c - head.radius + 1, .hi = c + head.radius - 1 };
        return out[0..h.heads.len];
    }
};

/// Whether a host may run `lc` (a core SpeechLaci): attention heads of this model with windows, and no guidance twin.
pub fn valid(lc: anytype, guided: bool) bool {
    if (guided or lc.heads.len == 0 or lc.heads.len > cfg.heads * (cfg.layers / cfg.gdn_period)) return false;
    if (!(lc.frames_per_token > 0) or !std.math.isFinite(lc.frames_per_token)) return false;
    if (lc.short) |sh| if (sh.weights.len != C * cfg.codebook_size or sh.filler_token == 0) return false;
    for (lc.heads) |h| if (h.layer >= cfg.layers or cfg.usesGdn(h.layer) or h.head >= cfg.heads or h.radius < 1 or h.radius > cfg.max_seqlen) return false;
    return true;
}

/// What an attempt's decode loop reads and the session sets: the masks' centre and whether to stop after this row.
pub const Probe = struct {
    text_len: usize,
    /// The ACI windows' centre while the attempt is on probation; null: no head is masked.
    center: ?usize = null,
    stop: bool = false,
};

/// A rolled-back request's regeneration: the prompt without its prefill rows, then stream rows [0, rows).
pub const Continuation = struct { rows: usize, history_rows: usize, seed: ?u64, max_tokens: usize };

pub const End = enum { text_done, eos, max_tokens, short_text, short_text_forced };

/// A failure as the session records it; `rollback_row` is where the regeneration resumed.
pub const Recorded = struct { failure: Failure, rollback_row: usize, clamped: bool, retry_seed: ?u64, ignored: bool };

/// One request under LACI: its stream, detector, retries and ACI probation (LaciSession, without duration targets).
pub const Session = struct {
    a: Allocator,
    heads: Heads,
    c: Config,
    n_prefill: usize,
    seed: ?u64,
    max_tokens: usize,
    /// A whole decode: nothing goes downstream before the result is final, so no rollback is ever clamped.
    hold: bool,
    /// The prefill rows, then the generated rows.
    stream: std.ArrayList([W]i32) = .empty,
    /// The stream row of the current attempt's first generated row.
    attempt_start: usize,
    /// Generated rows handed downstream.
    emitted: usize = 0,
    detector: ?Detector = null,
    text_len: usize = 0,
    attempts: usize = 0,
    failures: std.ArrayList(Recorded) = .empty,
    probation: ?struct { failure: Failure, aci: AciCenter } = null,
    pending: ?Failure = null,
    text_done: bool = false,
    /// The current attempt's stream EOS row, once the model emitted EOS.
    eos_row: ?usize = null,
    generated_total: usize = 0,
    end: ?struct { kind: End, row: ?usize } = null,
    ignored: std.ArrayList(struct { kind: Kind, bin: usize, onset_row: usize }) = .empty,
    /// A short text's cut; null: an ordinary request.
    short: ?short_text.Cut = null,
    short_retries: usize = 0,

    pub fn init(a: Allocator, heads: Heads, c: Config, prefill: []const [W]i32, seed: ?u64, max_tokens: usize, hold: bool, short: ?short_text.Cut) !Session {
        var s: Session = .{ .a = a, .heads = heads, .c = c, .n_prefill = prefill.len, .seed = seed, .max_tokens = max_tokens, .hold = hold, .attempt_start = prefill.len, .short = short };
        try s.stream.appendSlice(a, prefill);
        return s;
    }

    pub fn deinit(s: *Session) void {
        s.stream.deinit(s.a);
        if (s.detector) |*d| d.deinit();
        if (s.probation) |*p| p.aci.deinit();
        if (s.short) |*cut| cut.deinit();
        s.failures.deinit(s.a);
        s.ignored.deinit(s.a);
    }

    /// The ACI windows' centre of the next attempt's probe; null: no mask.
    pub fn probeCenter(s: *const Session) ?usize {
        return if (s.probation) |p| p.aci.center else null;
    }

    /// The logit the sampler adds to the stop id for the next row.
    pub fn stopBoost(s: *const Session) f32 {
        const det = s.detector orelse return 0;
        const done = det.done_row orelse return 0;
        return if (s.stream.items.len < done + s.c.stop_boost_delay) 0 else s.c.stop_boost;
    }

    /// One decode step of the current attempt: the reading position `pos` and centroid `m`.
    pub fn onStep(s: *Session, probe: *Probe, step: usize, pos: usize, m: f64) !void {
        const row = s.attempt_start + step;
        if (s.detector == null) {
            s.text_len = probe.text_len;
            s.detector = try Detector.init(s.a, probe.text_len, s.heads.frames_per_token, s.c, s.n_prefill);
        }
        const det = &s.detector.?;
        if (s.probation) |*p| {
            probe.center = try p.aci.update(m);
            if (det.frontier > p.failure.frontier) {
                p.aci.deinit();
                s.probation = null;
                probe.center = null;
            }
        }
        if (try det.observe(row, pos)) |f| {
            if (s.attempts < s.c.max_retries) {
                s.pending = f;
                probe.stop = true;
                return;
            }
            try s.ignore(f);
        }
        if (s.short) |*cut| {
            cut.observeReading(row - s.n_prefill, pos);
            if (cut.decided != null) {
                probe.stop = true;
                return;
            }
        }
        // the rows the delayed codebooks of the last kept frame need, with a margin
        if (det.stopRow()) |stop| if (row >= stop + C + 2) {
            s.text_done = true;
            probe.stop = true;
        };
    }

    fn ignore(s: *Session, f: Failure) !void {
        for (s.ignored.items) |k| if (k.kind == f.kind and k.bin == f.bin and k.onset_row == f.onset_row) return;
        try s.ignored.append(s.a, .{ .kind = f.kind, .bin = f.bin, .onset_row = f.onset_row });
        try s.failures.append(s.a, .{ .failure = f, .rollback_row = f.rollback_row, .clamped = false, .retry_seed = null, .ignored = true });
    }

    /// One row the current attempt emitted; `eos_frame`: its aligned EOS frame in its own rows, once EOS was emitted.
    pub fn append(s: *Session, row: [W]i32, eos_frame: ?usize) !void {
        try s.stream.append(s.a, row);
        s.generated_total += 1;
        if (eos_frame) |f| s.eos_row = s.attempt_start + f;
        if (s.short) |*cut| try cut.observeRow(row[0..C].*);
    }

    /// Stream rows below this are final: outside every rollback the detector's state can still report.
    pub fn commitRow(s: *const Session) usize {
        const n = s.stream.items.len;
        if (s.end != null) return n;
        if (s.hold or s.short != null) return s.n_prefill;
        var row = n;
        if (s.attempts < s.c.max_retries) {
            // a truncation: EOS at a stream row from the EOS-bearing row's - (codebooks - 1) on
            const eos = s.eos_row orelse n -| (C - 1);
            row = @max(eos -| s.c.rollback_margin_frames, s.n_prefill);
            if (s.detector) |*det| {
                if (det.commitRow(n)) |r| row = @min(row, r);
            } else row = s.n_prefill;
        }
        return @max(s.n_prefill, @min(row, n - 1));
    }

    /// The generated rows that became final since the last drain (all that remain when `final`).
    pub fn drain(s: *Session, final: bool) []const [W]i32 {
        const upto = if (final) s.stream.items.len else s.commitRow();
        const start = s.n_prefill + s.emitted;
        if (upto <= start) return &.{};
        s.emitted += upto - start;
        return s.stream.items[start..upto];
    }

    /// The current attempt finished: a rollback to regenerate from, or null when the result is final.
    pub fn finishAttempt(s: *Session) !?Continuation {
        var f = s.pending;
        // a short text's filler is there to be cut: a take that ends before it keeps that end
        if (f == null and s.eos_row != null and s.detector != null and s.short == null) {
            f = s.detector.?.onEos(s.eos_row.?);
            if (f != null and s.attempts >= s.c.max_retries) {
                try s.ignore(f.?);
                f = null;
            }
        }
        if (f == null and s.short != null) f = try s.shortRunOn();
        // rows already handed downstream are final: a rollback below them restarts at the first unsent row
        const restart: usize = if (f) |x| @max(x.rollback_row, s.n_prefill + s.emitted) else 0;
        if (f != null and s.max_tokens <= restart - s.n_prefill) {
            try s.ignore(f.?);
            f = null;
        }
        const failure = f orelse {
            if (s.short != null and s.short.?.enter != null) {
                s.end = s.shortEnd();
            } else if (s.text_done) {
                s.end = .{ .kind = .text_done, .row = s.detector.?.stopRow() };
            } else if (s.eos_row) |eos| {
                s.end = .{ .kind = .eos, .row = eos };
            } else s.end = .{ .kind = .max_tokens, .row = null };
            return null;
        };
        s.attempts += 1;
        const seed: ?u64 = if (s.seed) |x| x + 1000 * s.attempts else null;
        try s.failures.append(s.a, .{ .failure = failure, .rollback_row = restart, .clamped = restart != failure.rollback_row, .retry_seed = seed, .ignored = false });
        s.stream.shrinkRetainingCapacity(restart);
        try s.detector.?.truncate(restart);
        if (s.short) |*cut| cut.truncate(restart - s.n_prefill);
        if (s.probation) |*p| p.aci.deinit();
        s.probation = null;
        if (failure.kind == .short_run_on) {
            // a fresh take of a short text that ran into its filler
            s.short_retries += 1;
        } else s.probation = .{ .failure = failure, .aci = try AciCenter.init(s.a, s.detector.?.lastPosBefore(restart), s.text_len) };
        s.pending = null;
        s.text_done = false;
        s.eos_row = null;
        s.attempt_start = restart;
        return .{ .rows = restart, .history_rows = restart - s.n_prefill, .seed = seed, .max_tokens = s.max_tokens - (restart - s.n_prefill) };
    }

    /// A run-on to regenerate as a fresh take while retries are left; an exhausted one is recorded and ignored.
    fn shortRunOn(s: *Session) !?Failure {
        const cut = &s.short.?;
        const decided = cut.decided orelse return null;
        if (decided.kind != .run_on) return null;
        const det = &s.detector.?;
        const f: Failure = .{ .kind = .short_run_on, .detect_row = s.stream.items.len - 1, .onset_row = s.n_prefill + decided.frame, .rollback_row = s.n_prefill, .bin = det.frontier, .frontier = det.frontier };
        if (s.short_retries < cut.c.max_retries and s.attempts < s.c.max_retries) return f;
        try s.ignore(f);
        return null;
    }

    /// The end of a short-text take whose reading reached the filler: the cut, else where the reading reached it.
    fn shortEnd(s: *const Session) @TypeOf(@as(Session, undefined).end) {
        const cut = &s.short.?;
        if (cut.decided) |d| if (d.kind == .cut) return .{ .kind = .short_text, .row = s.n_prefill + d.frame };
        var frames = @min(cut.enter.?, (s.stream.items.len - s.n_prefill) -| (C - 1));
        if (s.eos_row) |eos| frames = @min(frames, eos - s.n_prefill);
        return .{ .kind = .short_text_forced, .row = s.n_prefill + frames };
    }

    /// The final result's EOS as an aligned frame in generated rows; null: no EOS, the budget ran out.
    pub fn eosFrame(s: *const Session) ?usize {
        return if (s.end.?.row) |row| row - s.n_prefill else null;
    }

    /// The final result's generated rows.
    pub fn finalRows(s: *const Session) usize {
        return (s.end.?.row orelse s.stream.items.len) - s.n_prefill;
    }
};

test "ACI centre: follows the centroid forward, a step at a time, and never back" {
    const t = std.testing;
    var aci = try AciCenter.init(t.allocator, 4, 10);
    defer aci.deinit();
    try t.expectEqual(@as(usize, 4), try aci.update(4.2));
    try t.expectEqual(@as(usize, 5), try aci.update(7.0));
    try t.expectEqual(@as(usize, 6), try aci.update(7.0));
    try t.expectEqual(@as(usize, 6), try aci.update(2.0));
    var far = try AciCenter.init(t.allocator, 99, 10);
    defer far.deinit();
    try t.expectEqual(@as(usize, 9), try far.update(3.0));
}

test "heads: a file's heads and pace; windows around a centre" {
    const t = std.testing;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const h = try Heads.parse(arena.allocator(), "{\"frames_per_token\": 3.5, \"heads\": [{\"layer\": 7, \"head\": 14, \"radius\": 6}, {\"layer\": 3, \"head\": 2, \"radius\": 4}]}");
    try t.expectEqual(@as(f64, 3.5), h.frames_per_token);
    var buf: [2]Window = undefined;
    try t.expectEqualSlices(Window, &.{ .{ .layer = 7, .head = 14, .lo = 5, .hi = 15 }, .{ .layer = 3, .head = 2, .lo = 7, .hi = 13 } }, h.windows(10, &buf));
    // layer 6 is a GDN layer: no attention head to probe
    try t.expectError(error.LaciHeadsOfAnotherModel, Heads.parse(arena.allocator(), "{\"frames_per_token\": 1, \"heads\": [{\"layer\": 6, \"head\": 1, \"radius\": 4}]}"));
    try t.expectError(error.BadLaciHeads, Heads.parse(arena.allocator(), "{\"frames_per_token\": 1, \"heads\": []}"));
}
