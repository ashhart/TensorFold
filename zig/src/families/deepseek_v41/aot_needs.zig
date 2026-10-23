//! `tf-dsv41 aot-needs SIGS OUT`: every Triton variant the served config can launch, from the engine's own plan.

const std = @import("std");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const block_prefill = @import("block_prefill.zig");
const dspark_emit = @import("dspark_emit.zig");
const check = @import("calls_check.zig");
const graphs = @import("graphs.zig");
const rowtab = @import("rowtab.zig");
const rowmode = @import("rowmode.zig");
const triton_call = @import("triton_call.zig");
const Config = @import("config.zig").Config;
const ced = @import("ced.zig");
const kvs = @import("cuda_kv.zig");

pub const Param = struct { name: []const u8, constexpr: bool, nospec: bool, noalign: bool };
pub const Fn = struct { function: []const u8, params: []Param };
pub const Sigs = std.json.ArrayHashMap([]Fn);
const SigsJson = struct { kernels: Sigs };

pub const Error = error{ NoKernel, AmbiguousKernel, UnknownParam, UnsupportedArg };

/// The variants seen so far: key (JSON) -> launches.
pub const Needs = struct {
    gpa: std.mem.Allocator,
    sigs: *const Sigs,
    keys: std.StringArrayHashMapUnmanaged(u64) = .empty,
    programs: u64 = 0,

    pub fn deinit(n: *Needs) void {
        for (n.keys.keys()) |k| n.gpa.free(k);
        n.keys.deinit(n.gpa);
    }

    /// Adds every Triton launch of a program; returns its fingerprint, the set of its variants (counts aside).
    pub fn program(n: *Needs, cs: []const calls.Call) !u64 {
        var arena = std.heap.ArenaAllocator.init(n.gpa);
        defer arena.deinit();
        var ids: std.ArrayList(u64) = .empty;
        for (cs) |*c| if (c.triton) {
            const key = try n.reduce(arena.allocator(), c);
            const got = try n.keys.getOrPut(n.gpa, key);
            if (!got.found_existing) {
                got.key_ptr.* = try n.gpa.dupe(u8, key);
                got.value_ptr.* = 0;
            }
            got.value_ptr.* += 1;
            try ids.append(arena.allocator(), got.index);
        };
        n.programs += 1;
        std.mem.sort(u64, ids.items, {}, std.sort.asc(u64));
        var h = std.hash.Wyhash.init(0);
        var last: ?u64 = null;
        for (ids.items) |id| {
            if (last != null and last.? == id) continue;
            h.update(std.mem.asBytes(&id));
            last = id;
        }
        return h.final();
    }

    /// The kernel function a launch names: the one whose parameters hold every argument the call passes.
    fn resolve(n: *const Needs, c: *const calls.Call) !*const Fn {
        const fns = n.sigs.map.get(c.name) orelse return error.NoKernel;
        var hit: ?*const Fn = null;
        for (fns) |*f| {
            const fits = for (c.args) |x| {
                if (paramOf(f, x.name) == null) break false;
            } else true;
            if (!fits) continue;
            if (hit != null) return error.AmbiguousKernel;
            hit = f;
        }
        return hit orelse error.NoKernel;
    }

    /// One launch's variant key: {"fn", "function", "consts": {...}, "runtime": {...}} (sorted by the call's order).
    pub fn reduce(n: *const Needs, a: std.mem.Allocator, c: *const calls.Call) ![]const u8 {
        const f = try n.resolve(c);
        var consts: std.Io.Writer.Allocating = .init(a);
        var rt: std.Io.Writer.Allocating = .init(a);
        var nc: usize = 0;
        var nr: usize = 0;
        for (c.args) |x| {
            const p = paramOf(f, x.name) orelse return error.UnknownParam;
            const cw = &consts.writer;
            const rw = &rt.writer;
            const as_const: ?Const = if (p.constexpr) try constOf(x.arg) else switch (x.arg) {
                .none => .null,
                .b => |v| .{ .b = v },
                // Triton folds a runtime int 1 into a constexpr unless the parameter is do_not_specialize
                .i => |v| if (v == 1 and !p.nospec) Const{ .i = 1 } else null,
                else => null,
            };
            if (as_const) |k| {
                try cw.print("{s}\"{s}\":", .{ if (nc > 0) "," else "", x.name });
                try k.write(cw);
                nc += 1;
                continue;
            }
            try rw.print("{s}\"{s}\":", .{ if (nr > 0) "," else "", x.name });
            nr += 1;
            switch (x.arg) {
                .t, .opaque_table => |t| try rw.print("{{\"type\":\"{s}\",\"div16\":{}}}", .{ triton_call.ptrType(t.dt), !p.noalign and @mod(t.offset, 16) == 0 }),
                .i => |v| {
                    const ty = if (v >= std.math.minInt(i32) and v <= std.math.maxInt(i32)) "i32" else "i64";
                    try rw.print("{{\"type\":\"{s}\",\"div16\":{}}}", .{ ty, !p.nospec and @mod(v, 16) == 0 });
                },
                .f => try rw.writeAll("{\"type\":\"fp32\"}"),
                else => return error.UnsupportedArg,
            }
        }
        return std.fmt.allocPrint(a, "{{\"fn\":\"{s}\",\"function\":\"{s}\",\"consts\":{{{s}}},\"runtime\":{{{s}}}}}", .{ c.name, f.function, consts.written(), rt.written() });
    }
};

const Const = union(enum) {
    null,
    b: bool,
    i: i64,
    f: f64,

    fn write(k: Const, w: *std.Io.Writer) !void {
        switch (k) {
            .null => try w.writeAll("null"),
            .b => |v| try w.print("{}", .{v}),
            .i => |v| try w.print("{d}", .{v}),
            // the exact fp64: triton_fill.unjson reads the bits
            .f => |v| try w.print("{{\"fp64_bits\":\"0x{x:0>16}\"}}", .{@as(u64, @bitCast(v))}),
        }
    }
};

fn constOf(x: calls.Arg) !Const {
    return switch (x) {
        .none => .null,
        .b => |v| .{ .b = v },
        .i => |v| .{ .i = v },
        .f => |v| .{ .f = v },
        else => error.UnsupportedArg,
    };
}

fn paramOf(f: *const Fn, name: []const u8) ?Param {
    for (f.params) |p| if (std.mem.eql(u8, p.name, name)) return p;
    return null;
}

// -- the served config's programs ---------------------------------------------------------------------------------

fn envInt(name: [:0]const u8, default: i64) !i64 {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(i64, std.mem.span(v), 10);
}

fn envOn(name: [:0]const u8, default: bool) bool {
    const v = std.c.getenv(name) orelse return default;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

pub const Served = struct {
    o: block.Options,
    slots: u32,
    cap: u32,
    drafts: bool,
    slot_drafts: bool,
    group: u32,
    graphs: graphs.Settings,
    /// TF_DSV41_PREFILL=replay: the decoder pass's programs too
    replay: bool = false,
    /// TF_DSV41_PIECE_RUNS / TF_DSV41_REPLAY_RUNS (several slots, replay): the multi-segment runs' programs too
    runs: bool = false,
    replay_runs: bool = false,
    /// TF_DSV41_IMAGES=native: image segments (Engram's keep); TF_DSV41_BIAS_VL set: their MoE in two calls
    images: bool = false,
    image_bias: bool = false,

    /// The engine's knobs as it reads them (PF_DENSE is CUDA, so the `_gemm` variants it replaces are listed too).
    pub fn fromEnv() !Served {
        const knobs = @import("serve_knobs.zig");
        var o: block.Options = .{ .world = @intCast(try envInt("TF_TP_WORLD", 2)), .rank = @intCast(try envInt("TF_TP_RANK", 0)), .r1 = knobs.r1(&knobs.env) };
        if (std.c.getenv("TF_DSV41_EXPERT_TOPP")) |v| o.expert_topp = try std.fmt.parseFloat(f64, std.mem.span(v));
        o.limit = try envInt("TF_DSV41_CONTEXT", o.limit);
        o.rope_rows = o.limit + 2048;
        if (try knobs.prefillRows(&knobs.env)) |r| o.prefill_rows = r;
        // TF_DSV41_PF_4K: 4,096-row segments (their Triton variants at those rows), x3gm v2 as apply has it
        o.pf4k = try knobs.pf4k(&knobs.env);
        o.gm_v2 = try @import("gm2pf.zig").mode(&knobs.env);
        // TF_DSV41_PF_OVERLAP: its pieces' projections (their own row counts) where the side stream is on
        o.pf_overlap = try knobs.pfOverlap(&knobs.env);
        o.pf_overlap_site = o.pf_overlap > 0 and try knobs.pfOverlapSite(&knobs.env); // the site pieces' `_finish_k`
        o.stream_rb = try knobs.streamRb(&knobs.env); // TF_DSV41_STREAM_RB: the twin's variants
        o.index_bound = try knobs.indexBound(&knobs.env); // TF_DSV41_INDEX_BOUND: `_scores_b` / `_dtopk_b` variants
        o.mhc_site_rows = try knobs.mhcSiteRows(&knobs.env); // the site pieces' `_finish_k` rows
        if (o.pf_overlap > 0) o.branches = (try @import("branches.zig").settings()).on;
        if (try knobs.indexBudget(&knobs.env)) |b| o.index_budget = b;
        // TF_DSV41_MHC_PFDEC: its norm is `_finish_k` without COEF at the windows it covers
        o.mhc_pf_rows = try knobs.mhcPfRows(&knobs.env);
        const drafts = envOn("TF_DSV41_DRAFTS", true);
        o.taps = drafts;
        o.pool = try poolFromEnv(o.limit, o.world);
        const slots: u32 = @intCast(try envInt("TF_DSV41_SLOTS", 1));
        if (slots > 1 and o.pool == null) return error.SlotsNeedPool;
        const rows_mod = @import("dspark_rows.zig");
        return .{
            .o = o,
            .slots = slots,
            .cap = @intCast(try envInt("TF_DSV41_ROWS_CAP", 16)),
            .drafts = drafts,
            .slot_drafts = drafts and slots > 1 and rows_mod.enabled(),
            .group = try rows_mod.groupFromEnv((Config{}).dspark_block),
            .graphs = try graphs.Settings.fromEnvOr(true),
            .replay = try knobs.prefillMode(&knobs.env) == .replay,
            .runs = slots > 1 and envOn("TF_DSV41_PIECE_RUNS", false) and try knobs.prefillMode(&knobs.env) == .replay,
            .replay_runs = slots > 1 and envOn("TF_DSV41_REPLAY_RUNS", false) and try knobs.prefillMode(&knobs.env) == .replay,
            .images = if (std.c.getenv("TF_DSV41_IMAGES")) |v| std.mem.eql(u8, std.mem.trim(u8, std.mem.span(v), " "), "native") else false,
            .image_bias = if (std.c.getenv("TF_DSV41_BIAS_VL")) |v| std.mem.trim(u8, std.mem.span(v), " ").len > 0 else false,
        };
    }
};

/// The served pool as model.zig opens it (kv_state.optionsFromEnv; null: no pool); TF_DSV41_NEEDS_POOL=1 / 0 forces it.
pub fn poolFromEnv(limit: i64, world: u32) !?block.Pool {
    const force: ?bool = if (std.c.getenv("TF_DSV41_NEEDS_POOL")) |v| std.mem.eql(u8, std.mem.span(v), "1") else null;
    const ko = try kvs.optionsFromEnv(@intCast(limit), world);
    if (!(force orelse (ko != null))) return null;
    const pages = @divExact(limit, block.Pool.page);
    const split = if (ko) |k| k.split else envOn("TF_DSV41_KV_SPLIT", false);
    return .{
        .comp_pages = if (split) @divExact(pages, 2) + 2 else pages + 1,
        .ik_pages = pages + 1,
        .pts = pages,
        .split = split,
        .early = split and earlyOn(),
        .union_cap = if (ko) |k| @intCast(k.union_cap) else 128 << 20,
    };
}

/// kv/split.zig's earlyOn (that file is the kv module's): TF_DSV41_KV_SPLIT_EARLY=1, blanks trimmed.
fn earlyOn() bool {
    const v = std.mem.span(std.c.getenv("TF_DSV41_KV_SPLIT_EARLY") orelse return false);
    return std.mem.eql(u8, std.mem.trim(u8, v, " \t"), "1");
}

const period: i64 = 256;

/// A program family over one position: `emit(end)` (the window or segment ending there).
const Family = struct {
    ctx: *const anyopaque,
    emit: *const fn (ctx: *const anyopaque, a: std.mem.Allocator, end: i64) anyerror![]const calls.Call,
};

/// Ends [lo, hi]: the first 512 each, then every 256th, plus 256 on each side where two samples' launches differ.
fn scan(n: *Needs, fam: Family, lo: i64, hi: i64) !void {
    if (hi < lo) return;
    const fp = struct {
        fn at(nd: *Needs, f: Family, e: i64) !u64 {
            var arena = std.heap.ArenaAllocator.init(nd.gpa);
            defer arena.deinit();
            return nd.program(try f.emit(f.ctx, arena.allocator(), e));
        }
    }.at;
    var e = lo;
    while (e <= @min(hi, lo + 2 * period - 1)) : (e += 1) _ = try fp(n, fam, e);
    if (e > hi) return;
    var prev = try fp(n, fam, e - 1);
    var at = e - 1 + period;
    while (at <= hi) : (at += period) {
        const cur = try fp(n, fam, at);
        if (cur != prev) {
            // a threshold crossed between the samples: every position around it
            var x = @max(lo, at - 2 * period + 1);
            while (x <= @min(hi, at + period)) : (x += 1) _ = try fp(n, fam, x);
        }
        prev = cur;
    }
    var x = @max(lo, hi - period + 1);
    while (x <= hi) : (x += 1) _ = try fp(n, fam, x);
}

const Ctx = struct {
    cfg: *const Config,
    w: *const block.Widths,
    o: block.Options,
    layers: []const u32,
    rows: i64,
};

fn decodeAt(ctx: *const anyopaque, a: std.mem.Allocator, end: i64) anyerror![]const calls.Call {
    const c: *const Ctx = @ptrCast(@alignCast(ctx));
    return block.emit(a, c.cfg, c.w, c.o, c.layers, c.rows, end - c.rows, true);
}

fn decoderAt(ctx: *const anyopaque, a: std.mem.Allocator, end: i64) anyerror![]const calls.Call {
    const c: *const Ctx = @ptrCast(@alignCast(ctx));
    return block_prefill.emitReplay(a, c.cfg, c.w, c.o, c.layers, c.rows, end - c.rows, .decoder);
}

fn prefillAt(ctx: *const anyopaque, a: std.mem.Allocator, end: i64) anyerror![]const calls.Call {
    const c: *const Ctx = @ptrCast(@alignCast(ctx));
    return block_prefill.emitPrefill(a, c.cfg, c.w, c.o, c.layers, c.rows, end - c.rows, true);
}

/// One multi-segment run's program (block_prefill.emitMulti); a run the options refuse adds nothing.
fn run(n: *Needs, cfg: *const Config, w: *const block.Widths, o: block.Options, layers: []const u32, part: block_prefill.Part, spans: []const block_prefill.Span) !void {
    var pa = std.heap.ArenaAllocator.init(n.gpa);
    defer pa.deinit();
    const cs = block_prefill.emitMulti(pa.allocator(), cfg, w, o, layers, spans, part) catch |e| switch (e) {
        error.Unsupported => return,
        else => return e,
    };
    try aligned(cs);
    _ = try n.program(cs);
}

/// A run's every Triton view 16-byte aligned from its role's base (a misaligned slice is a variant no fill holds).
fn aligned(cs: []const calls.Call) !void {
    for (cs) |c| {
        if (!c.triton) continue;
        for (c.args) |x| switch (x.arg) {
            .t, .opaque_table => |t| if (@mod(t.offset, 16) != 0) {
                std.log.err("aot-needs: a multi-segment run's {s} {s} at offset {d} is not 16-byte aligned", .{ c.name, x.name, t.offset });
                return error.MisalignedRun;
            },
            else => {},
        };
    }
}

/// Decoder replay runs at a position: a 127-row tail ending at `end` beside a 33-row tail from 0 (`first`: 127 first).
const MultiCtx = struct { c: Ctx, first: bool };

fn tailsAt(ctx: *const anyopaque, a: std.mem.Allocator, end: i64) anyerror![]const calls.Call {
    const m: *const MultiCtx = @ptrCast(@alignCast(ctx));
    const long: block_prefill.Span = .{ .slot = 0, .start = end - m.c.rows, .n = m.c.rows };
    const short: block_prefill.Span = .{ .slot = 1, .start = 0, .n = 33 };
    const spans = if (m.first) [_]block_prefill.Span{ long, short } else [_]block_prefill.Span{ short, long };
    return block_prefill.emitMulti(a, m.c.cfg, m.c.w, m.c.o, m.c.layers, &spans, .decoder) catch |e| switch (e) {
        error.Unsupported => &.{},
        else => e,
    };
}

/// The context buckets' ends up to the limit (graphs.bucketEnd).
fn bucketEnds(a: std.mem.Allocator, s: graphs.Settings, limit: u64) ![]u64 {
    var out: std.ArrayList(u64) = .empty;
    var b: u32 = 0;
    while (true) : (b += 1) {
        const e = graphs.bucketEnd(b, s.bucket, limit, s.grow);
        try out.append(a, e);
        if (e + 1 >= limit) break;
    }
    return out.items;
}

pub const Report = struct { programs: u64, variants: usize };

/// Every program of the served config into `n` (progress on `log`).
pub fn enumerate(gpa: std.mem.Allocator, n: *Needs, cfg: *const Config, w: *const block.Widths, sv: Served, log: *std.Io.Writer) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var backbone: [64]u32 = undefined;
    for (backbone[0..cfg.layers], 0..) |*l, i| l.* = @intCast(i);
    const layers = backbone[0..cfg.layers];
    const limit = sv.o.limit;
    const ends = try bucketEnds(a, sv.graphs, @intCast(limit));
    const max_rows: i64 = @max(16, cfg.dspark_block + 1);
    // one slot: eager at every end, graphed at every bucket's end
    var r: i64 = 1;
    while (r <= max_rows) : (r += 1) {
        const c: Ctx = .{ .cfg = cfg, .w = w, .o = sv.o, .layers = layers, .rows = r };
        try scan(n, .{ .ctx = &c, .emit = decodeAt }, r, limit);
        for (ends) |e| if (@as(i64, @intCast(e)) + 1 >= r) {
            var pa = std.heap.ArenaAllocator.init(gpa);
            defer pa.deinit();
            _ = try n.program(try block.emit(pa.allocator(), cfg, w, sv.o, layers, r, @as(i64, @intCast(e)) + 1 - r, true));
        };
        try log.print("aot-needs: decode {d} rows, {d} programs, {d} variants\n", .{ r, n.programs, n.keys.count() });
        try log.flush();
    }
    // row mode: every bucket up to the cap at every context bucket
    if (sv.slots > 1) {
        var o = sv.o;
        o.rows = true;
        for (rowtab.buckets) |R| {
            if (R > sv.cap) break;
            for (ends) |e| {
                const start = @as(i64, @intCast(e)) + 1 - R;
                if (start < 0) continue;
                var pa = std.heap.ArenaAllocator.init(gpa);
                defer pa.deinit();
                const one = try block.emit(pa.allocator(), cfg, w, o, layers, R, start, true);
                _ = try n.program(try rowmode.transform(pa.allocator(), one, R, .{ .slots = sv.slots, .rmax = sv.cap, .pts = sv.o.pool.?.pts }));
            }
        }
        try log.print("aot-needs: row mode, {d} programs, {d} variants\n", .{ n.programs, n.keys.count() });
        try log.flush();
    }
    // prefill: the full segment's regimes, then every row count at each regime's first start
    const P = sv.o.prefill_rows;
    {
        const full: Ctx = .{ .cfg = cfg, .w = w, .o = sv.o, .layers = layers, .rows = P };
        // a start whose full segment launches a new variant set opens a regime (each set swept once)
        var starts: std.ArrayList(i64) = .empty;
        var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
        var s: i64 = 0;
        while (s + P <= limit) : (s += P) {
            var pa = std.heap.ArenaAllocator.init(gpa);
            defer pa.deinit();
            const f = try n.program(try prefillAt(&full, pa.allocator(), s + P));
            if (!(try seen.getOrPut(a, f)).found_existing) try starts.append(a, s);
        }
        for (starts.items) |s0| {
            var rows: i64 = 1;
            while (rows <= P and s0 + rows <= limit) : (rows += 1) {
                var pa = std.heap.ArenaAllocator.init(gpa);
                defer pa.deinit();
                _ = try n.program(try block_prefill.emitPrefill(pa.allocator(), cfg, w, sv.o, layers, rows, s0, true));
            }
            try log.print("aot-needs: prefill from {d}, {d} programs, {d} variants\n", .{ s0, n.programs, n.keys.count() });
            try log.flush();
        }
        if (sv.runs) {
            const enc = layers[0 .. try ced.decoderStart(cfg) + 1];
            for (starts.items) |s0| {
                var k: i64 = 17;
                while (k <= 64) : (k += 1) {
                    try run(n, cfg, w, sv.o, enc, .encoder, &.{ .{ .slot = 0, .start = s0, .n = 17 }, .{ .slot = 1, .start = s0, .n = k } });
                    try run(n, cfg, w, sv.o, enc, .encoder, &.{ .{ .slot = 0, .start = s0, .n = k }, .{ .slot = 1, .start = s0, .n = 17 } });
                }
                k = 17;
                while (k <= 48) : (k += 1) {
                    try run(n, cfg, w, sv.o, enc, .encoder, &.{ .{ .slot = 0, .start = s0, .n = k }, .{ .slot = 1, .start = s0, .n = 100 } });
                    try run(n, cfg, w, sv.o, enc, .encoder, &.{ .{ .slot = 0, .start = s0, .n = 100 }, .{ .slot = 1, .start = s0, .n = k } });
                }
                // a staged segment (past 129 rows: the staging ring) beside short ones, at every residue mod 4
                k = 17;
                while (k <= 20) : (k += 1) {
                    try run(n, cfg, w, sv.o, enc, .encoder, &.{ .{ .slot = 0, .start = s0, .n = k }, .{ .slot = 1, .start = s0, .n = 1500 + k } });
                    try run(n, cfg, w, sv.o, enc, .encoder, &.{ .{ .slot = 0, .start = s0, .n = 1500 + k }, .{ .slot = 1, .start = s0, .n = k } });
                }
            }
            try log.print("aot-needs: piece runs, {d} programs, {d} variants\n", .{ n.programs, n.keys.count() });
            try log.flush();
        }
        if (sv.images) try images(gpa, n, cfg, w, sv, layers, starts.items, log);
    }
    // CED replay's decoder pass: n > 128 replays 127 rows ending at n - 1 (the scan), shorter ones n - 1 rows from 0
    if (sv.replay) {
        const dec = layers[try ced.decoderStart(cfg)..];
        const c: Ctx = .{ .cfg = cfg, .w = w, .o = sv.o, .layers = dec, .rows = ced.keep };
        try scan(n, .{ .ctx = &c, .emit = decoderAt }, ced.keep, limit - 1);
        var m: i64 = 1;
        while (m < ced.keep) : (m += 1) {
            var pa = std.heap.ArenaAllocator.init(gpa);
            defer pa.deinit();
            _ = try n.program(try block_prefill.emitReplay(pa.allocator(), cfg, w, sv.o, dec, m, 0, .decoder));
        }
        try log.print("aot-needs: CED decoder, {d} programs, {d} variants\n", .{ n.programs, n.keys.count() });
        try log.flush();
        if (sv.replay_runs) {
            var k: i64 = 33;
            while (k <= ced.keep) : (k += 1) {
                try run(n, cfg, w, sv.o, dec, .decoder, &.{ .{ .slot = 0, .start = 0, .n = 33 }, .{ .slot = 1, .start = 0, .n = k } });
                try run(n, cfg, w, sv.o, dec, .decoder, &.{ .{ .slot = 0, .start = 0, .n = k }, .{ .slot = 1, .start = 0, .n = 33 } });
            }
            k = 33;
            while (k <= 48) : (k += 1) {
                try run(n, cfg, w, sv.o, dec, .decoder, &.{ .{ .slot = 0, .start = 0, .n = k }, .{ .slot = 1, .start = 0, .n = ced.keep } });
                try run(n, cfg, w, sv.o, dec, .decoder, &.{ .{ .slot = 0, .start = 0, .n = ced.keep }, .{ .slot = 1, .start = 0, .n = k } });
            }
            for ([_]bool{ false, true }) |first| {
                const mc: MultiCtx = .{ .c = c, .first = first };
                try scan(n, .{ .ctx = &mc, .emit = tailsAt }, ced.keep, limit - 1);
            }
            try log.print("aot-needs: replay runs, {d} programs, {d} variants\n", .{ n.programs, n.keys.count() });
            try log.flush();
        }
    }
    // DSpark
    if (sv.drafts) {
        var pa = std.heap.ArenaAllocator.init(gpa);
        defer pa.deinit();
        const pb = pa.allocator();
        _ = try n.program(try dspark_emit.emitPass(pb, cfg, w, sv.o, cfg.dspark_block));
        // ingests: a window's rows, and a prompt's prefilled tail in one call (target.handTaps: up to the window)
        var rows: i64 = 1;
        while (rows <= @max(max_rows, cfg.window)) : (rows += 1) {
            _ = try n.program(try dspark_emit.emitIngest(pb, cfg, w, sv.o, rows, 0));
            _ = try n.program(try dspark_emit.emitIngest(pb, cfg, w, sv.o, rows, 1));
        }
        if (sv.slot_drafts) {
            var k: i64 = 1;
            while (k <= sv.group) : (k += 1) _ = try n.program(try dspark_emit.emitPassSlots(pb, cfg, w, sv.o, cfg.dspark_block, k, sv.slots));
            // TF_DSV41_DRAFT_CHAIN's `_chain`, with and without the confidence head; WORLD x k candidates a row
            const world: i64 = sv.o.world;
            const c = world * @min(64, @divTrunc(128, world));
            for ([_]bool{ true, false }) |conf| _ = try n.program(try dspark_emit.emitChain(pb, cfg, w, sv.o, cfg.dspark_block, sv.slots, c, conf));
            try log.print("aot-needs: DSpark chain (TF_DSV41_DRAFT_CHAIN), N {d}, KP {d}, {d} candidates, with and without the confidence head\n", .{ cfg.dspark_block, dspark_emit.chain_kp, c });
        } else if (try @import("dspark_dev.zig").chainFromEnv()) {
            // the served config would boot without its chain (dspark_slots refuses it): a fill env that drifted
            try log.print("aot-needs: TF_DSV41_DRAFT_CHAIN=1 without slot drafts (TF_DSV41_SLOTS > 1, TF_DSV41_SLOT_DRAFTS=1, row mode): no _chain variant listed\n", .{});
            try log.flush();
            return error.ChainWithoutSlotDrafts;
        }
        try log.print("aot-needs: DSpark, {d} programs, {d} variants\n", .{ n.programs, n.keys.count() });
        try log.flush();
    }
}

/// TF_DSV41_IMAGES=native: image segments at every row count as the plan sizes them, CED's decoder, sampled splits.
fn images(gpa: std.mem.Allocator, n: *Needs, cfg: *const Config, w: *const block.Widths, sv: Served, layers: []const u32, starts: []const i64, log: *std.Io.Writer) !void {
    var o = sv.o;
    o.image_keep = true;
    if (sv.image_bias) o.image_rows = -1;
    const P = sv.o.prefill_rows;
    for (starts) |s0| {
        var rows: i64 = 1;
        while (rows <= P and s0 + rows <= sv.o.limit) : (rows += 1) {
            var pa = std.heap.ArenaAllocator.init(gpa);
            defer pa.deinit();
            _ = try n.program(try block_prefill.emitPrefill(pa.allocator(), cfg, w, o, layers, rows, s0, true));
        }
    }
    if (sv.replay) {
        const dec = layers[try ced.decoderStart(cfg)..];
        var m: i64 = 1;
        while (m <= ced.keep) : (m += 1) {
            var pa = std.heap.ArenaAllocator.init(gpa);
            defer pa.deinit();
            _ = try n.program(try block_prefill.emitReplay(pa.allocator(), cfg, w, o, dec, m, 0, .decoder));
        }
        const c: Ctx = .{ .cfg = cfg, .w = w, .o = o, .layers = dec, .rows = ced.keep };
        try scan(n, .{ .ctx = &c, .emit = decoderAt }, ced.keep, sv.o.limit - 1);
    }
    const listed = n.keys.count();
    try log.print("aot-needs: images (bias {}), {d} programs, {d} variants\n", .{ sv.image_bias, n.programs, listed });
    if (sv.image_bias) {
        for ([_]i64{ 17, 64, 1000, P }) |R| for ([_]i64{ 1, @divTrunc(R, 2), R - 1 }) |ni| {
            var pa = std.heap.ArenaAllocator.init(gpa);
            defer pa.deinit();
            var x = o;
            x.image_rows = ni;
            _ = try n.program(try block_prefill.emitPrefill(pa.allocator(), cfg, w, x, layers, R, starts[0], true));
        };
        try log.print("aot-needs: images, sampled image / text splits: {d} variants past the plan's\n", .{n.keys.count() - listed});
    }
    try log.flush();
}

/// `tf-dsv41 aot-needs SIGS OUT`.
pub fn main(gpa: std.mem.Allocator, io: std.Io, sigs_path: []const u8, out_path: []const u8, log: *std.Io.Writer) !void {
    const cwd = std.Io.Dir.cwd();
    const text = try cwd.readFileAlloc(io, sigs_path, gpa, .limited(1 << 26));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(SigsJson, gpa, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
    defer parsed.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    const cfg: Config = .{};
    // a prefill segment's x3gm instances (per-layer K2 sets): CUDA launches, no Triton variant depends on them
    for (0..cfg.layers) |L| try w.gm.put(a, @intCast(L), .{ (1 << 4) | (1 << 6) | (1 << 8), (1 << 4) | (1 << 10) });
    // DSpark's ingest main_proj (a DSpark block's width; its EXL3 linear is CUDA, as slot_drafts_test.zig sets it)
    try w.dense.put(a, "dspark.main_proj", try w.k2("L40.attn.wkv"));
    const sv = try Served.fromEnv();
    try log.print("aot-needs: limit {d}, {d} slot(s), rows cap {d}, prefill rows {d}, index budget {d} MiB, R1 {}, drafts {}, slot drafts {} (group {d}), world {d}, pool {s}\n", .{ sv.o.limit, sv.slots, sv.cap, sv.o.prefill_rows, sv.o.index_budget >> 20, sv.o.r1, sv.drafts, sv.slot_drafts, sv.group, sv.o.world, if (sv.o.pool) |p| (if (p.split) "split" else "on") else "none (contiguous slot)" });
    var n: Needs = .{ .gpa = gpa, .sigs = &parsed.value.kernels };
    defer n.deinit();
    try enumerate(gpa, &n, &cfg, &w, sv, log);
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    const ow = &out.writer;
    try ow.print("{{\"generator\":\"tf-dsv41 aot-needs\",\"limit\":{d},\"slots\":{d},\"programs\":{d},\"kernels\":[\n", .{ sv.o.limit, sv.slots, n.programs });
    for (n.keys.keys(), n.keys.values(), 0..) |k, launches, i| {
        try ow.print("{s}{s},\"launches\":{d}}}", .{ if (i > 0) ",\n" else "", k[0 .. k.len - 1], launches });
    }
    try ow.writeAll("\n]}\n");
    try cwd.writeFile(io, .{ .sub_path = out_path, .data = out.written() });
    try log.print("aot-needs: {d} programs, {d} variants -> {s}\n", .{ n.programs, n.keys.count(), out_path });
    try log.flush();
}

test {
    _ = @import("aot_needs_test.zig");
}
