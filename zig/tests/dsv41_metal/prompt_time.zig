//! Prompt timing on one real release layer (TF_DSV41_REAL): ms a stage per chunk, and a hash of each stage output.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const fixture = @import("fixture.zig");
const ref = @import("ref.zig");

const testing = std.testing;
const pl = pkg.prompt_layer;
const ps = pkg.prompt_scratch;
const Stage = pl.Stage;
const all_stages = [_]Stage{ .attn_in, .kv, .q, .ix_proj, .ix_score, .ix_select, .core, .out, .ffn_in, .route, .xe_plan, .xe_up, .xe_down, .experts, .shared, .none };

/// The stages up to TF_DSV41_PT_LAST (a stage name; default all).
fn upTo() []const Stage {
    const last = env("TF_DSV41_PT_LAST") orelse return &all_stages;
    for (all_stages, 0..) |st, i| if (std.mem.eql(u8, @tagName(st), last)) return all_stages[0 .. i + 1];
    return &all_stages;
}

fn env(name: [*:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name) orelse return null;
    return std.mem.span(v);
}

fn envInt(name: [*:0]const u8, def: u32) u32 {
    return if (env(name)) |v| std.fmt.parseInt(u32, v, 10) catch def else def;
}

fn now() u64 {
    var t: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t);
    return @as(u64, @intCast(t.sec)) * 1_000_000_000 + @as(u64, @intCast(t.nsec));
}

const Rig = struct {
    a: std.mem.Allocator,
    c: host.config.Config,
    device: mtl.Device,
    queue: mtl.Queue,
    p: pkg.pipelines.Pipelines,
    w: pkg.weights.Layer,
    tables: pkg.rope.Tables,
    slot: pkg.kv.Slot,
    s: ps.Scratch,
    io: [4]mtl.Buffer,
    /// the layer class timed (TF_DSV41_PT_CLASS) and the config a reuse layer's selection comes from
    class: Class = .source,
    src_c: host.config.Config = undefined,

    fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, layer: u32, rows: u32, limit: u32, class: Class) !*Rig {
        const r = try a.create(Rig);
        r.a = a;
        r.c = try host.config.Config.read(a, io, dir);
        r.class = class;
        class.apply(&r.c, layer);
        r.device = try mtl.Device.init();
        r.queue = try r.device.queue();
        r.p = try pkg.pipelines.Pipelines.load(a, r.device, &r.c);
        var pack = try host.pack.Pack.open(a, io, dir);
        defer pack.deinit();
        r.w = try pkg.weights.loadFrom(a, io, r.device, &pack, &r.c, layer, null);
        r.tables = try pkg.rope.Tables.init(r.device, &r.c, limit);
        r.slot = try pkg.kv.Slot.init(a, r.device, &r.c, limit);
        r.s = try ps.Scratch.init(r.device, &r.c, rows, envInt("TF_DSV41_PT_IXROWS", 64), limit);
        const sizes = [4]usize{ rows * 8 * r.c.hidden, rows * 8 * r.c.hidden, rows * 16, rows * 16 };
        for (&r.io, sizes) |*b, n| b.* = try r.device.buffer(n, mtl.ResourceOptions.shared);
        // after the slot's regions: a window layer keeps its pool unread, a reuse layer its own source's
        r.src_c = r.c;
        if (class == .window) r.c.compress_ratios.buf[layer] = 0;
        if (class == .reuse or class == .r1reuse) r.c.index_sources = dropped(r.c.index_sources, layer);
        return r;
    }

    fn px(r: *const Rig, a: u32, b: u32, stop: Stage) pl.Px {
        return .{ .c = &r.c, .p = &r.p, .s = &r.s, .slot = &r.slot, .tables = &r.tables, .a = a, .b = b, .stop = stop };
    }
};

/// The layer classes a release prompt runs, all on one layer's weights (a timing estimate, not a layer's own run).
const Class = enum {
    /// as the checkpoint has it (layer 2: ratio 2, KV, index and candidate-free source)
    source,
    /// ratio 2 reading an earlier source's pool and selection: no compressor, no indexer
    reuse,
    /// window only (ratio 0)
    window,
    /// ratio 1, KV, index and candidate source (layer 20; with its compressor = the encoder's publish)
    r1,
    /// ratio 1 index source masked by candidates (layers 24..36), no compressor
    r1masked,
    /// ratio 1 reading an earlier source's pool and selection
    r1reuse,

    fn of(name: ?[]const u8) Class {
        const n = name orelse return .source;
        return std.meta.stringToEnum(Class, n) orelse .source;
    }

    fn apply(class: Class, c: *host.config.Config, layer: u32) void {
        switch (class) {
            .source, .reuse => {},
            .window => {},
            .r1, .r1reuse => {
                c.compress_ratios.buf[layer] = 1;
                c.candidate_source = layer;
            },
            .r1masked => {
                c.compress_ratios.buf[layer] = 1;
                c.candidate_source = 0;
            },
        }
    }

    fn compresses(class: Class) bool {
        return class == .source or class == .r1;
    }
};

fn dropped(ids: host.config.Ids, v: u32) host.config.Ids {
    var out: host.config.Ids = .{};
    for (ids.items()) |x| if (x != v) {
        out.buf[out.len] = x;
        out.len += 1;
    };
    return out;
}

fn hashOf(h: *std.hash.Wyhash, bytes: []const u8) void {
    h.update(bytes);
}

/// A stage's characteristic output, hashed: what a bit-equal change must leave unchanged.
fn stageHash(r: *const Rig, st: Stage, rows: u32) u64 {
    var h = std.hash.Wyhash.init(0);
    const s = &r.s;
    const c = &r.c;
    const H: usize = c.hidden;
    const R: usize = rows;
    const QH: usize = @as(usize, c.heads) * c.head_dim;
    const IX: usize = @as(usize, c.index_heads) * c.index_dim;
    switch (st) {
        .attn_in => hashOf(&h, s.bytes(s.hn, R * H * 2)),
        .kv => hashOf(&h, s.bytes(s.kvr, R * c.head_dim * 2)),
        .q => hashOf(&h, s.bytes(s.qrope, R * QH * 2)),
        .ix_proj => {
            hashOf(&h, s.bytes(s.ixqc, R * IX / 2));
            hashOf(&h, s.bytes(s.ixqs, R * IX / 32));
            hashOf(&h, s.bytes(s.ixw, R * c.index_heads * 2));
        },
        .ix_score => hashOf(&h, s.bytes(s.scores, @as(usize, s.ix_rows) * s.keys * 4)),
        .ix_select => hashOf(&h, s.bytes(s.sel, R * pkg.pipelines.attn.max_selected * 4)),
        .core => hashOf(&h, s.bytes(s.o, R * QH * 2)),
        .out => hashOf(&h, s.bytes(s.aout, R * H * 2)),
        .ffn_in => hashOf(&h, s.bytes(s.hn2, R * H * 2)),
        .route => {
            hashOf(&h, s.bytes(s.picks, R * c.topk * 4));
            hashOf(&h, s.bytes(s.wts, R * c.topk * 4));
        },
        .xe_up => hashOf(&h, s.bytes(s.act, R * c.topk * c.expert_width * 2)),
        .xe_down => hashOf(&h, s.bytes(s.eout, R * c.topk * H * 2)),
        .experts => hashOf(&h, s.bytes(s.yr, R * H * 4)),
        .shared => hashOf(&h, s.bytes(s.mout, R * H * 2)),
        .none => {
            hashOf(&h, r.io[1].contents()[0 .. R * 8 * H]);
            hashOf(&h, r.io[3].contents()[0 .. R * 16]);
        },
        else => {},
    }
    return h.final();
}

fn bfRow(out: []u16, rng: std.Random, scale: f32) void {
    for (out) |*v| v.* = ref.toBf((rng.float(f32) * 2 - 1) * scale);
}

/// One layer's random past before p0, as fixture.fillPast draws it for every layer (only this layer's here).
fn fillPast(r: *const Rig, layer: u32, p0: u32, seed: u64) void {
    const c = &r.c;
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const D = c.head_dim;
    const reg = r.slot.regions[layer];
    const win = reg.window.slice(u16, 2 * @as(usize, pkg.kv.windowSlots(c)) * D);
    for (p0 -| pkg.kv.windowSlots(c)..p0) |p| {
        const rows = pkg.kv.windowRows(c, @intCast(p));
        bfRow(win[rows[0] * D ..][0..D], rng, 1.5);
        @memcpy(win[rows[1] * D ..][0..D], win[rows[0] * D ..][0..D]);
    }
    if (c.compressRatio(layer) == 0) return;
    const n = p0 / c.compressRatio(layer);
    bfRow(reg.pool.?.slice(u16, n * D), rng, 1.5);
    const keys = reg.keys.?.contents()[0 .. n * pkg.kv.index_key_bytes];
    rng.bytes(keys);
    for (0..n) |k| for (64..68) |j| {
        keys[k * 68 + j] = 124 + rng.uintLessThan(u8, 5);
    };
    if (reg.carry) |cb| for (cb.slice(f32, 4 * D)) |*v| {
        v.* = rng.float(f32) * 2 - 1;
    };
}

fn fillInput(r: *const Rig, rows: usize, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    bfRow(r.io[0].slice(u16, rows * 4 * r.c.hidden), rng, 2);
    for (r.io[2].slice(f32, rows * 4)) |*v| v.* = 0.2 + 0.8 * rng.float(f32);
}

/// Encodes the layer up to `stop` and runs it; the commit-to-completion time in ns.
fn runTo(r: *Rig, a: u32, rows: u32, stop: Stage) !u64 {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = r.queue.commandBuffer();
    const e = cb.compute(.serial);
    const x = r.px(a, a + rows, stop);
    const io: pl.Io = .{ .x = .{ .buf = r.io[0] }, .pre = .{ .buf = r.io[2] }, .x_out = .{ .buf = r.io[1] }, .pre_out = .{ .buf = r.io[3] } };
    pl.encode(&x, e, &r.w, io, .{ .k = a, .o = a, .compress = r.class.compresses() }) catch |err| if (err != error.Stopped) return err;
    e.end();
    const t0 = now();
    cb.commit();
    cb.wait();
    const dt = now() - t0;
    if (cb.failure()) |msg| {
        std.debug.print("command buffer failed: {s}\n", .{msg});
        return error.CommandBufferFailed;
    }
    return dt;
}

fn chunk(r: *Rig, layer: u32, a: u32, rows: u32, reps: u32) !void {
    const D = r.c.head_dim;
    const carry = r.slot.regions[layer].carry;
    const cbytes: usize = if (carry != null) 4 * D * 4 else 0;
    fillPast(r, layer, a, 7 + a);
    fillInput(r, rows, 11);
    try r.s.writeRows(&r.c, a, rows, rows);
    const saved = try r.a.dupe(u8, if (carry) |cb| cb.contents()[0..cbytes] else &.{});
    defer r.a.free(saved);
    if (r.class == .reuse or r.class == .r1reuse) {
        // a source's selection for these rows, as the reuse layer reads it
        const keep = r.c;
        r.c = r.src_c;
        r.class = if (keep.compressRatio(layer) == 1) .r1 else .source;
        _ = try runTo(r, a, rows, .ix_select);
        r.class = Class.of(env("TF_DSV41_PT_CLASS"));
        r.c = keep;
        if (carry) |cb| @memcpy(cb.contents()[0..cbytes], saved);
    }
    const stages = upTo();
    var best: [all_stages.len]u64 = @splat(std.math.maxInt(u64));
    var hashes: [all_stages.len]u64 = undefined;
    for (0..reps) |rep| {
        for (stages, 0..) |st, i| {
            if (carry) |cb| @memcpy(cb.contents()[0..cbytes], saved);
            const t = try runTo(r, a, rows, st);
            best[i] = @min(best[i], t);
            if (rep == 0) hashes[i] = stageHash(r, st, rows);
        }
    }
    std.debug.print("chunk {d} rows at position {d} (layer {d}, {s}), ms: stage, cumulative, hash\n", .{ rows, a, layer, @tagName(r.class) });
    var prev: u64 = 0;
    for (stages, 0..) |st, i| {
        const cum = best[i];
        const d: f64 = @as(f64, @floatFromInt(cum -| prev)) / 1e6;
        prev = @max(prev, cum);
        std.debug.print("  {s:<10} {d:>8.2} {d:>8.2}  {x:0>16}\n", .{ if (st == .none) "post" else @tagName(st), d, @as(f64, @floatFromInt(cum)) / 1e6, hashes[i] });
    }
    const total: f64 = @as(f64, @floatFromInt(best[stages.len - 1])) / 1e6;
    std.debug.print("  total {d:.2} ms = {d:.0} rows/s\n", .{ total, @as(f64, @floatFromInt(rows)) / total * 1000 });
}

test "dsv41 metal prompt timing: one real layer, ms a stage, stage hashes" {
    const dir = env("TF_DSV41_REAL") orelse return error.SkipZigTest;
    const layer = envInt("TF_DSV41_REAL_LAYER", 2);
    const reps = envInt("TF_DSV41_PT_REPS", 3);
    var rows_buf: [8]u32 = undefined;
    var pos_buf: [16]u32 = undefined;
    // TF_DSV41_PT_ROWSL, TF_DSV41_PT_POSL: comma lists in place of ROWS0/1 and POS0/1
    const rows_list = list(env("TF_DSV41_PT_ROWSL"), &rows_buf) orelse &[_]u32{ envInt("TF_DSV41_PT_ROWS0", 2048), envInt("TF_DSV41_PT_ROWS1", 8192) };
    const pos_list = list(env("TF_DSV41_PT_POSL"), &pos_buf) orelse &[_]u32{ envInt("TF_DSV41_PT_POS0", 0), envInt("TF_DSV41_PT_POS1", 32768) };
    const max_rows = std.mem.max(u32, rows_list);
    const limit = std.mem.max(u32, pos_list) + max_rows + 128;
    const class = Class.of(env("TF_DSV41_PT_CLASS"));
    const r = try Rig.open(std.heap.page_allocator, testing.io, dir, layer, max_rows, limit, class);
    if (env("TF_DSV41_PT_ENGRAM") != null) {
        var bufs = try pkg.prompt.Bufs.init(r.device, &r.c, max_rows, 1, pkg.pipelines.attn.window);
        defer bufs.deinit();
        std.debug.print("prompt buffers for {d}-row chunks: scratch {d} MiB, chunk buffers {d} MiB\n", .{ max_rows, r.s.buf.length() >> 20, bufs.buf.length() >> 20 });
    }
    if (env("TF_DSV41_PT_ENGRAM") != null) for (rows_list) |rows| try engramWkv(r, rows, reps);
    for (rows_list) |rows| for (pos_list) |pos| if (rows != 0) try chunk(r, layer, pos, rows, reps);
}

/// Engram wkv of `rows` rows (release shape, random FP8 weights): the decode row kernel's launches vs one MMA.
fn engramWkv(r: *Rig, rows: u32, reps: u32) !void {
    const c = &r.c;
    const call = pkg.forward.engramCall(c, rows);
    const K: usize = @as(usize, call.cols) * pkg.pipelines.engram.head;
    const N: usize = call.n;
    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();
    const sh = mtl.ResourceOptions.shared;
    const x = try r.device.buffer(rows * K * 2, sh);
    const w = try r.device.buffer(N * K, sh);
    const sc = try r.device.buffer(N / 32 * (K / 32), sh);
    const ys = [2]mtl.Buffer{ try r.device.buffer(rows * N * 2, sh), try r.device.buffer(rows * N * 2, sh) };
    bfRow(x.slice(u16, rows * K), rng, 2);
    for (w.contents()[0 .. N * K]) |*b| b.* = rng.int(u8) & 0xF7;
    for (sc.contents()[0 .. N / 32 * (K / 32)]) |*b| b.* = 117 + rng.uintLessThan(u8, 8);
    var best: [2]u64 = @splat(std.math.maxInt(u64));
    for (0..reps) |_| for (0..2) |v| {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = r.queue.commandBuffer();
        const e = cb.compute(.serial);
        const R = pkg.pipelines.engram.Ref;
        if (v == 0) {
            try pkg.pipelines.engram.encodeWkv(&r.p.engram, e, .{ .buf = x }, .{ .buf = w }, .{ .buf = sc }, .{ .buf = x }, R{ .buf = ys[0] }, call);
        } else {
            const M = pkg.pipelines.fp8_mma;
            try M.encode(&r.p.mma, e, .{ .buf = x }, .{ .buf = w }, .{ .buf = sc }, .{ .buf = ys[1] }, .{ .k = @intCast(K), .n = @intCast(N), .rows = rows, .out = .bf16 });
        }
        e.end();
        const t0 = now();
        cb.commit();
        cb.wait();
        best[v] = @min(best[v], now() - t0);
    };
    const same = std.mem.eql(u8, ys[0].contents()[0 .. rows * N * 2], ys[1].contents()[0 .. rows * N * 2]);
    std.debug.print("engram wkv {d} rows ({d}x{d}): fp8_rows {d:.2} ms, fp8_mma {d:.2} ms, outputs {s}\n", .{ rows, N, K, @as(f64, @floatFromInt(best[0])) / 1e6, @as(f64, @floatFromInt(best[1])) / 1e6, if (same) "byte-equal" else "DIFFER" });
    if (!same) return error.EngramWkvDiffers;
}

fn list(text: ?[]const u8, buf: []u32) ?[]const u32 {
    var it = std.mem.tokenizeScalar(u8, text orelse return null, ',');
    var n: usize = 0;
    while (it.next()) |v| : (n += 1) buf[n] = std.fmt.parseInt(u32, v, 10) catch return null;
    return buf[0..n];
}
