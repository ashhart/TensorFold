//! Nemotron's paged K/V kernels under gates 1-4: each gives main's contiguous kernel's bytes through shuffled pages.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const paged = sources.nemotron_paged;

const D = 128;
const kvh = 2;
const heads = 32;
const G = heads / kvh;
const page = 256; // TF_PAGE_TOKENS
const page_bytes: u64 = 2 * kvh * page * D * 2; // one layer's keys then values a page
const row_bytes = D * 2;
const maxd = 64; // tf_tree_tail's MAXD

pub const Which = enum { write, compact, put, partial, tail, prompt };

/// The planted control: every position's page is the table's first, which gate 3 must catch.
const first_page = .{ .from = "table[pos / TF_PAGE_TOKENS]", .to = "table[0]" };

const page_edges: c.Tile = .{ .size = page, .step = 1, .min = 1 };
const write_shapes = [_]c.Shape{ .{ .rows = 1, .n = kvh * D, .k = 255 }, .{ .rows = 9, .n = kvh * D, .k = 250 }, .{ .rows = 16, .n = kvh * D, .k = 700 } };
const put_shapes = [_]c.Shape{ .{ .rows = 300, .n = kvh * D, .k = 1 }, .{ .rows = 100, .n = kvh * D, .k = 200 }, .{ .rows = 700, .n = kvh * D, .k = 37 } };
const partial_shapes = [_]c.Shape{ .{ .rows = 1, .n = heads * D, .k = 700 }, .{ .rows = 3, .n = heads * D, .k = 255 }, .{ .rows = 8, .n = heads * D, .k = 1100 }, .{ .rows = 16, .n = heads * D, .k = 513 } };
const tail_shapes = [_]c.Shape{ .{ .rows = 2, .n = heads * D, .k = 250 }, .{ .rows = 9, .n = heads * D, .k = 700 }, .{ .rows = 16, .n = heads * D, .k = 1024 } };
const prompt_shapes = [_]c.Shape{ .{ .rows = 300, .n = heads * D, .k = 1 }, .{ .rows = 64, .n = heads * D, .k = 200 }, .{ .rows = 128, .n = heads * D, .k = 448 }, .{ .rows = 219, .n = heads * D, .k = 37 } };

fn none(_: c.Shape) usize {
    return 0;
}

pub fn name(w: Which, control: bool) []const u8 {
    return switch (w) {
        inline else => |x| if (control) "control: metal nemotron paged first page " ++ @tagName(x) else "metal nemotron paged " ++ @tagName(x),
    };
}

pub fn spec(w: Which, control: bool) c.Spec {
    const what: []const u8 = switch (w) {
        .write => "tf_kv_write",
        .compact => "tf_kv_compact",
        .put => "prefill kv_put",
        .partial => "attn_partial_N",
        .tail => "tf_tree_tail",
        .prompt => "tf_attention_nax",
    };
    const served: []const c.Shape = switch (w) {
        .write, .compact => &write_shapes,
        .put => &put_shapes,
        .partial => &partial_shapes,
        .tail => &tail_shapes,
        .prompt => &prompt_shapes,
    };
    const out: c.Elem = if (w == .partial or w == .tail) .f32 else .bf16;
    return .{ .name = name(w, control), .accum = .none, .depth = none, .out = out, .served = served, .replaces = what, .tiles = .{ .{}, .{}, page_edges }, .nan = .{ .stated = "outputs equal the contiguous kernel's bytes, NaN for NaN" }, .sums = .{ .f16 = 0, .bf16 = 0, .why = "main's kernel's bits through the pages; it has no sums of its own" } };
}

fn value(rnd: std.Random) u16 {
    return @truncate(@as(u32, @bitCast(rnd.float(f32) * 2 - 1)) >> 16);
}

/// The twin's library text as the engine would build it on this GPU (a control: with its one edit).
fn twinText(a: std.mem.Allocator, device: mtl.Device, w: Which, control: bool) ![]u8 {
    const base: []const u8 = switch (w) {
        .write, .compact, .put => paged.glue,
        .partial => paged.partial,
        .tail => paged.tree,
        .prompt => paged.prefill,
    };
    const text = if (control) blk: {
        if (std.mem.count(u8, base, first_page.from) != 1) return error.ControlEditMissing;
        break :blk try std.mem.replaceOwned(u8, a, base, first_page.from, first_page.to);
    } else try a.dupe(u8, base);
    return built(a, device, w, text);
}

/// Main's library text for the kernel a twin replaces, built as on this GPU.
fn mainText(a: std.mem.Allocator, device: mtl.Device, w: Which, tiles: usize) ![]u8 {
    switch (w) {
        .write => return a.dupe(u8, sources.nemotron_glue),
        .compact, .tail => return built(a, device, w, try a.dupe(u8, sources.nemotron_tree)),
        .put, .prompt => {
            const file = if (w == .put) "glue" else "attention_nax";
            for (sources.prefill) |f| if (std.mem.eql(u8, f.name, file)) return built(a, device, w, try a.dupe(u8, f.text));
            return error.UnknownSource;
        },
        .partial => {
            var buf: [32]u8 = undefined;
            const key = try std.fmt.bufPrint(&buf, "attn_partial_{d}", .{tiles});
            for (sources.nemotron.all) |k| if (std.mem.eql(u8, k.key, key)) return built(a, device, w, try a.dupe(u8, k.source));
            return error.UnknownSource;
        },
    }
}

/// kernels.zig's and prefill_kernels.zig's per-GPU steps: the tree layout and attention rewrite, nax.h inlined.
fn built(a: std.mem.Allocator, device: mtl.Device, w: Which, text: []u8) ![]u8 {
    defer a.free(text);
    return switch (w) {
        .write => a.dupe(u8, text),
        .compact, .tail => if (device.tensorUnits() or std.mem.indexOf(u8, text, "tf_tree_tail") == null) a.dupe(u8, text) else std.mem.concat(a, u8, &.{ "#define TF_SIMD_LAYOUT 1\n", text }),
        .put, .prompt => tf.frags.source(device, a, text),
        .partial => if (device.tensorUnits()) a.dupe(u8, text) else tf.nemotron.simd_attention.rewrite(a, text),
    };
}

fn mainName(w: Which, tiles: usize, tag: [2]u8, buf: []u8) ![:0]const u8 {
    if (w == .prompt) {
        const s = try std.fmt.bufPrint(buf[0 .. buf.len - 1], "custom_kernel_tf_attention_nax_bf16_{c}_{c}_t_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_float_bfloat16_t", .{ tag[0], tag[1] });
        buf[s.len] = 0;
        return buf[0..s.len :0];
    }
    const s: [:0]const u8 = switch (w) {
        .write => "tf_kv_write",
        .compact => "tf_kv_compact",
        .tail => "tf_tree_tail",
        .put => "custom_kernel_tf_kv_put_bfloat16_t_bfloat16_t_int32_t_bfloat16_t",
        .partial => blk: {
            inline for (sources.nemotron.all) |k| if (comptime std.mem.startsWith(u8, k.key, "attn_partial_")) {
                if (std.fmt.parseInt(usize, k.key["attn_partial_".len..], 10) catch 0 == tiles) break :blk k.function;
            };
            return error.UnknownSource;
        },
        .prompt => unreachable,
    };
    return s;
}

fn twinName(w: Which, tiles: usize, tag: [2]u8) [:0]const u8 {
    return switch (w) {
        .write => paged.glue_names[0],
        .compact => paged.glue_names[1],
        .put => paged.glue_names[2],
        .partial => paged.partial_names[tiles - 1],
        .tail => paged.tree_name,
        .prompt => for (paged.prefill_tags, paged.prefill_names) |t, n| {
            if (t[0] == tag[0] and t[2] == tag[1]) break n;
        } else unreachable,
    };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    control: bool,
    twin_text: []u8,
    pipes: [2]?mtl.Pipeline = .{ null, null }, // the twin and main's kernel at the prepared shape
    key: usize = 0, // the shape's pipelines: tiles, or the prompt's variant
    rows: usize = 0,
    len: usize = 0,
    cap: usize = 0,
    tag: [2]u8 = .{ 'n', 'n' },
    ids: []u32 = &.{},
    image: []u8 = &.{}, // the arena as prepared: stale bytes and the stream's valid rows
    host: []u16 = &.{}, // the contiguous keys then values [2][kvh, cap, D]
    arena: rig_.Ref = undefined,
    table: rig_.Ref = undefined,
    kv: [2]rig_.Ref = undefined, // main's contiguous keys and values
    ins: [6]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,
    base: [3]rig_.Ref = undefined, // main's outputs
    out_len: [3]usize = .{ 0, 0, 0 },
    golden: []u8 = &.{},

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, w: Which, control: bool) !Probe {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = w, .control = control, .twin_text = try twinText(a, device, w, control) };
    }

    fn drop(p: *Probe) void {
        for (&p.pipes) |*x| if (x.*) |q| {
            q.deinit();
            x.* = null;
        };
        p.a.free(p.image);
        p.a.free(p.golden);
        p.a.free(p.ids);
        p.a.free(p.host);
        p.image = &.{};
        p.golden = &.{};
        p.ids = &.{};
        p.host = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.drop();
        p.rig.deinit();
        p.a.free(p.twin_text);
    }

    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .spec = spec(p.which, p.control), .control = if (p.control) .shapes else null };
    }

    fn writes(p: *const Probe) bool {
        return p.which == .write or p.which == .compact or p.which == .put;
    }

    /// Rows the stream holds before the call: the cached ones, and a window's or chunk's rows that attention reads.
    fn valid(p: *const Probe) usize {
        return switch (p.which) {
            .write, .put => p.len,
            .compact, .partial, .tail, .prompt => p.len + p.rows,
        };
    }

    fn at(p: *const Probe, pos: usize, h: usize, values: bool) usize {
        return p.ids[pos / page] * page_bytes + (@as(usize, @intFromBool(values)) * kvh + h) * page * row_bytes + (pos % page) * row_bytes;
    }

    fn contiguous(p: *const Probe, pos: usize, h: usize, values: bool) usize {
        return ((@as(usize, @intFromBool(values)) * kvh + h) * p.cap + pos) * D;
    }

    pub fn prepare(p: *Probe, s: c.Shape, f: g.Fill, seed: u64) !bool {
        if (s.rows == 0) return false;
        if (p.which == .tail and s.rows > 16) return false;
        p.drop();
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        p.rows = s.rows;
        p.len = s.k;
        p.cap = p.len + p.rows + 8;
        p.key = @min(p.rows, 16);
        p.tag = .{ if (p.rows % 64 == 0) 't' else 'n', if ((p.len + p.rows) % 32 == 0) 't' else 'n' };
        const npages = (p.cap + page - 1) / page;
        const arena_pages = npages + 2;
        p.ids = try p.a.alloc(u32, arena_pages);
        for (p.ids, 0..) |*id, i| id.* = @intCast(i);
        rnd.shuffle(u32, p.ids);
        p.host = try p.a.alloc(u16, 2 * kvh * p.cap * D);
        for (p.host) |*x| x.* = value(rnd);
        if (f == .nonfinite) {
            const mid = p.valid() / 2;
            p.host[p.contiguous(mid, 0, false) + 7] = 0x7fc0; // a NaN key
            p.host[p.contiguous(mid, 1, true) + 3] = 0x7f80; // +Inf value
            p.host[p.contiguous(mid, 1, true) + 4] = 0xff80; // -Inf value
        }
        p.image = try p.a.alloc(u8, arena_pages * page_bytes);
        rnd.bytes(p.image); // stale bytes, NaNs among them
        const h8 = std.mem.sliceAsBytes(p.host);
        for (0..p.valid()) |pos| for (0..kvh) |h| for ([_]bool{ false, true }) |v| {
            @memcpy(p.image[p.at(pos, h, v)..][0..row_bytes], h8[p.contiguous(pos, h, v) * 2 ..][0..row_bytes]);
        };
        p.table = try p.rig.input(std.mem.sliceAsBytes(p.ids[0..npages]));
        p.arena = if (p.writes()) try p.rig.output(p.image.len) else try p.rig.input(p.image);
        for (&p.kv) |*r| r.* = try p.rig.scratch(kvh * p.cap * D * 2);
        try p.inputs(rnd, f);
        const a = p.a;
        const twin = try rig_.pipeline(p.rig.device, p.twin_text, twinName(p.which, p.key, p.tag));
        p.pipes[0] = twin;
        const text = try mainText(a, p.rig.device, p.which, p.key);
        defer a.free(text);
        var buf: [160]u8 = undefined;
        p.pipes[1] = try rig_.pipeline(p.rig.device, text, try mainName(p.which, p.key, p.tag, &buf));
        return true;
    }

    fn bf16s(p: *Probe, rnd: std.Random, n: usize, f: g.Fill) !rig_.Ref {
        const x = try p.a.alloc(u16, n);
        defer p.a.free(x);
        for (x) |*v| v.* = value(rnd);
        if (f == .nonfinite and n > 9) {
            x[1] = 0x7fc0;
            x[n / 2] = 0x7f80;
            x[n - 3] = 0xff80;
        }
        return p.rig.input(std.mem.sliceAsBytes(x));
    }

    /// Each kernel's inputs past the cache, and its outputs (main's into scratch twins of them).
    fn inputs(p: *Probe, rnd: std.Random, f: g.Fill) !void {
        const t = p.rows;
        switch (p.which) {
            .write => p.ins[0] = try p.bf16s(rnd, t * (heads + 2 * kvh) * D, f),
            .compact => {
                var path: [16]i32 = @splat(0);
                var n: usize = 0;
                var r: usize = 0;
                while (r < t and n < 16) : (r += 1 + rnd.uintLessThan(usize, 2)) {
                    path[n] = @intCast(r);
                    n += 1;
                }
                p.ins[0] = try p.rig.input(std.mem.sliceAsBytes(path[0..@max(n, 8)]));
                p.key = n; // the path's length
            },
            .put => {
                p.ins[0] = try p.bf16s(rnd, t * kvh * D, f);
                p.ins[1] = try p.bf16s(rnd, kvh * (p.len + 1) * D, .random);
            },
            .partial => p.ins[0] = try p.bf16s(rnd, kvh * G * t * D, f),
            .tail => {
                var paths: [16 * maxd]i32 = @splat(0);
                var depths: [16]i32 = @splat(0);
                for (0..t) |node| {
                    depths[node] = @intCast(rnd.uintLessThan(usize, t));
                    for (0..@as(usize, @intCast(depths[node])) + 1) |j| paths[node * maxd + j] = @intCast(rnd.uintLessThan(usize, t));
                }
                p.ins[0] = try p.bf16s(rnd, kvh * 16 * t * D, f);
                p.ins[1] = try p.rig.input(std.mem.sliceAsBytes(paths[0 .. t * maxd]));
                p.ins[2] = try p.rig.input(std.mem.sliceAsBytes(depths[0..@max(t, 4)]));
                const chunks = p.nca();
                for (0..3) |i| {
                    const n = kvh * chunks * 16 * t * @as(usize, if (i == 0) D else 1);
                    const x = try p.a.alloc(f32, @max(n, 1));
                    defer p.a.free(x);
                    for (x) |*v| v.* = if (i == 2) 1 + rnd.float(f32) else rnd.float(f32) * 2 - 1;
                    p.ins[3 + i] = try p.rig.input(std.mem.sliceAsBytes(x));
                }
            },
            .prompt => p.ins[0] = try p.bf16s(rnd, t * heads * D, f),
        }
        p.out_len = switch (p.which) {
            .write, .compact, .put => .{ 0, 0, 0 },
            .partial => blk: {
                const slots = kvh * p.nch() * G * t;
                break :blk .{ slots * D * 4, slots * 4, slots * 4 };
            },
            .tail => blk: {
                const slots = kvh * p.ncb() * t * 16;
                break :blk .{ slots * D * 4, slots * 4, slots * 4 };
            },
            .prompt => .{ t * heads * D * 2, 0, 0 },
        };
        for (p.out_len, 0..) |n, i| if (n > 0) {
            p.outs[i] = try p.rig.output(n);
            p.base[i] = try p.rig.scratch(n);
        };
    }

    fn nch(p: *const Probe) usize {
        return (p.len + p.rows + 511) / 512;
    }

    fn pt(p: *const Probe) usize {
        return p.len / 64 * 64;
    }

    fn nca(p: *const Probe) usize {
        return @max((p.pt() + 511) / 512, 1);
    }

    fn ncb(p: *const Probe) usize {
        return (p.len + p.rows) / 512 - p.pt() / 512 + 1;
    }

    pub fn count(p: *Probe) usize {
        if (p.writes()) return p.image.len / 2;
        var n: usize = 0;
        for (p.out_len) |b| n += b;
        return n / @as(usize, if (p.which == .prompt) 2 else 4);
    }

    pub fn run(p: *Probe) !void {
        if (p.writes()) @memcpy(p.rig.bytes(p.arena, p.image.len), p.image);
        try p.launch(true);
    }

    /// One launch: the twin over the arena and table, or main's kernel over contiguous caches into scratch outputs.
    fn launch(p: *Probe, twin: bool) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipes[if (twin) 0 else 1].?);
        const t = p.rows;
        const k = if (twin) p.arena else p.kv[0];
        const v_off: usize = if (twin) kvh * page * row_bytes else 0;
        const v = if (twin) p.arena else p.kv[1];
        const o = if (twin) p.outs else p.base;
        const strides = [4]i64{ @intCast(kvh * p.cap * D), @intCast(p.cap * D), D, 1 };
        const scale: f32 = 1.0 / @sqrt(@as(f32, D));
        // the twin's (keys, values, table) slots and its table's first
        const slots: [3]usize = switch (p.which) {
            .write => .{ 1, 2, 4 },
            .compact => .{ 0, 1, 4 },
            .put => .{ 3, 3, 4 },
            .partial => .{ 1, 3, 10 },
            .tail => .{ 1, 3, 15 },
            .prompt => .{ 1, 2, 6 },
        };
        switch (p.which) {
            .write => {
                e.setBuffer(p.ins[0].buf, p.ins[0].off, 0);
                e.setValue([4]u32{ (heads + 2 * kvh) * D, heads * D, @intCast(p.cap), @intCast(p.len) }, 3);
            },
            .compact => {
                e.setBuffer(p.ins[0].buf, p.ins[0].off, 2);
                e.setValue([4]u32{ @intCast(p.len), @intCast(p.key), @intCast(p.cap), D }, 3);
            },
            .put => {
                e.setBuffer(p.ins[0].buf, p.ins[0].off, 0);
                e.setBuffer(p.ins[1].buf, p.ins[1].off, 1);
                e.setBytes(std.mem.asBytes(&[7]i32{ @intCast(t), kvh, D, @intCast(p.cap), @intCast(p.len), @intCast(p.len + 1), 0 }), 2);
            },
            .partial => {
                e.setBuffer(p.ins[0].buf, p.ins[0].off, 0);
                e.setBytes(std.mem.asBytes(&strides), 2);
                e.setBytes(std.mem.asBytes(&strides), 4);
                e.setValue(scale, 5);
                e.setBytes(std.mem.asBytes(&[5]i32{ @intCast(p.len + t), @intCast(p.nch()), @intCast(t), 1, @intCast(t) }), 6);
                for (o, 7..) |r, i| e.setBuffer(r.buf, r.off, i);
            },
            .tail => {
                e.setBuffer(p.ins[0].buf, p.ins[0].off, 0);
                e.setBytes(std.mem.asBytes(&strides), 2);
                e.setBytes(std.mem.asBytes(&strides), 4);
                e.setValue(scale, 5);
                e.setBytes(std.mem.asBytes(&[5]i32{ @intCast(p.len), @intCast(p.pt()), @intCast(p.nca()), @intCast(p.ncb()), @intCast(t) }), 6);
                for (p.ins[1..6], 7..) |r, i| e.setBuffer(r.buf, r.off, i);
                for (o, 12..) |r, i| e.setBuffer(r.buf, r.off, i);
            },
            .prompt => {
                const kl = p.len + t;
                const row = heads * D;
                const kvs = kvh * p.cap * D;
                var pr: [32]i32 = @splat(0);
                const head = [_]usize{ 1, heads, t, kl, G, (t + 63) / 64, (kl + 31) / 32, t / 64, kl / 32, t % 64, kl % 32, kl - t };
                const st = [12]usize{ t * row, D, row, kvs, p.cap * D, D, kvs, p.cap * D, D, t * row, D, row };
                for (head ++ st, 0..) |x, i| pr[i] = @intCast(x);
                var fl: [16]f32 = @splat(0);
                fl[0] = scale;
                e.setBuffer(p.ins[0].buf, p.ins[0].off, 0);
                e.setBytes(std.mem.sliceAsBytes(&pr), 3);
                e.setBytes(std.mem.sliceAsBytes(&fl), 4);
                e.setBuffer(o[0].buf, o[0].off, 5);
            },
        }
        if (p.which == .put) {
            // kv_put writes one cache a launch: keys, then values
            for ([_]bool{ false, true }) |values| {
                if (twin) e.setBuffer(p.arena.buf, p.arena.off + if (values) v_off else 0, 3) else e.setBuffer(p.kv[@intFromBool(values)].buf, p.kv[@intFromBool(values)].off, 3);
                if (twin) p.bindTable(e, slots[2]);
                e.dispatchThreads(mtl.Size.of(D, t, kvh), mtl.Size.of(128, 2, 1));
            }
            return p.rig.finish(pass);
        }
        e.setBuffer(k.buf, k.off, slots[0]);
        e.setBuffer(v.buf, v.off + v_off, slots[1]);
        if (twin) p.bindTable(e, slots[2]);
        switch (p.which) {
            .write => e.dispatchThreads(mtl.Size.of(D, kvh, t), mtl.Size.of(D, 1, 1)),
            .compact => e.dispatchThreads(mtl.Size.of(D, kvh, 1), mtl.Size.of(D, 1, 1)),
            .partial => e.dispatchThreads(mtl.Size.of(kvh * 32 * p.key, p.nch(), (t + p.key - 1) / p.key), mtl.Size.of(32 * p.key, 1, 1)),
            .tail => e.dispatchThreads(mtl.Size.of(kvh * 32, p.ncb(), t), mtl.Size.of(32, 1, 1)),
            .prompt => e.dispatchThreads(mtl.Size.of((t + 63) / 64 * 32, heads * 4, 1), mtl.Size.of(32, 4, 1)),
            .put => unreachable,
        }
        try p.rig.finish(pass);
    }

    fn bindTable(p: *const Probe, e: mtl.ComputeEncoder, slot: usize) void {
        e.setBuffer(p.table.buf, p.table.off, slot);
        e.setValue(page_bytes, slot + 1);
    }

    /// Main's kernel on the same inputs; for a write, its rows placed where the true table puts them.
    pub fn replaced(p: *Probe) ![]const u8 {
        const h8 = std.mem.sliceAsBytes(p.host);
        for (p.kv, [_]bool{ false, true }) |r, values| @memcpy(p.rig.bytes(r, kvh * p.cap * D * 2), h8[@as(usize, @intFromBool(values)) * kvh * p.cap * D * 2 ..][0 .. kvh * p.cap * D * 2]);
        try p.launch(false);
        p.a.free(p.golden);
        p.golden = &.{};
        if (p.writes()) {
            p.golden = try p.a.dupe(u8, p.image);
            const upto = p.len + p.rows;
            for (0..upto) |pos| for (0..kvh) |h| for ([_]bool{ false, true }, p.kv) |values, r| {
                const src = p.rig.bytes(r, kvh * p.cap * D * 2)[(h * p.cap + pos) * row_bytes ..][0..row_bytes];
                @memcpy(p.golden[p.at(pos, h, values)..][0..row_bytes], src);
            };
            return p.golden;
        }
        var all: std.ArrayList(u8) = .empty;
        for (p.out_len, 0..) |n, i| if (n > 0) try all.appendSlice(p.a, p.rig.bytes(p.base[i], n));
        p.golden = try all.toOwnedSlice(p.a);
        return p.golden;
    }

    pub fn reference(_: *Probe, want: []f64, mag: []f64) !void {
        @memset(want, 0); // the check compares bytes with main's kernel's
        @memset(mag, 0);
    }

    /// Gate 1's stated rule: the twin's bytes are main's kernel's on the same planted inputs, NaN for NaN.
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const got = try p.a.dupe(u8, try p.rig.outputBytes());
        defer p.a.free(got);
        const want = try p.replaced();
        if (!std.mem.eql(u8, got, want)) return "its bytes differ from the contiguous kernel's";
        return null;
    }
};

/// Every twin and its control, as tf-kernel-contract collects them.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(Which)) |w| for ([_]bool{ false, true }) |control| {
            if (only) |o| if (std.mem.indexOf(u8, name(w, control), o) == null) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, w, control);
            try s.probes.append(s.a, p);
            try s.kernels.append(s.a, p.kernel());
        };
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
