//! Main's Nemotron attention kernels and their K/V glue under gates 1-4 against float64, each with a planted control.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const ref = tf.attention_ref;

const D = 128;
const kvh = 2;
const heads = 32;
const G = heads / kvh;
const maxd = 64; // tf_tree_tail's MAXD
const chunk = 512; // the partials' and the tail's keys a chunk
const scale: f32 = 1.0 / @sqrt(@as(f32, D));

/// decode: attn_partial_N then attn_merge; tree: a partial, tf_tree_tail and tf_tree_merge; prompt: tf_attention_nax.
pub const Which = enum { decode, tree, prompt, write, put, compact };

/// Each kernel's planted control: one source edit gate 3 must catch.
const Edit = struct { from: []const u8, to: []const u8 };
fn edit(w: Which) Edit {
    return switch (w) {
        .decode => .{ .from = "S[i] * scale[0]", .to = "S[i]" },
        .tree => .{ .from = "sraw[i] * scale[0]", .to = "sraw[i]" },
        .prompt => .{ .from = "F[0] * 1.44269504089f", .to = "F[0] * 14.4269504089f" },
        .write => .{ .from = "kc[dst] = qkv[src];", .to = "kc[dst] = qkv[src + 1];" },
        .put => .{ .from = "src[(long(row - P[4]) * P[1] + kv) * D + d]", .to = "src[(long(row - P[4]) * P[1] + kv) * D + d + 1]" },
        .compact => .{ .from = "= K[(size_t(hk) * cap + src) * hd + d];", .to = "= V[(size_t(hk) * cap + src) * hd + d];" },
    };
}

const decode_shapes = blk: {
    var out: [16][3]c.Shape = undefined;
    for (&out, 1..) |*s, n| s.* = .{ .{ .rows = n, .n = heads * D, .k = 700 }, .{ .rows = n, .n = heads * D, .k = 255 }, .{ .rows = n, .n = heads * D, .k = 1100 } };
    break :blk out;
};
const tree_shapes = [_]c.Shape{ .{ .rows = 2, .n = heads * D, .k = 250 }, .{ .rows = 9, .n = heads * D, .k = 700 }, .{ .rows = 16, .n = heads * D, .k = 1024 } };
const prompt_shapes = [_]c.Shape{ .{ .rows = 300, .n = heads * D, .k = 1 }, .{ .rows = 64, .n = heads * D, .k = 200 }, .{ .rows = 128, .n = heads * D, .k = 448 }, .{ .rows = 219, .n = heads * D, .k = 37 } };
const write_shapes = [_]c.Shape{ .{ .rows = 1, .n = kvh * D, .k = 255 }, .{ .rows = 9, .n = kvh * D, .k = 250 }, .{ .rows = 16, .n = kvh * D, .k = 700 } };
const put_shapes = [_]c.Shape{ .{ .rows = 300, .n = kvh * D, .k = 1 }, .{ .rows = 100, .n = kvh * D, .k = 200 }, .{ .rows = 700, .n = kvh * D, .k = 37 } };

var names: [16][2][64]u8 = undefined;

/// An entry's name; decode at `tiles` tiles.
pub fn name(w: Which, tiles: usize, control: bool) []const u8 {
    const lead = if (control) "control: metal nemotron attention " else "metal nemotron attention ";
    if (w == .decode) return std.fmt.bufPrint(&names[tiles - 1][@intFromBool(control)], "{s}partial_{d}+merge", .{ lead, tiles }) catch unreachable;
    return switch (w) {
        inline else => |x| if (control) "control: metal nemotron attention " ++ comptime label(x) else "metal nemotron attention " ++ comptime label(x),
    };
}

fn label(w: Which) []const u8 {
    return switch (w) {
        .decode => "partial+merge",
        .tree => "tree partial+tail+merge",
        .prompt => "prompt nax",
        .write => "kv_write",
        .put => "kv_put",
        .compact => "kv_compact",
    };
}

fn attending(w: Which) bool {
    return w == .decode or w == .tree or w == .prompt;
}

fn keysDepth(s: c.Shape) usize {
    return D + s.k + s.rows;
}

pub fn spec(w: Which, tiles: usize, control: bool) c.Spec {
    const served: []const c.Shape = switch (w) {
        .decode => &decode_shapes[tiles - 1],
        .tree => &tree_shapes,
        .prompt => &prompt_shapes,
        .write, .compact => &write_shapes,
        .put => &put_shapes,
    };
    if (!attending(w)) return .{ .name = name(w, tiles, control), .accum = .none, .depth = keysDepth, .out = .bf16, .served = served, .sums = .{ .f16 = 0, .bf16 = 0, .why = "it moves rows" } };
    const tiles_of: [3]c.Tile = switch (w) {
        .prompt => .{ .{ .size = 64 }, .{}, .{ .size = 32 } },
        else => .{ .{}, .{}, .{ .size = chunk } },
    };
    return .{ .name = name(w, tiles, control), .accum = .f32, .depth = keysDepth, .ops = 8, .out = .bf16, .tiles = tiles_of, .served = served };
}

fn value(rnd: std.Random) u16 {
    return c.toBf16(rnd.float(f32) * 2 - 1);
}

/// Gate 4's values for key `j` of `n`: big halves that cancel, four ones left, so only exact fp32 sums give 4.
fn stressValue(j: usize, n: usize) f32 {
    const half = n / 2;
    if (j >= 2 * half) return 0;
    const at = if (j < half) j else j - half;
    const big: f32 = if (at % 2 == 0) 127 else 16256;
    const one: f32 = if (at < 8 and at % 2 == 0) 1 else 0;
    return if (j < half) big + one else -big;
}

/// A library's text as the engine builds it on this GPU, then the control's edit when asked.
fn built(a: std.mem.Allocator, device: mtl.Device, w: Which, text: []const u8, control: bool) ![]u8 {
    const src = switch (w) {
        .decode => if (device.tensorUnits() or std.mem.indexOf(u8, text, "partial_direct") == null) try a.dupe(u8, text) else try tf.nemotron.simd_attention.rewrite(a, text),
        .tree, .compact => if (device.tensorUnits()) try a.dupe(u8, text) else try std.mem.concat(a, u8, &.{ "#define TF_SIMD_LAYOUT 1\n", text }),
        .prompt, .put => try tf.frags.source(device, a, text),
        .write => try a.dupe(u8, text),
    };
    if (!control) return src;
    defer a.free(src);
    const e = edit(w);
    if (std.mem.count(u8, src, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, src, e.from, e.to);
}

fn nemotronKernel(key: []const u8) !sources.nemotron.Kernel {
    for (sources.nemotron.all) |k| if (std.mem.eql(u8, k.key, key)) return k;
    return error.UnknownSource;
}

fn prefillText(file: []const u8) ![]const u8 {
    for (sources.prefill) |f| if (std.mem.eql(u8, f.name, file)) return f.text;
    return error.UnknownSource;
}

fn compile(a: std.mem.Allocator, device: mtl.Device, w: Which, text: []const u8, control: bool, fun: []const u8) !mtl.Pipeline {
    const t = try built(a, device, w, text, control);
    defer a.free(t);
    return rig_.pipeline(device, t, fun);
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    tiles: usize,
    control: bool,
    pipes: [3]?mtl.Pipeline = .{ null, null, null }, // decode: partial, merge; tree: tail, merge; glue: the kernel
    partials: [16]?mtl.Pipeline = @splat(null), // the tree's prefix partial at 1 to 16 tiles
    prompts: [4]?mtl.Pipeline = @splat(null), // tf_attention_nax by (whole query blocks, whole key blocks)
    rows: usize = 0,
    len: usize = 0,
    cap: usize = 0,
    stress: bool = false,
    q: []u16 = &.{}, // decode and tree: Qp [kvh, 16 * rows, D]; prompt: [rows, heads, D]; glue: the source rows
    host: []u16 = &.{}, // the caches: keys then values [2][kvh, cap, D]
    paths: []i32 = &.{},
    depths: []i32 = &.{},
    path: []i32 = &.{}, // compact's kept path
    srcs: [2]rig_.Ref = undefined,
    kv: [2]rig_.Ref = undefined,
    qref: rig_.Ref = undefined,
    tree_refs: [2]rig_.Ref = undefined,
    parts: [6]rig_.Ref = undefined, // PO, PM, PL of the partials; the tail's in the tree
    out: rig_.Ref = undefined,
    out_bytes: usize = 0,
    image: []u8 = &.{}, // glue: the caches as prepared, which each run starts from

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, w: Which, tiles: usize, control: bool) !Probe {
        var p: Probe = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = w, .tiles = tiles, .control = control };
        errdefer p.deinit();
        switch (w) {
            .decode => {
                var buf: [32]u8 = undefined;
                const k = try nemotronKernel(try std.fmt.bufPrint(&buf, "attn_partial_{d}", .{tiles}));
                p.pipes[0] = try compile(a, device, .decode, k.source, control, k.function);
                const m = try nemotronKernel("attn_merge");
                p.pipes[1] = try compile(a, device, .decode, m.source, false, m.function);
            },
            .tree => {
                p.pipes[0] = try compile(a, device, .tree, sources.nemotron_tree, control, "tf_tree_tail");
                p.pipes[1] = try compile(a, device, .tree, sources.nemotron_tree, false, "tf_tree_merge");
            },
            .prompt => {},
            .write => p.pipes[0] = try compile(a, device, .write, sources.nemotron_glue, control, "tf_kv_write"),
            .compact => p.pipes[0] = try compile(a, device, .compact, sources.nemotron_tree, control, "tf_kv_compact"),
            .put => p.pipes[0] = try compile(a, device, .put, try prefillText("glue"), control, "custom_kernel_tf_kv_put_bfloat16_t_bfloat16_t_int32_t_bfloat16_t"),
        }
        return p;
    }

    fn drop(p: *Probe) void {
        for ([_][]u16{ p.q, p.host }) |x| p.a.free(x);
        for ([_][]i32{ p.paths, p.depths, p.path }) |x| p.a.free(x);
        p.a.free(p.image);
        p.q = &.{};
        p.host = &.{};
        p.paths = &.{};
        p.depths = &.{};
        p.path = &.{};
        p.image = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.drop();
        p.rig.deinit();
        for (&p.pipes) |x| if (x) |q| q.deinit();
        for (&p.partials) |x| if (x) |q| q.deinit();
        for (&p.prompts) |x| if (x) |q| q.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .spec = spec(p.which, p.tiles, p.control), .control = if (p.control) .shapes else null };
    }

    fn at(p: *const Probe, values: bool, h: usize, pos: usize) usize {
        return ((@as(usize, @intFromBool(values)) * kvh + h) * p.cap + pos) * D;
    }

    /// The tree's prefix: whole 64-key tiles of the cached rows, their chunks, and the tail's chunks.
    fn pt(p: *const Probe) usize {
        return p.len / 64 * 64;
    }

    fn nca(p: *const Probe) usize {
        return @max((p.pt() + chunk - 1) / chunk, 1);
    }

    fn ncb(p: *const Probe) usize {
        return (p.len + p.rows) / chunk - p.pt() / chunk + 1;
    }

    /// Keys every query attends: decode's whole window, the prompt's rows to its own, a tree node's prefix and path.
    fn keysOf(p: *const Probe, qi: usize, out: *std.ArrayList(usize)) !void {
        out.clearRetainingCapacity();
        switch (p.which) {
            .decode => {
                const n = if (p.stress) p.len + p.rows else p.len + qi + 1;
                for (0..n) |j| try out.append(p.a, j);
            },
            .prompt => for (0..p.len + qi + 1) |j| try out.append(p.a, j),
            .tree => {
                for (0..p.len) |j| try out.append(p.a, j);
                for (0..@as(usize, @intCast(p.depths[qi])) + 1) |j| try out.append(p.a, p.len + @as(usize, @intCast(p.paths[qi * maxd + j])));
            },
            else => unreachable,
        }
    }

    pub fn prepare(p: *Probe, s: c.Shape, f: g.Fill, seed: u64) !bool {
        if (s.rows == 0) return false;
        if (p.which == .tree and s.rows > 16) return false;
        if (p.which == .decode and s.rows > 64) return false;
        if ((p.which == .decode or p.which == .tree) and s.k + s.rows > 2048 and f == .stress) return false;
        p.drop();
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        p.rows = s.rows;
        p.len = s.k;
        p.cap = p.len + p.rows + 8;
        p.stress = f == .stress;
        const t = p.rows;
        p.host = try p.a.alloc(u16, 2 * kvh * p.cap * D);
        for (p.host) |*x| x.* = value(rnd);
        if (attending(p.which)) {
            const live = if (p.which == .tree) p.len else p.len + t; // the keys gate 4's values cancel over
            if (p.stress) for (0..kvh) |h| for (0..p.cap) |j| for (0..D) |d| {
                p.host[p.at(false, h, j) + d] = 0;
                p.host[p.at(true, h, j) + d] = c.toBf16(stressValue(j, live));
            };
            if (f == .nonfinite) for (0..kvh) |h| for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }, 0..) |x, d| {
                p.host[p.at(true, h, 0) + d] = x; // key 0's values: every query attends it
            };
        }
        const qn = switch (p.which) {
            .decode, .tree => kvh * 16 * t * D,
            .prompt => t * heads * D,
            .write => t * (heads + 2 * kvh) * D,
            .put => 2 * t * kvh * D,
            .compact => 0,
        };
        p.q = try p.a.alloc(u16, qn);
        for (p.q) |*x| x.* = if (p.stress) 0 else value(rnd);
        if (f == .nonfinite and !attending(p.which) and qn > 0) {
            const first: usize = if (p.which == .write) heads * D else 0; // the stacked row's first key column, which it copies
            for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }, [_]usize{ 0, kvh * D + 1, qn - first - 3 }) |x, i| p.q[first + i] = x;
        }
        if (p.which == .tree) {
            p.paths = try p.a.dupe(i32, &@as([16 * maxd]i32, @splat(0)));
            p.depths = try p.a.dupe(i32, &@as([16]i32, @splat(0)));
            for (0..t) |node| {
                p.depths[node] = @intCast(rnd.uintLessThan(usize, t));
                for (0..@as(usize, @intCast(p.depths[node])) + 1) |j| p.paths[node * maxd + j] = @intCast(rnd.uintLessThan(usize, t));
            }
        }
        if (p.which == .compact) {
            var path: [16]i32 = @splat(0);
            var n: usize = 0;
            var r: usize = 0;
            while (r < t and n < 16) : (r += 1 + rnd.uintLessThan(usize, 2)) {
                path[n] = @intCast(r);
                n += 1;
            }
            p.path = try p.a.dupe(i32, path[0..n]);
            if (f == .nonfinite) for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }, 0..) |x, d| {
                p.host[p.at(false, 0, p.len + @as(usize, @intCast(p.path[n - 1]))) + d] = x;
            };
        }
        try p.bind();
        return true;
    }

    fn bind(p: *Probe) !void {
        const t = p.rows;
        const cache = kvh * p.cap * D * 2;
        const h8 = std.mem.sliceAsBytes(p.host);
        if (!attending(p.which)) {
            p.image = try p.a.dupe(u8, h8);
            if (p.which == .write) p.srcs[0] = try p.rig.input(std.mem.sliceAsBytes(p.q));
            if (p.which == .put) for (&p.srcs, 0..) |*r, i| {
                r.* = try p.rig.input(std.mem.sliceAsBytes(p.q[i * t * kvh * D ..][0 .. t * kvh * D]));
            };
            if (p.which == .compact) p.srcs[0] = try p.rig.input(std.mem.sliceAsBytes(p.path));
            p.out_bytes = 2 * cache;
            p.out = try p.rig.output(p.out_bytes);
            return;
        }
        for (&p.kv, 0..) |*r, i| r.* = try p.rig.input(h8[i * cache ..][0..cache]);
        p.qref = try p.rig.input(std.mem.sliceAsBytes(p.q));
        switch (p.which) {
            .decode => {
                const slots = kvh * ((p.len + t + chunk - 1) / chunk) * 16 * t;
                p.parts[0] = try p.rig.scratch(slots * D * 4);
                p.parts[1] = try p.rig.scratch(slots * 4);
                p.parts[2] = try p.rig.scratch(slots * 4);
            },
            .tree => {
                p.tree_refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.paths));
                p.tree_refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.depths));
                const a_slots = kvh * p.nca() * 16 * t;
                const b_slots = kvh * p.ncb() * 16 * t;
                for (0..2) |i| {
                    const slots = if (i == 0) a_slots else b_slots;
                    p.parts[3 * i] = try p.rig.scratch(slots * D * 4);
                    p.parts[3 * i + 1] = try p.rig.scratch(slots * 4);
                    p.parts[3 * i + 2] = try p.rig.scratch(slots * 4);
                }
                const tiles: usize = @min(t, 16);
                if (p.pt() > 0 and p.partials[tiles - 1] == null) {
                    var buf: [32]u8 = undefined;
                    const k = try nemotronKernel(try std.fmt.bufPrint(&buf, "attn_partial_{d}", .{tiles}));
                    p.partials[tiles - 1] = try compile(p.a, p.rig.device, .decode, k.source, false, k.function);
                }
            },
            .prompt => {
                const v = p.variant();
                if (p.prompts[v] == null) {
                    var buf: [160]u8 = undefined;
                    const tag = [2]u8{ if (v >= 2) 't' else 'n', if (v % 2 == 1) 't' else 'n' };
                    const s = try std.fmt.bufPrint(&buf, "custom_kernel_tf_attention_nax_bf16_{c}_{c}_t_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_float_bfloat16_t", .{ tag[0], tag[1] });
                    p.prompts[v] = try compile(p.a, p.rig.device, .prompt, try prefillText("attention_nax"), p.control, s);
                }
            },
            else => unreachable,
        }
        p.out_bytes = heads * t * D * 2;
        p.out = try p.rig.output(p.out_bytes);
    }

    fn variant(p: *const Probe) usize {
        return @as(usize, @intFromBool(p.rows % 64 == 0)) * 2 + @intFromBool((p.len + p.rows) % 32 == 0);
    }

    pub fn count(p: *Probe) usize {
        return p.out_bytes / 2;
    }

    pub fn run(p: *Probe) !void {
        if (!attending(p.which)) @memcpy(p.rig.bytes(p.out, p.out_bytes), p.image);
        const pass = p.rig.begin();
        const e = pass.e;
        const t = p.rows;
        const strides = [4]i64{ @intCast(kvh * p.cap * D), @intCast(p.cap * D), D, 1 };
        const kref = if (attending(p.which)) p.kv[0] else p.out;
        const v_off: usize = if (attending(p.which)) 0 else kvh * p.cap * D * 2;
        const vref = if (attending(p.which)) p.kv[1] else p.out;
        switch (p.which) {
            .decode => {
                const keys = p.len + t;
                const nch = (keys + chunk - 1) / chunk;
                // gate 4 runs the window unmasked (the tree's prefix pass): every query's keys are then the same
                const dims = [5]i32{ @intCast(keys), @intCast(nch), @intCast(t), @intFromBool(!p.stress), @intCast(t) };
                e.setPipeline(p.pipes[0].?);
                p.attnArgs(e, strides);
                e.setBytes(std.mem.asBytes(&dims), 6);
                for (p.parts[0..3], 7..) |r, i| e.setBuffer(r.buf, r.off, i);
                e.dispatchThreads(mtl.Size.of(kvh * 32 * p.tiles, nch, (t + p.tiles - 1) / p.tiles), mtl.Size.of(32 * p.tiles, 1, 1));
                e.barrier();
                e.setPipeline(p.pipes[1].?);
                for (p.parts[0..3], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
                e.setBytes(std.mem.asBytes(&dims), 3);
                e.setBuffer(p.out.buf, p.out.off, 4);
                e.dispatchThreads(mtl.Size.of(kvh * 32, G * t, 1), mtl.Size.of(32, 1, 1));
            },
            .tree => {
                const tiles: usize = @min(t, 16);
                if (p.pt() > 0) {
                    e.setPipeline(p.partials[tiles - 1].?);
                    p.attnArgs(e, strides);
                    e.setBytes(std.mem.asBytes(&[5]i32{ @intCast(p.pt()), @intCast(p.nca()), @intCast(t), 0, @intCast(t) }), 6);
                    for (p.parts[0..3], 7..) |r, i| e.setBuffer(r.buf, r.off, i);
                    e.dispatchThreads(mtl.Size.of(kvh * 32 * tiles, p.nca(), (t + tiles - 1) / tiles), mtl.Size.of(32 * tiles, 1, 1));
                    e.barrier();
                }
                const dims = [5]i32{ @intCast(p.len), @intCast(p.pt()), @intCast(p.nca()), @intCast(p.ncb()), @intCast(t) };
                e.setPipeline(p.pipes[0].?);
                p.attnArgs(e, strides);
                e.setBytes(std.mem.asBytes(&dims), 6);
                e.setBuffer(p.tree_refs[0].buf, p.tree_refs[0].off, 7);
                e.setBuffer(p.tree_refs[1].buf, p.tree_refs[1].off, 8);
                for (p.parts, 9..) |r, i| e.setBuffer(r.buf, r.off, i);
                e.dispatchThreads(mtl.Size.of(kvh * 32, p.ncb(), t), mtl.Size.of(32, 1, 1));
                e.barrier();
                e.setPipeline(p.pipes[1].?);
                for (p.parts, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
                e.setBytes(std.mem.asBytes(&dims), 6);
                e.setBuffer(p.out.buf, p.out.off, 7);
                e.dispatchThreads(mtl.Size.of(kvh * 32, G * t, 1), mtl.Size.of(32, 1, 1));
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
                e.setPipeline(p.prompts[p.variant()].?);
                e.setBuffer(p.qref.buf, p.qref.off, 0);
                e.setBuffer(p.kv[0].buf, p.kv[0].off, 1);
                e.setBuffer(p.kv[1].buf, p.kv[1].off, 2);
                e.setBytes(std.mem.sliceAsBytes(&pr), 3);
                e.setBytes(std.mem.sliceAsBytes(&fl), 4);
                e.setBuffer(p.out.buf, p.out.off, 5);
                e.dispatchThreads(mtl.Size.of((t + 63) / 64 * 32, heads * 4, 1), mtl.Size.of(32, 4, 1));
            },
            .write => {
                e.setPipeline(p.pipes[0].?);
                e.setBuffer(p.srcs[0].buf, p.srcs[0].off, 0);
                e.setBuffer(kref.buf, kref.off, 1);
                e.setBuffer(vref.buf, vref.off + v_off, 2);
                e.setValue([4]u32{ (heads + 2 * kvh) * D, heads * D, @intCast(p.cap), @intCast(p.len) }, 3);
                e.dispatchThreads(mtl.Size.of(D, kvh, t), mtl.Size.of(D, 1, 1));
            },
            .compact => {
                e.setPipeline(p.pipes[0].?);
                e.setBuffer(kref.buf, kref.off, 0);
                e.setBuffer(vref.buf, vref.off + v_off, 1);
                e.setBuffer(p.srcs[0].buf, p.srcs[0].off, 2);
                e.setValue([4]u32{ @intCast(p.len), @intCast(p.path.len), @intCast(p.cap), D }, 3);
                e.dispatchThreads(mtl.Size.of(D, kvh, 1), mtl.Size.of(D, 1, 1));
            },
            .put => for (p.srcs, [_]usize{ 0, v_off }) |src, off| {
                e.setPipeline(p.pipes[0].?);
                e.setBuffer(src.buf, src.off, 0);
                e.setBuffer(src.buf, src.off, 1);
                e.setBytes(std.mem.asBytes(&[7]i32{ @intCast(t), kvh, D, @intCast(p.cap), @intCast(p.len), @intCast(p.cap), 0 }), 2);
                e.setBuffer(p.out.buf, p.out.off + off, 3);
                e.dispatchThreads(mtl.Size.of(D, t, kvh), mtl.Size.of(128, 2, 1));
            },
        }
        try p.rig.finish(pass);
    }

    fn attnArgs(p: *const Probe, e: mtl.ComputeEncoder, strides: [4]i64) void {
        e.setBuffer(p.qref.buf, p.qref.off, 0);
        e.setBuffer(p.kv[0].buf, p.kv[0].off, 1);
        e.setBytes(std.mem.asBytes(&strides), 2);
        e.setBuffer(p.kv[1].buf, p.kv[1].off, 3);
        e.setBytes(std.mem.asBytes(&strides), 4);
        e.setValue(scale, 5);
    }

    /// Output index of query `qi`, head `h`: [heads, rows, D] after a merge, [rows, heads, D] from the prompt kernel.
    fn outAt(p: *const Probe, qi: usize, h: usize) usize {
        return if (p.which == .prompt) (qi * heads + h) * D else (h * p.rows + qi) * D;
    }

    /// Query `qi` of head `h` as float64: Qp's row qi * G + g of its kv head, or the prompt's row.
    fn query(p: *const Probe, qi: usize, h: usize, out: []f64) void {
        const at_ = if (p.which == .prompt) (qi * heads + h) * D else ((h / G) * 16 * p.rows + qi * G + h % G) * D;
        for (out, 0..) |*x, d| x.* = c.fromBf16(p.q[at_ + d]);
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        if (!attending(p.which)) return p.moved(want, mag);
        const a = p.a;
        const kf = try a.alloc(f64, p.cap * D);
        defer a.free(kf);
        const vf = try a.alloc(f64, p.cap * D);
        defer a.free(vf);
        var q: [D]f64 = undefined;
        var keys: std.ArrayList(usize) = .empty;
        defer keys.deinit(a);
        for (0..kvh) |hk| {
            for (0..p.cap * D) |i| {
                kf[i] = c.fromBf16(p.host[p.at(false, hk, 0) + i]);
                vf[i] = c.fromBf16(p.host[p.at(true, hk, 0) + i]);
            }
            for (0..p.rows) |qi| {
                try p.keysOf(qi, &keys);
                for (hk * G..(hk + 1) * G) |h| {
                    p.query(qi, h, &q);
                    const o = p.outAt(qi, h);
                    ref.head(&q, kf, vf, keys.items, D, scale, want[o..][0..D], mag[o..][0..D]);
                }
            }
        }
    }

    /// attention_ref's bound for every finite output (its bf16-P allowance); gate 4's exact sums keep the generic one.
    pub fn bounds(p: *Probe, want: []const f64, tol: []const f64, allow: []f64) !void {
        if (!attending(p.which)) return;
        for (want, tol, allow) |w, t, *l| l.* = if (p.stress) c.boundExact(spec(p.which, p.tiles, p.control), w) else t;
    }

    /// Gate 4: every score equal, so each output is its keys' values summed in `acc` over their count.
    pub fn simulate(p: *Probe, acc: c.Accum, order: contract.rows.Order, out: []f64) !void {
        if (!p.stress) return error.StressRequiresEqualScores;
        var keys: std.ArrayList(usize) = .empty;
        defer keys.deinit(p.a);
        var terms: std.ArrayList(f64) = .empty;
        defer terms.deinit(p.a);
        for (0..kvh) |hk| for (0..p.rows) |qi| {
            try p.keysOf(qi, &keys);
            for (0..D) |d| {
                terms.clearRetainingCapacity();
                for (keys.items) |j| try terms.append(p.a, c.fromBf16(p.host[p.at(true, hk, j) + d]));
                const v = contract.rows.sumIn(acc, order, terms.items) / @as(f64, @floatFromInt(keys.items.len));
                for (hk * G..(hk + 1) * G) |h| out[p.outAt(qi, h) + d] = v;
            }
        };
    }

    /// The glue's float64: the prepared caches with the kernel's rows moved, value for value.
    fn moved(p: *Probe, want: []f64, mag: []f64) void {
        @memset(mag, 0);
        const cache = p.a.dupe(u16, p.host) catch unreachable;
        defer p.a.free(cache);
        const t = p.rows;
        switch (p.which) {
            .write => for (0..t) |r| for (0..kvh) |h| for (0..D) |d| {
                const src = r * (heads + 2 * kvh) * D + heads * D + h * D + d;
                cache[p.at(false, h, p.len + r) + d] = p.q[src];
                cache[p.at(true, h, p.len + r) + d] = p.q[src + kvh * D];
            },
            .put => for (0..2) |i| for (0..t) |r| for (0..kvh) |h| for (0..D) |d| {
                cache[p.at(i == 1, h, p.len + r) + d] = p.q[i * t * kvh * D + (r * kvh + h) * D + d];
            },
            .compact => for (p.path, 0..) |x, j| {
                const src = p.len + @as(usize, @intCast(x));
                const dst = p.len + j;
                if (src == dst) continue;
                for (0..kvh) |h| for ([_]bool{ false, true }) |v| {
                    @memcpy(cache[p.at(v, h, dst)..][0..D], cache[p.at(v, h, src)..][0..D]);
                };
            },
            else => unreachable,
        }
        for (want, cache) |*w, x| w.* = c.fromBf16(x);
    }
};

/// Every kernel and its control, as tf-kernel-contract collects them.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(Which)) |w| for (1..if (w == .decode) 17 else 2) |tiles| for ([_]bool{ false, true }) |control| {
            if (only) |o| if (std.mem.indexOf(u8, name(w, tiles, control), o) == null) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, w, tiles, control);
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
