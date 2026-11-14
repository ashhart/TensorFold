//! window_vonly.cu under the contract: each kernel bit-equal to its K+V counterpart, each with a planted control.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const wc = @import("window_rows_check.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const wa = cuda.window_attention;
const qk = cuda.qk_rows;
const VOnly = cuda.window_vonly.VOnly;
const Vk = cuda.window_vonly.Vk;

const d = 512;

/// A written batch's k and v rows on the device.
const Kv = struct { k: u64, v: u64 };

/// keys: shape rows = keys, k = the first key's position; decode: rows, n = query heads, k = each stream's keys.
pub const Kind = enum { keys, decode };
/// Planted controls: flip a made key's bit, keys without bf16 rounding, or decode with another k norm.
pub const Ctl = enum { none, flip, unrounded, norm };

pub const Entry = struct {
    name: []const u8,
    kind: Kind,
    h: usize = 32,
    hk: usize = 4,
    tok: usize,
    ctl: Ctl = .none,
    served: []const c.Shape,
};

/// Positions 0, page edges, 2^16, a 64K context's angles and the 256K model limit (cosf's slow path).
const keys_served = &[_]c.Shape{
    .{ .rows = 512, .n = 4, .k = 0 },
    .{ .rows = 300, .n = 4, .k = 65400 },
    .{ .rows = 256, .n = 4, .k = 69900 },
    .{ .rows = 64, .n = 4, .k = 262100 },
};
const decode_served = &[_]c.Shape{
    .{ .rows = 4, .n = 32, .k = 4096 },
    .{ .rows = 16, .n = 32, .k = 3000 },
    .{ .rows = 1, .n = 32, .k = 300 },
};
const gqa5_served = &[_]c.Shape{ .{ .rows = 4, .n = 10, .k = 3000 }, .{ .rows = 1, .n = 10, .k = 300 } };
const decode = "cuda window_vonly decode tf_wrows_part_512_pg_vo + tf_wrows_merge_512_pg_vo, ";

pub const entries = [_]Entry{
    .{
        .name = "cuda window_vonly keys tf_vonly_keys, 4 key heads of 512 (Gemma 4 31B), pages of 256",
        .kind = .keys,
        .tok = 256,
        .served = keys_served,
    },
    .{
        .name = "cuda window_vonly keys tf_vonly_keys, 4 key heads of 512, pages of 37",
        .kind = .keys,
        .tok = 37,
        .served = keys_served,
    },
    .{
        .name = "control: tf_vonly_keys with one bit of one made key flipped",
        .kind = .keys,
        .tok = 256,
        .ctl = .flip,
        .served = keys_served,
    },
    .{
        .name = "control: keys without the bf16 rounding of w * V",
        .kind = .keys,
        .tok = 256,
        .ctl = .unrounded,
        .served = keys_served,
    },
    .{ .name = decode ++ "32/4 heads of 512, pages of 256", .kind = .decode, .tok = 256, .served = decode_served },
    .{ .name = decode ++ "32/4 heads of 512, pages of 37", .kind = .decode, .tok = 37, .served = decode_served },
    .{
        .name = decode ++ "10/2 heads of 512 (GQA 5), pages of 37",
        .kind = .decode,
        .h = 10,
        .hk = 2,
        .tok = 37,
        .served = gqa5_served,
    },
    .{
        .name = "control: V-only decode whose keys take another k norm's weight",
        .kind = .decode,
        .tok = 37,
        .ctl = .norm,
        .served = decode_served,
    },
};

fn none(_: c.Shape) usize {
    return 0;
}

pub fn spec(e: Entry) c.Spec {
    const what = switch (e.kind) {
        .keys => "qk_rows' tf_qk_rows_round K (Gemma 4's global layers), placed by tf_wrows_put",
        .decode => "tf_wrows_part_512_pg + tf_wrows_merge_512_pg over K+V pages of qk_round's keys",
    };
    const rule = "the K+V path's bits: NaN, infinities, signed zeros and subnormals included";
    const stated: c.Nan = .{ .stated = rule };
    return .{
        .name = e.name,
        .accum = .none,
        .depth = none,
        .out = .bf16,
        .replaces = what,
        .nan = stated,
        .served = e.served,
    };
}

/// The kernels a registration runs: the set's, and the controls' image for its planted kernel.
pub const Fns = struct {
    win: *const wa.WindowAttention,
    vo: *const VOnly,
    qk: *const qk.QkRows,
    unrounded: cuda.Function,
};

/// A registration's device buffers: q, the V-only pages, rows' positions and slots, partials, the output.
const Dev = struct { q: u64 = 0, vo: u64 = 0, pos: u64 = 0, slot: u64 = 0, scratch: u64 = 0, out: u64 = 0 };

pub const VoRow = struct {
    rig: rig_.Rig,
    e: Entry,
    f: *const Fns,
    arena: std.heap.ArenaAllocator,
    pg: wa.Pages = undefined,
    w: wa.Window = undefined,
    vk: Vk = undefined,
    other: Vk = undefined,
    dev: Dev = .{},
    k0: usize = 0,
    keys: usize = 0,
    want: []const u8 = &.{},
    placed: ?[]const u8 = null,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *VoRow) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *VoRow) g.Kernel {
        return .{ .spec = spec(self.e), .ctx = self, .vt = &vt, .control = if (self.e.ctl != .none) .shapes else null };
    }

    const vt = g.hooks(VoRow);

    /// bf16 bits of 0.25 .. 1.75: a k norm's weight row.
    fn norm(self: *VoRow, rng: std.Random) !u64 {
        var x: [d]u16 = undefined;
        for (&x) |*v| v.* = core.bf16_round.bits(0.25 + 1.5 * rng.float(f32));
        return self.rig.input(std.mem.sliceAsBytes(&x));
    }

    /// k and v rows [n, hk, 512] as the served writer makes them (tf_qk_rows_round, v from k) at `pos`.
    fn written(self: *VoRow, a: std.mem.Allocator, rng: std.Random, pos: []const i32, edges: bool) !Kv {
        const e = self.e;
        const n = pos.len;
        const raw_q = try a.alloc(u16, n * e.hk * d); // q heads: as many as key heads (only k and v are read)
        const raw_k = try a.alloc(u16, n * e.hk * d);
        var i: usize = 0;
        while (i < raw_q.len) : (i += d) wc.row(rng, raw_q[i..][0..d], 1.0 + rng.float(f32));
        i = 0;
        while (i < raw_k.len) : (i += d) wc.row(rng, raw_k[i..][0..d], 0.5 + 4 * rng.float(f32));
        if (edges) plantEdges(raw_k);
        var inv_h: [d / 2]f32 = undefined;
        core.rope_freq.proportional(1000000.0, d, 64, &inv_h); // Gemma 4's global table (cuda_forward)
        const inv = try self.rig.input(std.mem.sliceAsBytes(&inv_h));
        const qn = try self.norm(rng);
        self.vk = .{ .w = try self.norm(rng), .inv = inv };
        self.other = .{ .w = try self.norm(rng), .inv = inv };
        const dk = try self.rig.input(std.mem.sliceAsBytes(raw_k));
        const qw: i64 = @intCast(e.hk * d);
        const kw: i64 = @intCast(e.hk * d);
        const dq = try self.rig.input(std.mem.sliceAsBytes(raw_q));
        const src: qk.Src = .{
            .q = dq,
            .q_ld = qw,
            .q_hs = d,
            .k = dk,
            .k_ld = kw,
            .k_hs = d,
            .v = dk,
            .v_ld = kw,
            .v_hs = d,
        };
        const shape: qk.Shape = .{
            .norms = qn,
            .norm_k = self.vk.w,
            .norms_bf16 = true,
            .inv = inv,
            .rot = d,
            .eps = 1e-6,
            .round = true,
            .v_norm = true,
        };
        const bytes = n * e.hk * d * 2;
        const out: qk.Out = .{
            .q = try self.rig.scratch(bytes),
            .k = try self.rig.scratch(bytes),
            .v = try self.rig.scratch(bytes),
        };
        const dpos = try self.rig.input(std.mem.sliceAsBytes(pos));
        try self.f.qk.rowsFrom(self.rig.s, src, .{ .h = e.hk, .hk = e.hk, .dim = d }, shape, dpos, null, out, n);
        return .{ .k = out.k, .v = out.v };
    }

    /// An arena of `pages` pages, every byte 0xff (an unwritten row reads as NaN).
    fn pagesOf(self: *VoRow, pages: usize) !u64 {
        const bytes = pages * self.pg.page_elems * 2;
        const p = try self.rig.scratch(bytes);
        const buf: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = p, .len = bytes, .borrowed = true };
        try buf.fill8(0xff, self.rig.s.handle);
        return p;
    }

    pub fn prepare(self: *VoRow, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const e = self.e;
        if (shape.rows == 0 or (e.kind == .decode and (shape.n != e.h or shape.k == 0))) return false;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const streams: usize = if (e.kind == .keys) 1 else 2;
        const top = if (e.kind == .keys) shape.k + shape.rows else shape.k; // keys a stream's table covers
        const first = if (e.kind == .keys) shape.k else 0;
        const stride = (top + e.tok - 1) / e.tok;
        const lo = first / e.tok;
        const pages = streams * (stride - lo) + 3; // the pages written, and three no stream holds
        const ids = try a.alloc(i32, pages);
        for (ids, 0..) |*v, i| v.* = @intCast(i);
        rng.shuffle(i32, ids);
        const table = try a.alloc(i32, streams * stride);
        @memset(table, 0);
        for (0..streams) |sl| for (lo..stride) |j| {
            table[sl * stride + j] = ids[sl * (stride - lo) + j - lo];
        };
        self.pg = .{
            .table = try self.rig.input(std.mem.sliceAsBytes(table)),
            .page_elems = e.tok * e.hk * d,
            .tokens = e.tok,
            .stride = stride,
        };
        self.w = .{ .h = e.h, .hk = e.hk, .d = d, .win = 0, .ring = top, .scale = 1, .pg = self.pg };
        const n = streams * (top - first);
        const pos = try a.alloc(i32, n);
        const slot = try a.alloc(i32, n);
        for (pos, slot, 0..) |*p, *sl, i| {
            p.* = @intCast(first + i % (top - first));
            sl.* = @intCast(i / (top - first));
        }
        const kv = try self.written(a, rng, pos, fill == .nonfinite);
        const dpos = try self.rig.input(std.mem.sliceAsBytes(pos));
        const dslot = try self.rig.input(std.mem.sliceAsBytes(slot));
        const kc = try self.pagesOf(pages);
        const vc = try self.pagesOf(pages);
        self.dev.vo = try self.pagesOf(pages);
        try self.f.win.put(self.rig.s, kv.k, kv.v, kc, vc, dpos, dslot, self.w, n);
        try self.f.vo.put(self.rig.s, kv.v, self.dev.vo, dpos, dslot, self.w, n);
        const bytes = pages * self.pg.page_elems * 2;
        const want_v = try a.dupe(u8, try self.rig.bytes(vc, bytes));
        const same = std.mem.eql(u8, want_v, try self.rig.bytes(self.dev.vo, bytes));
        self.placed = if (same) null else "tf_wrows_put_v placed V differently from tf_wrows_put";
        self.k0 = first;
        self.keys = shape.rows;
        if (e.kind == .keys) {
            self.want = try a.dupe(u8, try self.rig.bytes(kc, bytes));
            self.dev.out = try self.rig.output(bytes);
            return true;
        }
        const q = try a.alloc(u16, shape.rows * e.h * d);
        var i: usize = 0;
        while (i < q.len) : (i += d) wc.row(rng, q[i..][0..d], 1.0);
        if (fill == .nonfinite and shape.rows > 1) q[e.h * d + 7] = 0x7f80; // row 1: +inf, the merge's float64 replay
        const rpos = try a.alloc(i32, shape.rows);
        const rslot = try a.alloc(i32, shape.rows);
        for (rpos, rslot, 0..) |*p, *sl, r| { // the first key alone, the cache's end, scattered
            p.* = @intCast(if (r == 0) 0 else if (r == shape.rows - 1) top - 1 else (r * 7919) % top);
            sl.* = @intCast(r % streams);
        }
        self.dev.q = try self.rig.input(std.mem.sliceAsBytes(q));
        self.dev.pos = try self.rig.input(std.mem.sliceAsBytes(rpos));
        self.dev.slot = try self.rig.input(std.mem.sliceAsBytes(rslot));
        self.dev.scratch = try self.rig.scratch(wa.WindowAttention.scratchBytes(self.w, shape.rows));
        const base = try self.rig.scratch(shape.rows * e.h * d * 2);
        try self.f.win.decode(
            self.rig.s,
            self.dev.q,
            kc,
            vc,
            base,
            self.dev.pos,
            self.dev.slot,
            self.w,
            self.dev.scratch,
            shape.rows,
        );
        self.want = try a.dupe(u8, try self.rig.bytes(base, shape.rows * e.h * d * 2));
        self.dev.out = try self.rig.output(shape.rows * e.h * d * 2);
        return true;
    }

    pub fn run(self: *VoRow) !void {
        const e = self.e;
        const s = self.rig.s;
        switch (e.kind) {
            .keys => {
                const f = if (e.ctl == .unrounded) self.f.unrounded else self.f.vo.keys_fn;
                const x: VOnly.Keys = .{
                    .vc = self.dev.vo,
                    .ks = self.dev.out,
                    .pg = self.pg,
                    .hk = e.hk,
                    .vk = self.vk,
                };
                try VOnly.keysWith(f, s, x, self.k0, self.keys);
                if (e.ctl == .flip) { // the planted flip: the lowest bit of the first made key's dim 3
                    var two: [2]u8 = undefined;
                    const row = @as(usize, @intCast(try self.firstPage())) * self.pg.tokens + self.k0 % e.tok;
                    const at = row * e.hk * d * 2 + 6;
                    const buf: cuda.DeviceBuffer = .{
                        .d = self.rig.gpu.d,
                        .ptr = self.dev.out,
                        .len = at + 2,
                        .borrowed = true,
                    };
                    try s.synchronize();
                    try buf.download(at, &two);
                    two[0] ^= 1;
                    try buf.upload(at, &two);
                }
            },
            .decode => {
                const vk = if (e.ctl == .norm) self.other else self.vk;
                try self.f.vo.decode(
                    s,
                    self.dev.q,
                    self.dev.vo,
                    self.dev.out,
                    self.dev.pos,
                    self.dev.slot,
                    self.w,
                    vk,
                    self.dev.scratch,
                    self.keys,
                );
            },
        }
        try s.synchronize();
    }

    /// The page id the stream's first key sits in.
    fn firstPage(self: *VoRow) !i32 {
        var id: [1]i32 = undefined;
        const b: cuda.DeviceBuffer = .{
            .d = self.rig.gpu.d,
            .ptr = self.pg.table,
            .len = self.pg.stride * 4,
            .borrowed = true,
        };
        try b.download(self.k0 / self.e.tok * 4, std.mem.sliceAsBytes(&id));
        return id[0];
    }

    pub fn count(self: *VoRow) usize {
        return self.want.len / 2;
    }

    pub fn reference(_: *VoRow, _: []f64, _: []f64) !void {
        return error.ReplacementUsesTheKvBits;
    }

    pub fn replaced(self: *VoRow) ![]const u8 {
        return self.want;
    }

    pub fn check(self: *VoRow, _: g.Gate) !?[]const u8 {
        if (self.placed) |why| return why;
        const got = try self.rig.outputBytes();
        const at = std.mem.indexOfDiff(u8, got, self.want) orelse return null;
        return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from the K+V path's", .{ at, got.len });
    }
};

/// Edge values in raw k rows, a row each in turn: +0s, -0s, a -0, subnormals, a subnormal V, NaN, +-inf, min normal.
fn plantEdges(raw: []u16) void {
    var r: usize = 0;
    while ((r + 1) * d <= raw.len) : (r += 1) {
        const x = raw[r * d ..][0..d];
        switch (r % 9) {
            0 => @memset(x, 0x0000),
            1 => @memset(x, 0x8000),
            2 => x[0] = 0x8000,
            3 => {
                x[5] = 0x0001;
                x[300] = 0x8001;
            },
            4 => x[1] = 0x000b, // ~1e-39: a subnormal V after the unit norm
            5 => x[3] = 0x7fc0,
            6 => x[9] = 0x7f80,
            7 => x[9] = 0xff80,
            else => x[64] = 0x0080, // the smallest normal, on an unrotated pair
        }
    }
}

/// The registrations, loaded with their kernels; each control after the kernels it breaks.
pub const Set = struct {
    gpa: std.mem.Allocator,
    win: wa.WindowAttention,
    vo: VOnly,
    qk: qk.QkRows,
    ctl: cuda.Module,
    fns: Fns = undefined,
    items: [entries.len]VoRow = undefined,

    const controls = struct {
        pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
    };

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var win = try wa.WindowAttention.load(gpu.d);
        errdefer win.unload();
        try win.loadPages(gpu.d);
        var vo = try VOnly.load(gpu.d);
        errdefer vo.unload();
        var k = try qk.QkRows.load(gpu.d);
        errdefer k.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .win = win, .vo = vo, .qk = k, .ctl = ctl };
        set.fns = .{
            .win = &set.win,
            .vo = &set.vo,
            .qk = &set.qk,
            .unrounded = try ctl.function("tf_ctl_vonly_keys_unrounded"),
        };
        for (&set.items, entries) |*x, e| x.* = .{
            .rig = .{ .gpu = gpu, .s = s },
            .e = e,
            .f = &set.fns,
            .arena = .init(gpu.gpa),
        };
        for (&set.items) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.items) |*x| x.deinit();
        set.ctl.unload();
        set.qk.unload();
        set.vo.unload();
        set.win.unload();
        set.gpa.destroy(set);
    }
};

test "every V-only control names its kernel's registration and plants one fault" {
    var kinds: [2]usize = .{ 0, 0 };
    for (entries) |e| {
        try std.testing.expectEqual(e.ctl != .none, std.mem.startsWith(u8, e.name, "control: "));
        if (e.ctl != .none) kinds[@backingInt(e.kind)] += 1;
        try std.testing.expect(e.ctl == .none or (e.ctl == .norm) == (e.kind == .decode));
    }
    try std.testing.expect(kinds[0] >= 2 and kinds[1] >= 1);
}
