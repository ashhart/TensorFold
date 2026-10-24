//! Contiguous prompt/verify attention is byte-equal to the union's general paging kernel on identical flat caches.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const e4m3 = core.quant.fp8.e4m3;

/// qwen35.cuh's HD and MAXCH, and Qwen3.8-27B's 24 query heads over 4 key heads.
const hd = 256;
const maxch = 64;
const qh = 24;
const kvh = 4;
const group = qh / kvh;
const scale: f32 = 1.0 / 16.0;
/// A stream's drafted window here (DFlash2's verify rows); past 64 rows the call is one prompt chunk's chain.
const width = 16;
const most_window = 64;

/// cuda_engine.zig's Geo, cuda_cache.zig's Seg and cuda_round.zig's Ancestor, as the kernels read them.
const Geo = extern struct { hid: c_int = 5120, nk: c_int = 16, nv: c_int = 48, qh: c_int = qh, kvh: c_int = kvh, nl: c_int = 1 };
const Seg = extern struct { rec: u64 = 0, conv: u64 = 0, kv: u64, tap: u64 = 0, log: u64 = 0, before: u64 = 0, cap: c_int, len: c_int, row0: c_int, rows: c_int, tap_rows: c_int = 0, taps: c_int = 0, delta: c_int = 0, pt: u64 = 0, ps: u64 = 0, ptok: c_int = 0 };
const Ancestor = extern struct { bits: u64, depth: u32, chain: u32 };

pub const Cache = enum { bf16, fp8 };

/// A registered form: its cache, and for the control a launch that drops each row's last chunk.
pub const Form = struct { cache: Cache, control: bool = false };
pub const forms = [_]Form{ .{ .cache = .bf16 }, .{ .cache = .fp8 }, .{ .cache = .bf16, .control = true } };

/// A row's merged output sums every key it sees, then the chunks' partials (at most 64).
fn keyChain(s: c.Shape) usize {
    return s.k + most_window + maxch;
}

/// Gate 4's live keys a stream: their sums stay below 2^24 in fp32; the rest score alike with values 0.
const stress_keys = 2048;

/// tf_q35_attn_merge skips a chunk whose l is not above 0, so a NaN there drops out rather than spreading.
const nan_key = "a NaN key in one of a row's several chunks leaves the row finite: tf_q35_attn_merge skips that chunk";

pub fn spec(f: Form) c.Spec {
    const name = switch (f.cache) {
        .bf16 => if (f.control) "control qwen35 contiguous attention dropping each row's last chunk" else "cuda qwen35 contiguous attention, bf16 cache (tf_q35_attn_partial_legacy, tf_q35_attn_merge)",
        .fp8 => "cuda qwen35 contiguous attention, fp8 cache (tf_q35_attn_partial8_legacy, tf_q35_attn_merge)",
    };
    // Verify windows, prompt chains, chunk-width seams and long cold prefixes keep their original key order.
    const served: []const c.Shape = &.{ .{ .rows = 16, .n = qh, .k = 2100 }, .{ .rows = 64, .n = qh, .k = 700 }, .{ .rows = 128, .n = qh, .k = 300 }, .{ .rows = 16, .n = qh, .k = 9000 }, .{ .rows = 1, .n = qh, .k = 32768 }, .{ .rows = 1, .n = qh, .k = 65536 } };
    return .{ .name = name, .accum = .f32, .depth = keyChain, .ops = 3, .out = .bf16, .replaces = "the union page-table kernel over the identical contiguous cache", .tiles = .{ .{ .size = width }, .{}, .{ .size = 128 } }, .served = served, .sums = .{ .why = attn.why }, .known = .{ .gate = .nonfinite, .finding = nan_key } };
}

pub const Q35 = struct {
    rig: rig_.Rig,
    form: Form,
    partial: cuda.Function,
    baseline: cuda.Function,
    merge: cuda.Function,
    r: ?attn.Ref = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    lens: std.ArrayList(usize) = .empty,
    firsts: std.ArrayList(usize) = .empty,
    rowseg: std.ArrayList(i32) = .empty,
    anc: std.ArrayList(Ancestor) = .empty,
    q: std.ArrayList(u16) = .empty,
    kv: std.ArrayList(std.ArrayList(u8)) = .empty,
    dev: [3]u64 = @splat(0),
    segs: u64 = 0,
    part: [3]u64 = @splat(0),
    base: [4]u64 = @splat(0),
    out: u64 = 0,
    chunks: usize = 0,

    pub fn deinit(self: *Q35) void {
        const a = self.rig.gpu.gpa;
        if (self.r) |*r| r.deinit();
        for (self.kv.items) |*x| x.deinit(a);
        self.kv.deinit(a);
        inline for (.{ "lens", "firsts", "rowseg", "anc", "q" }) |f| @field(self, f).deinit(a);
        self.rig.deinit();
    }

    pub fn kernel(self: *Q35) g.Kernel {
        return .{ .spec = spec(self.form), .ctx = self, .vt = &vt, .control = if (self.form.control) .shapes else null };
    }

    const vt = g.hooks(Q35);

    fn cap(self: *const Q35, si: usize) usize {
        return self.lens.items[si] + self.rows(si);
    }

    fn rows(self: *const Q35, si: usize) usize {
        const end = if (si + 1 < self.firsts.items.len) self.firsts.items[si + 1] else self.shape.rows;
        return end - self.firsts.items[si];
    }

    /// A stream's cache bytes for one layer: keys then values (bf16), or codes then fp32 scales (fp8).
    fn cacheBytes(self: *const Q35, si: usize) usize {
        const n = kvh * self.cap(si);
        return switch (self.form.cache) {
            .bf16 => 2 * n * hd * 2,
            .fp8 => 2 * n * hd + 2 * n * 4,
        };
    }

    pub fn prepare(self: *Q35, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (shape.n != qh or shape.rows == 0 or shape.k == 0) return false;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        const n = shape.rows;
        const chain = n > most_window;
        const ns = if (chain) 1 else (n + width - 1) / width;
        try self.lens.resize(a, ns);
        try self.firsts.resize(a, ns);
        for (self.lens.items, self.firsts.items, 0..) |*l, *f, si| {
            l.* = @max(1, shape.k - si * shape.k / ns);
            f.* = si * width;
        }
        try self.rowseg.resize(a, n);
        try self.anc.resize(a, n);
        var most: usize = 0;
        self.chunks = 0;
        for (0..ns) |si| {
            const f = self.firsts.items[si];
            for (0..self.rows(si)) |j| {
                self.rowseg.items[f + j] = @intCast(si);
                if (chain) {
                    self.anc.items[f + j] = .{ .bits = 0, .depth = @intCast(j), .chain = 1 };
                } else {
                    const parent: ?usize = if (j == 0 or rng.uintLessThan(u32, 6) == 0) null else rng.uintLessThan(usize, j);
                    const prior: u64 = if (parent) |p| self.anc.items[f + p].bits else 0;
                    self.anc.items[f + j] = .{ .bits = prior | (@as(u64, 1) << @intCast(j)), .depth = 0, .chain = 0 };
                }
                const keys = self.lens.items[si] + seen(self.anc.items[f + j]);
                const ck = @max(128, (keys + maxch * 128 - 1) / (maxch * 128) * 128);
                self.chunks = @max(self.chunks, (keys + ck - 1) / ck);
                most = @max(most, keys);
            }
        }
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.r = try attn.Ref.init(a, hd, most);
        try self.q.resize(a, n * qh * hd);
        for (self.q.items) |*x| x.* = attn.normal(rng);
        for (self.kv.items) |*x| x.deinit(a);
        self.kv.clearRetainingCapacity();
        for (0..ns) |si| {
            var buf: std.ArrayList(u8) = .empty;
            try buf.resize(a, self.cacheBytes(si));
            try self.kv.append(a, buf);
            self.fillRandom(si, rng);
        }
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => try self.stress(rng),
        }
        const segs = try a.alloc(Seg, ns);
        defer a.free(segs);
        for (segs, 0..) |*s, si| s.* = .{ .kv = try self.rig.input(self.kv.items[si].items), .cap = @intCast(self.cap(si)), .len = @intCast(self.lens.items[si]), .row0 = @intCast(self.firsts.items[si]), .rows = @intCast(self.rows(si)) };
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.q.items));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.rowseg.items));
        self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.anc.items));
        self.segs = try self.rig.input(std.mem.sliceAsBytes(segs));
        const slots = n * qh * maxch;
        for (&self.part, [_]usize{ 4, 4, 4 * hd }) |*p, w| p.* = try self.rig.scratch(slots * w);
        for (self.base[0..3], [_]usize{ 4, 4, 4 * hd }) |*p, w| p.* = try self.rig.scratch(slots * w);
        self.base[3] = try self.rig.scratch(n * qh * hd * 2);
        self.out = try self.rig.output(n * qh * hd * 2);
        return true;
    }

    fn seen(anc: Ancestor) usize {
        return if (anc.chain != 0) anc.depth + 1 else @popCount(anc.bits);
    }

    fn fillRandom(self: *Q35, si: usize, rng: std.Random) void {
        const b = self.kv.items[si].items;
        switch (self.form.cache) {
            .bf16 => for (std.mem.bytesAsSlice(u16, b)) |*x| {
                x.* = attn.normal(rng);
            },
            .fp8 => {
                const n = kvh * self.cap(si);
                for (b[0 .. 2 * n * hd]) |*x| {
                    const code = rng.int(u8);
                    x.* = if (code & 0x7f == 0x7f) code ^ 1 else code; // no NaN codes
                }
                for (std.mem.bytesAsSlice(f32, b[2 * n * hd ..])) |*s| s.* = 3.0 / 448.0 * (0.5 + rng.float(f32));
            },
        }
    }

    /// Key head h's key (or value) slot `pos` of stream si: its hd elements as f64, scale applied.
    fn row(self: *const Q35, si: usize, value: bool, h: usize, pos: usize, out: []f64) void {
        const b = self.kv.items[si].items;
        const n = kvh * self.cap(si);
        const r = h * self.cap(si) + pos + @as(usize, if (value) n else 0);
        switch (self.form.cache) {
            .bf16 => {
                const x = std.mem.bytesAsSlice(u16, b);
                for (out, 0..) |*o, e| o.* = c.fromBf16(x[r * hd + e]);
            },
            .fp8 => {
                const s = std.mem.bytesAsSlice(f32, b[2 * n * hd ..])[r];
                for (out, 0..) |*o, e| o.* = e4m3(b[r * hd + e]) * s;
            },
        }
    }

    /// Writes element e of key head h's key (or value) slot `pos` of stream si (bf16: the value; fp8: the scale).
    fn put(self: *Q35, si: usize, value: bool, h: usize, pos: usize, e: usize, v: f32) void {
        const b = self.kv.items[si].items;
        const n = kvh * self.cap(si);
        const r = h * self.cap(si) + pos + @as(usize, if (value) n else 0);
        switch (self.form.cache) {
            .bf16 => std.mem.bytesAsSlice(u16, b)[r * hd + e] = c.toBf16(v),
            .fp8 => std.mem.bytesAsSlice(f32, b[2 * n * hd ..])[r] = v,
        }
    }

    /// NaN in a query, +inf and -inf values, both in one column (bf16), and a NaN key in a row's last chunk.
    fn plant(self: *Q35) void {
        const p0 = self.lens.items[0];
        self.q.items[3] = c.toBf16(std.math.nan(f32)); // row 0, head 0
        self.put(0, true, 0, p0 - 1, 7, std.math.inf(f32));
        if (self.form.cache == .bf16 and p0 >= 3) {
            self.put(0, true, 1, p0 - 2, 11, std.math.inf(f32));
            self.put(0, true, 1, p0 - 3, 11, -std.math.inf(f32));
            self.put(0, true, 2, p0 - 2, 5, -std.math.inf(f32));
        }
        self.put(0, false, 3, p0 - 1, 0, std.math.nan(f32));
    }

    /// Gate 4: every key of a head alike (each p is 1), committed values mirrored per column, window values 0.
    fn stress(self: *Q35, rng: std.Random) !void {
        const a = self.rig.gpu.gpa;
        const col = try a.alloc(f32, self.r.?.keys.len);
        defer a.free(col);
        for (self.kv.items, 0..) |*buf, si| {
            const cp = self.cap(si);
            const p = @min(self.lens.items[si], stress_keys);
            const b = buf.items;
            for (0..kvh) |h| {
                switch (self.form.cache) {
                    .bf16 => {
                        const x = std.mem.bytesAsSlice(u16, b);
                        for (1..cp) |j| @memcpy(x[(h * cp + j) * hd ..][0..hd], x[h * cp * hd ..][0..hd]);
                        for (0..hd) |e| {
                            attn.mirrored(rng, col[0..p]);
                            for (0..cp) |j| x[(kvh * cp + h * cp + j) * hd + e] = if (j < p) c.toBf16(col[j]) else 0;
                        }
                    },
                    .fp8 => {
                        const n = kvh * cp;
                        const s = std.mem.bytesAsSlice(f32, b[2 * n * hd ..]);
                        for (1..cp) |j| @memcpy(b[(h * cp + j) * hd ..][0..hd], b[h * cp * hd ..][0..hd]);
                        const half = @max(1, p / 2);
                        for (0..cp) |j| {
                            s[h * cp + j] = s[h * cp];
                            s[n + h * cp + j] = if (j < 2 * half and j % half % 2 == 1) 1024 else 1;
                        }
                        for (0..hd) |e| {
                            mirroredE4m3(rng, col[0..p]);
                            for (0..cp) |j| b[(n + h * cp + j) * hd + e] = if (j < p) toCode(col[j]) else 0;
                        }
                    },
                }
            }
        }
    }

    pub fn run(self: *Q35) !void {
        const z = if (self.form.control) @max(1, self.chunks - 1) else maxch;
        try self.launchPartial(self.partial, self.shape.rows, 0, z, self.part);
        try self.launchMerge(self.part, self.out);
        try self.rig.s.synchronize();
    }

    /// Rows [m0, m0 + n) through the partial kernel, chunks fastest, into partials `p` (row m0's slots at m0).
    fn launchPartial(self: *Q35, fn_: cuda.Function, n: usize, m0: usize, z: usize, p: [3]u64) !void {
        var args: cuda.Args = .{};
        args.add(Geo{});
        args.add(self.dev[0] + m0 * qh * hd * 2);
        args.add(self.segs);
        args.add(self.dev[1] + m0 * 4);
        args.add(self.dev[2] + m0 * @sizeOf(Ancestor));
        args.add(@as(c_int, 0));
        args.add(@as(u64, 0));
        args.add(scale);
        const slots = m0 * qh * maxch;
        args.add(p[0] + slots * 4);
        args.add(p[1] + slots * 4);
        args.add(p[2] + slots * hd * 4);
        try cuda.launch.launch(fn_, .{ .grid = .{ .x = @intCast(z), .y = kvh, .z = @intCast(n) }, .block = .{ .x = 32 * group } }, self.rig.s, &args);
    }

    fn launchMerge(self: *Q35, p: [3]u64, out: u64) !void {
        var args: cuda.Args = .{};
        args.add(Geo{});
        for (p) |x| args.add(x);
        args.add(self.segs);
        args.add(self.dev[1]);
        args.add(self.dev[2]);
        args.add(@as(u64, 0));
        args.add(out);
        try cuda.launch.launch(self.merge, .{ .grid = .{ .x = qh, .y = @intCast(self.shape.rows) }, .block = .{ .x = 32 } }, self.rig.s, &args);
    }

    /// Run the original general paging entry point over precisely the same inputs.
    pub fn replaced(self: *Q35) ![]const u8 {
        const p = self.base[0..3].*;
        try self.launchPartial(self.baseline, self.shape.rows, 0, maxch, p);
        try self.launchMerge(p, self.base[3]);
        return self.rig.bytes(self.base[3], self.count() * 2);
    }

    pub fn count(self: *Q35) usize {
        return self.shape.rows * qh * hd;
    }

    /// Output head i (row i / qh, head i % qh): its stream's committed keys, then the window keys it sees in order.
    pub fn load(self: *Q35, i: usize, h: *attn.Head) !void {
        const m = i / qh;
        const head = i % qh;
        const kh = head / group;
        const si: usize = @intCast(self.rowseg.items[m]);
        const p = self.lens.items[si];
        const anc = self.anc.items[m];
        for (h.q, 0..) |*x, e| x.* = c.fromBf16(self.q.items[(m * qh + head) * hd + e]);
        const n = p + seen(anc);
        var bits = anc.bits;
        for (0..n) |t| {
            const pos = if (t < p) t else if (anc.chain != 0) t else blk: {
                const at = p + @ctz(bits);
                bits &= bits - 1;
                break :blk at;
            };
            self.row(si, false, kh, pos, h.k[t * hd ..][0..hd]);
            self.row(si, true, kh, pos, h.v[t * hd ..][0..hd]);
        }
        h.n = n;
    }

    fn heads(self: *Q35) usize {
        return self.shape.rows * qh;
    }

    pub fn check(self: *Q35, at: g.Gate) !?[]const u8 {
        return self.r.?.check(self, self.heads(), scale, try self.rig.outputBytes(), at);
    }

    pub fn reference(self: *Q35, want: []f64, mag: []f64) !void {
        return self.r.?.reference(self, self.heads(), scale, want, mag);
    }

    pub fn simulate(self: *Q35, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(self, self.heads(), scale, acc, order, out);
    }
};

/// Gate 4's fp8 codes: odd integers to 15 (odd keys scaled by 1024), the second half cancelling the first.
fn mirroredE4m3(rng: std.Random, out: []f32) void {
    const h = out.len / 2;
    for (0..h) |i| {
        const v: f32 = @floatFromInt(1 + 2 * rng.uintLessThan(u32, 8));
        out[i] = v;
        out[h + i] = (if (i < 8 and i % 2 == 0) @as(f32, 1) else 0) - v;
    }
    if (out.len % 2 == 1) out[out.len - 1] = 0;
}

/// The e4m3 code of an integer in [-15, 15].
fn toCode(v: f32) u8 {
    var code: u8 = 0;
    while (code < 0x7f) : (code += 1) if (e4m3(code) == @abs(v)) break;
    return code | @as(u8, if (v < 0) 0x80 else 0);
}

test "toCode inverts e4m3 on the stress integers" {
    var v: f32 = -15;
    while (v <= 15) : (v += 1) try std.testing.expectEqual(@as(f64, v), e4m3(toCode(v)));
}
