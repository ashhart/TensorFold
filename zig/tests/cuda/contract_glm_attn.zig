//! GLM's MLA chunk kernels on KV pages under the contract on CUDA: dense and sparse, each run through the chunk merge.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const rig_ = @import("contract_rig.zig");
const pages = @import("contract_glm_pages.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const lw = pages.lw;
const heads = 32;
const chunk = glm.latent.chunk;
const scale: f64 = 1.0 / 16.0; // qk 256's 1 / sqrt: exact in fp32
const zero_sentinel: u8 = 0x5a;
const soft_why = "online softmax over 512-key chunks, then the chunk merge: no input makes its sums exact in fp32";

/// A key's 512 products, then a chunk's 512 keys and the merge's chunks.
fn attnDepth(s: c.Shape) usize {
    return lw + chunk + s.k / chunk + 1;
}

pub const Form = struct { hb: u32, sparse: bool, fp8: bool = false };

/// rows x a row's outputs (heads x 512) x, dense, the page edge its rows straddle; sparse, a row's tokens.
pub fn spec(comptime f: Form) c.Spec {
    const name = std.fmt.comptimePrint("cuda glm tf_glm_chunks_{d}{s} {s} on pages out of order, with tf_glm_merge", .{ f.hb, if (f.fp8) "_fp8_paged" else "", if (f.sparse) "sparse" else "dense" });
    const served: []const c.Shape = switch (f.hb) {
        16 => if (f.sparse) &.{.{ .rows = 8, .n = heads * lw, .k = glm.sparse.width }} else &.{.{ .rows = 8, .n = heads * lw, .k = 2 * chunk }},
        else => if (f.sparse) &.{.{ .rows = 64, .n = heads * lw, .k = 300 }} else &.{.{ .rows = 64, .n = heads * lw, .k = chunk }},
    };
    return .{ .name = name, .accum = .f32, .depth = attnDepth, .ops = 8, .out = .bf16, .served = served, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = soft_why } };
}

/// One chunk kernel's calls: queries, the cache's rows on pages out of order, and for sparse each row's token list.
pub const Chunks = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    wrong_table: bool = false,
    merge: cuda.Function,
    form: Form,
    shape: c.Shape = undefined,
    q: []u16 = &.{},
    cache: []u16 = &.{}, // the latent rows in row order, as the pages hold them
    keys: []u32 = &.{}, // sparse: each row's tokens [rows][k]
    tol: []f64 = &.{},
    ids: []i32 = &.{},
    at: usize = 0,
    nch: usize = 0,
    in: [4]u64 = @splat(0),
    scales: u64 = 0,
    pos: u64 = 0,
    pt: u64 = 0,
    page_bytes: usize = pages.page_bytes,
    parts: [3]u64 = @splat(0),
    out: u64 = 0,

    pub fn deinit(self: *Chunks) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Chunks) void {
        const a = self.rig.gpu.gpa;
        a.free(self.q);
        a.free(self.cache);
        a.free(self.keys);
        a.free(self.tol);
        a.free(self.ids);
        self.q = &.{};
        self.cache = &.{};
        self.keys = &.{};
        self.tol = &.{};
        self.ids = &.{};
    }

    pub fn kernel(self: *Chunks) g.Kernel {
        var declared = switch (self.form.hb) {
            16 => if (self.form.sparse)
                if (self.form.fp8) spec(.{ .hb = 16, .sparse = true, .fp8 = true }) else spec(.{ .hb = 16, .sparse = true })
            else if (self.form.fp8) spec(.{ .hb = 16, .sparse = false, .fp8 = true }) else spec(.{ .hb = 16, .sparse = false }),
            else => if (self.form.sparse)
                if (self.form.fp8) spec(.{ .hb = 32, .sparse = true, .fp8 = true }) else spec(.{ .hb = 32, .sparse = true })
            else if (self.form.fp8) spec(.{ .hb = 32, .sparse = false, .fp8 = true }) else spec(.{ .hb = 32, .sparse = false }),
        };
        if (self.wrong_table) declared.name = if (self.form.fp8) switch (self.form.hb) {
            16 => if (self.form.sparse)
                "control: GLM chunks16_fp8_paged sparse uses swapped page rows"
            else
                "control: GLM chunks16_fp8_paged dense uses a swapped table",
            else => if (self.form.sparse)
                "control: GLM chunks32_fp8_paged sparse uses swapped page rows"
            else
                "control: GLM chunks32_fp8_paged dense uses a swapped table",
        } else switch (self.form.hb) {
            16 => if (self.form.sparse)
                "control: GLM chunks16 sparse uses swapped page rows"
            else
                "control: GLM chunks16 dense uses a swapped table",
            else => if (self.form.sparse)
                "control: GLM chunks32 sparse uses swapped page rows"
            else
                "control: GLM chunks32 dense uses a swapped table",
        };
        return .{ .spec = declared, .ctx = self, .vt = &vt, .control = if (self.wrong_table) .shapes else null };
    }

    const vt = g.hooks(Chunks);

    /// Keys row r sees: dense the rows through at + r, sparse its tokens.
    fn seen(self: *const Chunks, r: usize) usize {
        return if (self.form.sparse) self.shape.k else self.at + r + 1;
    }

    fn key(self: *const Chunks, r: usize, i: usize) usize {
        return if (self.form.sparse) self.keys[r * self.shape.k + i] else i;
    }

    fn zeroCountRow(self: *const Chunks) ?usize {
        return if (self.form.fp8 and self.form.sparse) self.shape.rows - 1 else null;
    }

    fn activeRows(self: *const Chunks) usize {
        return self.shape.rows - @intFromBool(self.zeroCountRow() != null);
    }

    /// Queries and rows in [-1, 1); dense from row k - rows / 2, across a page edge; sparse tokens under 4,096.
    pub fn prepare(self: *Chunks, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != heads * lw or shape.rows == 0 or shape.k == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const span: usize = if (self.form.sparse) 4096 else shape.k + shape.rows;
        self.at = if (self.form.sparse) 0 else shape.k - @min(shape.k, shape.rows / 2);
        self.nch = ((if (self.form.sparse) shape.k else self.at + shape.rows) + chunk - 1) / chunk;
        self.q = try a.alloc(u16, shape.rows * heads * lw);
        self.cache = try a.alloc(u16, span * lw);
        for (self.q) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (self.cache) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        if (fill == .nonfinite) {
            self.q[3] = c.toBf16(std.math.nan(f32));
            if (!self.form.sparse and !self.form.fp8) self.cache[9] = c.toBf16(std.math.inf(f32));
        }
        if (self.form.sparse) {
            self.keys = try a.alloc(u32, shape.rows * shape.k);
            for (0..shape.rows) |r| for (0..shape.k) |i| {
                self.keys[r * shape.k + i] = @intCast((i * 4096 / shape.k + rng.uintLessThan(usize, 4096 / shape.k)) % 4096);
            };
            if (fill == .nonfinite and shape.rows > 1 and !self.form.fp8) self.cache[self.keys[shape.k] * lw + 9] = c.toBf16(std.math.inf(f32));
        }
        const n_pages = (span + pages.rows_a_page - 1) / pages.rows_a_page;
        self.ids = try pages.shuffled(a, rng, n_pages);
        const pool_bytes: usize = (pages.rows_a_page / 4) * pages.dk * 2;
        const part_bytes: usize = if (self.form.fp8) pages.rows_a_page * (lw + 4) + pool_bytes else pages.part_bytes;
        self.page_bytes = 2 * part_bytes;
        const arena = try a.alloc(u8, (n_pages + 3) * self.page_bytes);
        defer a.free(arena);
        @memset(arena, 0);
        const encoded = try a.alloc(u8, if (self.form.fp8) span * lw else 0);
        defer a.free(encoded);
        const scale_values = try a.alloc(f32, if (self.form.fp8) span else 0);
        defer a.free(scale_values);
        for (0..span) |t| {
            const physical = @as(usize, @intCast(self.ids[t / pages.rows_a_page])) * self.page_bytes;
            if (self.form.fp8) {
                const codes = encoded[t * lw ..][0..lw];
                scale_values[t] = try glm.kv8_ref.pack(self.cache[t * lw ..][0..lw], codes);
                const code_at = part_bytes + physical + (t % pages.rows_a_page) * lw;
                @memcpy(arena[code_at..][0..lw], codes);
                const scale_at = part_bytes + physical + pages.rows_a_page * lw + (t % pages.rows_a_page) * 4;
                @memcpy(arena[scale_at..][0..4], std.mem.asBytes(&scale_values[t]));
                for (codes, 0..) |code, d| self.cache[t * lw + d] = glm.kv8_ref.decoded(code, scale_values[t]);
            } else {
                const at = pages.part_bytes + physical + (t % pages.rows_a_page) * lw * 2;
                @memcpy(arena[at..][0 .. lw * 2], std.mem.sliceAsBytes(self.cache[t * lw ..][0..lw]));
            }
        }
        const page_rows = self.page_bytes / (lw * 2);
        const tok = try a.alloc(i32, if (self.form.sparse) shape.rows * shape.k else 1);
        defer a.free(tok);
        const cnt = try a.alloc(i32, shape.rows);
        defer a.free(cnt);
        @memset(tok, 0);
        @memset(cnt, @intCast(shape.k));
        if (self.zeroCountRow()) |row| cnt[row] = 0;
        if (self.form.sparse) for (tok, self.keys) |*v, t| { // FP8-paged kernels consume logical positions; BF16 consumes physical rows.
            v.* = if (self.form.fp8) @intCast(t) else @intCast(@as(usize, @intCast(self.ids[t / pages.rows_a_page])) * page_rows + t % pages.rows_a_page);
        };
        if (self.wrong_table and self.form.sparse and !self.form.fp8) for (tok) |*v| {
            const row: usize = @intCast(v.*);
            v.* = @intCast(((row / page_rows + 1) % (n_pages + 3)) * page_rows + row % page_rows);
        };
        const p: i32 = @intCast(self.at);
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.q));
        self.in[1] = try self.rig.input(arena) + part_bytes;
        self.scales = if (self.form.fp8) self.in[1] + pages.rows_a_page * lw else 0;
        self.in[2] = try self.rig.input(std.mem.sliceAsBytes(tok));
        self.in[3] = try self.rig.input(std.mem.sliceAsBytes(cnt));
        self.pos = try self.rig.input(std.mem.asBytes(&p));
        self.pt = try pages.uploadTable(&self.rig, self.ids, self.wrong_table);
        const n = self.nch * shape.rows * heads;
        self.parts = .{ try self.rig.scratch(n * lw * 4), try self.rig.scratch(n * 4), try self.rig.scratch(n * 4) };
        const row_bytes = heads * lw * 2;
        self.out = try self.rig.output(shape.rows * row_bytes);
        if (self.zeroCountRow()) |row| {
            const sentinel: cuda.DeviceBuffer = .{
                .d = self.rig.gpu.d,
                .ptr = self.out + row * row_bytes,
                .len = row_bytes,
            };
            try sentinel.fill8(zero_sentinel, self.rig.s.handle);
        }
        return true;
    }

    pub fn run(self: *Chunks) !void {
        const rows: i64 = @intCast(self.shape.rows);
        const w: i64 = if (self.form.sparse) @intCast(self.shape.k) else 0;
        var a: cuda.Args = .{};
        a.add(self.in[0]);
        a.add(self.in[1]);
        if (self.form.fp8) a.add(self.scales);
        inline for (.{ self.pos, self.in[2], self.in[3], self.parts[0], self.parts[1], self.parts[2], rows, @as(i64, heads), w, @as(f32, @floatCast(scale)), @as(c_int, @intFromBool(self.form.sparse)), self.pt, @as(i64, @intCast(self.page_bytes)) }) |v| a.add(v);
        const shared = if (self.form.hb == 32) glm.glue.chunksSmem(32) else glm.glue.chunksSmem(16);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = (heads + self.form.hb - 1) / self.form.hb, .z = @intCast(self.nch) }, .block = .{ .x = 256 }, .shared = shared }, self.rig.s, &a);
        var m: cuda.Args = .{};
        inline for (.{ self.parts[0], self.parts[1], self.parts[2], self.out, self.in[3], rows, @as(i64, heads), @as(c_int, @intCast(self.nch)), @as(c_int, @intFromBool(self.form.sparse)) }) |v| m.add(v);
        try cuda.launch.launch(self.merge, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = heads }, .block = .{ .x = 128 } }, self.rig.s, &m);
        try self.rig.s.synchronize();
        if (self.zeroCountRow()) |row| {
            const row_bytes = heads * lw * 2;
            for (try self.rig.bytes(self.out + row * row_bytes, row_bytes)) |byte|
                if (byte != zero_sentinel) return error.ZeroCountRowChanged;
        }
    }

    pub fn count(self: *Chunks) usize {
        return self.activeRows() * heads * lw;
    }

    /// Row r, head h: softmax(q . c scale) . c over its keys, dots and sums in `acc` (null: float64), into `o`.
    fn attend(self: *Chunks, r: usize, h: usize, acc: ?c.Accum, order: g.Order, o: []f64, tol: ?[]f64) !void {
        const a = self.rig.gpu.gpa;
        const n = self.seen(r);
        const s = try a.alloc(f64, 3 * n);
        defer a.free(s);
        const ds = s[n .. 2 * n];
        const p = s[2 * n ..];
        var terms: [lw]f64 = undefined;
        var most = -std.math.inf(f64);
        for (0..n) |i| {
            var abs: f64 = 0;
            for (&terms, 0..) |*x, d| {
                x.* = c.fromBf16(self.q[(r * heads + h) * lw + d]) * c.fromBf16(self.cache[self.key(r, i) * lw + d]);
                abs += @abs(x.*);
            }
            s[i] = (if (acc) |k| pages.sumIn(k, order, &terms) else pages.sumIn(.none, .sequential, &terms)) * scale;
            ds[i] = abs * scale * 520 * 0x1p-24;
            most = pages.maxNan(most, s[i]);
        }
        for (p, s[0..n]) |*x, v| x.* = @exp(v - most);
        const l = if (acc) |k| pages.sumIn(k, order, p) else pages.sumIn(.none, .sequential, p);
        const col = try a.alloc(f64, n);
        defer a.free(col);
        for (o, 0..) |*od, d| {
            var spread: f64 = 0;
            for (col, p, 0..) |*x, w, i| {
                const v = c.fromBf16(self.cache[self.key(r, i) * lw + d]);
                x.* = w * v;
                spread += w * @abs(v) * (0x1p-8 + ds[i]);
            }
            od.* = (if (acc) |k| pages.sumIn(k, order, col) else pages.sumIn(.none, .sequential, col)) / l;
            if (tol) |t| t[d] = @abs(od.*) * 0x1p-8 + spread / l + 1e-30;
        }
    }

    /// float64 o = sum p c / sum p, p = exp(s - max); bound: bf16 out and p (2^-8 each), 520 2^-24 scale sum |q c|.
    pub fn reference(self: *Chunks, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        a.free(self.tol);
        self.tol = try a.alloc(f64, want.len);
        for (0..self.activeRows()) |r| for (0..heads) |h| {
            const at = (r * heads + h) * lw;
            try self.attend(r, h, null, .sequential, want[at..][0..lw], self.tol[at..][0..lw]);
        };
        for (mag, want) |*m, w| m.* = @abs(w);
    }

    pub fn bounds(self: *Chunks, _: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, self.tol) |*l, t| l.* = if (std.math.isFinite(t)) t else std.math.floatMax(f64);
    }

    pub fn simulate(self: *Chunks, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..self.activeRows()) |r| for (0..heads) |h| {
            const at = (r * heads + h) * lw;
            try self.attend(r, h, acc, order, out[at..][0..lw], null);
            for (out[at..][0..lw]) |*v| v.* = c.fromBf16(c.toBf16(@floatCast(v.*)));
        };
    }
};
