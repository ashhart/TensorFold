//! sparse_attention.cu under the contract on CUDA: rows over their own listed cache rows, a list a row and key head.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const tc = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const sparse = cuda.sparse_attention;

const D = 128;
const h = 48; // Kolibri 1.1's indexed layers: 48 query heads over 4 key heads
const hk = 4;
const G = h / hk;
const stride = hk * D + 64; // cache rows a little wider than the key heads, as a paged arena's may be

/// A listed key's P x V chain: its warp's slice of 32-key tiles, then the four slices' fold.
fn keyChain(s: c.Shape) usize {
    return s.k + 32 + 4;
}

/// How a test deviates from what it tells the kernel; `drop_last` is the planted control.
pub const Plant = enum { none, drop_last };

/// Shape: query rows (rows), query heads (n) and row 0's listed keys (k); later rows list fewer.
pub fn spec(plant: Plant) c.Spec {
    return .{
        .name = switch (plant) {
            .none => "cuda sparse attention tf_sparse_attention (Kolibri 1.1, 48 heads over 4)",
            .drop_last => "cuda sparse attention tf_sparse_attention, control: the kernel told one key fewer",
        },
        .accum = .f32,
        .depth = keyChain,
        .ops = 1,
        .out = .bf16,
        .tiles = .{ .{}, .{}, .{ .size = 32 } },
        // a decode row's 2,048 top keys, 128 local and 8 sinks; a verify window's rows; a prompt chunk's short lists
        .served = &.{ .{ .rows = 1, .n = h, .k = 2184 }, .{ .rows = 4, .n = h, .k = 700 }, .{ .rows = 17, .n = h, .k = 129 } },
        .sums = .{ .why = attn.why },
    };
}

pub const Sparse = struct {
    rig: rig_.Rig,
    sa: *const sparse.SparseAttention,
    plant_: Plant = .none,
    r: ?attn.Ref = null,
    rows: usize = 0,
    cap: usize = 0,
    pool: usize = 0, // cache rows; the last one is poison (NaN), where every unused list slot points
    counts: std.ArrayList(i32) = .empty,
    told: std.ArrayList(i32) = .empty,
    list: std.ArrayList(i32) = .empty,
    q: std.ArrayList(u16) = .empty,
    kc: std.ArrayList(u16) = .empty,
    vc: std.ArrayList(u16) = .empty,
    dev: [5]u64 = @splat(0),
    out: u64 = 0,

    fn scale() f32 {
        return @floatCast(1 / @sqrt(@as(f64, D)));
    }

    pub fn deinit(self: *Sparse) void {
        const a = self.rig.gpu.gpa;
        if (self.r) |*r| r.deinit();
        inline for (.{ "counts", "told", "list", "q", "kc", "vc" }) |f| @field(self, f).deinit(a);
        self.rig.deinit();
    }

    pub fn kernel(self: *Sparse) g.Kernel {
        return .{ .spec = spec(self.plant_), .ctx = self, .vt = &vt, .control = if (self.plant_ == .drop_last) .shapes else null };
    }

    const vt = g.hooks(Sparse);

    fn listed(self: *const Sparse, row: usize, kh: usize) usize {
        return @intCast(self.counts.items[row * hk + kh]);
    }

    /// List entry i of (row, kh): a cache row.
    fn entry(self: *const Sparse, row: usize, kh: usize, i: usize) usize {
        return @intCast(self.list.items[(row * hk + kh) * self.cap + i]);
    }

    fn kAt(self: *Sparse, cache_row: usize, kh: usize) []u16 {
        return self.kc.items[cache_row * stride + kh * D ..][0..D];
    }

    fn vAt(self: *Sparse, cache_row: usize, kh: usize) []u16 {
        return self.vc.items[cache_row * stride + kh * D ..][0..D];
    }

    fn qAt(self: *Sparse, row: usize, head: usize) []u16 {
        return self.q.items[(row * h + head) * D ..][0..D];
    }

    pub fn prepare(self: *Sparse, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (shape.n != h or shape.rows == 0 or shape.k == 0) return false;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.rows = shape.rows;
        self.cap = shape.k + 3; // slots past every count: they point at the poison row
        try self.counts.resize(a, self.rows * hk);
        // row j lists its own block of cache rows (each key head in its own order): gate 4 mirrors a list's values
        var first = try a.alloc(usize, self.rows + 1);
        defer a.free(first);
        first[0] = 0;
        for (0..self.rows) |j| {
            const n = @max(1, shape.k - j * shape.k / self.rows);
            for (0..hk) |kh| self.counts.items[j * hk + kh] = @intCast(if (kh == 3 and j > 0) @max(1, n / 2) else n);
            first[j + 1] = first[j] + n;
        }
        self.pool = first[self.rows] + 1;
        const poison = self.pool - 1;
        try self.list.resize(a, self.rows * hk * self.cap);
        @memset(self.list.items, @intCast(poison));
        for (0..self.rows) |j| for (0..hk) |kh| {
            const slots = self.list.items[(j * hk + kh) * self.cap ..][0 .. first[j + 1] - first[j]];
            for (slots, first[j]..) |*x, at| x.* = @intCast(at);
            rng.shuffle(i32, slots);
        };
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.r = try attn.Ref.init(a, D, shape.k);
        try self.q.resize(a, self.rows * h * D);
        try self.kc.resize(a, self.pool * stride);
        try self.vc.resize(a, self.pool * stride);
        for ([_][]u16{ self.q.items, self.kc.items, self.vc.items }) |x| for (x) |*e| {
            e.* = attn.normal(rng);
        };
        const nan = c.toBf16(std.math.nan(f32));
        for (0..hk) |kh| {
            @memset(self.kAt(poison, kh), nan);
            @memset(self.vAt(poison, kh), nan);
        }
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => try self.stress(rng),
        }
        try self.told.resize(a, self.counts.items.len);
        @memcpy(self.told.items, self.counts.items);
        if (self.plant_ == .drop_last) for (self.told.items) |*n| {
            n.* = @max(1, n.* - 1);
        };
        for ([_][]const u8{ std.mem.sliceAsBytes(self.q.items), std.mem.sliceAsBytes(self.kc.items), std.mem.sliceAsBytes(self.vc.items), std.mem.sliceAsBytes(self.list.items), std.mem.sliceAsBytes(self.told.items) }, &self.dev) |bytes, *p| p.* = try self.rig.input(bytes);
        self.out = try self.rig.output(self.rows * h * D * 2);
        return true;
    }

    /// NaN in a query and a key, +inf and -inf values two rows' heads see, and +inf at weights float32 rounds to 0.
    fn plant(self: *Sparse) void {
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        const ninf = c.toBf16(-std.math.inf(f32));
        const last = self.rows - 1;
        self.qAt(0, 0)[3] = nan;
        self.vAt(self.entry(0, 1, 0), 1)[7] = inf;
        if (self.listed(0, 1) > 1) self.vAt(self.entry(0, 1, self.listed(0, 1) - 1), 1)[9] = ninf;
        self.kAt(self.entry(last, 3, self.listed(last, 3) - 1), 3)[5] = nan;
        // key head 2 of the last row: one key all 8, the others all -8 with +inf at 21; its first head's query all 2
        @memset(self.qAt(last, 2 * G), c.toBf16(2));
        for (0..self.listed(last, 2)) |i| {
            @memset(self.kAt(self.entry(last, 2, i), 2), c.toBf16(if (i == 0) 8 else -8));
            if (i != 0) self.vAt(self.entry(last, 2, i), 2)[21] = inf;
        }
    }

    /// Gate 4: every key alike (each p is 1); each row's block of values mirrored, so its sums are exact in fp32.
    fn stress(self: *Sparse, rng: std.Random) !void {
        const a = self.rig.gpu.gpa;
        const key = try a.alloc(u16, D);
        defer a.free(key);
        for (key) |*e| e.* = attn.normal(rng);
        for (0..self.pool - 1) |r| for (0..hk) |kh| {
            @memcpy(self.kAt(r, kh), key);
        };
        for (0..self.rows) |j| for (0..hk) |kh| {
            const n = self.listed(j, kh);
            const col = try a.alloc(f32, n);
            defer a.free(col);
            for (0..D) |e| {
                attn.mirrored(rng, col);
                for (col, 0..) |y, i| self.vAt(self.entry(j, kh, i), kh)[e] = c.toBf16(y);
            }
        };
        // key head 3 lists half of row j's block: the rest of that block's values stay out of its sums
    }

    pub fn run(self: *Sparse) !void {
        try self.sa.run(self.rig.s, self.dev[0], self.dev[1], self.dev[2], self.dev[3], self.dev[4], self.out, .{ .rows = self.rows, .h = h, .hk = hk, .cap = self.cap, .stride = stride, .scale = scale() });
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Sparse) usize {
        return self.rows * h * D;
    }

    /// Output head i (row i / h, head i % h): the keys its key head's list names, in list order.
    pub fn load(self: *Sparse, i: usize, hd: *attn.Head) !void {
        const row = i / h;
        const kh = (i % h) / G;
        for (hd.q, self.qAt(row, i % h)) |*y, x| y.* = c.fromBf16(x);
        const n = self.listed(row, kh);
        for (0..n) |j| {
            const at = self.entry(row, kh, j);
            for (0..D) |e| {
                hd.k[j * D + e] = c.fromBf16(self.kAt(at, kh)[e]);
                hd.v[j * D + e] = c.fromBf16(self.vAt(at, kh)[e]);
            }
        }
        hd.n = n;
    }

    pub fn check(self: *Sparse, at: g.Gate) !?[]const u8 {
        return self.r.?.check(self, self.rows * h, scale(), try self.rig.outputBytes(), at);
    }

    pub fn reference(self: *Sparse, want: []f64, mag: []f64) !void {
        return self.r.?.reference(self, self.rows * h, scale(), want, mag);
    }

    pub fn simulate(self: *Sparse, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(self, self.rows * h, scale(), acc, order, out);
    }
};

/// The kernel and its planted control, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    sa: sparse.SparseAttention,
    tests: [2]Sparse,

    pub fn init(gpu: tc.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        set.gpa = gpu.gpa;
        set.sa = try sparse.SparseAttention.load(gpu.d);
        errdefer set.sa.unload();
        for (&set.tests, [_]Plant{ .none, .drop_last }) |*t, p| t.* = .{ .rig = .{ .gpu = gpu, .s = s }, .sa = &set.sa, .plant_ = p };
        for (&set.tests) |*t| try list.append(a, t.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.tests) |*t| t.deinit();
        set.sa.unload();
        set.gpa.destroy(set);
    }
};
