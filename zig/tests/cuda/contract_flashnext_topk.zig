//! fn_row_topk under the contract: each row's K best by drawRow's order against a host oracle, every byte.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;

const vocab = 248320; // Flash Next's score rows
const k_max = 64; // cuda_mtp_draw.topk_max
const pad = 8; // the row stride's extra columns

fn noSums(_: c.Shape) usize {
    return 0;
}

/// rows = score rows, n = the columns ranked (V), k = the candidates kept (K).
pub fn spec() c.Spec {
    return .{ .name = "cuda flashnext fn_row_topk (a row's top_k candidates for sampled draws)", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "bf16 bits as drawRow orders them (-0 as +0, NaN above +Inf); ties to the lowest columns; the values above the K-th, then the ties, each in column order" }, .tiles = .{ .{}, .{ .size = 1024, .step = 1, .min = 1 }, .{ .size = k_max, .step = 1, .min = 1 } }, .served = &.{ .{ .rows = 1, .n = vocab, .k = 20 }, .{ .rows = 9, .n = vocab, .k = k_max }, .{ .rows = 17, .n = vocab, .k = 1 } } };
}

/// drawRow's order as an unsigned key: larger value, larger key; -0 as +0.
fn key(u: u16) u32 {
    const v: u32 = if (u == 0x8000) 0 else u;
    return if (v & 0x8000 != 0) ~v & 0xffff else v | 0x8000;
}

pub const TopK = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    x: u64 = 0,
    ov: u64 = 0,
    oc: u64 = 0,
    want: []u8 = &.{},
    msg: [160]u8 = undefined,

    pub fn deinit(self: *TopK) void {
        self.rig.gpu.gpa.free(self.want);
        self.rig.deinit();
    }

    const vt = g.hooks(TopK);

    pub fn kernel(self: *TopK) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    /// Random rows, rows of few distinct values (many ties), and planted NaNs, infinities and signed zeros.
    pub fn prepare(self: *TopK, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.want);
        self.want = &.{};
        self.shape = shape;
        const m = shape.rows;
        const n = shape.n;
        const k = shape.k;
        if (m == 0 or n == 0 or k == 0 or k > k_max or k > n or n > vocab) return false;
        const ld = n + pad;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x746f706b);
        const r = prng.random();
        const rows = try a.alloc(u16, m * ld);
        defer a.free(rows);
        for (rows, 0..) |*v, i| {
            // the stride's pad holds NaNs, which rank first: any read past a row's n columns shows
            v.* = if (i % ld >= n) 0x7fc0 else if (fill == .stress) c.toBf16(@floatFromInt(r.uintLessThan(u32, 4))) else c.toBf16(r.float(f32) * 8 - 4);
        }
        if (fill == .nonfinite) for (0..m) |row| {
            const base = (row * 7919) % n;
            for ([_]u16{ 0x7fc1, 0xffc1, 0x7f80, 0xff80, 0x8000, 0x0000 }, 0..) |bits, j| rows[row * ld + (base + j) % n] = bits;
        };
        self.want = try oracle(a, rows, m, n, ld, k);
        self.x = try self.rig.input(std.mem.sliceAsBytes(rows));
        self.ov = try self.rig.output(m * k * 2);
        self.oc = try self.rig.output(m * k * 4);
        return true;
    }

    /// A row's K best by drawRow's order (ties to lower columns): those above the K-th, then the ties, by column.
    fn oracle(a: std.mem.Allocator, rows: []const u16, m: usize, n: usize, ld: usize, k: usize) ![]u8 {
        const vals = try a.alloc(u16, m * k);
        defer a.free(vals);
        const cols = try a.alloc(i32, m * k);
        defer a.free(cols);
        const order = try a.alloc(u32, n);
        defer a.free(order);
        for (0..m) |row| {
            const x = rows[row * ld ..][0..n];
            for (order, 0..) |*o, i| o.* = @intCast(i);
            const Ctx = struct {
                x: []const u16,
                fn less(ctx: @This(), p: u32, q: u32) bool {
                    const kp = key(ctx.x[p]);
                    const kq = key(ctx.x[q]);
                    return kp > kq or (kp == kq and p < q);
                }
            };
            std.sort.pdq(u32, order, Ctx{ .x = x }, Ctx.less);
            const kth = key(x[order[k - 1]]);
            var at: usize = row * k;
            for (x, 0..) |v, i| if (key(v) > kth) {
                vals[at] = v;
                cols[at] = @intCast(i);
                at += 1;
            };
            var need = k - (at - row * k);
            for (x, 0..) |v, i| if (need > 0 and key(v) == kth) {
                vals[at] = v;
                cols[at] = @intCast(i);
                at += 1;
                need -= 1;
            };
        }
        return std.mem.concat(a, u8, &.{ std.mem.sliceAsBytes(vals), std.mem.sliceAsBytes(cols) });
    }

    /// The launch as Head.topk makes it: a row a block of 1,024 threads.
    pub fn run(self: *TopK) !void {
        var a: cuda.Args = .{};
        a.add(self.x);
        a.add(@as(c_int, @intCast(self.shape.n)));
        a.add(@as(c_longlong, @intCast(self.shape.n + pad)));
        a.add(@as(c_int, @intCast(self.shape.k)));
        a.add(self.ov);
        a.add(self.oc);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows) }, .block = .{ .x = 1024 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *TopK) usize {
        return self.want.len;
    }

    pub fn reference(self: *TopK, want: []f64, mag: []f64) !void {
        for (want, mag, self.want) |*w, *m, byte| {
            w.* = @floatFromInt(byte);
            m.* = 0;
        }
    }

    /// Gates 1 to 3's stated rule: every byte equals the oracle's.
    pub fn check(self: *TopK, at: g.Gate) !?[]const u8 {
        _ = at;
        if (!try self.rig.intact()) return "fn_row_topk changed a buffer guard";
        const got = try self.rig.outputBytes();
        if (std.mem.eql(u8, got, self.want)) return null;
        return try std.fmt.bufPrint(&self.msg, "candidates differ from the oracle's from byte {d}", .{std.mem.indexOfDiff(u8, got, self.want).?});
    }
};
