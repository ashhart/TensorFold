//! tf_plan_kv_write_act under the contract on HIP: a round's keys of the activation kind into its slot's pages.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const d = 128;
const page = 64;
const first = 200; // the first row's position: the rows cross a page edge at 256

fn none(_: c.Shape) usize {
    return 0;
}

pub fn spec(k: kit.Kind) c.Spec {
    return .{
        .name = if (k == .bf16) "hip tf_plan_kv_write_act (bf16)" else "hip tf_plan_kv_write_act (fp16)",
        .accum = .none,
        .depth = none,
        .out = kit.elem(k),
        .replaces = "tf_cast then tf_plan_kv_write",
        .tiles = .{ .{ .size = 2 }, .{}, .{} },
        // a decode row and a draft window over Nano's two KV heads, and a 64-row window on a tp 2 rank's one head
        .served = &.{ .{ .rows = 1, .n = 2, .k = d }, .{ .rows = 4, .n = 2, .k = d }, .{ .rows = 64, .n = 1, .k = d } },
    };
}

/// shape.rows rows of shape.n KV heads: keys and values [rows][heads][d] into the pools at positions from `first`.
pub const KvWrite = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    keys: []f64 = &.{},
    vals: []f64 = &.{},
    table: []u32 = &.{},
    pages: usize = 0,
    plan: [2]hip.launches.PlanRef = undefined, // the launch's and the baseline's: their own pools
    in: [3]u64 = @splat(0), // keys of the kind, the same keys in fp32, values
    pools: [2][2]u64 = undefined, // the launch's key and value pools, the baseline's

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher) KvWrite {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *KvWrite) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *KvWrite) g.Kernel {
        return .{ .spec = spec(self.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(KvWrite);

    pub fn prepare(self: *KvWrite, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != d) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows * shape.n * d;
        self.keys = try a.alloc(f64, n);
        self.vals = try a.alloc(f64, n);
        for (self.keys, self.vals) |*x, *y| {
            x.* = kit.round(self.k, rng.float(f64) * 4 - 2);
            y.* = kit.round(self.k, rng.float(f64) * 4 - 2);
        }
        if (fill == .nonfinite) for ([_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }, 0..) |v, i| {
            self.keys[(i * 37) % n] = v;
            self.vals[(i * 53 + 1) % n] = v;
        };
        self.pages = (first + shape.rows + page - 1) / page;
        self.table = try a.alloc(u32, self.pages);
        for (self.table, 0..) |*p, i| p.* = @intCast(self.pages - 1 - i);
        const kb = try a.alloc(u8, n * 2);
        const vb = try a.alloc(u8, n * 2);
        for (self.keys, self.vals, 0..) |x, y, i| {
            kit.put(self.k, kb, i, x);
            kit.put(self.k, vb, i, y);
        }
        self.in = .{ try self.rig.input(kb), try self.rig.scratch(n * 4), try self.rig.input(vb) };
        const pool = shape.n * self.pages * page * d * 2;
        const poison = try a.alloc(u8, pool);
        @memset(poison, 0xff); // as the outputs start: a slot neither launch writes reads as NaN in both
        for (0..2) |v| for (0..2) |j| {
            self.pools[v][j] = if (v == 0) try self.rig.output(pool) else try self.rig.scratch(pool);
            if (v == 1) try self.rig.put(self.pools[v][j], poison);
        };
        const pos = try a.alloc(i32, shape.rows);
        for (pos, 0..) |*p, r| p.* = @intCast(first + r);
        const zeros = try a.alloc(i32, shape.rows);
        @memset(zeros, 0);
        const many = [1]i32{@intCast(shape.rows)};
        const args = .{
            try self.rig.input(std.mem.sliceAsBytes(pos)),
            try self.rig.input(std.mem.sliceAsBytes(zeros)),
            try self.rig.input(std.mem.sliceAsBytes(zeros[0..1])),
            try self.rig.input(std.mem.asBytes(&many)),
        };
        for (0..2) |v| {
            // one slot: descriptor [cached, kept, k pool, v pool], then its page table
            const words = 4;
            const desc = try a.alloc(u64, words + (self.pages + 1) / 2);
            @memset(desc, 0);
            desc[2] = self.pools[v][0];
            desc[3] = self.pools[v][1];
            @memcpy(std.mem.sliceAsBytes(desc[words..])[0 .. self.pages * 4], std.mem.sliceAsBytes(self.table));
            const dev = try self.rig.input(std.mem.sliceAsBytes(desc));
            const descs = try self.rig.input(std.mem.asBytes(&dev));
            self.plan[v] = .{ .rows = shape.rows, .slots = 1, .args = .{
                .pos = args[0],
                .slot = args[1],
                .first = args[2],
                .count = args[3],
                .desc = descs,
                .pages = words,
                .pool = @intCast(self.pages),
            } };
        }
        return true;
    }

    pub fn run(self: *KvWrite) !void {
        const s = self.shape;
        try self.l.planKvWriteAct(
            self.in[0],
            self.in[2],
            @backingInt(self.k),
            self.plan[0],
            0,
            s.n,
            d,
            self.rig.s.handle,
        );
        try self.rig.s.synchronize();
    }

    /// The keys cast to fp32, then the fp32 key write, into the baseline's pools.
    pub fn replaced(self: *KvWrite) ![]const u8 {
        const s = self.shape;
        const n = s.rows * s.n * d;
        const st = self.rig.s.handle;
        try self.l.tf_cast(@ptrFromInt(self.in[0]), @backingInt(self.k), @ptrFromInt(self.in[1]), 0, @intCast(n), st);
        try self.l.planKvWrite(self.in[1], self.in[2], @backingInt(self.k), self.plan[1], 0, s.n, d, st);
        const a = self.arena.allocator();
        const len = s.n * self.pages * page * d * 2;
        const keys = try a.dupe(u8, try self.rig.bytes(self.pools[1][0], len));
        return std.mem.concat(a, u8, &.{ keys, try self.rig.bytes(self.pools[1][1], len) });
    }

    pub fn count(self: *KvWrite) usize {
        return 2 * self.shape.n * self.pages * page * d;
    }

    /// Each pool element: the row's key or value at its page and slot, NaN where no row writes.
    pub fn reference(self: *KvWrite, want: []f64, mag: []f64) !void {
        const s = self.shape;
        const len = s.n * self.pages * page * d;
        @memset(want, std.math.nan(f64));
        @memset(mag, 0);
        for (0..s.rows) |r| for (0..s.n) |h| for (0..d) |j| {
            const p = first + r;
            const at = ((h * self.pages + self.table[p / page]) * page + p % page) * d + j;
            want[at] = self.keys[(r * s.n + h) * d + j];
            want[len + at] = self.vals[(r * s.n + h) * d + j];
        };
    }
};
