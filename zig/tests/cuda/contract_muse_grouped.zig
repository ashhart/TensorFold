//! muse_grouped2 under the contract: tf_exl3_grouped_2's bytes on random routings, widths and members.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const x3 = @import("contract_muse_exl3.zig");
const mu = @import("contract_muse_util.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const sx = cuda.exl3_experts;

const E = 16; // experts in the fixture
const slots = 8; // Muse Spark's top 8

pub const Grp = struct {
    rig: rig_.Rig,
    entry: x3.GrpEntry,
    f: cuda.Function,
    base: cuda.Function,
    p: usize = 0,
    rows: usize = 0,
    k: usize = 0,
    n: usize = 0,
    sk: usize = 1,
    maxu: usize = 0,
    dev: [10]u64 = @splat(0), // x0, x1, tp0, tp1, k2_0, k2_1, uids, ucount, members, z
    host: std.ArrayList(u8) = .empty,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Grp) void {
        self.host.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    const vt = g.hooks(Grp);

    pub fn kernel(self: *Grp) g.Kernel {
        return .{ .spec = x3.grpSpec(self.entry), .ctx = self, .vt = &vt };
    }

    fn zBytes(self: *const Grp) usize {
        return self.entry.mats * self.sk * self.p * self.n * 4;
    }

    pub fn prepare(self: *Grp, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k % 128 != 0 or shape.n % 128 != 0 or shape.k == 0 or shape.n == 0) return false;
        self.sk = sx.plan(shape.k, shape.n, self.entry.mats == 2) catch return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.rows = shape.rows;
        self.k = shape.k;
        self.n = shape.n;
        self.p = shape.rows * slots;
        // routing: 8 distinct experts a row; members row * 32 + slot in row order, -1 after the last
        const pick = try a.alloc(i32, self.p);
        defer a.free(pick);
        for (0..self.rows) |r| {
            var ids: [E]i32 = undefined;
            for (&ids, 0..) |*v, i| v.* = @intCast(i);
            rng.shuffle(i32, &ids);
            @memcpy(pick[r * slots ..][0..slots], ids[0..slots]);
        }
        var uids: std.ArrayList(i32) = .empty;
        defer uids.deinit(a);
        for (0..E) |e| if (std.mem.indexOfScalar(i32, pick, @intCast(e)) != null) try uids.append(a, @intCast(e));
        self.maxu = uids.items.len;
        const members = try a.alloc(i32, self.maxu * self.rows);
        defer a.free(members);
        @memset(members, -1);
        for (uids.items, 0..) |e, u| {
            var j: usize = 0;
            for (pick, 0..) |pe, i| if (pe == e) {
                members[u * self.rows + j] = @intCast((i / slots) * 32 + i % slots);
                j += 1;
            };
        }
        // inputs: rotated fp16 rows, one trellis a (matrix, expert) in the stored layout, widths per expert
        const xs = try a.alloc(f16, self.p * self.k);
        defer a.free(xs);
        var ptrs: [2][E]i64 = undefined;
        var k2s: [2][E]i32 = undefined;
        for (0..2) |m| {
            for (xs) |*v| v.* = @floatCast(rng.float(f32) * 2 - 1);
            if (fill == .nonfinite) {
                xs[m * 3] = if (m == 0) std.math.nan(f16) else std.math.inf(f16);
                if (self.p > 2) xs[2 * self.k + 7] = -std.math.inf(f16);
            }
            self.dev[m] = try self.rig.input(std.mem.sliceAsBytes(xs));
            for (0..E) |e| {
                const k2 = self.entry.k2s[e % 2];
                k2s[m][e] = @intCast(k2);
                const words = try a.alloc(u32, (self.k / 16) * (self.n / 16) * 4 * k2);
                defer a.free(words);
                for (words) |*w| w.* = rng.int(u32);
                ptrs[m][e] = @intCast(try self.rig.input(std.mem.sliceAsBytes(words)));
            }
        }
        if (self.entry.mats == 1) {
            self.dev[1] = self.dev[0];
            ptrs[1] = ptrs[0];
            k2s[1] = k2s[0];
        }
        for (0..2) |m| {
            self.dev[2 + m] = try self.rig.input(std.mem.sliceAsBytes(&ptrs[m]));
            self.dev[4 + m] = try self.rig.input(std.mem.sliceAsBytes(&k2s[m]));
        }
        self.dev[6] = try self.rig.input(std.mem.sliceAsBytes(uids.items));
        const cnt = [1]i32{@intCast(self.maxu)};
        self.dev[7] = try self.rig.input(std.mem.sliceAsBytes(&cnt));
        self.dev[8] = try self.rig.input(std.mem.sliceAsBytes(members));
        self.dev[9] = try self.rig.output(self.zBytes());
        return true;
    }

    /// exl3.zig grouped2's launch: grid (experts, N / 128, mats * sk * member tiles of 32).
    pub fn run(self: *Grp) !void {
        var a: cuda.Args = .{};
        inline for (0..10) |i| a.add(self.dev[i]);
        inline for (.{ self.k, self.n, self.p, self.sk, self.rows, slots }) |v| a.add(@as(i32, @intCast(v)));
        const z: u32 = @intCast(self.entry.mats * self.sk * ((self.rows + 31) / 32));
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.maxu), .y = @intCast(self.n / 128), .z = z }, .block = .{ .x = 128 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Grp) usize {
        return self.zBytes() / 4;
    }

    /// Never asked: a replacement's gates 1 and 3 compare bytes (check, replaced), and it declares no sums for gate 4.
    pub fn reference(_: *Grp, _: []f64, _: []f64) !void {
        return error.ReplacementHasNoFloat64Reference;
    }

    /// tf_exl3_grouped_2 as exl3_experts.run launches it, into a buffer poisoned as the rig's output is.
    pub fn replaced(self: *Grp) ![]const u8 {
        const bytes = self.zBytes();
        var tmp = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, bytes);
        defer tmp.free();
        try tmp.fill8(0xff, self.rig.s.handle);
        var a: cuda.Args = .{};
        inline for (0..9) |i| a.add(self.dev[i]);
        a.add(tmp.ptr);
        inline for (.{ self.k, self.n, self.p, self.sk, self.rows, slots, self.maxu, self.entry.mats }) |v| a.add(@as(i32, @intCast(v)));
        a.add(@as(i32, 0));
        a.add(@as(i32, 0));
        a.add(@as(i32, -1));
        const z: u32 = @intCast(self.entry.mats * self.sk * ((self.rows + 15) / 16));
        try cuda.launch.launch(self.base, .{ .grid = .{ .x = @intCast(self.maxu), .y = @intCast(self.n / 128), .z = z }, .block = .{ .x = 128 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
        try self.host.resize(self.rig.gpu.gpa, bytes);
        try tmp.download(0, self.host.items);
        return self.host.items;
    }

    /// Gate 1: tf_exl3_grouped_2's bytes on the planted inputs, some of them non-finite.
    pub fn check(self: *Grp, at: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        return mu.sameBytes(&self.msg, got, try self.replaced(), "tf_exl3_grouped_2", .f32, at);
    }
};
