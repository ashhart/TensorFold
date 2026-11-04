//! Flash Next EXL3 fn_exl3_drestore and fn_exl3_dcopy under gates 1-4.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const u = @import("contract_flashnext_exl3_dplan_util.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const noSums = u.noSums;
const put = u.put;
const fetch = u.fetch;
const same = u.same;
const experts = u.experts;

pub const Restore = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    ids: []i32 = &.{},
    tab0: [3][experts]i64 = undefined,
    k2t0: [3][experts]i32 = undefined,
    d: [8]u64 = @splat(0),
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Restore) void {
        self.rig.gpu.gpa.free(self.ids);
        self.rig.deinit();
    }

    const vt = g.hooks(Restore);

    pub fn kernel(self: *Restore) g.Kernel {
        return .{
            .spec = .{
                .name = "cuda Flash Next EXL3 fn_exl3_drestore",
                .accum = .none,
                .depth = noSums,
                .out = .raw,
                .nan = .{ .stated = "integer ids only: each listed expert's six table entries become 0 and every other entry keeps its value" },
                // the block strides the list by its 256 threads
                .tiles = .{ .{ .size = 256, .step = 1, .min = 1 }, .{}, .{} },
                .served = &.{ .{ .rows = 1, .n = 0, .k = 0 }, .{ .rows = 11, .n = 0, .k = 0 }, .{ .rows = 160, .n = 0, .k = 0 }, .{ .rows = 512, .n = 0, .k = 0 } },
            },
            .ctx = self,
            .vt = &vt,
        };
    }

    pub fn prepare(self: *Restore, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > experts) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.ids);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x72657374);
        const rng = prng.random();
        var order: [experts]i32 = undefined;
        for (&order, 0..) |*o, i| o.* = @intCast(i);
        if (fill != .random) std.mem.reverse(i32, &order) else rng.shuffle(i32, &order);
        self.ids = try a.dupe(i32, order[0..shape.rows]);
        for (0..3) |w| for (0..experts) |i| {
            self.tab0[w][i] = 0x2222 + @as(i64, @intCast(rng.uintLessThan(u16, 1000)));
        };
        for (0..3) |w| for (0..experts) |i| {
            self.k2t0[w][i] = 9;
        };
        const n: i32 = @intCast(shape.rows);
        self.d[0] = try self.rig.input(std.mem.sliceAsBytes(self.ids));
        self.d[1] = try self.rig.input(std.mem.asBytes(&n));
        for (0..3) |w| {
            self.d[2 + w] = try self.rig.output(experts * 8);
            self.d[5 + w] = try self.rig.output(experts * 4);
        }
        return true;
    }

    pub fn run(self: *Restore) !void {
        for (0..3) |w| {
            try put(&self.rig, self.d[2 + w], std.mem.asBytes(&self.tab0[w]));
            try put(&self.rig, self.d[5 + w], std.mem.asBytes(&self.k2t0[w]));
        }
        var a: cuda.Args = .{};
        for (self.d) |p| a.add(p);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Restore) usize {
        return self.ids.len;
    }

    pub fn reference(self: *Restore, want: []f64, mag: []f64) !void {
        for (self.ids, want, mag) |id, *w, *m| {
            w.* = @floatFromInt(id);
            m.* = @abs(w.*);
        }
    }

    pub fn check(self: *Restore, _: g.Gate) !?[]const u8 {
        if (!try self.rig.intact()) return "a payload guard changed";
        var tab = self.tab0;
        var k2t = self.k2t0;
        for (self.ids) |id| for (0..3) |w| {
            tab[w][@intCast(id)] = 0;
            k2t[w][@intCast(id)] = 0;
        };
        for (0..3) |w| {
            if (try same(&self.rig, i64, self.d[2 + w], &tab[w], "table", &self.msg)) |why| return why;
            if (try same(&self.rig, i32, self.d[5 + w], &k2t[w], "k2 table", &self.msg)) |why| return why;
        }
        return null;
    }
};

pub const Copy = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    src: []u8 = &.{},
    blocks: u32 = 1,
    dst: u64 = 0,
    work_ptrs: [2]u64 = .{ 0, 0 },
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Copy) void {
        self.rig.gpu.gpa.free(self.src);
        self.rig.deinit();
    }

    const vt = g.hooks(Copy);

    pub fn kernel(self: *Copy) g.Kernel {
        return .{
            .spec = .{
                .name = "cuda Flash Next EXL3 fn_exl3_dcopy",
                .accum = .none,
                .depth = noSums,
                .out = .raw,
                .nan = .{ .stated = "bytes only: a segment's 16-byte words arrive unchanged, NaN and infinity patterns included" },
                // the copy unrolls by four once a segment passes three strides of the grid (256 threads a block)
                .tiles = .{ .{}, .{ .size = 768, .step = 1, .min = 1 }, .{} },
                .served = &.{ .{ .rows = 1, .n = 8, .k = 0 }, .{ .rows = 3, .n = 800, .k = 0 }, .{ .rows = 9, .n = 4096, .k = 0 }, .{ .rows = 30, .n = 9000, .k = 0 } },
            },
            .ctx = self,
            .vt = &vt,
        };
    }

    /// The kernel copies `work_n` lines of three segments (gate, up, down): the shape's rows are lines.
    const per_line = 3;

    /// Words (16 bytes) of segment `i`: the shape's longest, then shorter ones.
    fn words(shape: c.Shape, i: usize) usize {
        return @max(1, shape.n - (i * 37) % @max(shape.n / 2, 1));
    }

    pub fn prepare(self: *Copy, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.rows > 64 or shape.n > 20000) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.src);
        var total: usize = 0;
        const segs = shape.rows * per_line;
        for (0..segs) |i| total += words(shape, i) * 16;
        self.src = try a.alloc(u8, total);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x636f7079);
        prng.random().bytes(self.src);
        if (fill != .random) for (0..self.src.len / 4) |i| if (i % 5 == 0) {
            const pattern = [_]u32{ 0x7fc00000, 0xffc00000, 0x7f800000, 0xff800000, 0x7fc01234 };
            std.mem.writeInt(u32, self.src[i * 4 ..][0..4], pattern[(i / 5) % pattern.len], .little);
        };
        self.blocks = if (seed & 2 == 0) 1 else 3;
        const dsrc = try self.rig.input(self.src);
        self.dst = try self.rig.output(total);
        const work = try a.alloc(i64, segs * 3);
        defer a.free(work);
        var at: usize = 0;
        for (0..segs) |i| {
            const len = words(shape, i) * 16;
            work[i * 3 + 0] = @intCast(dsrc + at);
            work[i * 3 + 1] = @intCast(self.dst + at);
            work[i * 3 + 2] = @intCast(len);
            at += len;
        }
        const n: i32 = @intCast(shape.rows);
        const dwork = try self.rig.input(std.mem.sliceAsBytes(work));
        const dn = try self.rig.input(std.mem.asBytes(&n));
        self.work_ptrs = .{ dwork, dn };
        return true;
    }

    pub fn run(self: *Copy) !void {
        var a: cuda.Args = .{};
        a.add(self.work_ptrs[0]);
        a.add(self.work_ptrs[1]);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = self.blocks }, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Copy) usize {
        return self.src.len;
    }

    pub fn reference(self: *Copy, want: []f64, mag: []f64) !void {
        for (self.src, want, mag) |b, *w, *m| {
            w.* = @floatFromInt(b);
            m.* = w.*;
        }
    }

    pub fn check(self: *Copy, _: g.Gate) !?[]const u8 {
        if (!try self.rig.intact()) return "a payload guard changed";
        if (!std.mem.eql(u8, self.src, (try self.rig.inputBytes())[0..self.src.len])) return "the source changed";
        return try same(&self.rig, u8, self.dst, self.src, "copied byte", &self.msg);
    }
};
