//! Flash Next EXL3's host tier: kvt_fetch and kvt_remap against host models byte for byte; kvt_plan is in its own file.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const plan_ = @import("contract_flashnext_exl3_kvt_plan.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const page_rows = plan_.page_rows;
const ratio: usize = plan_.ratio;
const idw: usize = plan_.idw;
// a K or V row (2 heads of 256 dims, bf16) and a page of one layer's K then V (flash.exl3.pages' layout)
pub const row_bytes = 1024;
const row16 = row_bytes / 16;
pub const kv_page = 2 * page_rows * row_bytes;
pub const value_at = page_rows * row_bytes;
const hpages = 16; // logical pages in the host store
const vpage = 12; // the table entries before the lent frames
const win0 = 40_000;
const nblocks = 200 * page_rows / ratio;
const none = std.math.maxInt(u32);
const garbage = [_]u32{ 0x80000000, 0x7fffffff, 0xffffffff, 0x7fc00000, 0x7f800000, 0xff800000, 0 };

fn noSums(_: c.Shape) usize {
    return 0;
}

fn sh(rows: usize, n: usize, k: usize) c.Shape {
    return .{ .rows = rows, .n = n, .k = k };
}

const fetch_rule = "bytes moved, never summed: a word's bits (the NaN and both infinities planted in the rows fetched) arrive as stored, " ++
    "and entries of the list past the count are never read";
const remap_rule = "integer positions only: a row without a selection, and keys past a row's count, are never read";

// fetches (the 128 blocks a grid strides: the first 129 gives a block two), lent frames of 64 slots, pages in the store
const fetch_served = [_]c.Shape{ sh(128, 4, hpages), sh(1, 4, hpages), sh(200, 8, hpages), sh(512, 8, hpages) };
// rows, keys a row (a row lists up to idw - 1, the key count the engine passes)
const remap_served = [_]c.Shape{ sh(16, idw - 1, 0), sh(1, idw - 1, 0), sh(512, idw - 1, 0), sh(4, 300, 0) };

fn fetchSpec(control: bool) c.Spec {
    return .{
        .name = if (control) "control: Flash Next EXL3 kvt_fetch reading the table one entry late" else "cuda Flash Next EXL3 host-tier fetch kvt_fetch",
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = fetch_rule },
        .tiles = .{ .{ .size = 128 }, .{}, .{} },
        .served = &fetch_served,
    };
}

fn remapSpec(control: bool) c.Spec {
    return .{
        .name = if (control) "control: Flash Next EXL3 kvt_remap with the slot rows one block late" else "cuda Flash Next EXL3 host-tier remap kvt_remap",
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = remap_rule },
        .tiles = .{ .{}, .{ .size = 256 }, .{} },
        .served = &remap_served,
    };
}

fn less(_: void, x: u32, y: u32) bool {
    return x < y;
}

fn put(rig: *rig_.Rig, ptr: u64, bytes: []const u8) !void {
    const b: cuda.DeviceBuffer = .{ .d = rig.gpu.d, .ptr = ptr, .len = bytes.len };
    try b.upload(0, bytes);
}

/// Fills `out` with distinct values below `limit`, in a random order.
fn distinct(rng: std.Random, a: std.mem.Allocator, limit: usize, out: []u32) !void {
    const all = try a.alloc(u32, limit);
    defer a.free(all);
    for (all, 0..) |*v, i| v.* = @intCast(i);
    rng.shuffle(u32, all);
    @memcpy(out, all[0..out.len]);
}

pub const Fetch = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    f: cuda.Function,
    control: bool = false,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    frames: usize = 0,
    store: []u8 = &.{},
    lids: []u32 = &.{},
    pt: []u32 = &.{},
    list: []u32 = &.{},
    ctl: [8]u32 = @splat(0),
    want: []u8 = &.{},
    d_store: u64 = 0,
    d_lids: u64 = 0,
    d_pt: u64 = 0,
    d_list: u64 = 0,
    d_ctl: u64 = 0,
    d_arena: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Fetch) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Fetch) void {
        const a = self.a;
        a.free(self.store);
        a.free(self.want);
        for ([_]*[]u32{ &self.lids, &self.pt, &self.list }) |p| {
            a.free(p.*);
            p.* = &.{};
        }
        self.store = &.{};
        self.want = &.{};
    }

    const vt = g.hooks(Fetch);

    pub fn kernel(self: *Fetch) g.Kernel {
        return .{ .spec = fetchSpec(self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    /// Random store words; `n` distinct blocks into `n` distinct slots of the lent frames, shuffled frames behind them.
    pub fn prepare(self: *Fetch, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try self.build(shape, fill, seed)) return false;
        self.rig.reset();
        self.d_store = try self.rig.input(self.store);
        self.d_lids = try self.rig.input(std.mem.sliceAsBytes(self.lids));
        self.d_pt = try self.rig.input(std.mem.sliceAsBytes(self.pt));
        self.d_list = try self.rig.input(std.mem.sliceAsBytes(self.list));
        self.d_ctl = try self.rig.input(std.mem.sliceAsBytes(&self.ctl));
        self.d_arena = try self.rig.output(self.want.len);
        return true;
    }

    fn build(self: *Fetch, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const n = shape.rows;
        self.frames = shape.n;
        if (n == 0 or self.frames == 0 or n > self.frames * 64 or shape.k != hpages or n > hpages * 64) return false;
        self.free();
        self.shape = shape;
        self.fill = fill;
        const a = self.a;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6b767466);
        const rng = prng.random();
        self.store = try a.alloc(u8, hpages * kv_page);
        rng.bytes(self.store);
        self.lids = try a.alloc(u32, hpages);
        try distinct(rng, a, hpages, self.lids);
        self.pt = try a.alloc(u32, vpage + self.frames + 1);
        @memset(self.pt[0..vpage], 0);
        try distinct(rng, a, 2 * self.frames, self.pt[vpage..]);
        const slots = self.frames * 64;
        self.list = try a.alloc(u32, 2 * (slots + 8));
        for (self.list, 0..) |*v, i| v.* = garbage[i % garbage.len];
        const blocks = try a.alloc(u32, n);
        defer a.free(blocks);
        const at = try a.alloc(u32, n);
        defer a.free(at);
        try distinct(rng, a, hpages * 64, blocks);
        try distinct(rng, a, slots, at);
        for (blocks, at, 0..) |b, s, i| {
            self.list[2 * i] = b;
            self.list[2 * i + 1] = s;
        }
        self.ctl = .{ 0x7fc00000, @intCast(n), 0xffffffff, 5, 6, 7, 8, 9 };
        self.want = try a.alloc(u8, 2 * self.frames * kv_page);
        @memset(self.want, 0xff);
        const per = ratio * row_bytes;
        const planes = [_]usize{ 0, value_at };
        for (blocks, at) |b, s| {
            const page = b * ratio / page_rows;
            const within = b * ratio % page_rows;
            const src = @as(usize, self.lids[page]) * kv_page + within * row_bytes;
            const dst = @as(usize, self.pt[vpage + s / 64]) * kv_page + (s % 64) * per;
            for (planes) |p| {
                if (fill == .nonfinite) for ([_]u32{ 0x7fc00000, 0x7f800000, 0xff800000 }, 0..) |w, i|
                    @memcpy(self.store[p + src + 4 * i ..][0..4], std.mem.asBytes(&w));
                @memcpy(self.want[p + dst ..][0..per], self.store[p + src ..][0..per]);
            }
        }
        return true;
    }

    pub fn run(self: *Fetch) !void {
        var f: cuda.Args = .{};
        const k: u64 = 0;
        const v: u64 = value_at;
        for ([_]u64{ self.d_store + k, self.d_store + v, self.d_arena + k, self.d_arena + v, self.d_ctl, self.d_list, self.d_lids, self.d_pt }) |p| f.add(p);
        const table_at: usize = if (self.control) vpage + 1 else vpage;
        f.add(@as(c_int, @intCast(table_at)));
        f.add(@as(i64, @intCast(kv_page / 16)));
        f.add(@as(c_int, @intCast(row16)));
        f.add(@as(c_int, @intCast(ratio)));
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = 128 }, .block = .{ .x = 256 } }, self.rig.s, &f);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Fetch) usize {
        return self.want.len;
    }

    pub fn reference(self: *Fetch, want: []f64, mag: []f64) !void {
        for (want, mag, self.want) |*w, *m, b| {
            w.* = @floatFromInt(b);
            m.* = w.*;
        }
    }

    /// Every byte of the lent frames: the rows fetched where the table puts them, the rest as poisoned.
    pub fn check(self: *Fetch, _: g.Gate) !?[]const u8 {
        const got = try self.rig.outputBytes();
        if (std.mem.indexOfDiff(u8, got, self.want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} of the lent frames differs from the host store's row", .{ at, got.len });
        if (self.fill != .nonfinite) return null;
        for (std.mem.bytesAsSlice(u32, got)) |w| if (w == 0x7f800000) return null;
        return "no fetched word is an infinity: the planted values prove nothing";
    }
};

/// The slot rows start after the stream's pages.
const vrow: u32 = 200 * page_rows;

pub const Remap = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    f: cuda.Function,
    control: bool = false,
    shape: c.Shape = undefined,
    ids: []u32 = &.{},
    nkr: []u32 = &.{},
    spr: []u32 = &.{},
    b2s: []u32 = &.{},
    ctl: [8]u32 = @splat(0),
    want: []u8 = &.{},
    d_ids: u64 = 0,
    d_nk: u64 = 0,
    d_spr: u64 = 0,
    d_b2s: u64 = 0,
    d_ctl: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Remap) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Remap) void {
        const a = self.a;
        for ([_]*[]u32{ &self.ids, &self.nkr, &self.spr, &self.b2s }) |p| {
            a.free(p.*);
            p.* = &.{};
        }
        a.free(self.want);
        self.want = &.{};
    }

    const vt = g.hooks(Remap);

    pub fn kernel(self: *Remap) g.Kernel {
        return .{ .spec = remapSpec(self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    /// Rows of 512 sorted blocks' keys; where a block is missing every selecting row has one, so the rows agree on it.
    pub fn prepare(self: *Remap, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try self.build(shape, fill, seed)) return false;
        self.rig.reset();
        self.d_spr = try self.rig.input(std.mem.sliceAsBytes(self.spr));
        self.d_b2s = try self.rig.input(std.mem.sliceAsBytes(self.b2s));
        self.d_ids = try self.rig.output(self.ids.len * 4);
        self.d_nk = try self.rig.output(shape.rows * 4);
        self.d_ctl = try self.rig.output(32);
        return true;
    }

    fn build(self: *Remap, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const rows = shape.rows;
        const nk = shape.n;
        if (rows == 0 or rows > 512 or nk == 0 or nk >= idw) return false;
        self.free();
        self.shape = shape;
        const a = self.a;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6b767472);
        const rng = prng.random();
        const before = win0 / ratio;
        // 0: rows select, every block in a slot; 1: the flag set; 2: each selecting row misses one; 3: one row does
        const variant = seed % 4;
        self.b2s = try a.alloc(u32, nblocks);
        for (self.b2s) |*s| s.* = rng.uintLessThan(u32, 8192);
        self.ids = try a.alloc(u32, rows * idw);
        self.nkr = try a.alloc(u32, rows);
        self.spr = try a.alloc(u32, rows);
        const only = rng.uintLessThan(usize, rows);
        for (0..rows) |r| {
            self.spr[r] = switch (variant) {
                0 => rng.uintLessThan(u32, 2),
                3 => @intFromBool(r == only),
                else => 1,
            };
            if (variant == 0 and r == 0) self.spr[r] = 1;
            self.nkr[r] = if (self.spr[r] != 0) @intCast(nk) else 1 + rng.uintLessThan(u32, idw - 1);
            const out = self.ids[r * idw ..][0..idw];
            const blocks = out[0 .. idw - 4];
            try distinct(rng, a, before, blocks[0 .. blocks.len / ratio]);
            std.mem.sort(u32, blocks[0 .. blocks.len / ratio], {}, less);
            var j = blocks.len / ratio;
            while (j > 0) {
                j -= 1;
                const b = blocks[j];
                for (0..ratio) |t| blocks[j * ratio + t] = b * @as(u32, @intCast(ratio)) + @as(u32, @intCast(t));
            }
            for (0..4) |t| out[idw - 4 + t] = win0 + 100 + @as(u32, @intCast(t));
            if (self.spr[r] != 0 and (variant == 2 or variant == 3)) self.b2s[out[rng.uintLessThan(usize, @min(nk, idw - 4))] / ratio] = none;
        }
        self.ctl = .{ 7, 8, @intFromBool(variant == 1), 9, 10, 11, 12, 13 };
        if (fill == .nonfinite) for (0..rows) |r| {
            const out = self.ids[r * idw ..][0..idw];
            for (out, 0..) |*e, i| if (self.spr[r] == 0 or i >= self.nkr[r]) {
                e.* = garbage[(r + i) % garbage.len];
            };
        };
        try self.model();
        return true;
    }

    /// The remapped lists, counts and flag, as the three outputs lie side by side.
    fn model(self: *Remap) !void {
        const a = self.a;
        const rows = self.shape.rows;
        var ctl2 = self.ctl[2];
        const ids = try a.dupe(u32, self.ids);
        defer a.free(ids);
        const nkr = try a.dupe(u32, self.nkr);
        defer a.free(nkr);
        for (0..rows) |r| {
            if (self.spr[r] == 0) continue;
            const row = ids[r * idw ..][0..idw];
            var missing = ctl2 != 0;
            for (row[0..nkr[r]]) |p| missing = missing or (p < win0 and self.b2s[p / ratio] == none);
            if (missing) {
                ctl2 = 1;
                nkr[r] = 0;
                continue;
            }
            for (row[0..nkr[r]]) |*p| if (p.* < win0) {
                p.* = vrow + self.b2s[p.* / ratio] * @as(u32, @intCast(ratio)) + p.* % @as(u32, @intCast(ratio));
            };
        }
        var ctl = self.ctl;
        ctl[2] = ctl2;
        a.free(self.want);
        self.want = try a.alloc(u8, ids.len * 4 + nkr.len * 4 + 32);
        @memcpy(self.want[0 .. ids.len * 4], std.mem.sliceAsBytes(ids));
        @memcpy(self.want[ids.len * 4 ..][0 .. nkr.len * 4], std.mem.sliceAsBytes(nkr));
        @memcpy(self.want[ids.len * 4 + nkr.len * 4 ..], std.mem.sliceAsBytes(&ctl));
    }

    pub fn run(self: *Remap) !void {
        try put(&self.rig, self.d_ids, std.mem.sliceAsBytes(self.ids));
        try put(&self.rig, self.d_nk, std.mem.sliceAsBytes(self.nkr));
        try put(&self.rig, self.d_ctl, std.mem.sliceAsBytes(&self.ctl));
        var r: cuda.Args = .{};
        for ([_]u64{ self.d_ids, self.d_nk, self.d_spr }) |v| r.add(v);
        for ([_]usize{ idw, win0, if (self.control) vrow + ratio else vrow }) |v| r.add(@as(c_int, @intCast(v)));
        r.add(self.d_b2s);
        r.add(@as(c_int, @intCast(ratio)));
        r.add(self.d_ctl);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows) }, .block = .{ .x = 256 } }, self.rig.s, &r);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Remap) usize {
        return self.want.len;
    }

    pub fn reference(self: *Remap, want: []f64, mag: []f64) !void {
        for (want, mag, self.want) |*w, *m, b| {
            w.* = @floatFromInt(b);
            m.* = w.*;
        }
    }

    /// The lists, counts and flag the model gives, byte for byte.
    pub fn check(self: *Remap, _: g.Gate) !?[]const u8 {
        const got = try self.rig.outputBytes();
        if (std.mem.indexOfDiff(u8, got, self.want)) |at| {
            const ids = self.ids.len * 4;
            const part = if (at < ids) "a key list" else if (at < ids + self.nkr.len * 4) "a key count" else "the overflow flag";
            return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} ({s}) differs from the model's", .{ at, got.len, part });
        }
        return null;
    }
};

/// Fetch and remap with their controls, and the plan's, over one loaded image.
pub const Group = struct {
    a: std.mem.Allocator,
    mod: cuda.Module,
    plan: plan_.Set,
    fetch: [2]Fetch,
    remap: [2]Remap,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, list: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        self.a = a;
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_exl3_kvtier);
        errdefer self.mod.unload();
        self.plan = try plan_.Set.init(gpu, s);
        errdefer self.plan.deinit();
        const ff = try self.mod.function("kvt_fetch");
        const fr = try self.mod.function("kvt_remap");
        for (&self.fetch, [2]bool{ false, true }) |*x, control| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .a = a, .f = ff, .control = control };
        errdefer for (&self.fetch) |*x| x.deinit();
        for (&self.remap, [2]bool{ false, true }) |*x, control| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .a = a, .f = fr, .control = control };
        errdefer for (&self.remap) |*x| x.deinit();
        try self.plan.add(a, list);
        for (&self.fetch) |*x| try list.append(a, x.kernel());
        for (&self.remap) |*x| try list.append(a, x.kernel());
        return self;
    }

    pub fn destroy(self: *Group) void {
        for (&self.remap) |*x| x.deinit();
        for (&self.fetch) |*x| x.deinit();
        self.plan.deinit();
        self.mod.unload();
        self.a.destroy(self);
    }
};

/// tf-cuda-test exl3-kvt-contract: these kernels through the gates and their controls alone.
pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(gpu.gpa);
    const group = try Group.create(gpu.gpa, gpu, s, &list);
    defer group.destroy();
    var t: g.Tally = .{};
    try g.runAll(gpu.gpa, list.items, 0x6b7674, &t);
    std.debug.print("SUMMARY exl3 kvt {d} kernels: {d} pass, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "exl3-kvt-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

test "kvt fetch and remap contract: every served call moves rows, and plants reach the model" {
    const a = std.testing.allocator;
    for (fetch_served) |shape| for ([_]g.Fill{ .random, .nonfinite }) |fill| {
        var cs: Fetch = .{ .a = a, .rig = undefined, .f = undefined };
        defer cs.free();
        try std.testing.expect(try cs.build(shape, fill, 7));
        var written: usize = 0;
        for (cs.want) |b| written += @intFromBool(b != 0xff);
        try std.testing.expect(written >= shape.rows * 2 * ratio * row_bytes / 2); // random words are mostly not 0xff
        var inf = false;
        for (std.mem.bytesAsSlice(u32, cs.want)) |w| inf = inf or w == 0x7f800000;
        try std.testing.expectEqual(fill == .nonfinite, inf);
    };
    var flagged = false;
    var zeroed = false;
    var moved = false;
    for (remap_served) |shape| for (0..8) |seed| {
        var cs: Remap = .{ .a = a, .rig = undefined, .f = undefined };
        defer cs.free();
        try std.testing.expect(try cs.build(shape, .nonfinite, seed));
        const ids = cs.ids.len * 4;
        const out = std.mem.bytesAsSlice(u32, cs.want[0..ids]);
        const nk = std.mem.bytesAsSlice(u32, cs.want[ids..][0 .. cs.nkr.len * 4]);
        flagged = flagged or std.mem.readInt(u32, cs.want[ids + cs.nkr.len * 4 + 8 ..][0..4], .little) != 0;
        for (nk, cs.nkr, cs.spr) |got, was, sparse| zeroed = zeroed or (sparse != 0 and got == 0 and was != 0);
        for (out, cs.ids) |got, was| moved = moved or (got != was and got >= vrow);
    };
    try std.testing.expect(flagged and zeroed and moved);
}
