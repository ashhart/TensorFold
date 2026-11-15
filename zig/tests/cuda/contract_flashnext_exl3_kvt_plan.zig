//! Flash Next EXL3's host-tier slot plan (kvt_plan) under the contract: its LRU choice against a host model, per call.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;

// the tier's geometry (the suite checks it against the engine's constants at compile time)
pub const ratio = 4;
pub const top = 512;
pub const idw = top * ratio + ratio;
pub const page_rows = 256;
const npages = 200;
const nblocks = npages * page_rows / ratio;
const win0 = 40_000; // the window starts here: blocks from win0 / ratio are never cached
pub const before = win0 / ratio;
pub const none = std.math.maxInt(u32);
const rule = "integers only: a row without a selection and the three keys after a block's first are never read, " ++
    "so extreme values and the bit patterns of NaN and the infinities planted there change nothing";

fn noSums(_: c.Shape) usize {
    return 0;
}

fn sh(rows: usize, slots: usize) c.Shape {
    return .{ .rows = rows, .n = slots, .k = top };
}

// rows a call, slots a layer (128, 16 and 9 lent frames of 64): a call plans `slots / top` sparse rows at most
pub const served = [_]c.Shape{ sh(16, 8192), sh(1, 8192), sh(64, 8192), sh(2, 1024), sh(1, 1024), sh(8, 1024), sh(1, 576), sh(3, 576) };

/// The slot plan: counts, epoch, who is needed, who misses, the stamps' multiset; raw: an image the harness compares.
pub fn spec(control: bool) c.Spec {
    return .{
        .name = if (control) "control: Flash Next EXL3 kvt_plan with the window one block early" else "cuda Flash Next EXL3 host-tier slot plan kvt_plan",
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = rule },
        .served = &served,
    };
}

/// Where a call starts: a steady cache, all hits, a fresh one, heavy ties, a wrapped clock, no rows, a set flag.
pub const Variant = enum { steady, hits, fresh, ties, wrap, idle, sticky };

const garbage = [_]u32{ 0x80000000, 0x7fffffff, 0xffffffff, 0x7fc00000, 0x7f800000, 0xff800000, 0 };

/// What one call leaves, as far as the order the atomics take does not matter, and the victims' old stamps.
pub const Model = struct {
    ctl: [8]u32 = @splat(0),
    need: []u32 = &.{},
    miss: []u32 = &.{},
    stamp: []u32 = &.{},
    mark: []u32 = &.{},
    hit: []bool = &.{}, // by slot: a slot the call uses
    victims: []u32 = &.{}, // the old stamps of the slots the misses take, ascending
    assigned: usize = 0,
    epoch: u32 = 0,
};

fn less(_: void, x: u32, y: u32) bool {
    return x < y;
}

pub const Case = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    control: bool = false,
    shape: c.Shape = undefined,
    ids: []u32 = &.{},
    spr: []u32 = &.{},
    b2s: []u32 = &.{},
    s2b: []u32 = &.{},
    stamp: []u32 = &.{},
    mark: []u32 = &.{},
    ctl: [8]u32 = @splat(0),
    model: Model = .{},
    f: cuda.Function,
    d_ids: u64 = 0,
    d_nk: u64 = 0,
    d_spr: u64 = 0,
    d_b2s: u64 = 0,
    d_s2b: u64 = 0,
    d_stamp: u64 = 0,
    d_mark: u64 = 0,
    d_ctl: u64 = 0,
    d_need: u64 = 0,
    d_miss: u64 = 0,
    d_fetch: u64 = 0,
    cap: usize = 0,
    msg: [200]u8 = undefined,
    image_buf: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Case) void {
        self.free();
        self.image_buf.deinit(self.a);
        self.rig.deinit();
    }

    pub fn free(self: *Case) void {
        const a = self.a;
        for ([_]*[]u32{ &self.ids, &self.spr, &self.b2s, &self.s2b, &self.stamp, &self.mark, &self.model.need, &self.model.miss, &self.model.stamp, &self.model.mark, &self.model.victims }) |p| {
            a.free(p.*);
            p.* = &.{};
        }
        a.free(self.model.hit);
        self.model.hit = &.{};
    }

    /// The harness reads the image, not the raw lists: a miss lands in whichever victim an atomic reaches first.
    const vt = blk: {
        var v = g.hooks(Case);
        v.output = image;
        break :blk v;
    };

    pub fn kernel(self: *Case) g.Kernel {
        return .{ .spec = spec(self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    /// A call's rows over a state from `seed`; non-finite: garbage where the kernel never reads.
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try self.build(shape, fill, seed)) return false;
        self.rig.reset();
        try self.stageDevice(shape.n);
        return true;
    }

    /// The call's host side: the state, the rows and the model of what the call leaves.
    pub fn build(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const slots = shape.n;
        if (shape.rows == 0 or shape.rows > 512 or shape.k != top or slots % 64 != 0 or slots < top or slots > before) return false;
        self.free();
        self.shape = shape;
        const a = self.a;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6b767470);
        const rng = prng.random();
        const variant: Variant = @fromBackingInt(@intCast(seed % std.enums.values(Variant).len));
        const rows = shape.rows;
        const per_plan = @max(1, slots / top);
        // overflow: one sparse row more than the slots hold, where the lists still fit (capacity 2,048 or the slots)
        const over = (per_plan + 1) * top <= @max(slots, 2048) and variant == .steady and slots < 8192;
        const sparse = if (variant == .idle) 0 else @min(rows, if (over) per_plan + 1 else per_plan);
        self.b2s = try a.alloc(u32, nblocks);
        self.s2b = try a.alloc(u32, slots);
        self.stamp = try a.alloc(u32, slots);
        self.mark = try a.alloc(u32, nblocks);
        @memset(self.b2s, none);
        @memset(self.s2b, none);
        @memset(self.stamp, 0);
        const clock: u32 = if (variant == .wrap) none else 1000;
        @memset(self.mark, 0);
        for (self.mark) |*m| if (rng.uintLessThan(u32, 8) == 0) {
            m.* = 1 + rng.uintLessThan(u32, 1000);
        };
        // the blocks before the window in a random order: the first `have` of them sit in slots
        const perm = try a.alloc(u32, before);
        defer a.free(perm);
        for (perm, 0..) |*p, i| p.* = @intCast(i);
        rng.shuffle(u32, perm);
        const have: usize = switch (variant) {
            .fresh => 0,
            .hits, .ties => slots,
            else => slots * 3 / 4,
        };
        const at = try a.alloc(u32, slots);
        defer a.free(at);
        for (at, 0..) |*p, i| p.* = @intCast(i);
        rng.shuffle(u32, at);
        for (perm[0..have], at[0..have]) |b, s| {
            self.b2s[b] = s;
            self.s2b[s] = b;
            self.stamp[s] = if (variant == .ties) 1 + rng.uintLessThan(u32, 4) else 1 + rng.uintLessThan(u32, 1000);
        }
        self.ctl = @splat(0);
        self.ctl[0] = clock;
        self.ctl[2] = @intFromBool(variant == .sticky);
        for (3..6) |i| self.ctl[i] = rng.uintLessThan(u32, 100); // counters add to what they held
        self.ids = try a.alloc(u32, rows * idw);
        self.spr = try a.alloc(u32, rows);
        const pre = top - 6; // blocks before the window a row lists, then six inside it (never cached)
        const row = try a.alloc(u32, top);
        defer a.free(row);
        for (0..rows) |r| {
            self.spr[r] = @intFromBool(r < sparse);
            const out = self.ids[r * idw ..][0..idw];
            const want_hits: usize = switch (variant) {
                .hits => pre,
                .fresh => 0,
                else => pre * 6 / 10,
            };
            const hits = @min(want_hits, have);
            try sample(rng, a, perm[0..have], hits, row[0..hits]);
            try sample(rng, a, perm[have..], pre - hits, row[hits..pre]);
            // the last block before the window is in every row, so a window that starts early misses it
            if (std.mem.indexOfScalar(u32, row[0..pre], before - 1) == null) row[0] = before - 1;
            for (row[pre..], 0..) |*e, i| e.* = @intCast(before + 1 + i);
            std.mem.sort(u32, row, {}, less);
            for (row, 0..) |b, j| for (0..ratio) |t| {
                out[j * ratio + t] = b * ratio + @as(u32, @intCast(t));
            };
            for (0..ratio) |t| out[top * ratio + t] = win0 + 100 + @as(u32, @intCast(t));
        }
        if (fill == .nonfinite) for (0..rows) |r| {
            const out = self.ids[r * idw ..][0..idw];
            for (out, 0..) |*e, i| if (self.spr[r] == 0 or (i < top * ratio and i % ratio != 0)) {
                e.* = garbage[(r + i) % garbage.len];
            };
        };
        try self.plan(slots, clock);
        return true;
    }

    fn stageDevice(self: *Case, slots: usize) !void {
        const rows = self.shape.rows;
        self.cap = @max(slots, 2048);
        self.d_ids = try self.rig.input(std.mem.sliceAsBytes(self.ids));
        self.d_spr = try self.rig.input(std.mem.sliceAsBytes(self.spr));
        const nk = try self.a.alloc(u32, rows);
        defer self.a.free(nk);
        @memset(nk, idw - 1);
        self.d_nk = try self.rig.input(std.mem.sliceAsBytes(nk));
        self.d_b2s = try self.rig.output(nblocks * 4);
        self.d_mark = try self.rig.output(nblocks * 4);
        self.d_ctl = try self.rig.output(8 * 4);
        self.d_s2b = try self.rig.output(slots * 4);
        self.d_stamp = try self.rig.output(slots * 4);
        self.d_need = try self.rig.output(self.cap * 4);
        self.d_miss = try self.rig.output(self.cap * 4);
        self.d_fetch = try self.rig.output(self.cap * 8);
    }

    fn put(self: *Case, ptr: u64, bytes: []const u8) !void {
        const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = ptr, .len = bytes.len };
        try b.upload(0, bytes);
    }

    pub fn run(self: *Case) !void {
        try self.put(self.d_b2s, std.mem.sliceAsBytes(self.b2s));
        try self.put(self.d_mark, std.mem.sliceAsBytes(self.mark));
        try self.put(self.d_ctl, std.mem.sliceAsBytes(&self.ctl));
        try self.put(self.d_s2b, std.mem.sliceAsBytes(self.s2b));
        try self.put(self.d_stamp, std.mem.sliceAsBytes(self.stamp));
        for ([_]struct { u64, usize }{ .{ self.d_need, self.cap * 4 }, .{ self.d_miss, self.cap * 4 }, .{ self.d_fetch, self.cap * 8 } }) |l| {
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = l[0], .len = l[1] };
            try b.fill8(0xff, null);
        }
        var a: cuda.Args = .{};
        for ([_]u64{ self.d_ids, self.d_nk, self.d_spr }) |v| a.add(v);
        const origin: usize = if (self.control) win0 - ratio else win0; // the fault: a window one block early
        for ([_]usize{ self.shape.rows, idw, top, ratio, origin }) |v| a.add(@as(c_int, @intCast(v)));
        a.add(self.d_ctl);
        a.add(self.d_b2s);
        a.add(self.d_b2s);
        a.add(@as(c_int, 0));
        for ([_]u64{ self.d_s2b, self.d_stamp, self.d_mark }) |v| a.add(v);
        a.add(@as(c_int, @intCast(self.shape.n)));
        for ([_]u64{ self.d_ctl, self.d_need, self.d_miss, self.d_fetch }) |v| a.add(v);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Case) usize {
        return self.imageWords() * 4;
    }

    /// The host model of one call on the prepared state (kvt_plan's body, one thread, any order of its atomics).
    fn plan(self: *Case, slots: usize, clock: u32) !void {
        const a = self.a;
        var m: Model = .{ .ctl = self.ctl };
        m.stamp = try a.dupe(u32, self.stamp);
        m.mark = try a.dupe(u32, self.mark);
        m.hit = try a.alloc(bool, slots);
        @memset(m.hit, false);
        self.model = m;
        const mo = &self.model;
        if (clock == none) {
            mo.ctl[1] = 0;
            mo.ctl[2] = 1;
            return;
        }
        mo.epoch = clock + 1;
        mo.ctl[0] = mo.epoch;
        mo.ctl[6] = mo.epoch;
        mo.ctl[1] = 0;
        if (self.ctl[2] != 0) return;
        var any = false;
        for (self.spr) |s| any = any or s != 0;
        if (!any) return;
        var need: std.ArrayList(u32) = .empty;
        defer need.deinit(a);
        for (0..self.shape.rows) |r| {
            if (self.spr[r] == 0) continue;
            for (0..top) |j| {
                const b = self.ids[r * idw + j * ratio] / ratio;
                if (b * ratio >= win0) continue;
                if (mo.mark[b] != mo.epoch) {
                    mo.mark[b] = mo.epoch;
                    try need.append(a, b);
                }
            }
        }
        var miss: std.ArrayList(u32) = .empty;
        defer miss.deinit(a);
        for (need.items) |b| {
            const s = self.b2s[b];
            if (s != none and s < slots) {
                mo.stamp[s] = mo.epoch;
                mo.hit[s] = true;
            } else try miss.append(a, b);
        }
        var avail: usize = 0;
        for (mo.stamp) |t| avail += @intFromBool(t != mo.epoch);
        mo.ctl[3] += if (miss.items.len > avail) 0 else @as(u32, @intCast(miss.items.len));
        mo.ctl[4] += @intCast(need.items.len);
        mo.ctl[5] += 1;
        mo.ctl[1] = if (miss.items.len > avail) 0 else @intCast(miss.items.len);
        if (miss.items.len > avail) mo.ctl[2] = 1;
        mo.need = try a.dupe(u32, need.items);
        mo.miss = try a.dupe(u32, miss.items);
        std.mem.sort(u32, mo.need, {}, less);
        std.mem.sort(u32, mo.miss, {}, less);
        if (miss.items.len > avail or miss.items.len == 0) return;
        // the n smallest stamps among the slots the call does not use: ties are equally old, which is the multiset
        var olds: std.ArrayList(u32) = .empty;
        defer olds.deinit(a);
        for (mo.stamp) |t| if (t != mo.epoch) try olds.append(a, t);
        std.mem.sort(u32, olds.items, {}, less);
        mo.assigned = miss.items.len;
        mo.victims = try a.dupe(u32, olds.items[0..mo.assigned]);
        // the multiset after: each victim's stamp becomes the epoch
        for (mo.victims) |t| mo.stamp[std.mem.indexOfScalar(u32, mo.stamp, t).?] = mo.epoch;
    }

    fn imageWords(self: *const Case) usize {
        const mo = &self.model;
        return 8 + 1 + mo.need.len + 1 + mo.miss.len + mo.stamp.len + mo.mark.len + 1 + mo.assigned;
    }

    fn word(buf: *std.ArrayList(u8), a: std.mem.Allocator, v: u32) !void {
        try buf.appendSlice(a, std.mem.asBytes(&std.mem.nativeToLittle(u32, v)));
    }

    /// The call's result by words: ctl, the needed blocks, the missing ones, the stamps and marks, the fetched blocks.
    fn words(self: *Case, buf: *std.ArrayList(u8), ctl: [8]u32, need: []const u32, miss: []const u32, stamp: []const u32, mark: []const u32, fetched: []const u32) !void {
        const a = self.a;
        buf.clearRetainingCapacity();
        for (ctl) |v| try word(buf, a, v);
        try word(buf, a, @intCast(need.len));
        for (need) |v| try word(buf, a, v);
        try word(buf, a, @intCast(miss.len));
        for (miss) |v| try word(buf, a, v);
        for (stamp) |v| try word(buf, a, v);
        for (mark) |v| try word(buf, a, v);
        try word(buf, a, @intCast(fetched.len));
        for (fetched) |v| try word(buf, a, v);
    }

    /// The model's image: its stamps' multiset sorted, the lists as the call's sets.
    fn modelImage(self: *Case, buf: *std.ArrayList(u8)) !void {
        const mo = &self.model;
        const sorted = try self.a.dupe(u32, mo.stamp);
        defer self.a.free(sorted);
        std.mem.sort(u32, sorted, {}, less);
        try self.words(buf, mo.ctl, mo.need, mo.miss, sorted, mo.mark, mo.miss[0..mo.assigned]);
    }

    /// The model's image, as float64 bytes for the harness's reference.
    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(self.a);
        try self.modelImage(&buf);
        for (want, mag, buf.items) |*w, *m, b| {
            w.* = @floatFromInt(b);
            m.* = w.*;
        }
    }

    const Read = struct {
        ctl: [8]u32 = undefined,
        b2s: []u32,
        s2b: []u32,
        stamp: []u32,
        mark: []u32,
        need: []u32,
        miss: []u32,
        fetch: []u32,
        a: std.mem.Allocator,

        fn deinit(r: Read) void {
            for ([_][]u32{ r.b2s, r.s2b, r.stamp, r.mark, r.need, r.miss, r.fetch }) |s| r.a.free(s);
        }
    };

    fn pull(self: *Case, ptr: u64, n: usize) ![]u32 {
        const out = try self.a.alloc(u32, n);
        errdefer self.a.free(out);
        const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = ptr, .len = n * 4 };
        try self.rig.s.synchronize();
        try b.download(0, std.mem.sliceAsBytes(out));
        return out;
    }

    fn read(self: *Case) !Read {
        const a = self.a;
        var r: Read = .{ .b2s = &.{}, .s2b = &.{}, .stamp = &.{}, .mark = &.{}, .need = &.{}, .miss = &.{}, .fetch = &.{}, .a = a };
        errdefer r.deinit();
        const ctl = try self.pull(self.d_ctl, 8);
        defer a.free(ctl);
        @memcpy(&r.ctl, ctl);
        r.b2s = try self.pull(self.d_b2s, nblocks);
        r.s2b = try self.pull(self.d_s2b, self.shape.n);
        r.stamp = try self.pull(self.d_stamp, self.shape.n);
        r.mark = try self.pull(self.d_mark, nblocks);
        r.need = try self.pull(self.d_need, self.cap);
        r.miss = try self.pull(self.d_miss, self.cap);
        r.fetch = try self.pull(self.d_fetch, 2 * self.cap);
        return r;
    }

    /// The device's image: the lists as long as the model says, each sorted (their order is the atomics').
    fn deviceImage(self: *Case, r: Read, buf: *std.ArrayList(u8)) !void {
        const a = self.a;
        const mo = &self.model;
        const stamp = try a.dupe(u32, r.stamp);
        defer a.free(stamp);
        std.mem.sort(u32, stamp, {}, less);
        const need = try a.dupe(u32, r.need[0..mo.need.len]);
        defer a.free(need);
        std.mem.sort(u32, need, {}, less);
        const miss = try a.dupe(u32, r.miss[0..mo.miss.len]);
        defer a.free(miss);
        std.mem.sort(u32, miss, {}, less);
        const fetched = try a.alloc(u32, mo.assigned);
        defer a.free(fetched);
        for (fetched, 0..) |*f, i| f.* = r.fetch[2 * i];
        std.mem.sort(u32, fetched, {}, less);
        try self.words(buf, r.ctl, need, miss, stamp, r.mark, fetched);
    }

    fn image(ctx: *anyopaque) anyerror![]const u8 {
        const self: *Case = @ptrCast(@alignCast(ctx));
        const r = try self.read();
        defer r.deinit();
        try self.deviceImage(r, &self.image_buf);
        return self.image_buf.items;
    }

    /// Every gate: the image equals the model's, then the placement is one a correct LRU may make.
    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const a = self.a;
        const mo = &self.model;
        const r = try self.read();
        defer r.deinit();
        var got: std.ArrayList(u8) = .empty;
        defer got.deinit(a);
        try self.deviceImage(r, &got);
        var want: std.ArrayList(u8) = .empty;
        defer want.deinit(a);
        try self.modelImage(&want);
        if (std.mem.indexOfDiff(u8, got.items, want.items)) |at| return try std.fmt.bufPrint(&self.msg, "the call's counters, lists or stamps differ from the model's at byte {d} (word {d} of {d})", .{ at, at / 4, want.items.len / 4 });
        // entries past the lists stay as poisoned
        for (r.need[mo.need.len..]) |v| if (v != none) return "kvt_plan wrote past its need list";
        for (r.miss[mo.miss.len..]) |v| if (v != none) return "kvt_plan wrote past its miss list";
        for (r.fetch[2 * mo.assigned ..]) |v| if (v != none) return "kvt_plan wrote past its fetch list";
        return self.placement(r);
    }

    /// The slots hold what the plan fetched, the evicted blocks are unmapped, and nothing else moved.
    fn placement(self: *Case, r: Read) !?[]const u8 {
        const a = self.a;
        const mo = &self.model;
        const slots = self.shape.n;
        const taken = try a.alloc(bool, slots);
        defer a.free(taken);
        @memset(taken, false);
        const gone = try a.alloc(bool, nblocks);
        defer a.free(gone);
        @memset(gone, false);
        const new = try a.alloc(bool, nblocks);
        defer a.free(new);
        @memset(new, false);
        const olds = try a.alloc(u32, mo.assigned);
        defer a.free(olds);
        for (0..mo.assigned) |i| {
            const b = r.fetch[2 * i];
            const s = r.fetch[2 * i + 1];
            if (s >= slots or b >= nblocks) return try std.fmt.bufPrint(&self.msg, "fetch entry {d} names block {d} slot {d}", .{ i, b, s });
            if (taken[s]) return try std.fmt.bufPrint(&self.msg, "slot {d} is given to two blocks", .{s});
            if (mo.hit[s]) return try std.fmt.bufPrint(&self.msg, "slot {d} is a slot this call uses and was taken", .{s});
            if (r.s2b[s] != b or r.b2s[b] != s or r.stamp[s] != mo.epoch) return try std.fmt.bufPrint(&self.msg, "block {d} and slot {d} disagree: block->slot {d}, slot->block {d}, stamp {d}", .{ b, s, r.b2s[b], r.s2b[s], r.stamp[s] });
            taken[s] = true;
            new[b] = true;
            olds[i] = self.stamp[s];
            if (self.s2b[s] != none) gone[self.s2b[s]] = true;
        }
        std.mem.sort(u32, olds, {}, less);
        if (!std.mem.eql(u32, olds, mo.victims)) return "the slots it took are not the least recently used";
        const missed = try a.alloc(bool, nblocks);
        defer a.free(missed);
        @memset(missed, false);
        for (mo.miss) |b| missed[b] = true;
        for (0..nblocks) |b| {
            const want: u32 = if (gone[b] or (missed[b] and !new[b])) none else self.b2s[b];
            if (new[b]) continue;
            if (r.b2s[b] != want) return try std.fmt.bufPrint(&self.msg, "block {d} maps to slot {d}, not {d}", .{ b, r.b2s[b], want });
        }
        for (0..slots) |s| {
            if (taken[s]) continue;
            if (r.s2b[s] != self.s2b[s]) return try std.fmt.bufPrint(&self.msg, "slot {d} holds block {d}, not {d}", .{ s, r.s2b[s], self.s2b[s] });
            const want = if (mo.hit[s]) mo.epoch else self.stamp[s];
            if (r.stamp[s] != want) return try std.fmt.bufPrint(&self.msg, "slot {d} has stamp {d}, not {d}", .{ s, r.stamp[s], want });
        }
        return null;
    }
};

/// `n` distinct picks from `pool` into `out`.
fn sample(rng: std.Random, a: std.mem.Allocator, pool: []const u32, n: usize, out: []u32) !void {
    const work = try a.dupe(u32, pool);
    defer a.free(work);
    for (0..n) |i| {
        const j = i + rng.uintLessThan(usize, work.len - i);
        std.mem.swap(u32, &work[i], &work[j]);
        out[i] = work[i];
    }
}

/// The plan and its control over one loaded image, for the runner's list.
pub const Set = struct {
    mod: cuda.Module,
    each: [2]Case,

    pub fn init(gpu: check.Gpu, s: cuda.Stream) !Set {
        var mod = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_exl3_kvtier);
        errdefer mod.unload();
        const f = try mod.function("kvt_plan");
        return .{ .mod = mod, .each = .{ .{ .a = gpu.gpa, .rig = .{ .gpu = gpu, .s = s }, .f = f }, .{ .a = gpu.gpa, .rig = .{ .gpu = gpu, .s = s }, .f = f, .control = true } } };
    }

    pub fn deinit(self: *Set) void {
        for (&self.each) |*x| x.deinit();
        self.mod.unload();
    }

    pub fn add(self: *Set, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        for (&self.each) |*x| try list.append(a, x.kernel());
    }
};
