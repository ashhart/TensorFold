//! Pending DeltaNet rows under the contract: fn_gdn_front_log and fn_gdn_rows_late, the replaced kernels' bytes.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const parts = flash.batch_state;

pub const Kind = enum { front, chain };

const nk = 16;
const nv = 48;
const dk = 128;
const dv = 128;
const taps = 4;
const conv_c = 2 * nk * dk + nv * dv; // a projection row's conv channels (q, k, v)
const proj_w = conv_c + nv * dv + 2 * nv; // a projection row: conv channels, z, b, a
const lr = parts.late_rows;
const k_row = nk * dk * 4; // a log row's bytes in each of its three parts
const v_row = nv * dv * 4;
const g_row = nv * 2 * 4;
const state_len = nv * dv * dk * 4;

fn noSums(_: c.Shape) usize {
    return 0;
}

/// A verify window's rows: one, a greedy window and the widest a window may be (mtp.max_rows).
const served = [_]c.Shape{
    .{ .rows = 1, .n = nv * dv, .k = dk },
    .{ .rows = 9, .n = nv * dv, .k = dk },
    .{ .rows = 32, .n = nv * dv, .k = dk },
};

pub fn spec(kind: Kind) c.Spec {
    return switch (kind) {
        .front => .{ .name = "cuda flashnext DeltaNet window rows into the pending log fn_gdn_front_log", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "q, and k, v and the gates in the log's slots after the pending rows, bit-equal to fn_gdn_front's; other slots untouched" }, .replaces = "fn_gdn_front", .served = &served },
        .chain => .{ .name = "cuda flashnext DeltaNet chain after pending rows fn_gdn_rows_late", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "ys, the end state and the stored state bit-equal to fn_gdn_rows_to over the pending rows, then over the window's" }, .replaces = "fn_gdn_rows_to", .in_place = true, .served = &served },
    };
}

pub const Fns = struct { front: cuda.Function, front_log: cuda.Function, rows_to: cuda.Function, rows_late: cuda.Function };

pub const Late = struct {
    rig: rig_.Rig,
    kind: Kind,
    f: *const Fns,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    ins: [6]u64 = @splat(0),
    outs: [3]u64 = @splat(0),
    lens: [3]usize = @splat(0),
    nout: usize = 0,
    base: u64 = 0,
    first: usize = 0,
    pending: usize = 0,
    /// The chain stores its state in place: every launch restores these bytes first.
    state: u64 = 0,
    fresh: ?[]u8 = null,
    seen: std.ArrayList(u8) = .empty,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Late) void {
        if (self.fresh) |f| self.rig.gpu.gpa.free(f);
        self.seen.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *Late) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Late);

    fn f32s(self: *Late, n: usize, r: std.Random, lo: f32, hi: f32) ![]f32 {
        const v = try self.rig.gpu.gpa.alloc(f32, n);
        for (v) |*x| x.* = lo + r.float(f32) * (hi - lo);
        return v;
    }

    fn bfs(self: *Late, n: usize, r: std.Random, lo: f32, hi: f32) ![]u16 {
        const v = try self.rig.gpu.gpa.alloc(u16, n);
        for (v) |*x| x.* = c.toBf16(lo + r.float(f32) * (hi - lo));
        return v;
    }

    fn out(self: *Late, len: usize) !u64 {
        const p = try self.rig.output(len);
        self.outs[self.nout] = p;
        self.lens[self.nout] = len;
        self.nout += 1;
        return p;
    }

    /// The log's first pending slot and pending count: none, at the store threshold, past it; the window may wrap.
    fn place(self: *Late, rows: usize, seed: u64) void {
        const want = ([_]usize{ 0, parts.late_store, parts.late_store + 5 })[seed % 3];
        self.pending = @min(want, lr - rows);
        self.first = (lr - 3 + seed / 3) % lr;
    }

    pub fn prepare(self: *Late, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (self.fresh) |f| self.rig.gpu.gpa.free(f);
        self.fresh = null;
        self.state = 0;
        self.nout = 0;
        self.shape = shape;
        const m = shape.rows;
        if (m == 0 or m > 32 or shape.n != nv * dv or shape.k != dk) return false;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6c617465);
        const r = prng.random();
        const a = self.rig.gpu.gpa;
        const bad = fill == .nonfinite;
        self.place(m, seed);
        const ctl = [4]u32{ @intCast(self.first | self.pending << 16), 0, 0, 0 };
        switch (self.kind) {
            .front => {
                const proj = try self.bfs(m * proj_w, r, -1, 1);
                defer a.free(proj);
                const cs = try self.bfs((taps - 1) * conv_c, r, -1, 1);
                defer a.free(cs);
                const cw = try self.bfs(conv_c * taps, r, -0.5, 0.5);
                defer a.free(cw);
                const a_log = try self.f32s(nv, r, -1, 1);
                defer a.free(a_log);
                const dt = try self.f32s(nv, r, -1, 1);
                defer a.free(dt);
                if (bad) {
                    proj[3] = c.toBf16(std.math.nan(f32));
                    proj[(m / 2) * proj_w + 2 * nk * dk + 7] = c.toBf16(std.math.inf(f32));
                    proj[(m - 1) * proj_w + conv_c + nv * dv + 1] = c.toBf16(-std.math.inf(f32));
                }
                for ([_][]const u8{ std.mem.sliceAsBytes(proj), std.mem.sliceAsBytes(cs), std.mem.sliceAsBytes(cw), std.mem.sliceAsBytes(a_log), std.mem.sliceAsBytes(dt), std.mem.sliceAsBytes(&ctl) }, 0..) |b, i| self.ins[i] = try self.rig.input(b);
                _ = try self.out(m * k_row); // q, as fn_gdn_front's
                _ = try self.out(lr * (k_row + v_row + g_row)); // the log: only the window's slots are written
            },
            .chain => {
                const qn = try self.f32s(m * nk * dk, r, -0.1, 0.1);
                defer a.free(qn);
                const ks = try self.f32s(lr * nk * dk, r, -0.1, 0.1);
                defer a.free(ks);
                const vs = try self.f32s(lr * nv * dv, r, -1, 1);
                defer a.free(vs);
                const gs = try self.f32s(lr * nv * 2, r, 0.5, 1); // a decay, then beta
                defer a.free(gs);
                const st = try self.f32s(nv * dv * dk, r, -0.5, 0.5);
                defer a.free(st);
                if (bad) {
                    qn[3] = std.math.nan(f32);
                    vs[((self.first + self.pending) % lr) * nv * dv + 5] = std.math.inf(f32);
                    st[dv * dk + 7] = -std.math.inf(f32);
                }
                const log = try std.mem.concat(a, u8, &.{ std.mem.sliceAsBytes(ks), std.mem.sliceAsBytes(vs), std.mem.sliceAsBytes(gs) });
                defer a.free(log);
                for ([_][]const u8{ std.mem.sliceAsBytes(qn), log, std.mem.sliceAsBytes(&ctl), std.mem.sliceAsBytes(st) }, 0..) |b, i| self.ins[i] = try self.rig.input(b);
                _ = try self.out(m * v_row); // ys
                self.state = try self.out(state_len); // the state read, and stored in place past the threshold
                _ = try self.out(state_len); // the state after the window
                self.fresh = try a.dupe(u8, std.mem.sliceAsBytes(st));
            },
        }
        var total: usize = 0;
        for (self.lens[0..self.nout]) |l| total += l;
        self.base = try self.rig.scratch(total);
        return true;
    }

    fn launchFront(self: *Late, fun: cuda.Function, q: u64, tail: []const u64, log: bool) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ self.ins[0], self.ins[1], self.ins[2], self.ins[3], self.ins[4], q }) |v| a.add(v);
        for (tail) |v| a.add(v);
        if (log) a.add(@as(c_int, @intCast(lr)));
        try cuda.launch.launch(fun, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = nk + nv }, .block = .{ .x = 256 } }, self.rig.s, &a);
    }

    fn rowsTo(self: *Late, q: u64, k: u64, v: u64, gb: u64, in: u64, to: u64, rows: usize, ys: u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ q, k, v, gb, in, to }) |x| a.add(x);
        a.add(@as(c_int, @intCast(rows)));
        a.add(ys);
        try cuda.launch.launch(self.f.rows_to, .{ .grid = .{ .x = nv, .y = dv / 16 }, .block = .{ .x = 128 } }, self.rig.s, &a);
    }

    fn copy(self: *Late, dst: u64, src: u64, len: usize) !void {
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = dst, .len = len }).copyFrom(0, src, len, self.rig.s.handle);
    }

    /// `n` rows from log slot `from` on (wrapping) to or from contiguous rows: the log's three parts.
    fn slots(self: *Late, log: u64, rows: [3]u64, from: usize, n: usize, into_log: bool) !void {
        const sizes = [3]usize{ k_row, v_row, g_row };
        var at = log;
        for (sizes, rows) |size, flat| {
            for (0..n) |i| {
                const slot = at + ((from + i) % lr) * size;
                if (into_log) try self.copy(slot, flat + i * size, size) else try self.copy(flat + i * size, slot, size);
            }
            at += lr * size;
        }
    }

    fn ours(self: *Late, o: []const u64) !void {
        switch (self.kind) {
            .front => try self.launchFront(self.f.front_log, o[0], &.{ o[1], self.ins[5] }, true),
            .chain => {
                var a: cuda.Args = .{};
                for ([_]u64{ self.ins[0], self.ins[1], self.ins[2] }) |v| a.add(v);
                a.add(@as(c_int, @intCast(lr)));
                a.add(@as(c_int, @intCast(parts.late_store)));
                a.add(o[1]);
                a.add(o[2]);
                a.add(@as(c_int, @intCast(self.shape.rows)));
                a.add(o[0]);
                try cuda.launch.launch(self.f.rows_late, .{ .grid = .{ .x = nv, .y = dv / 16 }, .block = .{ .x = 128 } }, self.rig.s, &a);
            },
        }
    }

    /// The replaced kernels: fn_gdn_front, rows copied to slots; fn_gdn_rows_to over the pending rows, then the window.
    fn theirs(self: *Late, o: []const u64) !void {
        const m = self.shape.rows;
        const after = self.first + self.pending;
        switch (self.kind) {
            .front => {
                const flat = [3]u64{ try self.rig.scratch(m * k_row), try self.rig.scratch(m * v_row), try self.rig.scratch(m * g_row) };
                try self.launchFront(self.f.front, o[0], &flat, false);
                try self.slots(o[1], flat, after, m, true);
            },
            .chain => {
                const mid = try self.rig.scratch(state_len);
                if (self.pending > 0) {
                    const np = self.pending;
                    const pend = [3]u64{ try self.rig.scratch(np * k_row), try self.rig.scratch(np * v_row), try self.rig.scratch(np * g_row) };
                    try self.slots(self.ins[1], pend, self.first, np, false);
                    // the pending rows' read-outs are dropped: their q is zero, their ys go to scratch
                    try self.rowsTo(try self.rig.scratch(np * k_row), pend[0], pend[1], pend[2], self.ins[3], mid, np, try self.rig.scratch(np * v_row));
                } else try self.copy(mid, self.ins[3], state_len);
                const win = [3]u64{ try self.rig.scratch(m * k_row), try self.rig.scratch(m * v_row), try self.rig.scratch(m * g_row) };
                try self.slots(self.ins[1], win, after, m, false);
                try self.rowsTo(self.ins[0], win[0], win[1], win[2], mid, o[2], m, o[0]);
                try self.copy(o[1], if (self.pending > parts.late_store) mid else self.ins[3], state_len);
            },
        }
    }

    fn restoreState(self: *Late) !void {
        const f = self.fresh orelse return;
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.state, .len = f.len }).upload(0, f);
    }

    pub fn run(self: *Late) !void {
        try self.restoreState();
        try self.ours(self.outs[0..self.nout]);
        try self.rig.s.synchronize();
    }

    /// The scratch the replaced kernels write: the outputs' layout back to back, poisoned like them (0xff).
    fn baseline(self: *Late) ![]const u8 {
        var at = self.base;
        var o: [3]u64 = undefined;
        var total: usize = 0;
        for (self.lens[0..self.nout], 0..) |l, i| {
            o[i] = at;
            at += l;
            total += l;
        }
        const ff = try self.rig.gpu.gpa.alloc(u8, total);
        defer self.rig.gpu.gpa.free(ff);
        @memset(ff, 0xff);
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.base, .len = total }).upload(0, ff);
        try self.theirs(o[0..self.nout]);
        try self.rig.s.synchronize();
        return self.rig.bytes(self.base, total);
    }

    pub fn replaced(self: *Late) ![]const u8 {
        return self.baseline();
    }

    /// Gate 1's stated rule: the outputs on the planted inputs equal the replaced kernels' bytes.
    pub fn check(self: *Late, gate: g.Gate) !?[]const u8 {
        _ = gate;
        self.seen.clearRetainingCapacity();
        try self.seen.appendSlice(self.rig.gpu.gpa, try self.rig.outputBytes());
        const want = try self.baseline();
        if (std.mem.eql(u8, self.seen.items, want)) return null;
        return try std.fmt.bufPrint(&self.msg, "bytes differ from the replaced kernels' from byte {d}", .{std.mem.indexOfDiff(u8, self.seen.items, want).?});
    }

    pub fn count(self: *Late) usize {
        var total: usize = 0;
        for (self.lens[0..self.nout]) |l| total += l;
        return total;
    }

    pub fn reference(_: *Late, _: []f64, _: []f64) !void {
        return error.NoFloat64Reference;
    }
};
