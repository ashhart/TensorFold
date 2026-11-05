//! A shared pass's lanes in one launch a DeltaNet step (fn_gdn_lanes_*) against the per-lane launches they replace.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const parts = flash.batch_state;
const gdn_lanes = flash.batch_gdn_lanes;

pub const Kind = enum { keep, front, chain, shift };

const nk = 16;
const nv = 48;
const dk = 128;
const dv = 128;
const taps = 4;
const conv_c = 2 * nk * dk + nv * dv; // a projection row's conv channels (q, k, v)
const proj_w = conv_c + nv * dv + 2 * nv; // a projection row: conv channels, z, b, a
const lr = parts.late_rows;
const k_row = nk * dk * 4;
const v_row = nv * dv * 4;
const g_row = nv * 2 * 4;
const log_len = lr * (k_row + v_row + g_row);
const state_len = nv * dv * dk * 4;
const hist = (taps - 1) * conv_c * 2;
const row_b = proj_w * 2;
const max_lanes = gdn_lanes.gdn_lanes_max;

/// A lane's DeltaNet: in place (decode), kept for a rollback (a state copy), or with pending rows (a log too).
const Mode = enum { plain, keep, late };
const Cfg = struct { mode: Mode = .plain, rows: usize = 0, row0: usize = 0, first: usize = 0, pending: usize = 0, slot: usize = 0 };

fn noSums(_: c.Shape) usize {
    return 0;
}

/// rows: a pass's lanes; one (a single stream's round), two, five (mixed modes), the most it batches; 1-32 rows each.
const served = [_]c.Shape{
    .{ .rows = 1, .n = nv * dv, .k = dk },
    .{ .rows = 2, .n = nv * dv, .k = dk },
    .{ .rows = 5, .n = nv * dv, .k = dk },
    .{ .rows = max_lanes, .n = nv * dv, .k = dk },
};

pub fn spec(kind: Kind) c.Spec {
    return switch (kind) {
        .keep => .{ .name = "cuda flashnext lanes' DeltaNet keep copies fn_gdn_lanes_keep", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "each kept lane's rows and conv window bit-equal to their copies; other lanes' none" }, .replaces = "a copy of each lane's rows and conv window", .served = &served },
        .front => .{ .name = "cuda flashnext lanes' DeltaNet front fn_gdn_lanes_front", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "every lane's q, k, v and gates (or log slots) bit-equal to its fn_gdn_front or fn_gdn_front_log" }, .replaces = "fn_gdn_front or fn_gdn_front_log a lane", .served = &served },
        .chain => .{ .name = "cuda flashnext lanes' DeltaNet chain fn_gdn_lanes_chain", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "every lane's ys and states bit-equal to its fn_gdn_rows, fn_gdn_rows_to or fn_gdn_rows_late" }, .replaces = "fn_gdn_rows, fn_gdn_rows_to or fn_gdn_rows_late a lane", .in_place = true, .served = &served },
        .shift => .{ .name = "cuda flashnext lanes' conv windows moved on fn_gdn_lanes_shift", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "every lane's conv window bit-equal to fn_shift_window over its rows" }, .replaces = "fn_shift_window a lane", .in_place = true, .served = &served },
    };
}

pub const Fns = struct {
    front: cuda.Function,
    front_log: cuda.Function,
    rows: cuda.Function,
    rows_to: cuda.Function,
    rows_late: cuda.Function,
    shift: cuda.Function,
    lanes_keep: cuda.Function,
    lanes_front: cuda.Function,
    lanes_chain: cuda.Function,
    lanes_shift: cuda.Function,
};

pub const Lanes = struct {
    rig: rig_.Rig,
    kind: Kind,
    f: *const Fns,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    cfg: [max_lanes]Cfg = @splat(.{}),
    total_rows: usize = 0,
    /// proj, conv windows, cw, a_log, dt, ctl (keep / front / shift); q, k, v, gb, logs, ctl (chain).
    ins: [6]u64 = @splat(0),
    out: u64 = 0,
    len: usize = 0,
    base: u64 = 0,
    /// The output's in-place part (states, conv windows): restored to these bytes before every launch.
    fresh_at: usize = 0,
    fresh: ?[]u8 = null,
    seen: std.ArrayList(u8) = .empty,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Lanes) void {
        if (self.fresh) |f| self.rig.gpu.gpa.free(f);
        self.seen.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *Lanes) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Lanes);

    fn f32s(self: *Lanes, n: usize, r: std.Random, lo: f32, hi: f32) ![]f32 {
        const v = try self.rig.gpu.gpa.alloc(f32, n);
        for (v) |*x| x.* = lo + r.float(f32) * (hi - lo);
        return v;
    }

    fn bfs(self: *Lanes, n: usize, r: std.Random, lo: f32, hi: f32) ![]u16 {
        const v = try self.rig.gpu.gpa.alloc(u16, n);
        for (v) |*x| x.* = c.toBf16(lo + r.float(f32) * (hi - lo));
        return v;
    }

    fn lanes(self: *const Lanes) []const Cfg {
        return self.cfg[0..self.shape.rows];
    }

    /// Modes cycle over the lanes; a late lane's pending rows: none, at the store threshold, past it, the log wrapping.
    fn plan(self: *Lanes, r: std.Random, seed: u64) void {
        var row0: usize = 0;
        var late: usize = 0;
        for (self.cfg[0..self.shape.rows], 0..) |*x, i| {
            x.* = .{ .mode = @fromBackingInt(@intCast((seed + i) % 3)), .rows = 1 + r.uintLessThan(usize, 32), .row0 = row0 };
            if (x.mode == .late) {
                const want = ([_]usize{ 0, parts.late_store, parts.late_store + 5 })[(seed / 3 + i) % 3];
                x.pending = @min(want, lr - x.rows);
                x.first = (lr - 3 + 7 * i + seed) % lr;
                x.slot = late;
                late += 1;
            }
            row0 += x.rows;
        }
        self.total_rows = row0;
    }

    fn ctlBytes(self: *Lanes) ![]u32 {
        const ctl = try self.rig.gpu.gpa.alloc(u32, max_lanes * 4);
        @memset(ctl, 0);
        for (self.lanes(), 0..) |x, i| ctl[i * 4] = @intCast(x.first | x.pending << 16);
        return ctl;
    }

    fn lateCount(self: *const Lanes) usize {
        var n: usize = 0;
        for (self.lanes()) |x| n += @intFromBool(x.mode == .late);
        return n;
    }

    pub fn prepare(self: *Lanes, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (self.fresh) |f| self.rig.gpu.gpa.free(f);
        self.fresh = null;
        self.shape = shape;
        self.len = 0;
        self.fresh_at = 0;
        if (shape.rows == 0 or shape.rows > max_lanes or shape.n != nv * dv or shape.k != dk) return false;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6c616e65);
        const r = prng.random();
        const a = self.rig.gpu.gpa;
        const bad = fill == .nonfinite;
        self.plan(r, seed);
        const m = self.total_rows;
        const nl = shape.rows;
        switch (self.kind) {
            .keep, .front, .shift => {
                const proj = try self.bfs(m * proj_w, r, -1, 1);
                defer a.free(proj);
                const cs = try self.bfs(nl * hist / 2, r, -1, 1);
                defer a.free(cs);
                const cw = try self.bfs(conv_c * taps, r, -0.5, 0.5);
                defer a.free(cw);
                const a_log = try self.f32s(nv, r, -1, 1);
                defer a.free(a_log);
                const dt = try self.f32s(nv, r, -1, 1);
                defer a.free(dt);
                const ctl = try self.ctlBytes();
                defer a.free(ctl);
                if (bad) {
                    proj[3] = c.toBf16(std.math.nan(f32));
                    proj[(m / 2) * proj_w + 2 * nk * dk + 7] = c.toBf16(std.math.inf(f32));
                    proj[(m - 1) * proj_w + conv_c + nv * dv + 1] = c.toBf16(-std.math.inf(f32));
                    cs[11] = c.toBf16(std.math.inf(f32));
                }
                for ([_][]const u8{ std.mem.sliceAsBytes(proj), std.mem.sliceAsBytes(cs), std.mem.sliceAsBytes(cw), std.mem.sliceAsBytes(a_log), std.mem.sliceAsBytes(dt), std.mem.sliceAsBytes(ctl) }, 0..) |b, i| self.ins[i] = try self.rig.input(b);
                switch (self.kind) {
                    .keep => {
                        // each kept lane's rows then its conv window, back to back (slot: the byte offset)
                        for (self.cfg[0..nl]) |*x| {
                            if (x.mode == .plain) continue;
                            x.slot = self.len;
                            self.len += x.rows * row_b + hist;
                        }
                        self.len = @max(self.len, 16);
                    },
                    .front => self.len = m * (2 * k_row + v_row + g_row) + self.lateCount() * log_len,
                    .shift => {
                        self.len = nl * hist;
                        self.fresh_at = 0;
                        self.fresh = try a.dupe(u8, std.mem.sliceAsBytes(cs));
                    },
                    .chain => unreachable,
                }
            },
            .chain => {
                const q = try self.f32s(m * nk * dk, r, -0.1, 0.1);
                defer a.free(q);
                const ks = try self.f32s(m * nk * dk, r, -0.1, 0.1);
                defer a.free(ks);
                const vs = try self.f32s(m * nv * dv, r, -1, 1);
                defer a.free(vs);
                const gs = try self.f32s(m * nv * 2, r, 0.5, 1); // a decay, then beta
                defer a.free(gs);
                const logs = try self.f32s(@max(1, self.lateCount()) * log_len / 4, r, 0.5, 1);
                defer a.free(logs);
                // each log's k and v parts as the front writes them, the gates kept in (0.5, 1)
                for (0..self.lateCount()) |j| {
                    const l = logs[j * log_len / 4 ..][0 .. log_len / 4];
                    for (l[0 .. lr * nk * dk]) |*x| x.* = (x.* - 0.75) * 0.4;
                    for (l[lr * nk * dk ..][0 .. lr * nv * dv]) |*x| x.* = (x.* - 0.75) * 4;
                }
                const st = try self.f32s(nl * state_len / 4, r, -0.5, 0.5);
                defer a.free(st);
                const ctl = try self.ctlBytes();
                defer a.free(ctl);
                if (bad) {
                    q[3] = std.math.nan(f32);
                    vs[(m / 2) * nv * dv + 5] = std.math.inf(f32);
                    st[dv * dk + 7] = -std.math.inf(f32);
                }
                for ([_][]const u8{ std.mem.sliceAsBytes(q), std.mem.sliceAsBytes(ks), std.mem.sliceAsBytes(vs), std.mem.sliceAsBytes(gs), std.mem.sliceAsBytes(logs), std.mem.sliceAsBytes(ctl) }, 0..) |b, i| self.ins[i] = try self.rig.input(b);
                // a kept or late lane's end state goes apart (slot: its index among those)
                var outs: usize = 0;
                for (self.cfg[0..nl]) |*x| {
                    if (x.mode == .plain) continue;
                    x.slot = outs;
                    outs += 1;
                }
                self.fresh_at = m * v_row;
                self.fresh = try a.dupe(u8, std.mem.sliceAsBytes(st));
                self.len = self.fresh_at + nl * state_len + outs * state_len;
            },
        }
        self.out = try self.rig.output(self.len);
        self.base = try self.rig.scratch(self.len);
        return true;
    }

    /// The table cuda_batch_gdn_lanes.once builds for these lanes, its outputs in the region at `o`.
    fn table(self: *const Lanes, o: u64) gdn_lanes.GdnLanes {
        var t: gdn_lanes.GdnLanes = .{ .n = @intCast(self.shape.rows) };
        const late_at = o + self.total_rows * (2 * k_row + v_row + g_row);
        for (self.lanes(), t.l[0..self.shape.rows], 0..) |x, *l, i| {
            l.* = .{ .row0 = @intCast(x.row0), .rows = @intCast(x.rows) };
            const ctl = self.ins[5] + i * 16;
            switch (self.kind) {
                .keep => {
                    l.proj = self.ins[0] + x.row0 * row_b;
                    l.conv = self.ins[1] + i * hist;
                    if (x.mode != .plain) {
                        l.keep_proj = o + x.slot;
                        l.keep_hist = o + x.slot + x.rows * row_b;
                    }
                },
                .front => {
                    l.proj = self.ins[0] + x.row0 * row_b;
                    l.conv = self.ins[1] + i * hist;
                    if (x.mode == .late) {
                        l.log = late_at + x.slot * log_len;
                        l.ctl = ctl;
                    }
                },
                .chain => {
                    l.state = o + self.fresh_at + i * state_len;
                    if (x.mode != .plain) l.state_out = o + self.fresh_at + (self.shape.rows + x.slot) * state_len;
                    if (x.mode == .late) {
                        l.log = self.ins[4] + self.lateSlot(i) * log_len;
                        l.ctl = ctl;
                    }
                },
                .shift => {
                    l.proj = self.ins[0] + x.row0 * row_b;
                    l.conv = o + i * hist;
                },
            }
        }
        return t;
    }

    fn lateSlot(self: *const Lanes, lane: usize) usize {
        var n: usize = 0;
        for (self.lanes()[0..lane]) |x| n += @intFromBool(x.mode == .late);
        return n;
    }

    fn cint(v: usize) c_int {
        return @intCast(v);
    }

    fn widest(self: *const Lanes) usize {
        var w: usize = 0;
        for (self.lanes()) |x| w = @max(w, x.rows);
        return w;
    }

    /// The four launches as cuda_batch_gdn_lanes.once makes them.
    fn ours(self: *Lanes, o: u64) !void {
        const t = self.table(o);
        const n: u32 = @intCast(self.shape.rows);
        const m = self.total_rows;
        var a: cuda.Args = .{};
        switch (self.kind) {
            .keep => {
                a.add(t);
                try cuda.launch.launch(self.f.lanes_keep, .{ .grid = .{ .x = 64, .y = n }, .block = .{ .x = 256 } }, self.rig.s, &a);
            },
            .front => {
                for ([_]u64{ self.ins[2], self.ins[3], self.ins[4], o, o + m * k_row, o + 2 * m * k_row, o + m * (2 * k_row + v_row) }) |v| a.add(v);
                a.add(cint(lr));
                a.add(t);
                try cuda.launch.launch(self.f.lanes_front, .{ .grid = .{ .x = @intCast(self.widest()), .y = nk + nv, .z = n }, .block = .{ .x = 256 } }, self.rig.s, &a);
            },
            .chain => {
                for ([_]u64{ self.ins[0], self.ins[1], self.ins[2], self.ins[3], o }) |v| a.add(v);
                a.add(cint(lr));
                a.add(cint(parts.late_store));
                a.add(t);
                try cuda.launch.launch(self.f.lanes_chain, .{ .grid = .{ .x = nv, .y = dv / 16, .z = n }, .block = .{ .x = 128 } }, self.rig.s, &a);
            },
            .shift => {
                a.add(t);
                try cuda.launch.launch(self.f.lanes_shift, .{ .grid = .{ .x = (conv_c + 255) / 256, .y = n }, .block = .{ .x = 256 } }, self.rig.s, &a);
            },
        }
    }

    fn copy(self: *Lanes, dst: u64, src: u64, len: usize) !void {
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = dst, .len = len }).copyFrom(0, src, len, self.rig.s.handle);
    }

    /// The per-lane launches cuda_batch.zig's lane loop makes, on the same table's buffers.
    fn theirs(self: *Lanes, o: u64) !void {
        const t = self.table(o);
        const m = self.total_rows;
        for (self.lanes(), t.l[0..self.shape.rows]) |x, l| {
            var a: cuda.Args = .{};
            const r0 = x.row0;
            switch (self.kind) {
                .keep => if (x.mode != .plain) {
                    try self.copy(l.keep_proj, l.proj, x.rows * row_b);
                    try self.copy(l.keep_hist, l.conv, hist);
                },
                .front => {
                    for ([_]u64{ l.proj, l.conv, self.ins[2], self.ins[3], self.ins[4], o + r0 * k_row }) |v| a.add(v);
                    if (l.log != 0) {
                        a.add(l.log);
                        a.add(l.ctl);
                        a.add(cint(lr));
                    } else for ([_]u64{ o + m * k_row + r0 * k_row, o + 2 * m * k_row + r0 * v_row, o + m * (2 * k_row + v_row) + r0 * g_row }) |v| a.add(v);
                    try cuda.launch.launch(if (l.log != 0) self.f.front_log else self.f.front, .{ .grid = .{ .x = @intCast(x.rows), .y = nk + nv }, .block = .{ .x = 256 } }, self.rig.s, &a);
                },
                .chain => {
                    const q = self.ins[0] + r0 * k_row;
                    const ys = o + r0 * v_row;
                    if (l.log != 0) {
                        for ([_]u64{ q, l.log, l.ctl }) |v| a.add(v);
                        a.add(cint(lr));
                        a.add(cint(parts.late_store));
                        a.add(l.state);
                        a.add(l.state_out);
                    } else {
                        for ([_]u64{ q, self.ins[1] + r0 * k_row, self.ins[2] + r0 * v_row, self.ins[3] + r0 * g_row, l.state }) |v| a.add(v);
                        if (l.state_out != 0) a.add(l.state_out);
                    }
                    a.add(cint(x.rows));
                    a.add(ys);
                    const fun = if (l.log != 0) self.f.rows_late else if (l.state_out != 0) self.f.rows_to else self.f.rows;
                    try cuda.launch.launch(fun, .{ .grid = .{ .x = nv, .y = dv / 16 }, .block = .{ .x = 128 } }, self.rig.s, &a);
                },
                .shift => {
                    a.add(l.conv);
                    a.add(l.proj);
                    a.add(@as(c_longlong, proj_w));
                    a.add(cint(x.rows));
                    a.add(cint(taps - 1));
                    a.add(cint(conv_c));
                    try cuda.launch.launch(self.f.shift, .{ .grid = .{ .x = (conv_c + 255) / 256 }, .block = .{ .x = 256 } }, self.rig.s, &a);
                },
            }
        }
    }

    fn restore(self: *Lanes, o: u64) !void {
        const f = self.fresh orelse return;
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = o + self.fresh_at, .len = f.len }).upload(0, f);
    }

    pub fn run(self: *Lanes) !void {
        try self.restore(self.out);
        try self.ours(self.out);
        try self.rig.s.synchronize();
    }

    /// The per-lane launches into scratch laid out as the output, poisoned like it (0xff), in-place parts restored.
    fn baseline(self: *Lanes) ![]const u8 {
        const ff = try self.rig.gpu.gpa.alloc(u8, self.len);
        defer self.rig.gpu.gpa.free(ff);
        @memset(ff, 0xff);
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.base, .len = self.len }).upload(0, ff);
        try self.restore(self.base);
        try self.theirs(self.base);
        try self.rig.s.synchronize();
        return self.rig.bytes(self.base, self.len);
    }

    pub fn replaced(self: *Lanes) ![]const u8 {
        return self.baseline();
    }

    /// Gate 1's stated rule: the outputs on the planted inputs equal the per-lane launches' bytes.
    pub fn check(self: *Lanes, gate: g.Gate) !?[]const u8 {
        _ = gate;
        self.seen.clearRetainingCapacity();
        try self.seen.appendSlice(self.rig.gpu.gpa, try self.rig.outputBytes());
        const want = try self.baseline();
        if (std.mem.eql(u8, self.seen.items, want)) return null;
        return try std.fmt.bufPrint(&self.msg, "bytes differ from the per-lane launches' from byte {d}", .{std.mem.indexOfDiff(u8, self.seen.items, want).?});
    }

    pub fn count(self: *Lanes) usize {
        return self.len;
    }

    pub fn reference(_: *Lanes, _: []f64, _: []f64) !void {
        return error.NoFloat64Reference;
    }
};
