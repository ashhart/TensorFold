//! A Mamba-2 window's launches under the contract on HIP, each bit-equal to the kernels it replaces row by row.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");
const mamba = @import("contract_mamba.zig");
const scan_k = @import("contract_scan.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

/// The launch's argument, recurrence/mamba.hip's MambaWindow.
const Window = @typeInfo(@TypeOf(hip.Launcher.tf_mamba_window)).@"fn".param_types[1].?;

/// A draft window's rows at tp 1 and 2, and 17 rows on a tp 4 rank: the scan's 16-row stage and one past it.
const served = [_]c.Shape{
    .{ .rows = 2, .n = 64, .k = 5 },
    .{ .rows = 4, .n = 32, .k = 5 },
    .{ .rows = 17, .n = 16, .k = 5 },
};

pub const Part = enum { conv, dt };

pub fn convSpec(part: Part, k: kit.Kind) c.Spec {
    var s = mamba.spec(if (part == .conv) .conv else .dt, k);
    const names = [2][2][]const u8{
        .{
            "hip mamba tf_mamba_conv_window (xc and tails, fp16)",
            "hip mamba tf_mamba_conv_window (xc and tails, bf16)",
        },
        .{ "hip mamba tf_mamba_conv_window (dt, fp16 in)", "hip mamba tf_mamba_conv_window (dt, bf16 in)" },
    };
    s.name = names[@intFromBool(part == .dt)][@intFromBool(k == .bf16)];
    s.replaces = if (part == .conv) "tf_mamba_conv" else "tf_mamba_dt";
    s.served = &served;
    return s;
}

/// tf_mamba_conv_window (the conv, each row's tail kept, dt): one part's outputs a registration, the other's scratch.
pub const ConvWindow = struct {
    m: mamba.Mamba,
    rig: *rig_.Rig,
    part: Part,
    tail_bytes: []u8 = &.{},
    bias: u64 = 0,
    xc: u64 = 0,
    tail: u64 = 0,
    snaps: u64 = 0,
    dt: u64 = 0,
    base: [3]u64 = @splat(0), // the replaced kernels' xc, tail and slots, or dt

    /// In place: the rig is the inner test's.
    pub fn init(
        self: *ConvWindow,
        a: std.mem.Allocator,
        rt: *const hip.Runtime,
        s: hip.Stream,
        l: *const hip.Launcher,
        part: Part,
    ) void {
        self.* = .{ .m = .init(a, rt, s, l, if (part == .conv) .conv else .dt), .rig = undefined, .part = part };
        self.rig = &self.m.rig;
    }

    pub fn deinit(self: *ConvWindow) void {
        self.m.deinit();
    }

    pub fn kernel(self: *ConvWindow) g.Kernel {
        return .{ .spec = convSpec(self.part, self.m.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(ConvWindow);

    fn tailLen(self: *const ConvWindow) usize {
        return 3 * self.m.d.cd() * kit.size(self.m.k);
    }

    /// The inner test's inputs, then the window's other buffers: its outputs, or scratch for the other part's.
    pub fn prepare(self: *ConvWindow, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try self.m.prepare(shape, fill, seed)) return false;
        const a = self.m.arena.allocator();
        const r = self.rig;
        const n = shape.rows;
        const cd = self.m.d.cd();
        const kb = kit.size(self.m.k);
        if (self.part == .conv) {
            self.xc = self.m.out[0];
            self.tail = self.m.out[1];
            const tb = try a.alloc(u8, self.tailLen());
            for (self.m.tail, 0..) |v, i| kit.put(self.m.k, tb, i, v);
            self.tail_bytes = tb;
            self.snaps = try r.output((n - 1) * self.tailLen());
            var prng = std.Random.DefaultPrng.init(seed ^ 0xb1a5);
            self.bias = try r.input(try kit.f32Bytes(a, try kit.floats(a, prng.random(), shape.n, -0.5, 0.5)));
            self.dt = try r.scratch(n * shape.n * 4);
            const lens = [_]usize{ n * cd * kb, self.tailLen(), (n - 1) * self.tailLen() };
            for (&self.base, lens) |*at, len| at.* = try r.scratch(len);
            return true;
        }
        // dt's part: zero x columns, conv weights and tail, so the conv's half of the launch reads finite zeros
        self.dt = self.m.out[0];
        self.bias = self.m.in[1];
        self.tail_bytes = try a.alloc(u8, self.tailLen());
        @memset(self.tail_bytes, 0);
        self.tail = try r.scratch(self.tailLen());
        self.xc = try r.scratch(n * cd * kb);
        self.snaps = try r.scratch((n - 1) * self.tailLen());
        self.m.in[2] = try r.scratch(4 * cd * 4);
        self.m.in[3] = try r.scratch(cd * 4);
        self.base[0] = try r.scratch(n * shape.n * 4);
        return true;
    }

    fn window(self: *ConvWindow) Window {
        const d = self.m.d;
        const stride = self.tailLen();
        return .{
            .p = self.m.in[0],
            .kind = @backingInt(self.m.k),
            .conv = self.tail,
            .ssm = 0,
            .xc = self.xc,
            .dt = self.dt,
            .y = 0,
            .cw = self.m.in[2],
            .cb = self.m.in[3],
            .dt_bias = self.bias,
            .a_log = 0,
            .dsk = 0,
            .snap_conv = if (self.m.shape.rows > 1) self.snaps - stride else 0,
            .stride = @intCast(stride),
            .rows = @intCast(self.m.shape.rows),
            .proj = @intCast(d.proj()),
            .xd = @intCast(d.xd()),
            .cd = @intCast(d.cd()),
            .heads = @intCast(d.heads),
            .dh = 64,
            .groups = @intCast(d.groups()),
            .lo = 0,
            .hi = std.math.inf(f32),
        };
    }

    /// The tail back to the prepared one, then one launch.
    pub fn run(self: *ConvWindow) !void {
        try self.rig.put(self.tail, self.tail_bytes);
        const w = self.window();
        var a: hip.launch.Args = .{};
        try a.add(w);
        const blocks = (self.m.d.cd() + 255) / 256 + (self.m.shape.rows * self.m.d.heads + 255) / 256;
        try hip.launch.launch(
            self.m.l.mamba.conv_window,
            .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = 256 } },
            self.rig.s,
            &a,
        );
        try self.rig.s.synchronize();
    }

    /// tf_mamba_conv over the rows into the baseline, and over the first r rows for slot r's tail; or tf_mamba_dt.
    pub fn replaced(self: *ConvWindow) ![]const u8 {
        const m = &self.m;
        const d = m.d;
        const st = self.rig.s.handle;
        const kind = @backingInt(m.k);
        const C = ?*const anyopaque;
        const P = ?*anyopaque;
        const CF = ?[*]const f32;
        const n = m.shape.rows;
        if (self.part == .dt) {
            try m.l.tf_mamba_dt(
                kit.ptr(C, m.in[0]),
                kind,
                kit.ptr(CF, self.bias),
                kit.ptr(?[*]f32, self.base[0]),
                @intCast(n),
                @intCast(d.proj()),
                @intCast(d.xd() + d.cd()),
                @intCast(d.heads),
                0,
                std.math.inf(f32),
                st,
            );
            return self.rig.bytes(self.base[0], n * d.heads * 4);
        }
        for (0..n) |r| {
            const rows = if (r == 0) n else r; // the whole window first, then each slot's rows
            const out = if (r == 0) self.base[1] else self.base[2] + (r - 1) * self.tailLen();
            try m.l.tf_mamba_conv(
                kit.ptr(C, m.in[0]),
                kind,
                kit.ptr(C, m.in[1]),
                kit.ptr(P, self.base[0]),
                kit.ptr(CF, m.in[2]),
                kit.ptr(CF, m.in[3]),
                kit.ptr(P, out),
                @intCast(rows),
                @intCast(d.proj()),
                @intCast(d.xd()),
                @intCast(d.cd()),
                st,
            );
        }
        const a = m.arena.allocator();
        var all: std.ArrayList(u8) = .empty;
        const lens = [3]usize{ n * d.cd() * kit.size(m.k), self.tailLen(), (n - 1) * self.tailLen() };
        for (self.base, lens) |at, len| try all.appendSlice(a, try self.rig.bytes(at, len));
        return all.items;
    }

    pub fn count(self: *ConvWindow) usize {
        if (self.part == .dt) return self.m.count();
        return self.m.count() + (self.m.shape.rows - 1) * 3 * self.m.d.cd();
    }

    /// Slot r's tail: the window's raw inputs r to r + 2.
    fn snapshots(self: *ConvWindow, out: []f64) void {
        const d = self.m.d;
        const n = self.m.shape.rows;
        for (1..n) |r| for (0..3) |j| for (0..d.cd()) |ch| {
            const t = r + j;
            const v = if (t < 3) self.m.tail[t * d.cd() + ch] else self.m.p[(t - 3) * d.proj() + d.xd() + ch];
            out[((r - 1) * 3 + j) * d.cd() + ch] = v;
        };
    }

    pub fn reference(self: *ConvWindow, want: []f64, mag: []f64) !void {
        const own = self.m.count();
        try self.m.reference(want[0..own], mag[0..own]);
        if (self.part == .dt) return;
        self.snapshots(want[own..]);
        @memset(mag[own..], 0);
    }

    pub fn simulate(self: *ConvWindow, acc: c.Accum, order: g.Order, out: []f64) !void {
        const own = self.m.count();
        try self.m.simulate(acc, order, out[0..own]);
        self.snapshots(out[own..]);
    }
};

pub fn scanSpec(part: scan_k.Part, k: kit.Kind) c.Spec {
    var s = scan_k.spec(part, k);
    const names = [2][2][]const u8{
        .{ "hip mamba tf_mamba_scan_snap (y, fp16)", "hip mamba tf_mamba_scan_snap (y, bf16)" },
        .{
            "hip mamba tf_mamba_scan_snap (the state and its slots)",
            "hip mamba tf_mamba_scan_snap (the state and its slots)",
        },
    };
    s.name = names[@intFromBool(part == .state)][@intFromBool(k == .bf16)];
    s.replaces = "tf_mamba_scan";
    s.served = &shapes_k128;
    return s;
}

const shapes_k128 = blk: {
    var x = served;
    for (&x) |*v| v.k = 128;
    break :blk x;
};

/// tf_mamba_scan_snap: the scan over the window's rows, each state but the last kept in its slot.
pub const ScanWindow = struct {
    m: scan_k.Scan,
    rig: *rig_.Rig,
    snaps: u64 = 0,
    base: [3]u64 = @splat(0), // the replaced scan's y, state and slots

    pub fn init(
        self: *ScanWindow,
        a: std.mem.Allocator,
        rt: *const hip.Runtime,
        s: hip.Stream,
        l: *const hip.Launcher,
        part: scan_k.Part,
    ) void {
        self.* = .{ .m = .init(a, rt, s, l, part), .rig = undefined };
        self.rig = &self.m.rig;
    }

    pub fn deinit(self: *ScanWindow) void {
        self.m.deinit();
    }

    pub fn kernel(self: *ScanWindow) g.Kernel {
        return .{ .spec = scanSpec(self.m.part, self.m.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(ScanWindow);

    pub fn prepare(self: *ScanWindow, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try self.m.prepare(shape, fill, seed)) return false;
        const r = self.rig;
        const each = self.m.st_bytes.len;
        const slots = (shape.rows - 1) * each;
        self.snaps = if (self.m.part == .state) try r.output(slots) else try r.scratch(slots);
        const lens = [3]usize{ shape.rows * self.m.d.xd() * kit.size(self.m.k), each, slots };
        for (&self.base, lens) |*at, len| at.* = try r.scratch(len);
        return true;
    }

    /// The state back to the prepared one, then one launch.
    pub fn run(self: *ScanWindow) !void {
        const m = &self.m;
        try self.rig.put(m.st_out, m.st_bytes);
        const d = m.d;
        const w: Window = .{
            .p = m.in[0],
            .kind = @backingInt(m.k),
            .conv = 0,
            .ssm = m.st_out,
            .xc = m.in[1],
            .dt = m.in[2],
            .y = m.y,
            .cw = 0,
            .cb = 0,
            .dt_bias = 0,
            .a_log = m.in[3],
            .dsk = m.in[4],
            .snap_conv = 0,
            .snap_ssm = if (m.shape.rows > 1) self.snaps - m.st_bytes.len else 0,
            .stride = @intCast(m.st_bytes.len),
            .rows = @intCast(m.shape.rows),
            .proj = @intCast(d.proj()),
            .xd = @intCast(d.xd()),
            .cd = @intCast(d.cd()),
            .heads = @intCast(d.heads),
            .dh = 64,
            .groups = @intCast(d.groups()),
            .lo = 0,
            .hi = std.math.inf(f32),
        };
        var a: hip.launch.Args = .{};
        try a.add(w);
        const grid: hip.launch.Dim3 = .{ .x = @intCast(d.heads), .y = 2 };
        try hip.launch.launch(m.l.mamba.scan_snap, .{ .grid = grid, .block = .{ .x = 128 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    /// tf_mamba_scan over the rows from the prepared state, and over the first r rows for slot r's state.
    pub fn replaced(self: *ScanWindow) ![]const u8 {
        const m = &self.m;
        const d = m.d;
        const n = m.shape.rows;
        const C = ?*const anyopaque;
        const CF = ?[*]const f32;
        for (0..n) |r| {
            const rows = if (r == 0) n else r;
            const st = if (r == 0) self.base[1] else self.base[2] + (r - 1) * m.st_bytes.len;
            try self.rig.put(st, m.st_bytes);
            const v = [_]usize{ rows, d.proj(), d.xd(), d.cd(), d.heads, 64, d.groups() };
            try m.l.tf_mamba_scan(
                kit.ptr(C, m.in[0]),
                kit.ptr(C, m.in[1]),
                @backingInt(m.k),
                kit.ptr(CF, m.in[2]),
                kit.ptr(?[*]f32, st),
                kit.ptr(CF, m.in[3]),
                kit.ptr(CF, m.in[4]),
                kit.ptr(?*anyopaque, self.base[0]),
                @intCast(v[0]),
                @intCast(v[1]),
                @intCast(v[2]),
                @intCast(v[3]),
                @intCast(v[4]),
                @intCast(v[5]),
                @intCast(v[6]),
                self.rig.s.handle,
            );
            if (r == 0 and m.part == .y) break;
        }
        if (m.part == .y) return self.rig.bytes(self.base[0], n * d.xd() * kit.size(m.k));
        const a = m.arena.allocator();
        const st = try a.dupe(u8, try self.rig.bytes(self.base[1], m.st_bytes.len));
        return std.mem.concat(a, u8, &.{ st, try self.rig.bytes(self.base[2], (n - 1) * m.st_bytes.len) });
    }

    pub fn count(self: *ScanWindow) usize {
        const own = self.m.count();
        return if (self.m.part == .y) own else own * self.m.shape.rows;
    }

    /// The inner test's outputs, then (the state's part) the state after each of the first rows - 1 rows.
    fn outputs(self: *ScanWindow, want: []f64, mag: []f64, sim: ?struct { c.Accum, g.Order }) !void {
        const m = &self.m;
        const own = m.count();
        if (sim) |how| try m.simulate(how[0], how[1], want[0..own]) else try m.reference(want[0..own], mag[0..own]);
        if (m.part == .y) return;
        const n = m.shape.rows;
        defer m.shape.rows = n;
        for (1..n) |r| {
            m.shape.rows = r;
            const to = want[r * own ..][0..own];
            if (sim) |how| try m.simulate(how[0], how[1], to) else try m.reference(to, mag[r * own ..][0..own]);
        }
    }

    pub fn reference(self: *ScanWindow, want: []f64, mag: []f64) !void {
        return self.outputs(want, mag, null);
    }

    pub fn simulate(self: *ScanWindow, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.outputs(out, out, .{ acc, order });
    }
};
