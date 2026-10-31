//! A verify window that holds its DeltaNet state, and the keep that replays it, under the contract on HIP.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");
const gdn = @import("contract_gdn.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

/// tf_gdn_hold_*: a window's y, the state left as it was; tf_gdn_keep_*: each layer's state after its kept rows.
pub const Which = enum { hold, keep };

/// Rows a keep's inputs hold a layer: Flash Next's cache.max_window.
const window = 8;
/// DeltaNet layers in one keep launch, each with its own inputs and state, the states one after another.
const layers = 2;

/// Windows of 2, 4 and 8 rows over Flash Next's 48 value heads (16 key heads).
fn holdServed(comptime dk: usize) [3]c.Shape {
    return .{ .{ .rows = 2, .n = 48, .k = dk }, .{ .rows = 4, .n = 48, .k = dk }, .{ .rows = 8, .n = 48, .k = dk } };
}

/// A keep of 4 rows, a full reject (the pending row alone) and a full accept of 8.
fn keepServed(comptime dk: usize) [3]c.Shape {
    return .{ .{ .rows = 4, .n = 48, .k = dk }, .{ .rows = 1, .n = 48, .k = dk }, .{ .rows = 8, .n = 48, .k = dk } };
}

const hold_128 = holdServed(128);
const hold_16 = holdServed(16);
const keep_128 = keepServed(128);
const keep_16 = keepServed(16);

pub fn spec(which: Which, dk: usize) c.Spec {
    const wide = dk == 128;
    const names = [2][2][]const u8{
        .{ "hip DeltaNet tf_gdn_hold_16 (y)", "hip DeltaNet tf_gdn_hold_128 (y)" },
        .{ "hip DeltaNet tf_gdn_keep_16 (the states)", "hip DeltaNet tf_gdn_keep_128 (the states)" },
    };
    return .{
        .name = names[@intFromBool(which == .keep)][@intFromBool(wide)],
        .accum = .f32,
        .depth = gdn.depth,
        .out = .f32,
        .replaces = if (wide) "gated_delta_tile<128, 8, 16>" else "gated_delta_tile<16, 8, 16>",
        // the scan stages 16 rows of k and q at a time
        .tiles = .{ .{ .size = 16 }, .{}, .{} },
        .served = switch (which) {
            .hold => if (wide) &hold_128 else &hold_16,
            .keep => if (wide) &keep_128 else &keep_16,
        },
    };
}

/// shape.rows window rows (hold) or kept rows (keep) over shape.n value heads, dk = dv = shape.k.
pub const Window = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    which: Which,
    dk: usize,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    p: [layers]gdn.Problem = undefined,
    ins: [layers][5]u64 = undefined,
    sets: usize = 1,
    in: u64 = 0, // a keep's inputs: each layer's k, v, gate and beta for `window` rows
    st: u64 = 0,
    st_bytes: []u8 = &.{},
    y: u64 = 0,
    base: [2]u64 = @splat(0), // the plain scan's states and y

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, which: Which, dk: usize) Window {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .which = which, .dk = dk, .arena = .init(a) };
    }

    pub fn deinit(self: *Window) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Window) g.Kernel {
        return .{ .spec = spec(self.which, self.dk), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Window);

    fn one(self: *const Window) usize {
        return self.shape.n * self.dk * self.dk;
    }

    pub fn prepare(self: *Window, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const most: usize = if (self.which == .hold) 16 else window;
        if (shape.k != self.dk or shape.rows == 0 or shape.rows > most or shape.n == 0 or shape.n % gdn.group != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        self.sets = if (self.which == .hold) 1 else layers;
        const tokens = if (self.which == .hold) shape.rows else window;
        var states: std.ArrayList(f64) = .empty;
        for (0..self.sets) |z| {
            self.p[z] = try gdn.Problem.make(a, tokens, shape.n, self.dk, fill, seed + z, .exact);
            self.ins[z] = try gdn.upload(&self.rig, a, &self.p[z]);
            try states.appendSlice(a, self.p[z].state);
        }
        self.st_bytes = try kit.f32Bytes(a, states.items);
        const yb = tokens * shape.n * self.dk * 4;
        if (self.which == .hold) {
            // the hold reads the state and must leave it: an input, which gate 2 checks unchanged
            self.st = try self.rig.input(self.st_bytes);
            self.y = try self.rig.output(yb);
        } else {
            var packed_in: std.ArrayList(f64) = .empty;
            for (self.p[0..self.sets]) |p| for ([_][]const f64{ p.k, p.v, p.gate, p.beta }) |x| try packed_in.appendSlice(a, x);
            self.in = try self.rig.input(try kit.f32Bytes(a, packed_in.items));
            self.st = try self.rig.output(self.st_bytes.len);
            self.y = 0;
        }
        self.base = .{ try self.rig.scratch(self.st_bytes.len), try self.rig.scratch(yb) };
        return true;
    }

    /// One launch: the hold over the window, or the keep of both layers (their states back to the prepared ones).
    pub fn run(self: *Window) !void {
        const CF = ?[*]const f32;
        const F = ?[*]f32;
        const n: c_int = @intCast(self.shape.n);
        const kh: c_int = @divExact(n, gdn.group);
        const d: c_int = @intCast(self.dk);
        const s = self.rig.s.handle;
        if (self.which == .hold) {
            const in = self.ins[0];
            try self.l.tf_gdn_hold(kit.ptr(CF, in[0]), kit.ptr(CF, in[1]), kit.ptr(CF, in[2]), kit.ptr(CF, in[3]), kit.ptr(CF, in[4]), kit.ptr(F, self.st), kit.ptr(F, self.y), @intCast(self.shape.rows), kh, n, d, d, s);
        } else {
            try self.rig.put(self.st, self.st_bytes);
            const p = &self.p[0];
            const in_stride = p.k.len + p.v.len + p.gate.len + p.beta.len;
            try self.l.tf_gdn_keep(kit.ptr(F, self.st), @intCast(self.one()), kit.ptr(CF, self.in), @intCast(in_stride), window, @intCast(self.shape.rows), layers, kh, n, d, d, s);
        }
        try self.rig.s.synchronize();
    }

    /// The plain scan over the window's rows from the prepared state (y), or each layer's over its kept rows (states).
    pub fn replaced(self: *Window) ![]const u8 {
        const each = self.one() * 4;
        for (0..self.sets) |z| {
            const st = self.base[0] + z * each;
            try self.rig.put(st, self.st_bytes[z * each ..][0..each]);
            try gdn.plain(self.l, &self.rig, self.ins[z], st, self.base[1], &self.p[z], self.shape.rows);
        }
        if (self.which == .hold) return self.rig.bytes(self.base[1], self.count() * 4);
        return self.rig.bytes(self.base[0], self.st_bytes.len);
    }

    pub fn count(self: *Window) usize {
        return if (self.which == .hold) self.shape.rows * self.shape.n * self.dk else self.sets * self.one();
    }

    pub fn reference(self: *Window, want: []f64, mag: []f64) !void {
        return self.outputs(want, mag, null);
    }

    pub fn simulate(self: *Window, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try self.rig.a.alloc(f64, out.len);
        defer self.rig.a.free(mag);
        return self.outputs(out, mag, .{ acc, order });
    }

    /// The hold's y over its rows; the keep's states after the kept rows, layer by layer.
    fn outputs(self: *Window, want: []f64, mag: []f64, sim: ?gdn.Sim) !void {
        const a = self.rig.a;
        const rows = self.shape.rows;
        if (self.which == .hold) return gdn.partOf(a, &self.p[0], rows, .y, want, mag, sim);
        const each = self.one();
        for (0..self.sets) |z| {
            try gdn.partOf(a, &self.p[z], rows, .state, want[z * each ..][0..each], mag[z * each ..][0..each], sim);
        }
    }
};
