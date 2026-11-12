//! The decode attention pair as one replacement operator, and its controls: a merge losing a slice, the old grid.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.spec;
const g = core.gates;
const ref = core.xpu_ref.attention;

const Buffers = struct { pm: ?*anyopaque, pl: ?*anyopaque, po: ?*anyopaque, out: ?*anyopaque };

/// The registration, or a planted control: the merge that loses the last slice, or attn_decode on the old grid.
pub const Plant = enum { none, drop_slice, old_grid };

/// Q rows, a bf16 K/V cache and a tree table to bf16 attention rows; no input is writable.
pub const Attn = struct {
    rig: rig_.Rig,
    decode: xpu.Kernel,
    merge: xpu.Kernel,
    old_decode: xpu.Kernel,
    old_merge: xpu.Kernel,
    plant: Plant,
    arena: std.heap.ArenaAllocator,
    f: ref.Fixture = undefined,
    q: ?*anyopaque = null,
    kc: ?*anyopaque = null,
    vc: ?*anyopaque = null,
    tree: ?*anyopaque = null,
    candidate: Buffers = undefined,
    baseline: Buffers = undefined,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, plant: Plant) !Attn {
        var decode = try dev.kernelFor(.decode, "attn_decode", a);
        errdefer decode.deinit();
        var merge = if (plant == .drop_slice)
            try dev.kernel(.gate, "tf_ctl_attn_drop_slice", .{ 16, 1, 1 })
        else
            try dev.kernelFor(.decode, "attn_merge", a);
        errdefer merge.deinit();
        var old_decode = try dev.kernel(.attn_upstream, "attn_decode", .{ 16, 1, 1 });
        errdefer old_decode.deinit();
        const old_merge = try dev.kernel(.attn_upstream, "attn_merge", .{ 16, 1, 1 });
        return .{
            .rig = .{ .a = a, .dev = dev },
            .decode = decode,
            .merge = merge,
            .old_decode = old_decode,
            .old_merge = old_merge,
            .plant = plant,
            .arena = .init(a),
        };
    }

    pub fn deinit(self: *Attn) void {
        self.old_merge.deinit();
        self.old_decode.deinit();
        self.merge.deinit();
        self.decode.deinit();
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Attn) g.Kernel {
        var s = ref.spec();
        switch (self.plant) {
            .none => {},
            .drop_slice => s.name = "control: xpu attention merge losing its last slice",
            .old_grid => s.name = "control: xpu attn_decode launched on the old (slices, heads, rows) grid",
        }
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.plant == .none) null else .shapes };
    }

    const vt = g.hooks(Attn);

    fn buffers(self: *Attn, output: bool) !Buffers {
        const parts = self.f.shape.rows * ref.q_heads * self.f.slices();
        return .{
            .pm = try self.rig.scratch(parts * 4),
            .pl = try self.rig.scratch(parts * 4),
            .po = try self.rig.scratch(parts * ref.head_dim * 4),
            .out = if (output) try self.rig.output(self.count() * 2) else try self.rig.scratch(self.count() * 2),
        };
    }

    pub fn prepare(self: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!ref.valid(shape)) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.f = try ref.Fixture.init(self.arena.allocator(), shape);
        self.f.fill(fill, seed);
        self.q = try self.rig.input(std.mem.sliceAsBytes(self.f.q));
        self.kc = try self.rig.input(std.mem.sliceAsBytes(self.f.k));
        self.vc = try self.rig.input(std.mem.sliceAsBytes(self.f.v));
        self.tree = try self.rig.input(std.mem.sliceAsBytes(self.f.tree));
        self.candidate = try self.buffers(true);
        self.baseline = try self.buffers(false);
        return true;
    }

    fn pair(self: *Attn, old: bool) !void {
        const b = if (old) self.baseline else self.candidate;
        const t: u32 = @intCast(self.f.shape.rows);
        const ns: u32 = @intCast(self.f.slices());
        const p0: i32 = @intCast(self.f.shape.k - 1);
        const nil: ?*anyopaque = null;
        // the old-grid control launches only where that grid stays inside the buffers: no more slices than rows
        const old_grid = old or (self.plant == .old_grid and ns <= t);
        const grid: [3]u32 = if (old_grid) .{ ns, ref.kv_heads, t } else .{ t, ref.kv_heads, ns };
        try self.rig.launch(if (old) self.old_decode else self.decode, grid, .{
            self.q,              self.kc,   self.vc, b.pm, b.pl,        b.po, p0, @as(i32, @intCast(ns)),
            @as(f32, ref.scale), self.tree, nil,     nil,  @as(i32, 0),
        });
        try self.rig.launch(if (old) self.old_merge else self.merge, .{ ref.q_heads, t, 1 }, .{
            b.pm, b.pl, b.po, b.out, p0, @as(i32, @intCast(ns)), self.tree, nil,
        });
    }

    pub fn run(self: *Attn) !void {
        try self.pair(false);
    }

    pub fn count(self: *Attn) usize {
        return self.f.shape.rows * ref.width;
    }

    pub fn reference(self: *Attn, want: []f64, mag: []f64) !void {
        try self.f.reference(want, mag);
    }

    pub fn simulate(self: *Attn, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.f.simulate(acc, order, out);
    }

    pub fn replaced(self: *Attn) ![]const u8 {
        try self.pair(true);
        return self.rig.bytes(self.baseline.out, self.count() * 2);
    }
};
