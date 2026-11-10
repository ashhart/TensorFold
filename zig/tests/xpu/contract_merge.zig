//! The decode attention's slice merge as a bit-equal replacement, and two merges that launch or weigh slices wrongly.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.spec;
const g = core.gates;
const ref = core.xpu_ref.merge;

/// The registered merge, the same kernel on the old one-subgroup grid, or a copy that weighs lanes past the last slice.
pub const Variant = enum { real, old_grid, unguarded };

/// Partials (max, sum, values a slice) and a tree table to bf16 attention rows; no input is writable.
pub const Merge = struct {
    rig: rig_.Rig,
    merge: xpu.Kernel,
    old: xpu.Kernel,
    variant: Variant,
    arena: std.heap.ArenaAllocator,
    f: ref.Fixture = undefined,
    pm: ?*anyopaque = null,
    pl: ?*anyopaque = null,
    po: ?*anyopaque = null,
    tree: ?*anyopaque = null,
    out: ?*anyopaque = null,
    base: ?*anyopaque = null,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, variant: Variant) !Merge {
        var merge = if (variant == .unguarded)
            try dev.kernel(.gate, "tf_ctl_attn_merge_unguarded", .{ 16, 1, 1 })
        else
            try dev.kernelFor(.decode, "attn_merge", a);
        errdefer merge.deinit();
        return .{
            .rig = .{ .a = a, .dev = dev },
            .merge = merge,
            .old = try dev.kernel(.merge_upstream, "attn_merge", .{ 16, 1, 1 }),
            .variant = variant,
            .arena = .init(a),
        };
    }

    pub fn deinit(self: *Merge) void {
        self.old.deinit();
        self.merge.deinit();
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Merge) g.Kernel {
        var s = ref.spec();
        switch (self.variant) {
            .real => {},
            .old_grid => s.name = "control: xpu attn_merge launched on the one-subgroup-a-head grid",
            .unguarded => s.name = "control: xpu attn_merge weighing past its last slice",
        }
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.variant == .real) null else .shapes };
    }

    const vt = g.hooks(Merge);

    pub fn prepare(self: *Merge, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!ref.valid(shape)) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.f = try ref.Fixture.init(self.arena.allocator(), shape);
        self.f.fill(fill, seed);
        self.pm = try self.rig.input(std.mem.sliceAsBytes(self.f.pm));
        self.pl = try self.rig.input(std.mem.sliceAsBytes(self.f.pl));
        self.po = try self.rig.input(std.mem.sliceAsBytes(self.f.po));
        self.tree = try self.rig.input(std.mem.sliceAsBytes(self.f.tree));
        self.out = try self.rig.output(self.count() * 2);
        self.base = try self.rig.scratch(self.count() * 2);
        return true;
    }

    fn launch(self: *Merge, old: bool, out: ?*anyopaque) !void {
        const nil: ?*anyopaque = null;
        const groups: u32 = if (old or self.variant == .old_grid) ref.q_heads else ref.q_heads * 16;
        try self.rig.launch(if (old) self.old else self.merge, .{ groups, @intCast(self.f.shape.rows), 1 }, .{
            self.pm,                                self.pl,                             self.po,   out,
            @as(i32, @intCast(self.f.shape.k - 1)), @as(i32, @intCast(self.f.slices())), self.tree, nil,
        });
    }

    pub fn run(self: *Merge) !void {
        try self.launch(false, self.out);
    }

    pub fn count(self: *Merge) usize {
        return self.f.shape.rows * ref.width;
    }

    pub fn reference(self: *Merge, want: []f64, mag: []f64) !void {
        try self.f.reference(want, mag);
    }

    pub fn simulate(self: *Merge, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.f.simulate(acc, order, out);
    }

    pub fn replaced(self: *Merge) ![]const u8 {
        try self.launch(true, self.base);
        return self.rig.bytes(self.base, self.count() * 2);
    }
};
