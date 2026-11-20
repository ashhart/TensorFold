//! Column scans retain the slot kernel's bytes and the independent parent-indexed FP64 recurrence.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const ref = contract.delta_slots_ref;
const sums = contract.q27_stack_sums;
fn none(_: c.Shape) usize {
    return 0;
}
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    arena: std.heap.ArenaAllocator,
    pipe: mtl.Pipeline,
    old: mtl.Pipeline,
    slots: usize,
    bad: bool,
    precision_bad: bool,
    name: []const u8,
    f: ref.Fixture = undefined,
    nonfinite: bool = false,
    ins: [8]rig_.Ref = undefined,
    steps: rig_.Ref = undefined,
    parents: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    base: rig_.Ref = undefined,
    scratch: rig_.Ref = undefined,
    before: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, d: mtl.Device, q: mtl.Queue, slots: usize, bad: bool, precision_bad: bool, name: []const u8) !Probe {
        const text = if (precision_bad) try std.mem.replaceOwned(u8, a, sources.core_deltanet, "result += t[part] * qi[part];", "result = float(half(result + t[part] * qi[part]));") else if (bad) try std.mem.replaceOwned(u8, a, sources.core_deltanet, "row = flow & 0xffffu;", "row = 0u;") else try a.dupe(u8, sources.core_deltanet);
        defer a.free(text);
        var buf: [80]u8 = undefined;
        const pipe = try rig_.pipeline(d, text, try std.fmt.bufPrint(&buf, "tf_delta_gdn_columns_{d}", .{slots}));
        errdefer pipe.deinit();
        const old = try rig_.pipeline(d, sources.core_deltanet, try std.fmt.bufPrint(&buf, "tf_delta_gdn_tree_slots_{d}", .{slots}));
        return .{ .a = a, .rig = .{ .a = a, .device = d, .queue = q }, .arena = .init(a), .pipe = pipe, .old = old, .slots = slots, .bad = bad, .precision_bad = precision_bad, .name = name };
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &g.hooks(Probe), .control = if (p.precision_bad) .sums else if (p.bad) .shapes else null, .spec = .{
            .name = p.name,
            .accum = .f32,
            .out = .bf16,
            .depth = none,
            .served = &.{ .{ .rows = 1, .n = 128, .k = 128 }, .{ .rows = 3, .n = 128, .k = 128 }, .{ .rows = 15, .n = 128, .k = 128 } },
            .replaces = "tf_delta_gdn_tree_slots on the same flow schedule",
            .nan = .{ .stated = "planted NaN and infinity classes and all finite bytes equal the slot recurrence" },
            .sums = .{ .exact = true, .f16 = 100, .bf16 = 0, .why = "exact read sums plus replacement bytes and independent FP64 recurrence; cancellation can hide BF16 sums" },
        } };
    }
    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.n != 128 or s.k != 128 or s.rows == 0 or s.rows > 15) return false;
        p.nonfinite = fill == .nonfinite;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const a = p.arena.allocator();
        p.f = try ref.make(tf.tree_slots, a, p.slots, s.rows, false, seed);
        if (fill == .nonfinite) {
            p.f.query[0] = c.toBf16(std.math.nan(f32));
            p.f.value[128] = c.toBf16(std.math.inf(f32));
        }
        if (fill == .stress) ref.exactRead(&p.f);
        const f = p.f;
        const flows = try a.alloc(u32, f.steps.len);
        for (f.segments) |seg| {
            const at = seg.first;
            const steps: []const tf.tree_slots.Step = @ptrCast(f.steps[at..][0..seg.rows]);
            if (!tf.tree_slots.flows(steps, flows[at..][0..seg.rows])) return error.InvalidFlowFixture;
        }
        p.ins = .{
            try p.rig.input(std.mem.sliceAsBytes(f.query)),  try p.rig.input(std.mem.sliceAsBytes(f.key)),
            try p.rig.input(std.mem.sliceAsBytes(f.value)),  try p.rig.input(std.mem.sliceAsBytes(f.decay)),
            try p.rig.input(std.mem.sliceAsBytes(f.mixing)), try p.rig.input(std.mem.sliceAsBytes(f.state)),
            try p.rig.input(std.mem.asBytes(&f.segments)),   try p.rig.input(std.mem.sliceAsBytes(flows)),
        };
        p.parents = try p.rig.input(std.mem.sliceAsBytes(f.parents));
        p.steps = try p.rig.input(std.mem.sliceAsBytes(f.steps));
        p.out = try p.rig.output(f.output.len * 2);
        p.base = try p.rig.scratch(f.output.len * 2);
        p.scratch = try p.rig.scratch(256);
        p.before = try p.rig.scratch(f.state.len * 4);
        return true;
    }
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (p.ins[0..7], 0..) |b, i| e.setBuffer(b.buf, b.off, i);
        e.setValue(p.f.p, 7);
        e.setBuffer(p.out.buf, p.out.off, 8);
        e.setBuffer(p.ins[7].buf, p.ins[7].off, 9);
        e.dispatchGroups(mtl.Size.of(1, 32, 96), mtl.Size.of(32, 1, 1));
        try p.rig.finish(pass);
    }
    pub fn replaced(p: *Probe) ![]const u8 {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.old);
        for (p.ins[0..6], 0..) |b, i| e.setBuffer(b.buf, b.off, i);
        e.setBuffer(p.parents.buf, p.parents.off, 6);
        e.setBuffer(p.ins[6].buf, p.ins[6].off, 7);
        e.setValue(p.f.p, 8);
        e.setBuffer(p.base.buf, p.base.off, 9);
        e.setBuffer(p.scratch.buf, p.scratch.off, 10);
        e.setBuffer(p.steps.buf, p.steps.off, 11);
        e.setBuffer(p.ins[5].buf, p.ins[5].off, 12);
        e.setBuffer(p.before.buf, p.before.off, 13);
        e.dispatchThreads(mtl.Size.of(32, 128, 96), mtl.Size.of(32, 4, 1));
        try p.rig.finish(pass);
        const bytes = p.rig.bytes(p.base, p.count() * 2);
        if (!p.nonfinite and !p.f.accepts(bytes)) return error.SlotBaselineOutsideFp64Bound;
        return bytes;
    }
    pub fn check(p: *Probe, gate: g.Gate) !?[]const u8 {
        const out = p.rig.bytes(p.out, p.count() * 2);
        const baseline = try p.replaced();
        if (!std.mem.eql(u8, out, baseline)) return "column bytes differ from register-slot bytes";
        if (gate != .nonfinite and !p.f.accepts(out)) return "column output leaves independent FP64 recurrence bound";
        return null;
    }
    pub fn deinit(p: *Probe) void {
        p.pipe.deinit();
        p.old.deinit();
        p.rig.deinit();
        p.arena.deinit();
    }
    pub fn count(p: *Probe) usize {
        return p.f.output.len;
    }
    pub fn simulate(_: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        @memset(out, sums.output(.columns, 0, acc, order));
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        @memcpy(want, p.f.output);
        @memset(mag, 0);
    }
};
