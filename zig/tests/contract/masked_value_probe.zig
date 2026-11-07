//! Shared contract wiring for the isolated panel, keeping kernel adapters and the binary64 data independent.
const std = @import("std");
const metal = @import("metal");
const ct = @import("contract");
const c = ct.spec;
const g = ct.gates;
const rig_ = @import("rig.zig");
const data = @import("masked_value_data.zig");
pub const control_source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\[[max_total_threads_per_threadgroup(32)]] [[kernel]] void tf_masked_control(device ushort* out [[buffer(0)]], constant uint2& p [[buffer(1)]], uint t [[thread_index_in_threadgroup]]) {
    \\  if (t == 0) out[p.y] = p.x == 1 ? ushort(0x7fc0) : ushort(0x3f80);
    \\}
;
fn depth(s: c.Shape) usize {
    return s.n + s.k + s.rows + 32;
}
const Control = enum { none, hidden, visible };
pub fn Types(comptime B: type) type {
    return struct {
        pub const Probe = struct {
            a: std.mem.Allocator,
            rig: rig_.Rig,
            kind: B.Kind,
            geo: data.Geometry,
            poison: data.Poison,
            label: [192]u8 = undefined,
            failure: [192]u8 = undefined,
            served: [2]c.Shape,
            shape: c.Shape = undefined,
            pipes: [3]?metal.Pipeline = @splat(null),
            owns: bool = true,
            control: Control = .none,
            control_pipe: ?metal.Pipeline = null,
            arena: ?std.heap.ArenaAllocator = null,
            fixture: data.Data = undefined,
            q: rig_.Ref = undefined,
            k: rig_.Ref = undefined,
            v: rig_.Ref = undefined,
            out: rig_.Ref = undefined,
            aux: [12]rig_.Ref = undefined,
            fn deinit(p: *Probe) void {
                p.rig.deinit();
                if (p.arena) |*a| a.deinit();
                if (p.owns) for (p.pipes) |pipe| if (pipe) |x| x.deinit();
                if (p.control_pipe) |x| x.deinit();
            }
            pub fn kernel(p: *Probe) g.Kernel {
                const name = std.fmt.bufPrint(&p.label, "{s}metal masked-V {s} {s} {s}", .{ if (p.control == .none) "" else "control: ", B.label(p.kind), @tagName(p.poison), if (p.control == .none) "" else @tagName(p.control) }) catch unreachable;
                return .{ .ctx = p, .vt = &vt, .control = if (p.control == .none) null else .nonfinite, .spec = .{
                    .name = name,
                    .accum = .f32,
                    .depth = depth,
                    .ops = 32,
                    .mids = 2,
                    .out = B.output(p.kind),
                    .served = &p.served,
                    .nan = .{ .stated = "Each output class and every finite hidden value matches binary64 over the actual visible key set." },
                    .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "equal-score unit-value mask panel; this entry does not discriminate accumulator precision" },
                    .known = if (p.control == .none and B.known(p.kind, p.poison)) .{ .gate = .nonfinite, .finding = B.finding(p.kind) } else null,
                } };
            }
            const vt = g.hooks(Probe);
            pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, _: u64) !bool {
                if (shape.rows == 0 or shape.rows > 32 or shape.n != p.geo.heads * p.geo.dim or (p.geo.ring == 0 and shape.k + shape.rows > p.geo.cap)) return false;
                p.rig.reset();
                if (p.arena) |*a| a.deinit();
                p.arena = .init(p.a);
                p.shape = shape;
                var geo = p.geo;
                geo.first = shape.k;
                p.fixture = try data.Data.init(p.arena.?.allocator(), geo, shape.rows, fill == .nonfinite, p.poison);
                try B.prepare(p);
                return true;
            }
            pub fn run(p: *Probe) !void {
                const pass = p.rig.begin();
                B.encode(p, pass.e);
                if (p.control != .none) {
                    pass.e.barrier();
                    pass.e.setPipeline(p.control_pipe.?);
                    pass.e.setBuffer(p.out.buf, p.out.off, 0);
                    pass.e.setValue([2]u32{ if (p.control == .hidden) 1 else 2, if (p.control == .hidden) 0 else @intCast(p.fixture.outputAt(1, 0, 7)) }, 1);
                    pass.e.dispatchGroups(.{ .width = 1 }, .{ .width = 32 });
                }
                try p.rig.finish(pass);
            }
            pub fn count(p: *Probe) usize {
                return p.fixture.count();
            }
            pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
                if (@hasDecl(B, "reference")) B.reference(p, want, mag) else p.fixture.reference(want, mag);
            }
            pub fn simulate(p: *Probe, _: c.Accum, _: g.Order, out: []f64) !void {
                const mag = try p.a.alloc(f64, out.len);
                defer p.a.free(mag);
                try p.reference(out, mag);
            }
            pub fn check(p: *Probe, gate: g.Gate) !?[]const u8 {
                const want = try p.a.alloc(f64, p.count());
                defer p.a.free(want);
                const mag = try p.a.alloc(f64, want.len);
                defer p.a.free(mag);
                try p.reference(want, mag);
                const bytes = try p.rig.outputBytes();
                const spec = p.kernel().spec;
                var hidden_bad: usize = 0;
                var visible_bad: usize = 0;
                var point_bad: usize = 0;
                var first: ?usize = null;
                var planted: usize = 0;
                for (want, 0..) |w, i| {
                    const got = c.read(spec.out, bytes, i);
                    if (!std.math.isFinite(w)) planted += 1;
                    const classes = c.classOf(got) != c.classOf(w);
                    const point = std.math.isFinite(w) and (!std.math.isFinite(got) or @abs(got - w) > c.bound(spec, p.shape, w, mag[i]));
                    if (classes or point) {
                        if (first == null) first = i;
                        if (i < p.geo.heads * p.geo.dim) hidden_bad += 1 else visible_bad += 1;
                        if (point) point_bad += 1;
                    }
                }
                if (gate == .nonfinite) {
                    std.debug.print("MASKED_DETAIL {s} hidden_bad={d} later_bad={d} finite_bad={d} nonfinite={d}/{d} first={any}\n", .{ spec.name, hidden_bad, visible_bad, point_bad, planted, want.len, first });
                    if (planted == 0 or planted == want.len) return "nonfinite fixture lacks both finite and exceptional reference outputs";
                }
                if (first) |i| return try std.fmt.bufPrint(&p.failure, "first={d} got={t}({d}) want={t}({d}); hidden={d} later={d} finite={d}", .{ i, c.classOf(c.read(spec.out, bytes, i)), c.read(spec.out, bytes, i), c.classOf(want[i]), want[i], hidden_bad, visible_bad, point_bad });
                return null;
            }
        };
        pub const Suite = struct {
            a: std.mem.Allocator,
            device: metal.Device,
            queue: metal.Queue,
            kernels: *std.ArrayList(g.Kernel),
            probes: std.ArrayList(*Probe) = .empty,
            pub fn collect(s: *Suite, only: ?[]const u8) !void {
                if (!s.device.tensorUnits()) return;
                for (std.enums.values(B.Kind)) |kind| {
                    if (!B.available(kind)) {
                        std.debug.print("MASKED_UNAVAILABLE {s}: source absent from this tree\n", .{B.label(kind)});
                        continue;
                    }
                    var first: ?*Probe = null;
                    for (std.enums.values(data.Poison)) |poison| {
                        var label: [192]u8 = undefined;
                        const name = try std.fmt.bufPrint(&label, "metal masked-V {s} {s}", .{ B.label(kind), @tagName(poison) });
                        if (only) |o| if (std.mem.indexOf(u8, name, o) == null) continue;
                        const p = try s.a.create(Probe);
                        const geo = B.geometry(kind);
                        p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = s.device, .queue = s.queue }, .kind = kind, .geo = geo, .poison = poison, .served = .{ .{ .rows = 8, .n = geo.heads * geo.dim, .k = geo.first }, .{ .rows = 32, .n = geo.heads * geo.dim, .k = geo.first + 15 } } };
                        errdefer {
                            p.deinit();
                            s.a.destroy(p);
                        }
                        if (first) |base| {
                            p.pipes = base.pipes;
                            p.owns = false;
                        } else {
                            try B.load(p);
                            first = p;
                        }
                        try s.probes.append(s.a, p);
                        try s.kernels.append(s.a, p.kernel());
                    }
                }
            }
            pub fn controls(s: *Suite, kind: B.Kind, only: ?[]const u8) !void {
                for ([_]Control{ .hidden, .visible }) |control| {
                    const p = try s.a.create(Probe);
                    const geo = B.geometry(kind);
                    p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = s.device, .queue = s.queue }, .kind = kind, .geo = geo, .poison = .inf, .control = control, .served = .{ .{ .rows = 8, .n = geo.heads * geo.dim, .k = geo.first }, .{ .rows = 8, .n = geo.heads * geo.dim, .k = geo.first } } };
                    if (only) |o| if (std.mem.indexOf(u8, p.kernel().spec.name, o) == null) {
                        s.a.destroy(p);
                        continue;
                    };
                    try B.load(p);
                    p.control_pipe = try rig_.pipeline(s.device, control_source, "tf_masked_control");
                    if (p.control_pipe.?.maxThreads() < 32 or rig_.tgMemory(p.control_pipe.?) != 0) return error.MaskedControlLaunchLimits;
                    try s.probes.append(s.a, p);
                    try s.kernels.append(s.a, p.kernel());
                }
            }
            pub fn deinit(s: *Suite) void {
                for (s.probes.items) |p| {
                    p.deinit();
                    s.a.destroy(p);
                }
                s.probes.deinit(s.a);
            }
        };
    };
}
