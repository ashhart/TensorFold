//! flashnext_batch.cu's lane kernels: a shared round's DeltaNet in one launch a kernel, the per-lane launches' bits.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const gates = core.contract.gates;
const c = core.contract.spec;
const table = flash.batch_lanes;
const gdn = flash.gdn;

const layer = 5; // a table's layers sit apart: a wrong stride reads another layer's zeros
const pw = gdn.proj_width;
const hist_n = (gdn.taps - 1) * gdn.conv_dim;
const state_n = gdn.nv * gdn.dv * gdn.dk;
const qk_n = gdn.nk * gdn.dk;
const v_n = gdn.nv * gdn.dv;
const gb_n = gdn.nv * 2;
const max_lanes = 32;

/// A lane's rows, whether it keeps a rollback, and whether its end state goes apart from its state.
pub const Lane = struct { rows: usize, keep: bool = false, apart: bool = false };
const sixteen: [16]Lane = @splat(.{ .rows = 16 });
const widest: [max_lanes]Lane = @splat(.{ .rows = 8, .keep = true });
const cases = [_][]const Lane{
    &.{ .{ .rows = 1 }, .{ .rows = 1 } },
    &.{ .{ .rows = 16, .keep = true, .apart = true }, .{ .rows = 1 }, .{ .rows = 7, .keep = true } },
    &.{ .{ .rows = 3, .apart = true }, .{ .rows = 16 }, .{ .rows = 5, .keep = true, .apart = true }, .{ .rows = 2 } },
    &.{.{ .rows = 5, .keep = true, .apart = true }}, // one stream
    &.{ .{ .rows = 4, .keep = true }, .{ .rows = 0 }, .{ .rows = 3, .apart = true } }, // a stream with no rows this round
    &sixteen, // sixteen streams
    &widest, // the table's lanes
};

/// Lanes for a gate's `rows`: up to 16 rows each, `base`'s rollbacks in turn, two lanes at least.
fn split(a: std.mem.Allocator, base: []const Lane, rows: usize) ![]Lane {
    const n = @max(2, (rows + 15) / 16);
    const out = try a.alloc(Lane, n);
    var left = rows;
    for (out, 0..) |*l, i| {
        const take = if (i + 1 == n) left else @min(left - (n - 1 - i), 16);
        l.* = base[i % base.len];
        l.rows = take;
        left -= take;
    }
    return out;
}

const Fixture = struct {
    lanes: []const Lane,
    conv: [][]u16,
    state: [][]f32,
    proj: []u16,
    cw: []u16,
    a_log: []f32,
    dt: []f32,

    fn rows(f: Fixture) usize {
        var n: usize = 0;
        for (f.lanes) |l| n += l.rows;
        return n;
    }

    fn make(a: std.mem.Allocator, lanes: []const Lane, seed: u64) !Fixture {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        var f: Fixture = .{ .lanes = lanes, .conv = try a.alloc([]u16, lanes.len), .state = try a.alloc([]f32, lanes.len), .proj = undefined, .cw = try a.alloc(u16, gdn.conv_dim * gdn.taps), .a_log = try a.alloc(f32, gdn.nv), .dt = try a.alloc(f32, gdn.nv) };
        for (f.conv, f.state) |*x, *st| {
            x.* = try a.alloc(u16, hist_n);
            for (x.*) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
            st.* = try a.alloc(f32, state_n);
            for (st.*) |*v| v.* = rng.float(f32) - 0.5;
        }
        f.proj = try a.alloc(u16, f.rows() * pw);
        for (f.proj) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
        for (f.cw) |*v| v.* = c.toBf16(rng.float(f32) - 0.5);
        for (f.a_log) |*v| v.* = rng.float(f32) - 2;
        for (f.dt) |*v| v.* = rng.float(f32) - 0.5;
        return f;
    }
};

fn cint(v: usize) c_int {
    return @intCast(v);
}

const Fns = struct { front: cuda.Function, chain: cuda.Function, chain_to: cuda.Function, shift: cuda.Function, front_lanes: cuda.Function, chain_lanes: cuda.Function, shift_lanes: cuda.Function, keep_lanes: cuda.Function };
const Way = enum { lanes, one_launch, crossed_table };

/// The round through `way`, its outputs, windows, states and keeps: `.crossed_table` has two lanes' windows swapped.
fn launch(gpu: check.Gpu, s: cuda.Stream, k: Fns, f: Fixture, way: Way, a: std.mem.Allocator) ![]const u8 {
    var rig = rig_.Rig{ .gpu = gpu, .s = s };
    defer {
        cuda.completion.drain(gpu.d, s, "the lane contract did not drain");
        rig.deinit();
    }
    const n = f.rows();
    const lanes = f.lanes.len;
    const conv = try a.alloc(u64, lanes);
    const st = try a.alloc(u64, lanes);
    const st_out = try a.alloc(u64, lanes);
    const kp = try a.alloc(u64, lanes);
    const kh = try a.alloc(u64, lanes);
    for (f.lanes, 0..) |l, i| {
        inline for (.{ .{ conv, f.conv[i] }, .{ st, f.state[i] } }) |p| {
            const bytes = std.mem.sliceAsBytes(p[1]);
            p[0][i] = try rig.scratch(bytes.len);
            try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = p[0][i], .len = bytes.len }).uploadAsync(0, bytes, s.handle);
        }
        st_out[i] = if (l.apart) try rig.output(state_n * 4) else 0;
        kp[i] = if (l.keep) try rig.output(l.rows * pw * 2) else 0;
        kh[i] = if (l.keep) try rig.output(hist_n * 2) else 0;
    }
    const proj = try rig.input(std.mem.sliceAsBytes(f.proj));
    const cw = try rig.input(std.mem.sliceAsBytes(f.cw));
    const a_log = try rig.input(std.mem.sliceAsBytes(f.a_log));
    const dt = try rig.input(std.mem.sliceAsBytes(f.dt));
    const gq = try rig.output(n * qk_n * 4);
    const gk = try rig.output(n * qk_n * 4);
    const gv = try rig.output(n * v_n * 4);
    const ggb = try rig.output(n * gb_n * 4);
    const ys = try rig.output(n * v_n * 4);
    var inputs: []const u8 = &.{};
    if (way == .lanes) {
        const d = gpu.d;
        var row0: usize = 0;
        for (f.lanes, 0..) |l, i| {
            if (l.rows == 0) continue; // nothing of a lane without rows is launched
            const pr = proj + row0 * pw * 2;
            const q = gq + row0 * qk_n * 4;
            const ky = gk + row0 * qk_n * 4;
            const v = gv + row0 * v_n * 4;
            const gb = ggb + row0 * gb_n * 4;
            if (l.keep) {
                try d.check(d.api.cuMemcpyDtoDAsync_v2(kp[i], pr, l.rows * pw * 2, s.handle), "cuMemcpyDtoDAsync");
                try d.check(d.api.cuMemcpyDtoDAsync_v2(kh[i], conv[i], hist_n * 2, s.handle), "cuMemcpyDtoDAsync");
            }
            var x: cuda.Args = .{};
            inline for (.{ pr, conv[i], cw, a_log, dt, q, ky, v, gb }) |p| x.add(p);
            try cuda.launch.launch(k.front, .{ .grid = .{ .x = @intCast(l.rows), .y = gdn.nk + gdn.nv }, .block = .{ .x = 256 } }, s, &x);
            x = .{};
            inline for (.{ q, ky, v, gb, st[i] }) |p| x.add(p);
            if (l.apart) x.add(st_out[i]);
            x.add(cint(l.rows));
            x.add(ys + row0 * v_n * 4);
            try cuda.launch.launch(if (l.apart) k.chain_to else k.chain, .{ .grid = .{ .x = gdn.nv, .y = gdn.dv / 16 }, .block = .{ .x = 128 } }, s, &x);
            x = .{};
            inline for (.{ conv[i], pr, @as(c_longlong, pw), cint(l.rows), cint(gdn.taps - 1), cint(gdn.conv_dim) }) |p| x.add(p);
            try cuda.launch.launch(k.shift, .{ .grid = .{ .x = (gdn.conv_dim + 255) / 256 }, .block = .{ .x = 256 } }, s, &x);
            row0 += l.rows;
        }
    } else {
        const tab = try a.alignedAlloc(u8, .@"16", flash.batch_state.lanes_bytes);
        @memset(tab, 0);
        var row0: usize = 0;
        for (f.lanes, 0..) |l, i| {
            const w = conv[if (way == .crossed_table) (i + 1) % lanes else i];
            table.putGdn(tab, layer, i, w, st[i], st_out[i], kp[i], kh[i]);
            table.putLane(tab, i, row0, 0, l.rows);
            row0 += l.rows;
        }
        const lp = try rig.input(tab);
        inputs = try a.dupe(u8, try rig.inputBytes());
        try table.gdnRound(k, s, .{ .lp = lp + flash.batch_state.lanes_attn_ptrs + layer * max_lanes * 5 * 8, .li = lp + flash.batch_state.lanes_ptrs, .proj = proj, .cw = cw, .a_log = a_log, .dt = dt, .gq = gq, .gk = gk, .gv = gv, .ggb = ggb, .ys = ys, .rows = n, .lanes = lanes });
    }
    var out: std.ArrayList(u8) = .empty;
    for ([_][2]u64{ .{ gq, n * qk_n * 4 }, .{ gk, n * qk_n * 4 }, .{ gv, n * v_n * 4 }, .{ ggb, n * gb_n * 4 }, .{ ys, n * v_n * 4 } }) |o| try out.appendSlice(a, try rig.bytes(o[0], o[1]));
    for (f.lanes, 0..) |l, i| {
        try out.appendSlice(a, try rig.bytes(conv[i], hist_n * 2));
        try out.appendSlice(a, try rig.bytes(st[i], state_n * 4));
        if (l.apart) try out.appendSlice(a, try rig.bytes(st_out[i], state_n * 4));
        if (l.keep) {
            try out.appendSlice(a, try rig.bytes(kp[i], l.rows * pw * 2));
            try out.appendSlice(a, try rig.bytes(kh[i], hist_n * 2));
        }
    }
    if (way != .lanes) try check.expect(std.mem.eql(u8, inputs, try rig.inputBytes()), "flashnext DeltaNet lanes: a launch changed an input", .{});
    try check.expect(try rig.intact() and !rig.overlap(), "flashnext DeltaNet lanes: a launch broke a guard or overlapped", .{});
    return out.items;
}

const State = struct {
    gpu: check.Gpu,
    stream: cuda.Stream,
    k: Fns,
    base: []const Lane,
    gate: gates.Gate = .shapes,
    inputs_arena: std.heap.ArenaAllocator,
    outputs_arena: std.heap.ArenaAllocator,
    fixture: ?Fixture = null,
    one: []const u8 = &.{},
    per_lane: []const u8 = &.{},
    checked: bool = false,

    fn of(p: *anyopaque) *State {
        return @ptrCast(@alignCast(p));
    }
    fn prepare(p: *anyopaque, shape: c.Shape, fill: gates.Fill, seed: u64) !bool {
        const s = of(p);
        s.inputs_arena.deinit();
        s.inputs_arena = .init(s.gpu.gpa);
        const a = s.inputs_arena.allocator();
        const lanes = if (s.gate == .shapes and shape.rows >= 2) try split(a, s.base, shape.rows) else s.base;
        s.fixture = try Fixture.make(a, lanes, seed);
        if (fill == .nonfinite) {
            const f = &s.fixture.?;
            f.proj[0] = c.toBf16(std.math.nan(f32));
            f.proj[1] = c.toBf16(std.math.inf(f32));
            f.proj[2] = c.toBf16(-std.math.inf(f32));
            f.conv[0][0] = c.toBf16(std.math.nan(f32));
            f.state[0][0] = std.math.inf(f32);
        }
        s.checked = false;
        return true;
    }
    fn runOne(p: *anyopaque) !void {
        const s = of(p);
        s.outputs_arena.deinit();
        s.outputs_arena = .init(s.gpu.gpa);
        const a = s.outputs_arena.allocator();
        s.per_lane = try launch(s.gpu, s.stream, s.k, s.fixture.?, .lanes, a);
        s.one = try launch(s.gpu, s.stream, s.k, s.fixture.?, .one_launch, a);
        s.checked = true;
    }
    fn output(p: *anyopaque) ![]const u8 {
        return of(p).one;
    }
    fn replaced(p: *anyopaque) ![]const u8 {
        return of(p).per_lane;
    }
    fn inputs(p: *anyopaque) ![]const u8 {
        const s = of(p);
        const f = s.fixture.?;
        var out: std.ArrayList(u8) = .empty;
        const a = s.outputs_arena.allocator();
        try out.appendSlice(a, std.mem.sliceAsBytes(f.proj));
        for (f.conv, f.state) |x, st| {
            try out.appendSlice(a, std.mem.sliceAsBytes(x));
            try out.appendSlice(a, std.mem.sliceAsBytes(st));
        }
        inline for (.{ "cw", "a_log", "dt" }) |field| try out.appendSlice(a, std.mem.sliceAsBytes(@field(f, field)));
        return out.items;
    }
    fn intact(p: *anyopaque) !bool {
        return of(p).checked; // launch checks every device input, guard and overlap before it returns
    }
    fn overlap(_: *anyopaque) bool {
        return false;
    }
    fn count(p: *anyopaque) usize {
        return of(p).one.len;
    }
    fn reference(_: *anyopaque, _: []f64, _: []f64) !void {
        return error.ReplacementUsesPerLaneBytes;
    }
    fn own(p: *anyopaque, _: gates.Gate) !?[]const u8 {
        const s = of(p);
        return if (s.checked and std.mem.eql(u8, s.one, s.per_lane)) null else "one-launch bytes differ from the per-lane kernels";
    }
    const vt: gates.VTable = .{ .prepare = prepare, .run = runOne, .output = output, .inputs = inputs, .intact = intact, .overlap = overlap, .count = count, .reference = reference, .check = own, .replaced = replaced };
};

fn depth(_: c.Shape) usize {
    return 0;
}

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var batch = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_batch);
    defer batch.unload();
    const k: Fns = .{ .front = try batch.function("fn_gdn_front"), .chain = try batch.function("fn_gdn_rows"), .chain_to = try batch.function("fn_gdn_rows_to"), .shift = try batch.function("fn_shift_window"), .front_lanes = try batch.function("fn_gdn_front_lanes"), .chain_lanes = try batch.function("fn_gdn_rows_lanes"), .shift_lanes = try batch.function("fn_shift_window_lanes"), .keep_lanes = try batch.function("fn_gdn_keep_lanes") };
    var red = false;
    for (cases, 0..) |lanes, ci| {
        var arena = std.heap.ArenaAllocator.init(gpu.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const f = try Fixture.make(a, lanes, 0xd47a + ci);
        const per_lane = try launch(gpu, s, k, f, .lanes, a);
        try check.sameBytes("flashnext DeltaNet lanes: one launch against per-lane", try launch(gpu, s, k, f, .one_launch, a), per_lane);
        red = red or !std.mem.eql(u8, try launch(gpu, s, k, f, .crossed_table, a), per_lane);
        var st: State = .{ .gpu = gpu, .stream = s, .k = k, .base = lanes, .inputs_arena = .init(gpu.gpa), .outputs_arena = .init(gpu.gpa) };
        defer st.inputs_arena.deinit();
        defer st.outputs_arena.deinit();
        const served = [_]c.Shape{.{ .rows = f.rows(), .n = lanes.len, .k = 1 }};
        const kernel: gates.Kernel = .{ .ctx = &st, .vt = &State.vt, .spec = .{
            .name = "cuda Flash Next DeltaNet, a shared round's lanes in one launch a kernel",
            .accum = .none,
            .depth = depth,
            .out = .raw,
            .replaces = "the same programs launched a lane at a time",
            .nan = .{ .stated = "all outputs, windows, states and keeps match the per-lane kernels, including NaN and both infinities" },
            .served = &served,
            .tiles = .{ .{ .size = 16, .step = 1, .min = 2 }, .{}, .{} },
        } };
        for ([_]gates.Gate{ .nonfinite, .repeat, .shapes }) |gate| {
            st.gate = gate;
            try check.expect(try gates.one(gpu.gpa, gate, kernel, 0xd400 + ci) == .pass, "Flash Next DeltaNet lanes gate {t} failed for case {d}", .{ gate, ci });
        }
        std.debug.print("PASS flashnext DeltaNet lanes: {d} lanes, {d} rows: one launch == per-lane (fronts, readouts, windows, states, keeps)\n", .{ lanes.len, f.rows() });
    }
    try check.expect(red, "flashnext DeltaNet lanes: a crossed lane table went unseen (control missed)", .{});
    std.debug.print("PASS flashnext DeltaNet lanes: control RED as planted (crossed lane table)\n", .{});
}

test "the lane table holds each lane's DeltaNet addresses after the attention layers'" {
    const tab = try std.testing.allocator.alignedAlloc(u8, .@"16", flash.batch_state.lanes_bytes);
    defer std.testing.allocator.free(tab);
    @memset(tab, 0);
    table.putGdn(tab, 1, 2, 11, 22, 0, 44, 55);
    table.putGdn(tab, 1, 3, 11, 22, 33, 0, 0);
    const ptrs: []const u64 = @alignCast(std.mem.bytesAsSlice(u64, tab[flash.batch_state.lanes_attn_ptrs..flash.batch_state.lanes_ptrs]));
    try std.testing.expectEqualSlices(u64, &.{ 11, 22, 22, 44, 55 }, ptrs[(max_lanes + 2) * 5 ..][0..5]);
    try std.testing.expectEqualSlices(u64, &.{ 11, 22, 33, 0, 0 }, ptrs[(max_lanes + 3) * 5 ..][0..5]);
}
