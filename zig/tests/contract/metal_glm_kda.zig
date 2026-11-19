//! GLM's windowed KDA steps (glm_kda_window.metal, glm_kda_parts.metal) against kda_rows on the same window and state.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const bits = @import("metal_glm_bits.zig");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const kernels = tf.glm.kernels;

const D = 128;
const TAPS = 4;
const lower_bound: f32 = -5.0; // config.zig's gate_lower_bound and rms_norm_eps
const eps: f32 = 1e-5;

/// The window step (all heads, one Mac's 32 heads at TP2), and an Ultra's two groups a head with the norm after.
pub const Form = enum { window, window_tp, parts };

/// kernels.zig's KDA library: kda_rows' source, then the prompt passes, the window step and the parts appended.
pub fn text(a: std.mem.Allocator) ![]u8 {
    return std.mem.concat(a, u8, &.{ comptime kernels.generatedSource("kda_rows"), sources.glm_kda_prompt, sources.glm_kda_window, sources.glm_kda_parts });
}

fn heads(f: Form) usize {
    return if (f == .window_tp) 32 else 64;
}

/// Rows: the window's rows (one row keeps kda_rows); n: the heads; k: the head width. RB 8: 8 and 9 are edges.
const served_rows = [_]usize{ 16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
fn servedOf(comptime f: Form) [served_rows.len]c.Shape {
    var out: [served_rows.len]c.Shape = undefined;
    for (&out, served_rows) |*s, r| s.* = .{ .rows = r, .n = heads(f), .k = D };
    return out;
}
const served = [3][served_rows.len]c.Shape{ servedOf(.window), servedOf(.window_tp), servedOf(.parts) };

/// The step's buffers in kda_rows' binding order (P, its shape is a value), then Y, the state and the window out.
const Slots = enum(u8) { p, cs, cw, fbw, fbs, fbb, gbw, gbs, gbb, a, dtb, st, onw, y, st_out, cs_out };

pub const Case = struct {
    form: Form,
    control: bool,
    new: mtl.Pipeline,
    norm: ?mtl.Pipeline = null,
    old: mtl.Pipeline,

    pub fn init(s: *bits.Suite, form: Form, control: bool) !Case {
        const loader = try s.production();
        const own = try text(s.a);
        defer s.a.free(own);
        _ = try loader.require(own);
        const body = if (!control) try s.a.dupe(u8, own) else if (form == .parts) try bits.edited(s.a, own, "void glm_kda_parts(", "const uint dv0 = part * (uint)DV;", "const uint dv0 = 0u * part;") else try bits.edited(s.a, own, "void glm_kda_window(", "P[(size_t)(r0 + (int)tid) * PS + BO + h]", "P[(size_t)r0 * PS + BO + h]");
        defer s.a.free(body);
        const fun = switch (form) {
            .window => "glm_kda_window",
            .window_tp => "glm_kda_window_tp",
            .parts => "glm_kda_parts",
        };
        const new = try bits.pipeline(s.device, body, fun);
        errdefer new.deinit();
        const norm = if (form == .parts) try bits.pipeline(s.device, body, "glm_kda_norm") else null;
        errdefer if (norm) |p| p.deinit();
        return .{ .form = form, .control = control, .new = new, .norm = norm, .old = try oldPipe(s, form == .window_tp) };
    }
    pub fn deinit(x: *Case) void {
        x.new.deinit();
        if (x.norm) |p| p.deinit();
        x.old.deinit();
    }
    pub fn name(form: Form, control: bool) []const u8 {
        return switch (form) {
            .window => if (control) "control: metal glm launch kda glm_kda_window, every row's beta read from the first row" else "metal glm launch kda glm_kda_window 64 heads",
            .window_tp => if (control) "control: metal glm launch kda glm_kda_window_tp, every row's beta read from the first row" else "metal glm launch kda glm_kda_window_tp 32 heads",
            .parts => if (control) "control: metal glm launch kda glm_kda_parts, the second group on the first group's value columns" else "metal glm launch kda glm_kda_parts and glm_kda_norm 64 heads",
        };
    }
    pub fn spec(x: *const Case) c.Spec {
        return bits.spec(name(x.form, x.control), if (x.form == .window_tp) "kda_rows_tp" else "kda_rows", "kda_rows' bytes in the outputs, the next state and the next conv window, NaN payloads included", &served[@backingInt(x.form)], "kda_rows' sums in its order: quad dots, conv taps, simd sums and the serial recurrence");
    }
    fn proj(f: Form) usize {
        return 3 * heads(f) * D + 2 * D + heads(f);
    }
    pub fn layout(x: *const Case, s: c.Shape) ?bits.Layout {
        if (s.rows < 2 or s.rows > 16 or s.n != heads(x.form) or s.k != D) return null;
        return kdaLayout(s.rows, heads(x.form), x.form == .parts);
    }
    pub fn fill(x: *const Case, s: c.Shape, f: g.Fill, seed: u64, data: [][]u8) !void {
        kdaFill(s.rows, heads(x.form), f, seed, data);
    }
    /// forward.zig's kdaStep: the parts take the gate rows at buffer 13 and 16 rows of lanes, then the norm.
    pub fn encode(x: *const Case, e: mtl.ComputeEncoder, r: []const rig_.Ref, s: c.Shape, new: bool) void {
        const h = heads(x.form);
        if (!new or x.form != .parts) return kdaEncode(e, r, if (new) x.new else x.old, s.rows, h, 32, null);
        kdaEncode(e, r, x.new, s.rows, h, 16, r[16]);
        e.setPipeline(x.norm.?);
        e.setBuffer(r[13].buf, r[13].off, 0);
        e.setBuffer(r[16].buf, r[16].off, 1);
        e.setBuffer(r[12].buf, r[12].off, 2);
        e.setValue(eps, 3);
        e.dispatchGroups(mtl.Size.of(h / 8, s.rows, 1), mtl.Size.of(256, 1, 1));
    }
};

/// The generated kda_rows (all 64 heads) or kda_rows_tp (one Mac's 32), from the loader's own library.
pub fn oldPipe(s: *bits.Suite, tp: bool) !mtl.Pipeline {
    const loader = try s.production();
    if (tp) return bits.pipeline(s.device, try loader.require(comptime kernels.generatedSource("kda_rows_tp")), comptime kernels.generatedFunction("kda_rows_tp"));
    return bits.pipeline(s.device, try loader.require(comptime kernels.generatedSource("kda_rows")), comptime kernels.generatedFunction("kda_rows"));
}

/// The step's slots for `rows` rows of `h` heads (`gate`: kda_parts' gate rows after them).
pub fn kdaLayout(rows: usize, h: usize, gate: bool) bits.Layout {
    const W = h * D;
    var l: bits.Layout = .{};
    _ = l.add(.input, rows * (3 * W + 2 * D + h) * 2, .bf16);
    _ = l.add(.input, (TAPS - 1) * 3 * W * 2, .bf16);
    _ = l.add(.input, TAPS * 3 * W * 4, .f32);
    for (0..2) |_| {
        _ = l.add(.input, W * D * 4 / 8, .int);
        _ = l.add(.input, W * (D / 64) * 2, .bf16);
        _ = l.add(.input, W * (D / 64) * 2, .bf16);
    }
    _ = l.add(.input, h * 4, .f32);
    _ = l.add(.input, W * 4, .f32);
    _ = l.add(.input, h * D * D * 4, .f32);
    _ = l.add(.input, D * 4, .f32);
    _ = l.add(.output, rows * W * 2, .bf16);
    _ = l.add(.output, h * D * D * 4, .f32);
    _ = l.add(.output, (TAPS - 1) * 3 * W * 2, .bf16);
    if (gate) _ = l.add(.scratch, rows * W * 2, .bf16);
    return l;
}

fn uniform(rnd: std.Random, lo: f32, hi: f32) f32 {
    return lo + rnd.float(f32) * (hi - lo);
}

/// tf-glm-kda-speed's draws; odd seeds add -0 and large inputs; gate 1 and every third shape plant NaN and infinities.
pub fn kdaFill(rows: usize, h: usize, f: g.Fill, seed: u64, data: [][]u8) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const W = h * D;
    const P = 3 * W + 2 * D + h;
    for ([_]Slots{ .p, .cs }) |i| for (std.mem.bytesAsSlice(u16, data[@backingInt(i)])) |*v| {
        v.* = bits.bf(uniform(rnd, -2, 2));
    };
    for (std.mem.bytesAsSlice(f32, data[@backingInt(Slots.cw)])) |*v| v.* = uniform(rnd, -0.5, 0.5);
    for ([_]Slots{ .fbw, .gbw }) |i| rnd.bytes(data[@backingInt(i)]);
    for ([_]Slots{ .fbs, .gbs }) |i| for (std.mem.bytesAsSlice(u16, data[@backingInt(i)])) |*v| {
        v.* = bits.bf(uniform(rnd, 0.001, 0.02));
    };
    for ([_]Slots{ .fbb, .gbb }) |i| for (std.mem.bytesAsSlice(u16, data[@backingInt(i)])) |*v| {
        v.* = bits.bf(uniform(rnd, -0.05, 0.05));
    };
    for (std.mem.bytesAsSlice(f32, data[@backingInt(Slots.a)])) |*v| v.* = uniform(rnd, 0.5, 2);
    for (std.mem.bytesAsSlice(f32, data[@backingInt(Slots.dtb)])) |*v| v.* = uniform(rnd, -1, 1);
    for (std.mem.bytesAsSlice(f32, data[@backingInt(Slots.st)])) |*v| v.* = uniform(rnd, -0.1, 0.1);
    for (std.mem.bytesAsSlice(f32, data[@backingInt(Slots.onw)])) |*v| v.* = uniform(rnd, 0.5, 1.5);
    const p = std.mem.bytesAsSlice(u16, data[@backingInt(Slots.p)]);
    if (seed % 2 == 1) for (p, 0..) |*v, i| {
        if (i % 97 == 0) v.* = 0x8000 else if (i % 89 == 0) v.* = bits.bf(40);
    };
    if (f == .stress or (f == .random and seed % 3 != 2)) return;
    // q of row 0, k of row 1, v of row 2, f_b's input in row 3, beta of row 4: NaN, +Inf, -Inf, both, NaN
    const at = [_][2]usize{ .{ 0, 5 }, .{ 1, W + D + 7 }, .{ 2, 2 * W + 3 * D + 9 }, .{ 3, 3 * W + 11 }, .{ 3, 3 * W + 40 }, .{ 4, 3 * W + 2 * D + 1 } };
    const vals = [_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x7f80, 0xff80, 0x7fc0 };
    for (at, vals) |q, v| if (q[0] < rows) {
        p[q[0] * P + q[1]] = v;
    };
}

/// forward.zig's kdaStep for the generated gate: `pipe` over a (32, 32, heads) grid of (32, `ty`) groups.
pub fn kdaEncode(e: mtl.ComputeEncoder, r: []const rig_.Ref, pipe: mtl.Pipeline, rows: usize, h: usize, ty: usize, gate: ?rig_.Ref) void {
    e.setPipeline(pipe);
    e.setBuffer(r[0].buf, r[0].off, 0);
    const shape = [2]i32{ @intCast(rows), @intCast(3 * h * D + 2 * D + h) };
    e.setBytes(std.mem.asBytes(&shape), 1);
    for (1..13) |i| e.setBuffer(r[i].buf, r[i].off, i + 1);
    e.setValue(lower_bound, 14);
    e.setValue(eps, 15);
    for (13..16) |i| e.setBuffer(r[i].buf, r[i].off, i + 3);
    if (gate) |q| e.setBuffer(q.buf, q.off, 13);
    e.dispatchThreads(mtl.Size.of(32, 32, h), mtl.Size.of(32, ty, 1));
}

pub fn collect(s: *bits.Suite, only: ?[]const u8) !void {
    for (std.enums.values(Form)) |form| for ([_]bool{ false, true }) |control| {
        if (s.wanted(only, Case.name(form, control))) try s.add(Case, try Case.init(s, form, control));
    };
}

/// Gate 5's registrations `glm launch kda <function>`: the loader's KDA library.
pub const prefix = "glm launch kda ";
pub fn limitsText(a: std.mem.Allocator, _: mtl.Device, _: []const u8) !bits.Texts {
    const t = try text(a);
    return .{ .here = t, .pre_m5 = t };
}
