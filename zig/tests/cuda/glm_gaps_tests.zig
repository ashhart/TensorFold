//! The glm_glue2 kernels and MoE per-slot epilogue against the Python engine's bytes (manifest "gaps").
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const ref = @import("glm_kda_ref.zig");
const glue = cuda.glm_glue;
const glue2 = cuda.glm_glue2;
const exl3e = cuda.exl3_experts;

const Rig = quant_ref_tests.RigT;
const Tol = @import("glm_glue_tests.zig").Tol;
const bf = quant_ref_tests.helpers.bfOf;

/// The manifest's "gaps" object, or null (the caller skips) when the fixture dir has none.
pub fn section(r: *Rig, dir: []const u8) !?std.json.ObjectMap {
    if (std.mem.eql(u8, dir, "none")) return null;
    var man = try ref.Manifest.open(r.a, r.gpu.io, dir);
    return if (ref.Manifest.root(&man).get("gaps")) |g| g.object else null;
}

fn up(r: *Rig, dir: []const u8, name: []const u8) !u64 {
    return r.dev(try ref.readFile(r.a, r.gpu.io, dir, name));
}

fn f32s(bytes: []const u8) []const f32 {
    return ref.f32s(bytes);
}

/// bf16 bytes as f64 values (exact).
fn bf4(a: std.mem.Allocator, bytes: []const u8) ![]f64 {
    const u = std.mem.bytesAsSlice(u16, bytes);
    const out = try a.alloc(f64, u.len);
    for (u, 0..) |v, i| out[i] = bf(v);
    return out;
}

// -- glm.glue.rmsnorm, the 1536 and 512 wide DSA layernorms ----------------------------------------
pub fn rmsnorm(r: *Rig, dir: []const u8) !void {
    const gp = (try section(r, dir)) orelse return;
    const o = gp.get("rms").?.object;
    var k = try glue2.Glue2.load(r.gpu.d);
    defer k.unload();
    const rows: usize = @intCast(ref.int(o, "rows"));
    const eps: f32 = @floatCast(ref.float(o, "eps"));
    const dp = try up(r, dir, ref.str(o, "dp"));
    for ([_]struct { d: usize, xoff: usize, nw: []const u8, out: []const u8, sums: []const u8 }{
        .{ .d = 1536, .xoff = 0, .nw = ref.str(o, "nw1"), .out = ref.str(o, "out1"), .sums = ref.str(o, "sums1") },
        .{ .d = 512, .xoff = 1536, .nw = ref.str(o, "nw2"), .out = ref.str(o, "out2"), .sums = ref.str(o, "sums2") },
    }) |case| {
        const w = try up(r, dir, case.nw);
        const want = try ref.readFile(r.a, r.gpu.io, dir, case.out);
        const want_sums = try ref.readFile(r.a, r.gpu.io, dir, case.sums);
        const out = try r.zeros(want.len);
        const sums = try r.zeros(want_sums.len);
        try k.rmsnorm2(r.s, dp + case.xoff * 2, 2048, w, out, @intCast(case.d), sums, eps, rows, case.d);
        try check.sameBytes("glm.glue.rmsnorm", try r.back(out, want.len), want);
        try check.sameBytes("glm.glue.rmsnorm sums", try r.back(sums, want_sums.len), want_sums);
        // f64: x starts at element case.xoff of dp's 2048-wide rows; w is the case's own.
        const dpv = try bf4(r.a, try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "dp")));
        const wv = try bf4(r.a, try ref.readFile(r.a, r.gpu.io, dir, case.nw));
        const gotv = try bf4(r.a, try r.back(out, want.len));
        const off: usize = if (case.d == 1536) 0 else 1536; // 512's x starts at element 1536 of the 2048-wide row
        var t = Tol{};
        for (0..rows) |i| {
            var ss: f64 = 0;
            for (0..case.d) |j| ss += dpv[i * 2048 + off + j] * dpv[i * 2048 + off + j];
            const rinv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(case.d)) + eps);
            for (0..case.d) |j| t.f64v(gotv[i * case.d + j], wv[j] * dpv[i * 2048 + off + j] * rinv, 1.0 / 64.0, 3e-3);
        }
        try t.done("glm.glue.rmsnorm gaps");
        check.pass("glm.glue.rmsnorm", .{});
    }
}

// -- glm.glue.combine, fp32 y -----------------------------------------------------------------------
pub fn combine(r: *Rig, dir: []const u8) !void {
    const gp = (try section(r, dir)) orelse return;
    const o = gp.get("combine").?.object;
    var k = try glue2.Glue2.load(r.gpu.d);
    defer k.unload();
    const rows: usize = @intCast(ref.int(o, "rows"));
    const d: usize = @intCast(ref.int(o, "d"));
    const y = try up(r, dir, ref.str(o, "y"));
    const wts = try up(r, dir, ref.str(o, "wts"));
    const want = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "out"));
    const out = try r.zeros(want.len);
    try k.combineF32(r.s, y, wts, out, rows, d, 9);
    try check.sameBytes("glm.glue.combine", try r.back(out, want.len), want);
    const yv = f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "y")));
    const wv = f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "wts")));
    const gotv = f32s(try r.back(out, want.len));
    var t = Tol{};
    for (0..rows) |i| for (0..d) |j| {
        var acc: f64 = 0;
        for (0..9) |kk| acc += wv[i * 9 + kk] * yv[(i * 9 + kk) * d + j];
        t.f32v(gotv, i * d + j, acc, 1.0 / 1024.0, 1e-5);
    };
    try t.done("glm.glue.combine gaps");
    check.pass("glm.glue.combine", .{});
}

// -- glm.glue.combine, the real MoE layer: per-slot down epilogue, then the fp32 combine
pub fn combineMoe(r: *Rig, dir: []const u8) !void {
    const gp = (try section(r, dir)) orelse return;
    const o = gp.get("moe").?.object;
    const count: usize = @intCast(ref.int(o, "experts"));
    const dims: usize = @intCast(ref.int(o, "dims"));
    const width: usize = @intCast(ref.int(o, "width"));
    const slots: usize = @intCast(ref.int(o, "slots"));
    var exp = try exl3e.Experts.load(r.gpu.d);
    defer exp.unload();
    var gate_p = try r.a.alloc(u64, count);
    var up_p = try r.a.alloc(u64, count);
    var down_p = try r.a.alloc(u64, count);
    var gk2 = try r.a.alloc(i32, count);
    var uk2 = try r.a.alloc(i32, count);
    var dk2 = try r.a.alloc(i32, count);
    for (o.get("gate_k2").?.array.items, 0..) |v, i| gk2[i] = @intCast(v.integer);
    for (o.get("up_k2").?.array.items, 0..) |v, i| uk2[i] = @intCast(v.integer);
    for (o.get("down_k2").?.array.items, 0..) |v, i| dk2[i] = @intCast(v.integer);
    for (0..count) |i| {
        gate_p[i] = try up(r, dir, o.get("gate").?.array.items[i].string);
        up_p[i] = try up(r, dir, o.get("up").?.array.items[i].string);
        down_p[i] = try up(r, dir, o.get("down").?.array.items[i].string);
    }
    const layer: exl3e.Layer = .{
        .gate = try r.dev(std.mem.sliceAsBytes(gate_p)),
        .up = try r.dev(std.mem.sliceAsBytes(up_p)),
        .down = try r.dev(std.mem.sliceAsBytes(down_p)),
        .gate_k2 = try r.dev(std.mem.sliceAsBytes(gk2)),
        .up_k2 = try r.dev(std.mem.sliceAsBytes(uk2)),
        .down_k2 = try r.dev(std.mem.sliceAsBytes(dk2)),
        .suh_g = try up(r, dir, ref.str(o, "suh_g")),
        .suh_u = try up(r, dir, ref.str(o, "suh_u")),
        .svh_g = try up(r, dir, ref.str(o, "svh_g")),
        .svh_u = try up(r, dir, ref.str(o, "svh_u")),
        .suh_d = try up(r, dir, ref.str(o, "suh_d")),
        .svh_d = try up(r, dir, ref.str(o, "svh_d")),
        .experts = count,
        .dims = dims,
        .width = width,
        .codebook = .mul1,
        .limit = @floatCast(ref.float(o, "limit")),
        .act_mode = @intCast(ref.int(o, "act_mode")),
    };
    var k = try glue2.Glue2.load(r.gpu.d);
    defer k.unload();
    const xmap = o.get("x").?.object;
    const pickmap = o.get("pick").?.object;
    const wtsmap = o.get("wts").?.object;
    const ymap = o.get("y").?.object;
    const symap = o.get("sy").?.object;
    const partmap = o.get("part").?.object;
    const key = struct {
        fn f(a: std.mem.Allocator, m: i64) ![]const u8 {
            return std.fmt.allocPrint(a, "{d}", .{m});
        }
    }.f;
    for (o.get("rows").?.array.items) |rv| {
        const m: usize = @intCast(rv.integer);
        const kk = try key(r.a, rv.integer);
        const dx = try up(r, dir, xmap.get(kk).?.string);
        const dpick = try up(r, dir, pickmap.get(kk).?.string);
        const dwts = try up(r, dir, wtsmap.get(kk).?.string);
        // ey: 0xff everywhere except the shared expert's slot, which starts at the fixture's sy
        const eyb = try r.a.alloc(u8, m * slots * dims * 4);
        @memset(eyb, 0xff);
        const sy = try ref.readFile(r.a, r.gpu.io, dir, symap.get(kk).?.string);
        for (0..m) |row| @memcpy(eyb[(row * slots + slots - 1) * dims * 4 ..][0 .. dims * 4], sy[row * dims * 4 ..][0 .. dims * 4]);
        const dey = try r.dev(eyb);
        const sb = try r.zeros(try exl3e.scratchBytes(layer, m, slots));
        const sc = try exl3e.scratch(sb, layer, m, slots);
        try exp.runPerSlot(r.s, layer, dx, dims, .bf16, dpick, m, dey, sc);
        const want_y = try ref.readFile(r.a, r.gpu.io, dir, ymap.get(kk).?.string);
        try check.sameBytes("glm.glue.combine moe ey", try r.back(dey, want_y.len), want_y);
        const want = try ref.readFile(r.a, r.gpu.io, dir, partmap.get(kk).?.string);
        const out = try r.zeros(want.len);
        try k.combineF32(r.s, dey, dwts, out, m, dims, slots);
        try check.sameBytes("glm.glue.combine moe", try r.back(out, want.len), want);
        // f64 over the fixture's y (fp32, exact): the same ordered sum in float64
        const yv = f32s(try ref.readFile(r.a, r.gpu.io, dir, ymap.get(kk).?.string));
        const wv = f32s(try ref.readFile(r.a, r.gpu.io, dir, wtsmap.get(kk).?.string));
        const gotv = f32s(try r.back(out, want.len));
        var t = Tol{};
        for (0..m) |i| for (0..dims) |j| {
            var acc: f64 = 0;
            for (0..slots) |q| acc += wv[i * slots + q] * yv[(i * slots + q) * dims + j];
            t.f32v(gotv, i * dims + j, acc, 1.0 / 1024.0, 1e-5);
        };
        try t.done("glm.glue.combine moe f64");
        check.pass("glm.glue.combine", .{});
    }
}
