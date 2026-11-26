//! A visit's narrow projections (q, gate, r, k, v) as one rotation launch and one linear launch, each member's bits.
const std = @import("std");
const cuda = @import("cuda");

const exl3 = cuda.exl3;
const Buf = cuda.DeviceBuffer;

/// The most members and rows a group launch takes (linear_body's one-pass rows).
pub const max_members = 5;
pub const max_rows = 16;
const widths = [_]u32{ 8, 10, 12 }; // halves: 4, 5 and 6 bits, mul1

/// exl3_linear.cu's LinearArgs, GroupArgs5 and RotGroup5Args as the kernels read them.
const LinearArgsC = extern struct { xh: u64, t: u64, stride_k: i64, stride_nb: i64, svh: u64, bias: u64, y: u64, z: u64, counters: u64, y_dtype: i32, n: i32, sk: i32, m: i32, k: i32 };
const GroupC = extern struct { l: [max_members]LinearArgsC, ends: [max_members]i32 };
const RotC = extern struct { x: [max_members]u64, suh: [max_members]u64, xh: [max_members]u64, x_dtype: i32, m: i32, k: i32 };

/// One member: its weight, input rows (bf16) and output (y_dtype).
pub const Member = struct { w: exl3.Weight, x: u64, y: u64, dt: exl3.DType };

pub const Group = struct {
    d: *const cuda.Driver,
    rot: cuda.Function,
    fns: [widths.len][2]cuda.Function, // by width, then 4 or 8 warps
    pdl: bool,
    xh: ?Buf = null,
    z: ?Buf = null,
    counters: ?Buf = null,

    pub fn init(d: *const cuda.Driver, lin: *const exl3.Linear) !Group {
        var g: Group = .{ .d = d, .rot = try lin.mod.function("tf_exl3_rot_group5"), .fns = undefined, .pdl = lin.pdl };
        inline for (widths, 0..) |hv, i| inline for (.{ 4, 8 }, 0..) |wk, j| {
            g.fns[i][j] = try lin.mod.function(std.fmt.comptimePrint("tf_exl3_group5_{d}_2_{d}", .{ hv, wk }));
        };
        return g;
    }

    pub fn deinit(g: *Group) void {
        for ([_]*?Buf{ &g.xh, &g.z, &g.counters }) |b| if (b.*) |*x| {
            x.free();
            b.* = null;
        };
    }

    /// Whether one launch takes these members at `rows`: one K, width and warp count, whole 128-column blocks.
    pub fn fits(ms: []const Member, rows: usize) bool {
        if (ms.len < 2 or ms.len > max_members or rows < 1 or rows > max_rows) return false;
        const w0 = ms[0].w;
        const wk = exl3.planOf(w0).wk;
        if ((wk != 4 and wk != 8) or std.mem.indexOfScalar(u32, &widths, w0.halves) == null) return false;
        for (ms) |m| {
            const w = m.w;
            const p = exl3.planOf(w);
            if (w.halves != w0.halves or w.codebook != .mul1 or w.k != w0.k or p.wk != wk or m.dt == .f16) return false;
            if (w.k % 128 != 0 or w.n % 128 != 0 or (w.k / 16) % (p.sk * p.wk) != 0) return false;
        }
        return true;
    }

    fn grown(g: *Group, slot: *?Buf, bytes: usize, zero: ?cuda.Stream) !u64 {
        if (slot.*) |b| if (b.len >= bytes) return b.ptr;
        const nb = try Buf.alloc(g.d, bytes);
        if (slot.*) |*old| old.free();
        slot.* = nb;
        // counters start at zero; each block's last program resets its own
        if (zero) |s| try slot.*.?.fill8(0, s.handle);
        return nb.ptr;
    }

    /// Scratch for a group of `ws` at max_rows, grown before any graph capture (a call inside one allocates nothing).
    pub fn reserve(g: *Group, s: cuda.Stream, ws: []const exl3.Weight) !void {
        var zb: usize = 0;
        var cb: usize = 0;
        for (ws) |w| {
            zb += exl3.Linear.zBytes(w, max_rows);
            cb += exl3.Linear.counterBytes(w);
        }
        _ = try g.grown(&g.xh, ws.len * max_rows * ws[0].k * 2, null);
        if (zb > 0) _ = try g.grown(&g.z, zb, null);
        _ = try g.grown(&g.counters, cb, s);
    }

    /// Every member's y from its x on `s`: the rotations in one launch, then the linears in one.
    pub fn run(g: *Group, s: cuda.Stream, ms: []const Member, rows: usize) !void {
        if (!fits(ms, rows)) return error.Exl3Group;
        var ws: [max_members]exl3.Weight = undefined;
        for (ms, 0..) |m, i| ws[i] = m.w;
        try g.reserve(s, ws[0..ms.len]);
        const k = ms[0].w.k;
        var rot: RotC = .{ .x = @splat(0), .suh = @splat(0), .xh = @splat(0), .x_dtype = @backingInt(exl3.DType.bf16), .m = @intCast(rows), .k = @intCast(k) };
        var ga: GroupC = .{ .l = undefined, .ends = @splat(0) };
        var zat = if (g.z) |b| b.ptr else 0;
        var cat = g.counters.?.ptr;
        var end: i32 = 0;
        for (ms, 0..) |m, i| {
            const w = m.w;
            const p = exl3.planOf(w);
            rot.x[i] = m.x;
            rot.suh[i] = w.suh;
            rot.xh[i] = g.xh.?.ptr + i * max_rows * k * 2;
            ga.l[i] = .{ .xh = rot.xh[i], .t = w.trellis, .stride_k = @intCast(w.stride_k), .stride_nb = @intCast(w.stride_nb), .svh = w.svh, .bias = w.bias, .y = m.y, .z = if (p.sk > 1) zat else 0, .counters = cat, .y_dtype = @backingInt(m.dt), .n = @intCast(w.n), .sk = @intCast(p.sk), .m = @intCast(rows), .k = @intCast(k) };
            zat += exl3.Linear.zBytes(w, max_rows);
            cat += exl3.Linear.counterBytes(w);
            end += @intCast(w.n / 128 * p.sk);
            ga.ends[i] = end;
        }
        for (ms.len..max_members) |i| {
            ga.l[i] = ga.l[0];
            ga.ends[i] = end;
        }
        var ra: cuda.Args = .{};
        ra.add(rot);
        try cuda.launch.launch(g.rot, .{ .grid = .{ .x = @intCast((k / 128 + 3) / 4), .y = @intCast(rows), .z = @intCast(ms.len) }, .block = .{ .x = 128 }, .pdl = g.pdl }, s, &ra);
        var la: cuda.Args = .{};
        la.add(ga);
        const wk = exl3.planOf(ms[0].w).wk;
        const wi = std.mem.indexOfScalar(u32, &widths, ms[0].w.halves).?;
        try cuda.launch.launch(g.fns[wi][if (wk == 4) 0 else 1], .{ .grid = .{ .x = @intCast(end) }, .block = .{ .x = wk * 32 }, .shared = exl3.Linear.sharedBytes(ms[0].w, rows), .pdl = g.pdl }, s, &la);
    }
};

test "proj_group: members of one width, K and warps fit; others do not" {
    const w: exl3.Weight = .{ .trellis = 0, .suh = 0, .svh = 0, .k = 2304, .n = 1536, .halves = 12, .codebook = .mul1, .stride_k = 0, .stride_nb = 0 };
    var r = w;
    r.n = 384;
    const ms = [_]Member{ .{ .w = w, .x = 0, .y = 0, .dt = .bf16 }, .{ .w = r, .x = 0, .y = 0, .dt = .f32 } };
    try std.testing.expect(Group.fits(&ms, 1));
    try std.testing.expect(Group.fits(&ms, 16));
    try std.testing.expect(!Group.fits(&ms, 17));
    var odd = ms;
    odd[1].w.halves = 8;
    try std.testing.expect(!Group.fits(&odd, 1));
    try std.testing.expectEqual(@as(usize, 136), @sizeOf(RotC));
    try std.testing.expectEqual(@as(usize, 504), @sizeOf(GroupC));
}
