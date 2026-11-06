//! One rank's DeltaNet heads inside the full layout: the ranges each buffer keeps, their gather, the kernels' launches.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");

pub const Heads = flash.Heads;
const parts = flash.batch_state;
const full = Heads.full;

pub const dk = 128;
pub const dv = 128;
pub const taps = 4;
pub const eps: f32 = 1e-6;

/// A DeltaNet block buffer, by what one of its rows holds.
pub const Part = enum {
    /// A projection row: q, k and v channels, then z, b and a.
    proj,
    /// A conv window row: the q, k and v channels.
    conv,
    /// The conv taps, channel-major: one row of channels x taps.
    taps,
    /// a_log or dt_bias: one value a value head.
    gates,
    /// Normed q or k: dk values a key head.
    keys,
    /// v, the readouts or the gated-norm rows: dv values a value head.
    values,
    /// Decay and beta: two values a value head.
    gb,
    /// The recurrent state, one row: dv x dk values a value head.
    state,
    /// The gated norm's 32-wide sums: dv / 32 values a value head.
    sums,
};

/// Elements of one row of `part` for the heads h.
pub fn width(part: Part, h: Heads) usize {
    return switch (part) {
        .proj => h.projWidth(),
        .conv => h.convDim(),
        .taps => h.convDim() * taps,
        .gates => h.nv,
        .keys => h.nk * dk,
        .values => h.nv * dv,
        .gb => h.nv * 2,
        .state => h.nv * dv * dk,
        .sums => h.nv * dv / 32,
    };
}

/// Rows of `part` in a pass over m token rows: the conv window's taps - 1, one for the taps, gates and state.
pub fn rowsIn(part: Part, m: usize) usize {
    return switch (part) {
        .conv => taps - 1,
        .taps, .gates, .state => 1,
        else => m,
    };
}

/// Bytes of a whole buffer of `part` (elements `elem` bytes each) over m token rows, for the heads h.
pub fn bytes(part: Part, elem: usize, m: usize, h: Heads) usize {
    return rowsIn(part, m) * width(part, h) * elem;
}

pub const Cut = struct { from: usize, len: usize };

pub const Cuts = struct {
    buf: [6]Cut = undefined,
    len: usize = 0,

    fn add(c: *Cuts, from: usize, len: usize) void {
        c.buf[c.len] = .{ .from = from, .len = len };
        c.len += 1;
    }

    pub fn slice(c: *const Cuts) []const Cut {
        return c.buf[0..c.len];
    }
};

/// Rank r's ranges of a full-layout row of `part`, in the order its local row (heads h) holds them.
pub fn cuts(part: Part, h: Heads, r: usize) Cuts {
    var c: Cuts = .{};
    switch (part) {
        .proj, .conv, .taps => {
            const t: usize = if (part == .taps) taps else 1;
            const qk = h.nk * dk;
            const vw = h.nv * dv;
            c.add(r * qk * t, qk * t);
            c.add((full.nk * dk + r * qk) * t, qk * t);
            c.add((2 * full.nk * dk + r * vw) * t, vw * t);
            if (part == .proj) {
                const z = full.convDim();
                c.add(z + r * vw, vw);
                c.add(z + full.nv * dv + r * h.nv, h.nv);
                c.add(z + full.nv * dv + full.nv + r * h.nv, h.nv);
            }
        },
        else => {
            const w = width(part, h);
            c.add(r * w, w);
        },
    }
    return c;
}

/// The full-layout row index of element i of rank r's local row of `part`.
pub fn fullIndex(part: Part, h: Heads, r: usize, i: usize) usize {
    var left = i;
    const cs = cuts(part, h, r);
    for (cs.slice()) |c| {
        if (left < c.len) return c.from + left;
        left -= c.len;
    }
    unreachable;
}

/// Rank r's elements (`elem` bytes each) of a full-layout buffer of `part`, in the local layout of heads h.
pub fn gather(a: std.mem.Allocator, src: []const u8, elem: usize, part: Part, h: Heads, r: usize) ![]u8 {
    const fw = width(part, full) * elem;
    const lw = width(part, h) * elem;
    if (src.len % fw != 0) return error.LayoutLength;
    const n = src.len / fw;
    const out = try a.alloc(u8, n * lw);
    const cs = cuts(part, h, r);
    for (0..n) |row| {
        var at = row * lw;
        for (cs.slice()) |c| {
            @memcpy(out[at..][0 .. c.len * elem], src[row * fw + c.from * elem ..][0 .. c.len * elem]);
            at += c.len * elem;
        }
    }
    return out;
}

pub const Kind = enum { front, norm, rows, rows_to, tpr, fused };

/// The kernel of `kind`: the full layer's, or (half) its twin on one rank's 8 key and 24 value heads.
pub fn symbol(kind: Kind, half: bool) [:0]const u8 {
    return switch (kind) {
        .front => if (half) "fn_gdn_front_h8" else "fn_gdn_front",
        .norm => if (half) "fn_gdn_norm_h8" else "fn_gdn_norm",
        .rows => if (half) "fn_gdn_rows_h8" else "fn_gdn_rows",
        .rows_to => if (half) "fn_gdn_rows_to_h8" else "fn_gdn_rows_to",
        .tpr => if (half) "fn_gdn_chain_tpr_h8" else "fn_gdn_chain_tpr",
        .fused => if (half) "fn_gdn_fused_h8" else "fn_gdn_fused",
    };
}

/// A full kernel and its twin, by kind, from the flashnext_batch image (dynamic shared bytes allowed).
pub const Fns = struct {
    full: [6]cuda.Function,
    half: [6]cuda.Function,

    pub fn load(batch: cuda.Module) !Fns {
        var f: Fns = undefined;
        for (std.enums.values(Kind)) |k| {
            const i = @backingInt(k);
            f.full[i] = try batch.function(symbol(k, false));
            f.half[i] = try batch.function(symbol(k, true));
            const smem: u32 = switch (k) {
                .tpr => parts.chain_tpr_smem,
                .fused => parts.gdn_fused_smem,
                else => continue,
            };
            try f.full[i].allowDynamicShared(smem);
            try f.half[i].allowDynamicShared(smem);
        }
        return f;
    }

    /// Replace only the full-head oracle; candidate half handles stay initialized and owned by their original module.
    pub fn withFull(f: Fns, batch: cuda.Module) !Fns {
        var out = f;
        for (std.enums.values(Kind)) |k| {
            const i = @backingInt(k);
            out.full[i] = try batch.function(symbol(k, false));
            const smem: u32 = switch (k) {
                .tpr => parts.chain_tpr_smem,
                .fused => parts.gdn_fused_smem,
                else => continue,
            };
            try out.full[i].allowDynamicShared(smem);
        }
        return out;
    }

    /// The kernel of `kind` built for heads h.
    pub fn of(f: *const Fns, kind: Kind, h: Heads) cuda.Function {
        return if (h.eql(.full)) f.full[@backingInt(kind)] else f.half[@backingInt(kind)];
    }
};

test "a frozen full-head oracle replaces no candidate half handle" {
    const Fake = struct {
        var shared: usize = 0;
        fn get(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) c_int {
            for (std.enums.values(Kind)) |k| if (std.mem.eql(u8, std.mem.span(name), symbol(k, false))) {
                out.* = @ptrFromInt(0x300 + @as(usize, @backingInt(k)));
                return 0;
            };
            return 1;
        }
        fn attribute(_: cuda.abi.Function, _: cuda.abi.FunctionAttribute, _: c_int) callconv(.c) c_int {
            shared += 1;
            return 0;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuModuleGetFunction = Fake.get;
    driver.api.cuFuncSetAttribute = Fake.attribute;
    var candidate: Fns = undefined;
    for (&candidate.full, &candidate.half, 0..) |*f, *h, i| {
        f.* = .{ .d = &driver, .handle = @ptrFromInt(0x100 + i) };
        h.* = .{ .d = &driver, .handle = @ptrFromInt(0x200 + i) };
    }
    Fake.shared = 0;
    const oracle = try candidate.withFull(.{ .d = &driver, .handle = @ptrFromInt(0x10) });
    for (candidate.full, candidate.half, oracle.full, oracle.half, 0..) |f, h, of, oh, i| {
        try std.testing.expectEqual(@as(usize, 0x100 + i), @intFromPtr(f.handle));
        try std.testing.expectEqual(@as(usize, 0x300 + i), @intFromPtr(of.handle));
        try std.testing.expectEqual(h.handle, oh.handle);
    }
    try std.testing.expectEqual(@as(usize, 2), Fake.shared);
}

fn cint(n: usize) c_int {
    return @intCast(n);
}

/// fn_gdn_front: projection rows p after the conv window cs into normed q and k, v, decay and beta.
pub fn front(f: cuda.Function, s: cuda.Stream, h: Heads, rows: usize, p: u64, cs: u64, cw: u64, a_log: u64, dt: u64, qn: u64, kn: u64, v: u64, gb: u64) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ p, cs, cw, a_log, dt, qn, kn, v, gb }) |x| a.add(x);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = h.nk + h.nv }, .block = .{ .x = 256 } }, s, &a);
}

/// The chain over `rows` rows: in place (rows, st_out unused), to st_out (rows_to), or two threads a state row (tpr).
pub fn chain(f: cuda.Function, s: cuda.Stream, kind: Kind, h: Heads, rows: usize, qn: u64, kn: u64, v: u64, gb: u64, st_in: u64, st_out: u64, ys: u64) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ qn, kn, v, gb, st_in }) |x| a.add(x);
    if (kind != .rows) a.add(st_out);
    a.add(cint(rows));
    a.add(ys);
    const cfg: cuda.launch.Config = switch (kind) {
        .tpr => .{ .grid = .{ .x = h.nv }, .block = .{ .x = 256 }, .shared = parts.chain_tpr_smem },
        .rows, .rows_to => .{ .grid = .{ .x = h.nv, .y = dv / 16 }, .block = .{ .x = 128 } },
        else => return error.NotAChain,
    };
    try cuda.launch.launch(f, cfg, s, &a);
}

/// fn_gdn_norm: the readouts ys gated by z of the projection rows p, into out (bf16) and its sums xs.
pub fn norm(f: cuda.Function, s: cuda.Stream, h: Heads, rows: usize, ys: u64, p: u64, nw: u64, out: u64, xs: u64) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ ys, p, nw }) |x| a.add(x);
    a.add(eps);
    for ([_]u64{ out, xs }) |x| a.add(x);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = h.nv }, .block = .{ .x = dv } }, s, &a);
}

/// fn_gdn_fused: front, chain and gated norm in one launch a value head; ys 0 keeps no fp32 readouts.
pub fn fused(f: cuda.Function, s: cuda.Stream, h: Heads, rows: usize, p: u64, cs: u64, cw: u64, a_log: u64, dt: u64, st_in: u64, st_out: u64, nw: u64, out: u64, xs: u64, ys: u64) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ p, cs, cw, a_log, dt, st_in, st_out }) |x| a.add(x);
    a.add(cint(rows));
    a.add(nw);
    a.add(eps);
    for ([_]u64{ out, xs, ys }) |x| a.add(x);
    try cuda.launch.launch(f, .{ .grid = .{ .x = h.nv }, .block = .{ .x = parts.gdn_fused_threads }, .shared = parts.gdn_fused_smem }, s, &a);
}
