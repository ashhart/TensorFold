//! GLM-5.3-Flash's KDA on CUDA: decode chain and prompt chunks against float64 and the Python engine's bytes.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3_tests = @import("exl3_tests.zig");
const ref = @import("glm_kda_ref.zig");
const block_tests = @import("glm_kda_block.zig");
const kda = cuda.glm_kda;
const chunked_mod = cuda.glm_kda_chunk;

const Run = exl3_tests.Run;
const Rig = quant_ref_tests.RigT;
const Failed = check.Failed;

pub fn run(r: *Rig, state: *Run, dir: []const u8) !void {
    chain(r, state, dir) catch |e| state.fail(e, "glm.kda.chain");
    chunked(r, state, dir) catch |e| state.fail(e, "glm.kda.chunked");
    block_tests.block(r, state, dir) catch |e| state.fail(e, "glm.kda.block");
}

pub fn skip(id: []const u8) void {
    std.debug.print("SKIP {s}\n", .{id});
}

/// The largest difference as a fraction of its tolerance, over one check's float64 comparisons.
pub const T = struct {
    worst: f64 = 0,
    at: usize = 0,

    pub fn close(self: *T, name: []const u8, got: []const u8, want: []const f64, is_f32: bool, scale: f64, floor: f64) Failed!void {
        for (want, 0..) |w, i| {
            const g: f64 = if (is_f32)
                @as(f32, @bitCast(std.mem.readInt(u32, got[4 * i ..][0..4], .little)))
            else
                ref.bf(std.mem.readInt(u16, got[2 * i ..][0..2], .little));
            const q = @abs(g - w) / (@abs(w) * scale + floor);
            if (q > self.worst) {
                self.worst = q;
                self.at = i;
            }
        }
        if (self.worst > 1.0) {
            std.debug.print("FAIL {s}: f64 worst {d:.4} of tolerance at element {d}\n", .{ name, self.worst, self.at });
            return error.TestFailed;
        }
    }
};

/// The fixture's params (shared by every check).
pub const Params = struct {
    h: u32,
    b_off: i32,
    p_stride: i64,
    eps: f32,
    lower: f32,
    per: usize, // h * dk

    pub fn read(o: std.json.ObjectMap) !Params {
        const p = o.get("params").?.object;
        const h: u32 = @intCast(ref.int(p, "H"));
        return .{
            .h = h,
            .b_off = @intCast(ref.int(p, "b_off")),
            .p_stride = ref.int(p, "p_stride"),
            .eps = @floatCast(ref.float(p, "eps")),
            .lower = @floatCast(ref.float(p, "lower")),
            .per = @as(usize, h) * ref.DK,
        };
    }
};

fn chain(r: *Rig, state: *Run, dir: []const u8) !void {
    if (std.mem.eql(u8, dir, "none")) {
        _ = state;
        skip("glm.kda.chain");
        return;
    }
    var man = try ref.Manifest.open(r.a, r.gpu.io, dir);
    defer man.deinit();
    const o = man.kda() orelse {
        skip("glm.kda.chain");
        return;
    };
    const q = try Params.read(o);
    var k = try kda.Kda.load(r.gpu.d);
    defer k.unload();

    const a_log = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, "kda_a_log.bin"));
    const dt_bias = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, "kda_dt_bias.bin"));
    const norm_w = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, "kda_norm.bin"));
    const conv_w = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, "kda_conv.bin"));

    var t = T{};
    const chains = o.get("chain").?.object;
    for ([_][]const u8{ "1", "3", "8", "64", "128" }) |key| {
        const rows: u32 = @intCast(try std.fmt.parseInt(u32, key, 10));
        const e = chains.get(key).?.object;
        const p = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "p")));
        const a = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "a")));
        const g = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "g")));
        const conv0 = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "conv0")));
        const state0 = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "state0")));
        const want_out = try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "out"));
        const want_state = try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "state_out"));
        const out = try r.zeros(want_out.len);
        const st_out = try r.zeros(want_state.len);

        const sc = kda.Scratch{
            .k = try r.zeros(@as(usize, rows) * q.per * 4),
            .v = try r.zeros(@as(usize, rows) * q.per * 2),
            .g = try r.zeros(@as(usize, rows) * q.per * 4),
            .b = try r.zeros(@as(usize, rows) * @as(usize, q.h) * 4),
            .q = try r.zeros(@as(usize, rows) * q.per * 4),
            .y = try r.zeros(@as(usize, rows) * q.per * 2),
        };
        if (rows >= 64) {
            try k.chainWide(&r.s, q.h, p, q.p_stride, @as(i32, @intCast(q.b_off)), a, @as(i64, @intCast(q.per)), g, @as(i64, @intCast(q.per)), conv0, conv_w, state0, a_log, dt_bias, norm_w, q.eps, q.lower, rows, out, st_out, sc);
        } else {
            try k.chain(&r.s, q.h, p, q.p_stride, @as(i32, @intCast(q.b_off)), a, @as(i64, @intCast(q.per)), g, @as(i64, @intCast(q.per)), conv0, conv_w, state0, a_log, dt_bias, norm_w, q.eps, q.lower, rows, out, st_out, sc);
        }
        const got_out = try r.back(out, want_out.len);
        try check.sameBytes(try std.fmt.allocPrint(r.a, "glm.kda.chain r{d} out", .{rows}), got_out, want_out);
        const got_state = try r.back(st_out, want_state.len);
        try check.sameBytes(try std.fmt.allocPrint(r.a, "glm.kda.chain r{d} state", .{rows}), got_state, want_state);

        if (rows == 8) {
            const st2 = try r.zeros(want_state.len);
            try k.replay(&r.s, q.h, state0, .{ .k = sc.k, .v = sc.v, .g = sc.g, .b = sc.b }, 8, st2);
            const got_replay = try r.back(st2, want_state.len);
            try check.sameBytes("glm.kda.chain replay state", got_replay, want_state);
            const sk8 = e.get("scratch").?.object;
            inline for (.{ "k", "v", "g", "b" }) |sk_name| {
                const want_sc = try ref.readFile(r.a, r.gpu.io, dir, ref.str(sk8, sk_name));
                const got_sc = try r.back(@field(sc, sk_name), want_sc.len);
                try check.sameBytes(try std.fmt.allocPrint(r.a, "glm.kda.chain scratch {s}", .{sk_name}), got_sc, want_sc);
            }
            try rowInvariant(r, dir, &k, q, p, a, g, conv0, conv_w, state0, a_log, dt_bias, norm_w, want_out, want_state);
        }

        const out64 = try r.a.alloc(f64, @as(usize, rows) * q.per);
        const st64 = try r.a.alloc(f64, q.per * ref.DK);
        try ref.serialRef(r.a, ref.f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "q"))), ref.f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "k"))), ref.f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "v"))), ref.f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "lg"))), ref.f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "beta"))), ref.f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "state0"))), ref.bf16s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "g"))), ref.bf16s(try ref.readFile(r.a, r.gpu.io, dir, "kda_norm.bin")), q.eps, q.h, rows, out64, st64);
        try t.close(try std.fmt.allocPrint(r.a, "glm.kda.chain r{d} out f64", .{rows}), got_out, out64, false, 1.0 / 128.0, 2e-3);
        try t.close(try std.fmt.allocPrint(r.a, "glm.kda.chain r{d} state f64", .{rows}), got_state, st64, true, 2e-2, 2e-3);
    }
    std.debug.print("RESULT glm.kda.chain: f64 worst {d:.4} of tolerance\n", .{t.worst});
    check.pass("glm.kda.chain", .{});
}

/// The same 8 rows one at a time, conv windows shifted in between: every row's bits and the final state.
fn rowInvariant(r: *Rig, dir: []const u8, k: *kda.Kda, q: Params, p: u64, a: u64, g: u64, conv0: u64, conv_w: u64, state0: u64, a_log: u64, dt_bias: u64, norm_w: u64, want_out: []const u8, want_state: []const u8) !void {
    const row_bytes = want_out.len / 8;
    var s_in = try r.zeros(want_state.len);
    var s_out = try r.zeros(want_state.len);
    const buf: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = s_in, .len = want_state.len };
    try buf.copyFrom(0, state0, want_state.len, r.s.handle);
    const conv = try r.zeros(9 * q.per * 2);
    const cbuf: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = conv, .len = 9 * q.per * 2 };
    try cbuf.copyFrom(0, conv0, 9 * q.per * 2, r.s.handle);
    for (0..8) |i| {
        const out_i = try r.zeros(row_bytes);
        try k.chain(&r.s, q.h, p + @as(u64, @intCast(@as(i64, @intCast(i)) * q.p_stride * 2)), q.p_stride, @as(i32, @intCast(q.b_off)), a + @as(u64, @intCast(i * q.per * 2)), @as(i64, @intCast(q.per)), g + @as(u64, @intCast(i * q.per * 2)), @as(i64, @intCast(q.per)), conv, conv_w, s_in, a_log, dt_bias, norm_w, q.eps, q.lower, 1, out_i, s_out, .{});
        const got = try r.back(out_i, row_bytes);
        try check.sameBytes(try std.fmt.allocPrint(r.a, "glm.kda.chain row {d} (one row at a time)", .{i}), got, want_out[i * row_bytes ..][0..row_bytes]);
        const t = s_in;
        s_in = s_out;
        s_out = t;
        if (i < 7) try k.convShift(&r.s, conv, p + @as(u64, @intCast(@as(i64, @intCast(i)) * q.p_stride * 2)), 1, 1, 9 * @as(i64, @intCast(q.per)), q.p_stride, q.p_stride, @intCast(q.per * 3));
    }
    const rw_state = try ref.readFile(r.a, r.gpu.io, dir, "chain_r8_rowwise_state.bin");
    const got_state = try r.back(s_in, rw_state.len);
    try check.sameBytes("glm.kda.chain rowwise state", got_state, rw_state);
}

fn chunked(r: *Rig, state: *Run, dir: []const u8) !void {
    _ = state;
    if (std.mem.eql(u8, dir, "none")) {
        skip("glm.kda.chunked");
        return;
    }
    var man = try ref.Manifest.open(r.a, r.gpu.io, dir);
    defer man.deinit();
    const o = man.kda() orelse {
        skip("glm.kda.chunked");
        return;
    };
    const q = try Params.read(o);
    var c = try chunked_mod.Chunk.load(r.gpu.d);
    defer c.unload();

    const a_log = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, "kda_a_log.bin"));
    const dt_bias = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, "kda_dt_bias.bin"));
    const norm_w = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, "kda_norm.bin"));
    const conv_w = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, "kda_conv.bin"));

    const e = o.get("chunked").?.object;
    const rows: u32 = @intCast(ref.int(e, "rows"));
    const p_bytes = try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "p"));
    const p = try r.dev(p_bytes);
    const a = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "a")));
    const g = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "g")));
    const conv0 = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "conv0")));
    const state0 = try r.dev(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "state0")));
    const want_out = try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "out"));
    const want_state = try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "state_out"));
    const out = try r.zeros(want_out.len);
    const st = try r.zeros(want_state.len);
    try c.chain(&r.s, q.h, p, q.p_stride, @as(i32, @intCast(q.b_off)), a, @as(i64, @intCast(q.per)), g, @as(i64, @intCast(q.per)), conv0, conv_w, state0, a_log, dt_bias, norm_w, q.eps, q.lower, rows, 0, out, st);
    const got_out = try r.back(out, want_out.len);
    try check.sameBytes("glm.kda.chunked out", got_out, want_out);
    const got_state = try r.back(st, want_state.len);
    try check.sameBytes("glm.kda.chunked state", got_state, want_state);

    const out64 = try r.a.alloc(f64, @as(usize, rows) * q.per);
    const st64 = try r.a.alloc(f64, q.per * ref.DK);
    try ref.chunkedRef(r.a, ref.f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "q"))), ref.f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "k"))), ref.f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "v"))), ref.f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "lg"))), ref.f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "beta"))), ref.f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "state0"))), ref.bf16s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(e, "g"))), ref.bf16s(try ref.readFile(r.a, r.gpu.io, dir, "kda_norm.bin")), q.eps, q.h, rows, 0, out64, st64);
    var t = T{};
    try t.close("glm.kda.chunked out f64", got_out, out64, false, 1.0 / 128.0, 2e-3);
    try t.close("glm.kda.chunked state f64", got_state, st64, true, 2e-2, 2e-3);

    // the cuts: segment 1's state, then rows n.. from it through the shifted conv window reproduce the whole run
    const splits = e.get("splits").?.object;
    for ([_][]const u8{ "32", "96", "1024", "2016" }) |key| {
        const n: u32 = @intCast(try std.fmt.parseInt(u32, key, 10));
        const want_st1 = try ref.readFile(r.a, r.gpu.io, dir, ref.str(splits.get(key).?.object, "st1"));
        const out1 = try r.zeros(@as(usize, n) * q.per * 2);
        const st1 = try r.zeros(want_state.len);
        try c.chain(&r.s, q.h, p, q.p_stride, @as(i32, @intCast(q.b_off)), a, @as(i64, @intCast(q.per)), g, @as(i64, @intCast(q.per)), conv0, conv_w, state0, a_log, dt_bias, norm_w, q.eps, q.lower, n, 0, out1, st1);
        const got_st1 = try r.back(st1, want_st1.len);
        try check.sameBytes(try std.fmt.allocPrint(r.a, "glm.kda.chunked cut {d} state", .{n}), got_st1, want_st1);

        const stride: usize = @intCast(q.p_stride);
        const ch: usize = q.per * 3;
        const conv_mid = try r.a.alloc(u8, 3 * ch * 2);
        for (0..3) |ri| @memcpy(conv_mid[ri * ch * 2 ..][0 .. ch * 2], p_bytes[(n - 3 + ri) * stride * 2 ..][0 .. ch * 2]);
        const out2 = try r.zeros((@as(usize, rows) - n) * q.per * 2);
        const st2 = try r.zeros(want_state.len);
        try c.chain(&r.s, q.h, p + @as(u64, n) * @as(u64, stride) * 2, q.p_stride, q.b_off, a + @as(u64, n) * @as(u64, q.per) * 2, @as(i64, @intCast(q.per)), g + @as(u64, n) * @as(u64, q.per) * 2, @as(i64, @intCast(q.per)), try r.dev(conv_mid), conv_w, st1, a_log, dt_bias, norm_w, q.eps, q.lower, rows - n, 0, out2, st2);
        const got_out1 = try r.back(out1, @as(usize, n) * q.per * 2);
        const got_out2 = try r.back(out2, (@as(usize, rows) - n) * q.per * 2);
        const cat = try r.a.alloc(u8, got_out1.len + got_out2.len);
        @memcpy(cat[0..got_out1.len], got_out1);
        @memcpy(cat[got_out1.len..], got_out2);
        try check.sameBytes(try std.fmt.allocPrint(r.a, "glm.kda.chunked cut {d} rows", .{n}), cat, want_out);
        const got_st2 = try r.back(st2, want_state.len);
        try check.sameBytes(try std.fmt.allocPrint(r.a, "glm.kda.chunked cut {d} final state", .{n}), got_st2, want_state);
    }
    std.debug.print("RESULT glm.kda.chunked: f64 worst {d:.4} of tolerance\n", .{t.worst});
    check.pass("glm.kda.chunked", .{});
}
