//! Flash Next's prompt glue replacing #952's: the DeltaNet chain, the indexer scores and the hyper-connection read-out.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const parts = flash.batch_state;
const tk = flash.tile_kernels;

pub const Kind = enum { chain, scores, readout, fused };

const nk = 16;
const nv = 48;
const dk = 128;
const dv = 128;
const di = 128;
const heads = 4;
const ratio = 4;
const top = 512;
const dims = 2560;
const streams = 4;
const wide = streams * dims;
const low = 320;
const n_down = low + streams;
const hc_sk = 32;
const nc = dims / 256;
const eps: f32 = 1e-6;
const taps = 4;
const conv_c = 2 * nk * dk + nv * dv; // a projection row's conv channels (q, k, v)
const proj_w = conv_c + nv * dv + 2 * nv; // a projection row: conv channels, z, b, a

fn noSums(_: c.Shape) usize {
    return 0;
}

/// rows = token rows; n = outputs a row (value heads x width, pooled blocks, or the read-out's mixed dims).
pub fn spec(kind: Kind) c.Spec {
    return switch (kind) {
        .chain => .{ .name = "cuda flashnext DeltaNet prompt chain fn_gdn_chain_tpr", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "ys and the state bit-equal to fn_gdn_rows_to's, non-finite inputs included" }, .replaces = "fn_gdn_rows_to", .tiles = .{ .{ .size = 16 }, .{}, .{} }, .served = &.{ .{ .rows = 64, .n = nv * dv, .k = dk }, .{ .rows = 300, .n = nv * dv, .k = dk } } },
        .scores => .{ .name = "cuda flashnext indexer scores qsa_tile::scores<8, 4>", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "every score byte equal to fn_qsa_scores', non-finite inputs included" }, .replaces = "fn_qsa_scores", .tiles = .{ .{ .size = 8 }, .{ .size = 256, .step = 1, .min = 1 }, .{} }, .served = &.{ .{ .rows = 64, .n = 600, .k = di }, .{ .rows = 300, .n = 2100, .k = di } } },
        .fused => .{ .name = "cuda flashnext DeltaNet prompt block fn_gdn_fused (front, chain, gated norm)", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "gated-norm rows, xs, the state and the readouts bit-equal to fn_gdn_front -> fn_gdn_rows_to -> fn_gdn_norm, non-finite inputs included" }, .replaces = "fn_gdn_front, fn_gdn_chain_tpr and fn_gdn_norm", .tiles = .{ .{ .size = 16 }, .{}, .{} }, .served = &.{ .{ .rows = 64, .n = nv * dv, .k = dk }, .{ .rows = 300, .n = nv * dv, .k = dk } } },
        .readout => .{ .name = "cuda flashnext hyper-connection read-out on prompt tiles (hcf down, reduce, hcf up, rinv, mix_h)", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "act, xs, inj, mixed and xsm bit-equal to #952's chain, non-finite inputs included" }, .replaces = "fn_hc_down, fn_hc_reduce_act and fn_hc_upmix", .tiles = .{ .{ .size = 64 }, .{}, .{} }, .served = &.{ .{ .rows = 64, .n = dims, .k = wide }, .{ .rows = 130, .n = dims, .k = wide } } },
    };
}

pub const Fns = struct { tpr: cuda.Function, rows_to: cuda.Function, tile: cuda.Function, scores: cuda.Function, hcf_down: cuda.Function, hcf_up: cuda.Function, front: cuda.Function, norm: cuda.Function, fused: cuda.Function };

pub const Glue = struct {
    rig: rig_.Rig,
    kind: Kind,
    f: *const Fns,
    glue: *const flash.Glue,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    ins: [12]u64 = @splat(0),
    outs: [5]u64 = @splat(0),
    lens: [5]usize = @splat(0),
    nout: usize = 0,
    base: u64 = 0,
    pos0: usize = 0,
    seen: std.ArrayList(u8) = .empty,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Glue) void {
        self.seen.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *Glue) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Glue);

    fn rng(seed: u64) std.Random.DefaultPrng {
        return std.Random.DefaultPrng.init(seed ^ 0x676c7565);
    }

    fn f32s(self: *Glue, n: usize, r: std.Random, lo: f32, hi: f32) ![]f32 {
        const v = try self.rig.gpu.gpa.alloc(f32, n);
        for (v) |*x| x.* = lo + r.float(f32) * (hi - lo);
        return v;
    }

    fn bfs(self: *Glue, n: usize, r: std.Random, lo: f32, hi: f32) ![]u16 {
        const v = try self.rig.gpu.gpa.alloc(u16, n);
        for (v) |*x| x.* = c.toBf16(lo + r.float(f32) * (hi - lo));
        return v;
    }

    fn in(self: *Glue, i: usize, bytes: []const u8) !void {
        self.ins[i] = try self.rig.input(bytes);
    }

    fn out(self: *Glue, len: usize) !void {
        self.outs[self.nout] = try self.rig.output(len);
        self.lens[self.nout] = len;
        self.nout += 1;
    }

    pub fn prepare(self: *Glue, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        self.nout = 0;
        self.shape = shape;
        if (shape.rows == 0) return false;
        var prng = rng(seed);
        const r = prng.random();
        const a = self.rig.gpu.gpa;
        const bad = fill == .nonfinite;
        switch (self.kind) {
            .chain => {
                if (shape.n != nv * dv or shape.k != dk) return false;
                const m = shape.rows;
                const qn = try self.f32s(m * nk * dk, r, -0.1, 0.1);
                defer a.free(qn);
                const kn = try self.f32s(m * nk * dk, r, -0.1, 0.1);
                defer a.free(kn);
                const v = try self.f32s(m * nv * dv, r, -1, 1);
                defer a.free(v);
                const gb = try self.f32s(m * nv * 2, r, 0, 1);
                defer a.free(gb);
                for (0..m * nv) |i| gb[2 * i] = -gb[2 * i]; // a log decay, then beta in (0, 1)
                const st = try self.f32s(nv * dv * dk, r, -0.5, 0.5);
                defer a.free(st);
                if (bad) {
                    qn[3] = std.math.nan(f32);
                    v[(m / 2) * nv * dv + 5] = std.math.inf(f32);
                    st[dv * dk + 7] = -std.math.inf(f32);
                }
                try self.in(0, std.mem.sliceAsBytes(qn));
                try self.in(1, std.mem.sliceAsBytes(kn));
                try self.in(2, std.mem.sliceAsBytes(v));
                try self.in(3, std.mem.sliceAsBytes(gb));
                try self.in(4, std.mem.sliceAsBytes(st));
                try self.out(m * nv * dv * 4);
                try self.out(nv * dv * dk * 4);
            },
            .scores => {
                if (shape.k != di or shape.n < 2 or shape.n > 4096) return false;
                const m = shape.rows;
                const nb = shape.n;
                // the last row sees nb complete blocks of `ratio`; rows past `top` blocks score
                if (nb * ratio < m) return false;
                self.pos0 = nb * ratio - m;
                if ((self.pos0 + m) / ratio <= top) return false;
                const iq = try self.bfs(m * heads * di, r, -1, 1);
                defer a.free(iq);
                const pooled = try self.bfs(nb * di, r, -1, 1);
                defer a.free(pooled);
                if (bad) {
                    iq[7] = c.toBf16(std.math.nan(f32));
                    pooled[di * 3 + 1] = c.toBf16(std.math.inf(f32));
                }
                const pos = [4]i32{ @intCast(self.pos0), 0, 0, 0 };
                try self.in(0, std.mem.sliceAsBytes(iq));
                try self.in(1, std.mem.sliceAsBytes(pooled));
                try self.in(2, std.mem.sliceAsBytes(&pos));
                try self.out(m * nb * 4);
            },
            .readout => {
                if (shape.n != dims or shape.k != wide) return false;
                try self.readoutInputs(shape.rows, r, bad);
            },
            .fused => {
                if (shape.n != nv * dv or shape.k != dk) return false;
                const m = shape.rows;
                const proj = try self.bfs(m * proj_w, r, -1, 1);
                defer a.free(proj);
                const cs = try self.bfs((taps - 1) * conv_c, r, -1, 1);
                defer a.free(cs);
                const cw = try self.bfs(conv_c * taps, r, -0.5, 0.5);
                defer a.free(cw);
                const a_log = try self.f32s(nv, r, -1, 1);
                defer a.free(a_log);
                const dt = try self.f32s(nv, r, -1, 1);
                defer a.free(dt);
                const st = try self.f32s(nv * dv * dk, r, -0.5, 0.5);
                defer a.free(st);
                const nw = try self.bfs(dv, r, 0.5, 1.5);
                defer a.free(nw);
                if (bad) {
                    proj[3] = c.toBf16(std.math.nan(f32));
                    proj[(m / 2) * proj_w + conv_c + 7] = c.toBf16(std.math.inf(f32));
                    st[dv * dk + 7] = -std.math.inf(f32);
                }
                try self.in(0, std.mem.sliceAsBytes(proj));
                try self.in(1, std.mem.sliceAsBytes(cs));
                try self.in(2, std.mem.sliceAsBytes(cw));
                try self.in(3, std.mem.sliceAsBytes(a_log));
                try self.in(4, std.mem.sliceAsBytes(dt));
                try self.in(5, std.mem.sliceAsBytes(st));
                try self.in(6, std.mem.sliceAsBytes(nw));
                try self.out(m * nv * dv * 2); // gated-norm rows (bf16)
                try self.out(m * (nv * dv / 32) * 4); // their 32-wide sums
                try self.out(nv * dv * dk * 4); // the state after the pass
                try self.out(m * nv * dv * 4); // the chain's readouts
            },
        }
        self.base = 0;
        var total: usize = 0;
        for (self.lens[0..self.nout]) |l| total += l;
        self.base = try self.rig.scratch(total);
        return true;
    }

    /// The streams and their sums of squares, the norm's scale, the down rows (inject stacked) and the up rows.
    fn readoutInputs(self: *Glue, m: usize, r: std.Random, bad: bool) !void {
        const a = self.rig.gpu.gpa;
        const h = try self.bfs(m * wide, r, -1, 1);
        defer a.free(h);
        if (bad) h[11] = c.toBf16(std.math.nan(f32));
        const pss = try a.alloc(f32, m * nc * streams);
        defer a.free(pss);
        for (0..m) |row| for (0..nc) |cc| for (0..streams) |s| {
            var t: f32 = 0;
            for (0..256) |i| {
                const v: f32 = @floatCast(c.fromBf16(h[row * wide + s * dims + cc * 256 + i]));
                t += v * v;
            }
            pss[(row * nc + cc) * streams + s] = t;
        };
        const scale = try self.f32s(wide, r, 0.5, 1.5);
        defer a.free(scale);
        try self.in(0, std.mem.sliceAsBytes(h));
        try self.in(1, std.mem.sliceAsBytes(pss));
        try self.in(2, std.mem.sliceAsBytes(scale));
        for ([_][2]usize{ .{ n_down, wide }, .{ wide, low } }, 0..) |nk_, wi| {
            const n = nk_[0];
            const k = nk_[1];
            const words = try a.alloc(u32, n * k / 8);
            defer a.free(words);
            for (words) |*w| w.* = r.int(u32);
            const sc = try self.bfs(n * k / 32, r, 0.001, if (wi == 0) 0.004 else 0.05);
            defer a.free(sc);
            const bi = try self.bfs(n * k / 32, r, -0.01, if (wi == 0) 0.01 else 0.1);
            defer a.free(bi);
            const tw = try flash.hc.tileWords(a, std.mem.sliceAsBytes(words), n, k / 8);
            defer a.free(tw);
            const ts = try flash.hc.transposeBf16(a, std.mem.sliceAsBytes(sc), n, k / 32);
            defer a.free(ts);
            const tb = try flash.hc.transposeBf16(a, std.mem.sliceAsBytes(bi), n, k / 32);
            defer a.free(tb);
            try self.in(3 + wi * 3, tw);
            try self.in(4 + wi * 3, ts);
            try self.in(5 + wi * 3, tb);
        }
        // act, xs, inj (the reduce's), then mixed and xsm (the mix's)
        try self.out(m * low * 2);
        try self.out(m * (low / 32) * 4);
        try self.out(m * streams * 2);
        try self.out(m * dims * 2);
        try self.out(m * (dims / 32) * 4);
    }

    fn sweepOf(m: usize, bm: usize, k: usize) usize {
        return @max(@as(usize, 1), @min((m + bm - 1) / bm, (12 << 20) / (bm * k * 2)));
    }

    /// Our launches as the pass makes them, into `o` (the outputs in order).
    fn ours(self: *Glue, o: []const u64) !void {
        const s = self.rig.s;
        const m = self.shape.rows;
        switch (self.kind) {
            .chain => {
                var a: cuda.Args = .{};
                for ([_]u64{ self.ins[0], self.ins[1], self.ins[2], self.ins[3], self.ins[4], o[1] }) |v| a.add(v);
                a.add(@as(c_int, @intCast(m)));
                a.add(o[0]);
                try cuda.launch.launch(self.f.tpr, .{ .grid = .{ .x = nv }, .block = .{ .x = 256 }, .shared = parts.chain_tpr_smem }, s, &a);
            },
            .scores => {
                var a: cuda.Args = .{};
                for ([_]u64{ self.ins[0], self.ins[1], self.ins[2], o[0] }) |v| a.add(v);
                for ([_]usize{ self.shape.n, ratio, top, m }) |v| a.add(@as(c_int, @intCast(v)));
                try cuda.launch.launch(self.f.tile, .{ .grid = .{ .x = @intCast((m + 7) / 8), .y = @intCast((self.shape.n + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
            },
            .fused => {
                var a: cuda.Args = .{};
                for ([_]u64{ self.ins[0], self.ins[1], self.ins[2], self.ins[3], self.ins[4], self.ins[5], o[2] }) |v| a.add(v);
                a.add(@as(c_int, @intCast(m)));
                a.add(self.ins[6]);
                a.add(eps);
                for ([_]u64{ o[0], o[1], o[3] }) |v| a.add(v);
                try cuda.launch.launch(self.f.fused, .{ .grid = .{ .x = nv }, .block = .{ .x = parts.gdn_fused_threads }, .shared = parts.gdn_fused_smem }, s, &a);
            },
            .readout => {
                const part = try self.rig.scratch(m * n_down * 4);
                const rinv = try self.rig.scratch(m * streams * 4);
                const up = try self.rig.scratch(m * wide * 2);
                var a: cuda.Args = .{};
                for ([_]u64{ self.ins[0], 0, self.ins[3], self.ins[4], self.ins[5], part }) |v| a.add(v);
                for ([_]usize{ m, n_down, wide, hc_sk, wide, sweepOf(m, 32, wide) }) |v| a.add(@as(c_int, @intCast(v)));
                a.add(self.ins[1]);
                a.add(self.ins[2]);
                a.add(eps);
                for ([_]usize{ dims, nc, streams }) |v| a.add(@as(c_int, @intCast(v)));
                try cuda.launch.launch(self.f.hcf_down, .{ .grid = .{ .x = @intCast((m + 31) / 32 * ((n_down + 191) / 192)) }, .block = .{ .x = 256 }, .shared = tk.hcf_down_smem }, s, &a);
                try self.glue.hcReduceAct(s, part, o[0], o[1], o[2], 1, m, streams, low, n_down, 1);
                var b: cuda.Args = .{};
                for ([_]u64{ o[0], o[1], self.ins[6], self.ins[7], self.ins[8], up }) |v| b.add(v);
                for ([_]usize{ m, wide, low, 1, low, sweepOf(m, 64, low) }) |v| b.add(@as(c_int, @intCast(v)));
                b.add(@as(u64, 0));
                b.add(@as(u64, 0));
                b.add(@as(f32, 0));
                for ([_]usize{ dims, nc, streams }) |v| b.add(@as(c_int, @intCast(v)));
                try cuda.launch.launch(self.f.hcf_up, .{ .grid = .{ .x = @intCast((m + 63) / 64 * (wide / 128)) }, .block = .{ .x = 256 }, .shared = tk.hcf_up_smem }, s, &b);
                try self.glue.hcRinv(s, self.ins[1], rinv, eps, m, dims, nc, streams);
                try self.glue.hcMixH(s, up, self.ins[0], rinv, self.ins[2], o[3], o[4], m, dims, streams);
            },
        }
    }

    /// #952's launches on the same inputs, into `o`.
    fn theirs(self: *Glue, o: []const u64) !void {
        const s = self.rig.s;
        const m = self.shape.rows;
        switch (self.kind) {
            .chain => {
                var a: cuda.Args = .{};
                for ([_]u64{ self.ins[0], self.ins[1], self.ins[2], self.ins[3], self.ins[4], o[1] }) |v| a.add(v);
                a.add(@as(c_int, @intCast(m)));
                a.add(o[0]);
                try cuda.launch.launch(self.f.rows_to, .{ .grid = .{ .x = nv, .y = dv / 16 }, .block = .{ .x = 128 } }, s, &a);
            },
            .scores => {
                var a: cuda.Args = .{};
                for ([_]u64{ self.ins[0], self.ins[1], self.ins[2], o[0] }) |v| a.add(v);
                for ([_]usize{ self.shape.n, heads, ratio, top }) |v| a.add(@as(c_int, @intCast(v)));
                try cuda.launch.launch(self.f.scores, .{ .grid = .{ .x = @intCast(m), .y = @intCast((self.shape.n + 63) / 64) }, .block = .{ .x = 256 } }, s, &a);
            },
            .fused => {
                const gq = try self.rig.scratch(m * nk * dk * 4);
                const gk = try self.rig.scratch(m * nk * dk * 4);
                const gv = try self.rig.scratch(m * nv * dv * 4);
                const ggb = try self.rig.scratch(m * nv * 2 * 4);
                var f: cuda.Args = .{};
                for ([_]u64{ self.ins[0], self.ins[1], self.ins[2], self.ins[3], self.ins[4], gq, gk, gv, ggb }) |v| f.add(v);
                try cuda.launch.launch(self.f.front, .{ .grid = .{ .x = @intCast(m), .y = nk + nv }, .block = .{ .x = 256 } }, s, &f);
                var a: cuda.Args = .{};
                for ([_]u64{ gq, gk, gv, ggb, self.ins[5], o[2] }) |v| a.add(v);
                a.add(@as(c_int, @intCast(m)));
                a.add(o[3]);
                try cuda.launch.launch(self.f.rows_to, .{ .grid = .{ .x = nv, .y = dv / 16 }, .block = .{ .x = 128 } }, s, &a);
                var b: cuda.Args = .{};
                for ([_]u64{ o[3], self.ins[0], self.ins[6] }) |v| b.add(v);
                b.add(eps);
                b.add(o[0]);
                b.add(o[1]);
                try cuda.launch.launch(self.f.norm, .{ .grid = .{ .x = @intCast(m), .y = nv }, .block = .{ .x = dv } }, s, &b);
            },
            .readout => {
                const normed = try self.rig.scratch(m * wide * 2);
                const dn = try self.rig.scratch(m * (low + 4) * 2);
                const part = try self.rig.scratch(hc_sk * m * n_down * 4);
                try self.glue.hcDown(s, self.ins[0], self.ins[1], self.ins[2], normed, self.ins[3], self.ins[4], self.ins[5], dn, part, eps, m, n_down, wide, dims, nc, streams, hc_sk);
                try self.glue.hcReduceAct(s, part, o[0], o[1], o[2], hc_sk, m, streams, low, n_down, 1);
                try self.glue.hcUpmix(s, o[0], o[1], self.ins[6], self.ins[7], self.ins[8], normed, o[3], o[4], m, wide, low, dims, streams);
            },
        }
    }

    pub fn run(self: *Glue) !void {
        try self.ours(self.outs[0..self.nout]);
        try self.rig.s.synchronize();
    }

    /// The scratch the baseline writes: the outputs' layout back to back, poisoned like them (0xff).
    fn baseline(self: *Glue) ![]const u8 {
        var at = self.base;
        var o: [5]u64 = undefined;
        var total: usize = 0;
        for (self.lens[0..self.nout], 0..) |l, i| {
            o[i] = at;
            at += l;
            total += l;
        }
        const ff = try self.rig.gpu.gpa.alloc(u8, total);
        defer self.rig.gpu.gpa.free(ff);
        @memset(ff, 0xff);
        const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.base, .len = total };
        try b.upload(0, ff);
        try self.theirs(o[0..self.nout]);
        try self.rig.s.synchronize();
        return self.rig.bytes(self.base, total);
    }

    pub fn replaced(self: *Glue) ![]const u8 {
        return self.baseline();
    }

    /// Gate 1's stated rule: the outputs on the planted inputs equal #952's bytes.
    pub fn check(self: *Glue, gate: g.Gate) !?[]const u8 {
        _ = gate;
        self.seen.clearRetainingCapacity();
        try self.seen.appendSlice(self.rig.gpu.gpa, try self.rig.outputBytes());
        const want = try self.baseline();
        if (std.mem.eql(u8, self.seen.items, want)) return null;
        return try std.fmt.bufPrint(&self.msg, "bytes differ from #952's from byte {d}", .{std.mem.indexOfDiff(u8, self.seen.items, want).?});
    }

    pub fn count(self: *Glue) usize {
        var total: usize = 0;
        for (self.lens[0..self.nout]) |l| total += l;
        return total;
    }

    pub fn reference(_: *Glue, _: []f64, _: []f64) !void {
        return error.NoFloat64Reference;
    }
};
