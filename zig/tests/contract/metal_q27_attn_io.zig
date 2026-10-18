//! qwen27's attention_io kernels under the contract on Metal: packing and cache moves copy bf16 bits, checked exactly.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const at_ = @import("metal_q27_attn.zig");

const c = contract.spec;
const g = contract.gates;
const attn = tf.qwen27.attention;
const planning = tf.qwen27.attention_plan;

const D = 256;
const rw = mtl.ResourceUsage.read | mtl.ResourceUsage.write;

/// pack: queries into prefix rows and tail blocks; append: the round's keys into the caches; keep: kept paths moved.
pub const Which = enum { pack, append, keep };

pub const Reg = struct { heads: at_.Heads, which: Which, control: bool = false };

const rule = "bits only: NaN and both infinities copy as they are";

fn noSums(_: c.Shape) usize {
    return 0;
}

fn nameOf(comptime r: Reg) []const u8 {
    if (r.control) return "control: qwen27 attention io append one row late";
    const what = comptime switch (r.which) {
        .pack => "pack q27_attn_pack_a q27_attn_pack_b",
        .append => "append q27_attn_append",
        .keep => "keep q27_attn_gather q27_attn_scatter",
    };
    return std.fmt.comptimePrint("metal qwen27 attention {s} io {s}", .{ comptime at_.label(r.heads), what });
}

pub fn spec(r: Reg) c.Spec {
    return switch (r.heads) {
        inline else => |h| switch (r.which) {
            inline else => |w| switch (r.control) {
                inline else => |k| .{ .name = nameOf(.{ .heads = h, .which = w, .control = k }), .accum = .none, .depth = noSums, .out = .raw, .in_place = w == .keep, .nan = .{ .stated = rule }, .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 512 } }, .served = at_.served(h, .decode) },
            },
        },
    };
}

pub const Io = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    reg: Reg,
    op: attn.Attention,
    tensor: bool,
    st: ?at_.State = null,
    rd: at_.Round = .{},
    host: [3][]u16 = .{ &.{}, &.{}, &.{} },
    paths: [planning.max_streams][]u32 = @splat(&.{}),
    path_buf: [planning.max_rows]u32 = undefined,
    kept: u32 = 0,
    ins: [2]rig_.Ref = undefined,
    outs: [2 * planning.max_streams]rig_.Ref = undefined,
    caches: [planning.max_streams]attn.Cache = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, r: Reg) !Io {
        const tensor = device.tensorUnits();
        const edit: ?at_.Edit = if (r.control) .{ .from = "row = uint(meta[base + 4]) + local;", .to = "row = uint(meta[base + 4]) + local + 1;" } else null;
        const text = try at_.source(a, r.heads, tensor, false, edit);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var op = try attn.Attention.init(device, lib, tensor);
        op.geometry = at_.geometry(r.heads, tensor);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .reg = r, .op = op, .tensor = tensor };
    }

    pub fn deinit(t: *Io) void {
        t.release();
        t.op.deinit();
        t.rig.deinit();
    }

    fn release(t: *Io) void {
        if (t.st) |*s| s.deinit();
        t.st = null;
        for (&t.host) |*h| {
            t.a.free(h.*);
            h.* = &.{};
        }
    }

    pub fn kernel(t: *Io) g.Kernel {
        return .{ .spec = spec(t.reg), .ctx = t, .vt = &vt, .control = if (t.reg.control) .shapes else null };
    }

    const vt = g.hooks(Io);

    fn geo(t: *const Io) planning.Geometry {
        return at_.geometry(t.reg.heads, t.tensor);
    }

    fn bits(rng: std.Random, xs: []u16) void {
        for (xs) |*x| x.* = c.toBf16(rng.float(f32) * 4 - 2);
    }

    /// Inputs (pack: queries; append: keys and values; keep: the caches as they start), at gate 1 with NaN and Inf.
    pub fn prepare(t: *Io, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const gm = t.geo();
        const heads: usize = gm.heads;
        const hk: usize = gm.kv_heads;
        if (shape.n != heads or shape.rows == 0 or shape.rows > planning.max_rows) return false;
        t.release();
        t.rig.reset();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.rd = at_.Round.init(shape, false, true, rng);
        const r = &t.rd;
        const per = hk * r.cap * D;
        switch (t.reg.which) {
            .pack => t.host[0] = try t.a.alloc(u16, r.rows * heads * D),
            .append => for (t.host[0..2]) |*h| {
                h.* = try t.a.alloc(u16, r.rows * hk * D);
            },
            .keep => for (t.host[0..2]) |*h| {
                h.* = try t.a.alloc(u16, r.streams * per);
            },
        }
        for (t.host) |h| bits(rng, h);
        if (fill == .nonfinite) for (t.host[0..if (t.reg.which == .pack) 1 else 2], 0..) |h, i| {
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (put, 0..) |v, j| h[(i * 7 + j * 131) % h.len] = c.toBf16(v);
        };
        if (t.reg.which != .keep) for (t.host, 0..) |h, i| if (h.len > 0) {
            t.ins[i] = try t.rig.input(std.mem.sliceAsBytes(h));
        };
        // the caches are what append and keep write; pack's binding needs them only to exist
        for (0..r.streams) |s| {
            var kv: [2]rig_.Ref = undefined;
            for (&kv) |*x| x.* = if (t.reg.which == .pack) try t.rig.scratch(per * 2) else try t.rig.output(per * 2);
            t.outs[2 * s] = kv[0];
            t.outs[2 * s + 1] = kv[1];
            t.caches[s] = .{ .keys = at_.ref(kv[0]), .values = at_.ref(kv[1]), .capacity = @intCast(r.cap) };
        }
        t.st = try at_.State.init(t.a, t.rig.device, r, gm, t.caches[0..r.streams]);
        if (t.reg.which == .pack) {
            t.outs[0] = try t.rig.output(hk * t.st.?.plan.tiles * 16 * D * 2);
            t.outs[1] = try t.rig.output(hk * r.rows * 16 * D * 2);
        }
        if (t.reg.which == .keep) {
            t.pick(rng, &t.path_buf, &t.paths);
            const st = &t.st.?;
            t.kept = try st.binding.uploadKeep(st.plan, t.caches[0..r.streams], t.paths[0..r.streams]);
        }
        return true;
    }

    /// Each stream's kept rows: a random row's path from the root, as a committed draft keeps.
    fn pick(t: *const Io, rng: std.Random, buf: *[planning.max_rows]u32, paths: *[planning.max_streams][]u32) void {
        const r = &t.rd;
        var used: usize = 0;
        for (0..r.streams) |s| {
            var x: i32 = @intCast(rng.uintLessThan(usize, r.counts[s]));
            var n: usize = 0;
            var rev: [planning.max_rows]u32 = undefined;
            while (x >= 0) : (x = r.parents[r.firsts[s] + @as(usize, @intCast(x))]) {
                rev[n] = @intCast(x);
                n += 1;
            }
            for (0..n) |i| buf[used + i] = rev[n - 1 - i];
            paths[s] = buf[used..][0..n];
            used += n;
        }
    }

    /// The engine's keep of `kept` rows on the caches as they are now.
    fn keep(t: *Io, kept: u32) !void {
        const st = &t.st.?;
        const p = t.rig.begin();
        t.op.keep(p.e, &st.scratch, &st.binding, kept, t.caches[0..t.rd.streams]) catch |err| {
            p.e.end();
            p.pool.pop();
            return err;
        };
        try t.rig.finish(p);
    }

    pub fn run(t: *Io) !void {
        const st = &t.st.?;
        const r = &t.rd;
        const gm = t.geo();
        const per = gm.kv_heads * r.cap * D;
        // keep moves rows within the caches it reads, so each run starts from the caches as prepared
        if (t.reg.which == .keep) {
            for (0..r.streams) |s| for (0..2) |kv| {
                @memcpy(t.rig.bytes(t.outs[2 * s + kv], per * 2), std.mem.sliceAsBytes(t.host[kv][s * per ..][0..per]));
            };
            return t.keep(t.kept);
        }
        const p = t.rig.begin();
        const e = p.e;
        const args = attn.PackArgs{ .rows = @intCast(r.rows), .packed_rows = st.plan.tiles * 16, .heads = gm.heads, .kv_heads = gm.kv_heads, .group = gm.group() };
        switch (t.reg.which) {
            .pack => {
                e.setPipeline(t.op.io[0]);
                for ([_]mtl.Buffer{ t.ins[0].buf, st.scratch.get(.q_rows).buffer, t.outs[0].buf }, [_]usize{ t.ins[0].off, 0, t.outs[0].off }, 0..) |b, off, i| e.setBuffer(b, off, i);
                e.setValue(args, 3);
                e.dispatchThreads(mtl.Size.of(256, args.packed_rows, gm.kv_heads), mtl.Size.of(32, 4, 1));
                e.setPipeline(t.op.io[1]);
                e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
                e.setBuffer(t.outs[1].buf, t.outs[1].off, 1);
                e.setValue(args, 2);
                e.dispatchThreads(mtl.Size.of(256, 16, gm.kv_heads * r.rows), mtl.Size.of(32, 4, 1));
            },
            .append => {
                for (st.binding.buffers) |b| e.useResource(b, rw);
                for (t.outs[0 .. 2 * r.streams]) |o| e.useResource(o.buf, rw);
                e.setPipeline(t.op.io[2]);
                for ([_]mtl.Buffer{ t.ins[0].buf, t.ins[1].buf, st.binding.get(.caches).buffer, st.binding.get(.meta).buffer, st.scratch.get(.nodes).buffer }, [_]usize{ t.ins[0].off, t.ins[1].off, 0, 0, 0 }, 0..) |b, off, i| e.setBuffer(b, off, i);
                e.setValue(args, 5);
                e.dispatchThreads(mtl.Size.of(256, gm.kv_heads, r.rows), mtl.Size.of(32, 4, 1));
            },
            .keep => unreachable,
        }
        try t.rig.finish(p);
    }

    pub fn count(_: *Io) usize {
        return 0;
    }

    pub fn reference(_: *Io, _: []f64, _: []f64) !void {}

    /// An output's bf16 words (the rig's payloads start 64 bytes in, so they are aligned).
    fn words(t: *Io, r: rig_.Ref, n: usize) []const u16 {
        return @alignCast(std.mem.bytesAsSlice(u16, t.rig.bytes(r, n * 2)));
    }

    fn say(t: *Io, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.bufPrint(&t.msg, fmt, args);
    }

    /// Every element against its copy: packed rows (zero past a block's heads), appended rows, kept rows by depth.
    pub fn check(t: *Io, _: g.Gate) !?[]const u8 {
        const gm = t.geo();
        const heads: usize = gm.heads;
        const hk: usize = gm.kv_heads;
        const group: usize = gm.group();
        const r = &t.rd;
        const per = hk * r.cap * D;
        switch (t.reg.which) {
            .pack => {
                const plan = t.st.?.plan;
                const packed_rows = plan.tiles * 16;
                const qa = t.words(t.outs[0], hk * packed_rows * D);
                const qb = t.words(t.outs[1], hk * r.rows * 16 * D);
                for (0..hk) |h| for (0..packed_rows) |y| {
                    const m: usize = @intCast(plan.q_rows[y]);
                    const from = t.host[0][((m / group) * heads + h * group + m % group) * D ..][0..D];
                    if (!std.mem.eql(u16, qa[(h * packed_rows + y) * D ..][0..D], from)) return try t.say("pack_a key head {d} row {d} differs from query row {d}", .{ h, y, m });
                };
                for (0..hk) |h| for (0..r.rows) |node| for (0..16) |y| {
                    const got = qb[((h * r.rows + node) * 16 + y) * D ..][0..D];
                    const ok = if (y < group) std.mem.eql(u16, got, t.host[0][(node * heads + h * group + y) * D ..][0..D]) else std.mem.allEqual(u16, got, 0);
                    if (!ok) return try t.say("pack_b key head {d} row {d} block row {d} is not its copy", .{ h, node, y });
                };
            },
            .append => for (0..r.streams) |s| for (0..2) |kv| {
                const got = t.words(t.outs[2 * s + kv], per);
                for (0..hk) |h| for (0..r.cap) |row| {
                    const cell = got[(h * r.cap + row) * D ..][0..D];
                    const local = row -% r.starts[s];
                    const ok = if (local < r.counts[s]) std.mem.eql(u16, cell, t.host[kv][((r.firsts[s] + local) * hk + h) * D ..][0..D]) else std.mem.allEqual(u16, cell, 0xffff);
                    if (!ok) return try t.say("stream {d} {s} head {d} cache row {d}: {s}", .{ s, if (kv == 0) "keys" else "values", h, row, if (local < r.counts[s]) "not the round's row" else "written outside the round" });
                };
            },
            .keep => {
                if (try t.moved(.{ t.host[0], t.host[1] }, t.paths[0..r.streams], "")) |why| return why;
                // a second keep, other paths, on the caches the first left (no restore): keep reads them as they are
                var now: [2][]u16 = undefined;
                for (&now, 0..) |*x, kv| {
                    x.* = try t.a.alloc(u16, r.streams * per);
                    for (0..r.streams) |s| @memcpy(x.*[s * per ..][0..per], t.words(t.outs[2 * s + kv], per));
                }
                defer for (now) |x| t.a.free(x);
                var prng = std.Random.DefaultPrng.init(t.kept);
                var buf: [planning.max_rows]u32 = undefined;
                var paths: [planning.max_streams][]u32 = undefined;
                t.pick(prng.random(), &buf, &paths);
                const st = &t.st.?;
                try t.keep(try st.binding.uploadKeep(st.plan, t.caches[0..r.streams], paths[0..r.streams]));
                const why = try t.moved(.{ now[0], now[1] }, paths[0..r.streams], "after a first keep, ");
                t.kept = try st.binding.uploadKeep(st.plan, t.caches[0..r.streams], t.paths[0..r.streams]);
                return why;
            },
        }
        return null;
    }

    /// The caches against `was` with each stream's kept path moved to its window's start, the rest as they were.
    fn moved(t: *Io, was_: [2][]const u16, paths: []const []u32, comptime when: []const u8) !?[]const u8 {
        const r = &t.rd;
        const hk: usize = t.geo().kv_heads;
        const per = hk * r.cap * D;
        for (0..r.streams) |s| for (0..2) |kv| {
            const got = t.words(t.outs[2 * s + kv], per);
            const was = was_[kv][s * per ..][0..per];
            const path = paths[s];
            for (0..hk) |h| for (0..r.cap) |row| {
                const depth = row -% r.starts[s];
                const from = if (depth < path.len) r.starts[s] + path[depth] else row;
                if (!std.mem.eql(u16, got[(h * r.cap + row) * D ..][0..D], was[(h * r.cap + from) * D ..][0..D])) return try t.say(when ++ "stream {d} {s} head {d} cache row {d} is not row {d} as it was", .{ s, if (kv == 0) "keys" else "values", h, row, from });
            };
        };
        return null;
    }
};
