//! DeepSeek V4.1's Engram under the contract on Metal: four stages against float64, controls, whole-step invariance.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_engram_ref.zig");

const c = contract.spec;
const g = contract.gates;
const eg = tf.dsv41_engram;
const base = "dsv41 engram ";

/// What a registration varies: the stage and a planted fault (TF_CTL in the source).
pub const Variant = struct { stage: ref.Stage, ctl: u8 = 0 };

fn s3(rows: usize, n: usize, k: usize) c.Shape {
    return .{ .rows = rows, .n = n, .k = k };
}

const rowsK = blk: {
    var out: [17]usize = undefined;
    for (0..16) |i| out[i] = i + 1;
    out[16] = 33;
    break :blk out;
};

fn served(comptime stage: ref.Stage) []const c.Shape {
    const list = comptime blk: {
        var out: [rowsK.len + 8]c.Shape = undefined;
        var n: usize = 0;
        switch (stage) {
            .embed => {
                out[n] = s3(8, 6144, 6144);
                n += 1;
                for (rowsK) |r| {
                    out[n] = s3(r, 6144, 6144);
                    n += 1;
                }
                out[n] = s3(3, 512, 512);
                n += 1;
            },
            .aq => {
                out[n] = s3(8, 6144, 6144);
                n += 1;
                for (rowsK) |r| {
                    out[n] = s3(r, 6144, 6144);
                    n += 1;
                }
                out[n] = s3(3, 512, 512);
                n += 1;
            },
            .wkv => {
                out[n] = s3(8, 192, 6144);
                n += 1;
                for (rowsK) |r| {
                    out[n] = s3(r, 64, 1536);
                    n += 1;
                }
                for ([_]c.Shape{ s3(1, 512, 6144), s3(5, 96, 2560), s3(16, 128, 256), s3(2, 32, 6144), s3(4, 5120, 6144) }) |x| {
                    out[n] = x;
                    n += 1;
                }
            },
            .gate => {
                out[n] = s3(8, 4, 5120);
                n += 1;
                for (rowsK) |r| {
                    out[n] = s3(r, 4, 5120);
                    n += 1;
                }
                out[n] = s3(2, 4, 32);
                n += 1;
                out[n] = s3(5, 4, 1056);
                n += 1;
            },
            .add => {
                out[n] = s3(8, 20480, 5120);
                n += 1;
                for (rowsK) |r| {
                    out[n] = s3(r, 20480, 5120);
                    n += 1;
                }
                out[n] = s3(2, 128, 32);
                n += 1;
            },
        }
        var fin: [n]c.Shape = undefined;
        @memcpy(&fin, out[0..n]);
        break :blk fin;
    };
    return &list;
}

fn depthOne(_: c.Shape) usize {
    return 1;
}

fn depthGate(s: c.Shape) usize {
    return s.k / 32 + 8;
}

fn depthAdd(_: c.Shape) usize {
    return 2;
}

pub fn spec(v: Variant, name: []const u8) c.Spec {
    return switch (v.stage) {
        .embed => .{ .name = name, .accum = .none, .depth = depthOne, .out = .bf16, .served = served(.embed) },
        .aq => .{ .name = name, .accum = .none, .depth = depthOne, .out = .bf16, .served = served(.aq) },
        .wkv => .{ .name = name, .accum = .none, .depth = depthOne, .out = .bf16, .served = served(.wkv) },
        .gate => .{
            .name = name,
            .accum = .f32,
            .depth = depthGate,
            .out = .f32,
            .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 32, .step = 32, .min = 32 } },
            .served = served(.gate),
        },
        .add => .{ .name = name, .accum = .none, .depth = depthAdd, .out = .bf16, .served = served(.add) },
    };
}

pub const Proj = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    pipes: eg.Pipelines,
    d: ?ref.Data = null,
    ins: [5]rig_.Ref = undefined,
    scr: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Proj {
        var p: Proj = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name, .pipes = undefined };
        if (v.ctl == 0) {
            p.pipes = try eg.Pipelines.load(device);
        } else {
            const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ v.ctl, sources.deepseek_v41_engram });
            defer a.free(text);
            p.pipes = try eg.Pipelines.loadSource(device, text);
        }
        return p;
    }

    pub fn deinit(p: *Proj) void {
        p.drop();
        p.pipes.deinit();
        p.rig.deinit();
    }

    fn drop(p: *Proj) void {
        if (p.d) |*d| d.deinit();
        p.d = null;
        p.rig.reset();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        const control: ?g.Gate = switch (p.v.ctl) {
            1, 7 => .shapes,
            5 => .sums,
            4 => .nonfinite,
            6 => .repeat,
            else => null,
        };
        return .{ .spec = spec(p.v, p.name), .ctx = p, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Proj);

    fn call(p: *const Proj) eg.Call {
        const s = p.d.?.shape;
        const rows: u32 = @intCast(s.rows);
        return switch (p.v.stage) {
            .embed, .aq => .{ .rows = rows, .cols = @intCast(s.k / ref.head), .n = 32, .dim = 32 },
            .wkv => .{ .rows = rows, .cols = @intCast(s.k / ref.head), .n = @intCast(s.n), .dim = 32 },
                        .gate, .add => .{ .rows = rows, .cols = 1, .n = 32, .dim = @intCast(s.k) },
        };
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        const wide = p.v.stage != .gate and p.v.stage != .add;
        if (shape.k % 32 != 0 or (wide and shape.k % ref.head != 0) or shape.rows == 0 or shape.n == 0 or (p.v.stage == .wkv and shape.n % 32 != 0)) return false;
        var d = try ref.Data.init(p.a, p.v.stage, shape);
        d.fill(fill, seed);
        p.d = d;
        try p.upload();
        return true;
    }

    fn upload(p: *Proj) !void {
        const d = p.d.?;
        const bytes = d.outputs() * c.size(spec(p.v, p.name).out);
        const b = std.mem.sliceAsBytes;
        switch (p.v.stage) {
            .embed => {
                p.ins[0] = try p.rig.input(d.codes);
                p.ins[1] = try p.rig.input(d.scales);
            },
            .aq => p.ins[0] = try p.rig.input(b(d.emb)),
            .wkv => {
                p.ins[0] = try p.rig.input(b(d.emb));
                p.ins[1] = try p.rig.input(d.f.?.codes);
                p.ins[2] = try p.rig.input(d.f.?.scales);
                const c0 = p.call();
                p.scr = try p.rig.scratch(2 * @as(usize, c0.rows) * d.shape.k + eg.fp8Scratch(c0));
            },
            .gate => {
                p.ins[0] = try p.rig.input(b(d.x));
                p.ins[1] = try p.rig.input(b(d.kv));
                p.ins[2] = try p.rig.input(b(d.q));
                p.ins[3] = try p.rig.input(b(d.kw));
                p.ins[4] = try p.rig.input(d.mask);
            },
            .add => {
                p.ins[0] = try p.rig.input(b(d.x));
                p.ins[1] = try p.rig.input(b(d.kv));
                p.ins[2] = try p.rig.input(b(d.g));
            },
        }
        p.out = try p.rig.output(bytes);
    }

    fn at(r: rig_.Ref) eg.Ref {
        return .{ .buf = r.buf, .off = r.off };
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const i = p.ins;
        switch (p.v.stage) {
            .embed => try eg.encodeEmbed(&p.pipes, e, at(i[0]), at(i[1]), at(p.out), p.call()),
            .aq => try eg.encodeAq(&p.pipes, e, at(i[0]), at(p.out), p.call()),
            .wkv => {
                const cl = p.call();
                const roots = eg.Ref{ .buf = p.scr.buf, .off = p.scr.off + 2 * cl.rows * cl.cols * ref.head };
                try eg.encodeAq(&p.pipes, e, at(i[0]), at(p.scr), cl);
                try eg.encodeWkv(&p.pipes, e, at(p.scr), at(i[1]), at(i[2]), roots, at(p.out), cl);
            },
            .gate => try eg.encodeGate(&p.pipes, e, at(i[0]), at(i[1]), at(i[2]), at(i[3]), at(i[4]), at(p.out), p.call()),
            .add => try eg.encodeAdd(&p.pipes, e, at(i[0]), at(i[1]), at(i[2]), at(p.out), p.call()),
        }
        try p.rig.finish(pass);
    }

    pub fn count(p: *Proj) usize {
        return p.d.?.outputs();
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        return p.d.?.reference(want, mag);
    }

    pub fn bounds(p: *Proj, want: []const f64, mag: []const f64, allow: []f64) !void {
        return p.d.?.bounds(want, mag, allow);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.d.?.simulate(acc, order, out);
    }
};

/// Every registered variant, and the whole-step probes that run after the gates.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: std.ArrayList(*Proj) = .empty,

    const list = [_]struct { v: Variant, name: []const u8 }{
        .{ .v = .{ .stage = .embed }, .name = base ++ "embed" },
        .{ .v = .{ .stage = .aq }, .name = base ++ "aq" },
        .{ .v = .{ .stage = .wkv }, .name = base ++ "wkv (act_quant, fp8_rows)" },
        .{ .v = .{ .stage = .gate }, .name = base ++ "gate" },
        .{ .v = .{ .stage = .add }, .name = base ++ "add" },
        .{ .v = .{ .stage = .embed, .ctl = 1 }, .name = base ++ "control: embed reads one scale for 64 codes" },
        .{ .v = .{ .stage = .aq, .ctl = 7 }, .name = base ++ "control: act_quant truncates its codes" },
        .{ .v = .{ .stage = .gate, .ctl = 4 }, .name = base ++ "control: gate clamp drops a NaN" },
        .{ .v = .{ .stage = .gate, .ctl = 5 }, .name = base ++ "control: gate sums rounded to fp16" },
        .{ .v = .{ .stage = .add, .ctl = 6 }, .name = base ++ "control: add writes past its output" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a, .device = device, .queue = queue };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            p.* = try Proj.init(a, device, queue, entry.v, entry.name);
            try s.probes.append(a, p);
            try kernels.append(a, p.kernel());
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }

    /// The whole step on 40 rows (some masked): any chunk or a launch a row gives the same bytes, also add in place.
    pub fn invariance(s: *Suite, seed: u64) !usize {
        if (s.probes.items.len == 0 or s.probes.items[0].v.ctl != 0) return 0;
        const pipes = &s.probes.items[0].pipes;
        const cols = 24;
        const dim = 640;
        const k = cols * ref.head;
        const n = (ref.hc + 1) * dim;
        const total = 40;
        const xrow = ref.hc * dim;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const codes = try s.a.alloc(u8, total * k);
        defer s.a.free(codes);
        for (codes) |*q| {
            const b = rng.int(u8);
            q.* = if (b & 0x7f == 0x7f) b & 0xfe else b;
        }
        const scales = try s.a.alloc(u8, total * k / 32);
        defer s.a.free(scales);
        for (scales) |*q| q.* = 119 + rng.uintLessThan(u8, 7);
        const w8 = try s.a.alloc(u8, n * k);
        defer s.a.free(w8);
        for (w8) |*q| {
            const b = rng.int(u8);
            q.* = if (b & 0x7f == 0x7f) b & 0xfe else b;
        }
        const ws = try s.a.alloc(u8, n / 32 * (k / 32));
        defer s.a.free(ws);
        for (ws) |*q| q.* = 108 + rng.uintLessThan(u8, 7);
        const x = try s.a.alloc(u16, total * xrow);
        defer s.a.free(x);
        for (x) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        const qk = try s.a.alloc(u16, 2 * xrow);
        defer s.a.free(qk);
        for (qk) |*w| w.* = c.toBf16(rng.float(f32) * 2 - 1);
        var mask: [total]u8 = undefined;
        for (&mask, 0..) |*m, r| m.* = @intFromBool(r % 5 != 3);
        const ones: [total]u8 = @splat(1);

        var rig: rig_.Rig = .{ .a = s.a, .device = s.device, .queue = s.queue };
        defer rig.deinit();
        const b = std.mem.sliceAsBytes;
        const rc = try rig.input(codes);
        const rs = try rig.input(scales);
        const rw = try rig.input(w8);
        const rws = try rig.input(ws);
        const rx = try rig.input(b(x));
        const rq = try rig.input(b(qk[0..xrow]));
        const rk = try rig.input(b(qk[xrow..]));
        const rm = try rig.input(&mask);
        const r1 = try rig.input(&ones);
        const rxio = try rig.input(b(x));
        const ry = try rig.output(2 * total * xrow);
        const call: eg.Call = .{ .rows = total, .cols = cols, .n = n, .dim = dim };
        const rscratch = try rig.scratch(eg.scratchBytes(call));
        const whole = try s.a.alloc(u8, 2 * total * xrow);
        defer s.a.free(whole);
        const Run = struct {
            fn go(pp: *const eg.Pipelines, rg: *rig_.Rig, from: usize, rows: u32, m: rig_.Ref, y: rig_.Ref, xin: rig_.Ref, ins: [7]rig_.Ref, sc: rig_.Ref) !void {
                const pass = rg.begin();
                const cl: eg.Call = .{ .rows = rows, .cols = cols, .n = n, .dim = dim };
                const o = struct {
                    fn f(r: rig_.Ref, extra: usize) eg.Ref {
                        return .{ .buf = r.buf, .off = r.off + extra };
                    }
                }.f;
                try eg.encode(pp, pass.e, o(ins[0], from * k), o(ins[1], from * k / 32), o(ins[2], 0), o(ins[3], 0), o(xin, 2 * from * xrow), o(ins[4], 0), o(ins[5], 0), o(m, from), o(sc, 0), o(y, 2 * from * xrow), cl);
                try rg.finish(pass);
            }
        };
        const ins = [7]rig_.Ref{ rc, rs, rw, rws, rq, rk, rx };
        var bad: usize = 0;
        var launches: usize = 0;
        @memset(rig.bytes(ry, whole.len), 0xff);
        try Run.go(pipes, &rig, 0, total, rm, ry, rx, ins, rscratch);
        @memcpy(whole, rig.bytes(ry, whole.len));
        for ([_]u32{ 1, 2, 3, 4, 8, 15, 16, 17, 33 }) |r| {
            @memset(rig.bytes(ry, whole.len), 0xff);
            try Run.go(pipes, &rig, 0, r, rm, ry, rx, ins, rscratch);
            launches += 1;
            if (!std.mem.eql(u8, whole[0 .. 2 * r * xrow], rig.bytes(ry, 2 * r * xrow))) {
                bad += 1;
                std.debug.print("FAIL row invariance {s}: {d} rows differ from the {d}-row launch\n", .{ base, r, total });
            }
        }
        for (0..total) |r| {
            @memset(rig.bytes(ry, whole.len), 0xff);
            try Run.go(pipes, &rig, r, 1, rm, ry, rx, ins, rscratch);
            launches += 1;
            const got = rig.bytes(ry, 2 * (r + 1) * xrow)[2 * r * xrow ..];
            if (!std.mem.eql(u8, whole[2 * r * xrow ..][0 .. 2 * xrow], got)) {
                bad += 1;
                std.debug.print("FAIL row invariance {s}: row {d} alone differs from the {d}-row launch\n", .{ base, r, total });
            }
        }
        // In place: the add reads each element and writes it from one thread.
        try Run.go(pipes, &rig, 0, total, rm, rxio, rxio, ins, rscratch);
        launches += 1;
        if (!std.mem.eql(u8, whole, rig.bytes(rxio, whole.len))) {
            bad += 1;
            std.debug.print("FAIL in place {s}: y == x differs from the out-of-place bytes\n", .{base});
        }
        // Image rows: a masked row's streams come out as they went in, and no row depends on another's mask.
        var through: usize = 0;
        for (0..total) |r| if (mask[r] == 0) {
            through += 1;
            if (!std.mem.eql(u8, b(x)[2 * r * xrow ..][0 .. 2 * xrow], whole[2 * r * xrow ..][0 .. 2 * xrow])) {
                bad += 1;
                std.debug.print("FAIL image row {s}: masked row {d} changed\n", .{ base, r });
            }
        };
        @memset(rig.bytes(ry, whole.len), 0xff);
        try Run.go(pipes, &rig, 0, total, r1, ry, rx, ins, rscratch);
        launches += 1;
        for (0..total) |r| {
            const same = std.mem.eql(u8, whole[2 * r * xrow ..][0 .. 2 * xrow], rig.bytes(ry, whole.len)[2 * r * xrow ..][0 .. 2 * xrow]);
            if (mask[r] != 0 and !same) {
                bad += 1;
                std.debug.print("FAIL image row {s}: unmasked row {d} depends on the mask of the others\n", .{ base, r });
            }
        }
        const word = if (bad == 0) "PASS" else "FAIL";
        std.debug.print("{s} row invariance {s}: {d} launches ({d}-row, chunks 1..33, one a row, in place, {d} masked rows pass through), {d} differ\n", .{ word, base, launches, total, through, bad });
        return bad;
    }
};
