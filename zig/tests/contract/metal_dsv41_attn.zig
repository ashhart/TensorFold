//! DeepSeek V4.1 CSA2 attention under the contract on Metal: float64 reference, controls, row invariance, timing.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_attn_ref.zig");

const c = contract.spec;
const g = contract.gates;
const attn = tf.dsv41_attn;
const D = attn.head_dim;
const base = "dsv41_attn ";

/// What a registration varies: the output, the launch shape, a planted fault.
pub const Variant = struct { out: attn.Out = .f32, layout: ref.Layout = .decode, ctl: u8 = 0 };

fn sh(rows: usize, heads: usize, k: usize) c.Shape {
    return .{ .rows = rows, .n = heads * D, .k = k };
}

const decode_served = [_]c.Shape{ sh(8, 8, 16), sh(4, 8, 512), sh(1, 8, 512), sh(2, 8, 0), sh(4, 64, 512), sh(16, 8, 37), sh(32, 8, 512), sh(3, 2, 5) };
const prompt_served = [_]c.Shape{ sh(8, 4, 16), sh(4, 8, 512), sh(16, 4, 64), sh(17, 2, 0), sh(3, 8, 512), sh(32, 4, 128), sh(2, 2, 512), sh(8192, 2, 32), sh(1, 2, 5) };

pub fn spec(v: Variant, name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = ref.depth,
        .out = if (v.out == .f32) .f32 else .bf16,
        .tiles = .{ .{ .size = 16 }, .{ .size = 2 * D, .step = 2 * D, .min = 2 * D }, .{ .size = 4, .step = 1, .min = 0 } },
        .served = if (v.layout == .decode) &decode_served else &prompt_served,
    };
}

pub const Proj = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    pipes: attn.Pipelines,
    f: ?ref.Attn = null,
    b: attn.Bind = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Proj {
        var p: Proj = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name, .pipes = undefined };
        if (v.ctl == 0) {
            p.pipes = try attn.Pipelines.load(device);
        } else {
            const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ v.ctl, sources.deepseek_v41_csa2_attn });
            defer a.free(text);
            p.pipes = try attn.Pipelines.loadSource(device, text);
        }
        return p;
    }

    pub fn deinit(p: *Proj) void {
        p.drop();
        p.pipes.deinit();
        p.rig.deinit();
    }

    fn drop(p: *Proj) void {
        if (p.f) |*f| f.deinit();
        p.f = null;
        p.rig.reset();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        const control: ?g.Gate = switch (p.v.ctl) {
            1 => .nonfinite,
            2 => .shapes,
            3 => .sums,
            4 => .repeat,
            else => null,
        };
        return .{ .spec = spec(p.v, p.name), .ctx = p, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Proj);

    fn call(p: *const Proj) attn.Call {
        const f = p.f.?;
        return .{ .rows = @intCast(f.rows), .heads = @intCast(f.heads), .out = p.v.out };
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        attn.check(.{ .rows = @intCast(shape.rows), .heads = @intCast(shape.n / D), .out = p.v.out }) catch return false;
        var f = ref.Attn.init(p.a, shape, .{ .layout = p.v.layout }) catch return false;
        errdefer f.deinit();
        try f.fill(fill, seed);
        p.f = f;
        try p.upload(&f);
        return true;
    }

    pub fn upload(p: *Proj, f: *const ref.Attn) !void {
        const r = &p.rig;
        const es = c.size(spec(p.v, p.name).out);
        p.b = .{
            .q = at(try r.input(std.mem.sliceAsBytes(f.q))),
            .w = at(try r.input(std.mem.sliceAsBytes(f.win))),
            .c = at(try r.input(std.mem.sliceAsBytes(f.comp))),
            .i = at(try r.input(std.mem.sliceAsBytes(f.idx))),
            .m = at(try r.input(std.mem.sliceAsBytes(f.meta))),
            .sk = at(try r.input(std.mem.sliceAsBytes(f.sink))),
            .cs = at(try r.input(std.mem.sliceAsBytes(f.cs))),
            .o = at(try r.output(f.outputs() * es)),
        };
    }

    fn at(r: rig_.Ref) attn.Ref {
        return .{ .buf = r.buf, .off = r.off };
    }

    /// Rows [row, row + n) of the prepared call as their own launch: the row-indexed buffers move, the rest stay.
    pub fn encode(p: *Proj, e: mtl.ComputeEncoder, row: usize, n: usize, layout: ref.Layout) !void {
        const f = p.f.?;
        var b = p.b;
        const es = c.size(spec(p.v, p.name).out);
        b.q.off += row * f.heads * D * 2;
        b.i.off += row * attn.max_selected * 4;
        b.m.off += row * @sizeOf(attn.Meta);
        b.o.off += row * f.heads * D * es;
        var cl = p.call();
        cl.rows = @intCast(n);
        if (layout == .decode) return attn.encodeDecode(&p.pipes, e, b, cl);
        return attn.encodePrompt(&p.pipes, e, b, cl);
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        try p.encode(pass.e, 0, p.f.?.rows, p.v.layout);
        try p.rig.finish(pass);
    }

    pub fn count(p: *Proj) usize {
        return p.f.?.outputs();
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        return p.f.?.reference(want, mag);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.f.?.simulate(acc, order, out);
    }
};

/// Every registered variant, and the probes that run after the gates: invariance and the timing of real shapes.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: std.ArrayList(*Proj) = .empty,
    only: ?[]const u8,

    const list = [_]struct { v: Variant, name: []const u8 }{
        .{ .v = .{}, .name = base ++ "decode f32" },
        .{ .v = .{ .out = .bf16 }, .name = base ++ "decode bf16" },
        .{ .v = .{ .layout = .prompt }, .name = base ++ "prompt f32" },
        .{ .v = .{ .layout = .prompt, .out = .bf16 }, .name = base ++ "prompt bf16" },
        .{ .v = .{ .ctl = 1 }, .name = base ++ "control: NaN scores dropped" },
        .{ .v = .{ .ctl = 2 }, .name = base ++ "control: the last selected row dropped" },
        .{ .v = .{ .ctl = 3 }, .name = base ++ "control: scores summed in fp16" },
        .{ .v = .{ .ctl = 4 }, .name = base ++ "control: writes past its output" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a, .device = device, .queue = queue, .only = only };
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

    /// Chunks of `size` rows after a first chunk of `lead` rows, each its own launch, into the output buffer.
    fn chunked(p: *Proj, n: usize, lead: usize, size: usize, layout: ref.Layout) !void {
        const pass = p.rig.begin();
        var row: usize = 0;
        var step = if (lead > 0) lead else size;
        while (row < n) : (step = size) {
            const m = @min(step, n - row);
            try p.encode(pass.e, row, m, layout);
            row += m;
        }
        try p.rig.finish(pass);
    }

    fn same(p: *Proj, whole: []const u8, what: []const u8, bad: *usize, launches: *usize, verbose: bool) void {
        launches.* += 1;
        if (!std.mem.eql(u8, whole, p.rig.bytes(rr(p.b.o), whole.len))) {
            bad.* += 1;
            if (verbose) std.debug.print("FAIL row invariance {s}: {s} differs from the whole launch\n", .{ p.name, what });
        }
    }

    /// Row bits: whole launch, chunk splits at several offsets and decode on both windows agree; the control must not.
    pub fn invariance(s: *Suite, io: std.Io, seed: u64, t: *g.Tally) !void {
        _ = io;
        if (s.probes.items.len == 0) return;
        var bad: usize = 0;
        var launches: usize = 0;
        const n: usize = 8192;
        for ([_]attn.Out{ .f32, .bf16 }) |out| for ([_]u64{ 1, 2 }) |sd| {
            var p = try Proj.init(s.a, s.device, s.queue, .{ .layout = .prompt, .out = out }, base ++ "invariance");
            defer p.deinit();
            if (!try p.prepare(sh(n, 4, 512), .random, seed + sd)) return error.Refused;
            const es = c.size(spec(p.v, p.name).out);
            const whole = try s.a.dupe(u8, blk: {
                try p.run();
                break :blk p.rig.bytes(rr(p.b.o), n * 4 * D * es);
            });
            defer s.a.free(whole);
            const full = whole.len;
            for ([_][2]usize{ .{ 0, 1 }, .{ 0, 3 }, .{ 0, 16 }, .{ 0, 2048 }, .{ 5, 16 }, .{ 7, 3 }, .{ 13, 2048 } }) |lt| {
                @memset(p.rig.bytes(rr(p.b.o), full), 0xff);
                try chunked(&p, n, lt[0], lt[1], .prompt);
                same(&p, whole, "chunks", &bad, &launches, true);
            }
            for ([_]usize{ 1, 32 }) |size| {
                @memset(p.rig.bytes(rr(p.b.o), full), 0xff);
                try chunked(&p, n, 0, size, .decode);
                same(&p, whole, "decode rows on the linear window", &bad, &launches, true);
            }
            if (sd == 2) bad += try ringsEqual(&p, whole, &launches);
        };
        const word = if (bad == 0) "PASS" else "FAIL";
        std.debug.print("{s} row invariance {s}: prompt rows 8192 x 4 heads x 512 selected, f32 and bf16, chunk splits 1/3/16/2048 at offsets 0/5/7/13, decode on linear and ring windows ({d} comparisons) byte-equal to the whole launch, {d} differ\n", .{ word, base, launches, bad });
        t.failed += bad;
        t.controls += 1;
        var p = try Proj.init(s.a, s.device, s.queue, .{ .layout = .prompt, .ctl = 5 }, base ++ "control: scores depend on the slot");
        defer p.deinit();
        if (!try p.prepare(sh(64, 4, 64), .random, seed)) return error.Refused;
        try p.run();
        const whole = try s.a.dupe(u8, p.rig.bytes(rr(p.b.o), 64 * 4 * D * 4));
        defer s.a.free(whole);
        var cbad: usize = 0;
        var cl: usize = 0;
        try chunked(&p, 64, 0, 16, .prompt);
        same(&p, whole, "chunks", &cbad, &cl, false);
        try chunked(&p, 64, 1, 16, .prompt);
        same(&p, whole, "chunks", &cbad, &cl, false);
        if (cbad > 0) {
            t.caught += 1;
            std.debug.print("CONTROL {s}: red at row invariance, as planted\n", .{p.name});
        } else {
            t.missed += 1;
            std.debug.print("CONTROL MISSED {s}: row invariance did not go red\n", .{p.name});
        }
    }

    /// The first 32 rows' windows copied into per-row rings: the decode kernel must give the linear window's bytes.
    fn ringsEqual(p: *Proj, whole: []const u8, launches: *usize) !usize {
        const f = &p.f.?;
        const rows: usize = 32;
        const ring = try p.a.alloc(u16, rows * attn.window * D);
        defer p.a.free(ring);
        const meta = try p.a.alloc(attn.Meta, rows);
        defer p.a.free(meta);
        for (meta, 0..) |*m, r| {
            m.* = f.meta[r];
            const lo = @max(0, m.pos - (attn.window - 1));
            var t = lo;
            while (t <= m.pos) : (t += 1) {
                const src: usize = @intCast(m.wofs + t - m.wbase);
                const dst = r * attn.window + @as(usize, @intCast(t & 127));
                @memcpy(ring[dst * D ..][0..D], f.win[src * D ..][0..D]);
            }
            m.wofs = @intCast(r * attn.window);
            m.wbase = -1;
        }
        const saved = p.b;
        defer p.b = saved;
        p.b.w = Proj.at(try p.rig.input(std.mem.sliceAsBytes(ring)));
        p.b.m = Proj.at(try p.rig.input(std.mem.sliceAsBytes(meta)));
        const es = c.size(spec(p.v, p.name).out);
        @memset(p.rig.bytes(rr(saved.o), whole.len), 0xff);
        const pass = p.rig.begin();
        try p.encode(pass.e, 0, rows, .decode);
        try p.rig.finish(pass);
        launches.* += 1;
        const n = rows * f.heads * D * es;
        if (std.mem.eql(u8, whole[0..n], p.rig.bytes(rr(saved.o), n))) return 0;
        std.debug.print("FAIL row invariance {s}: decode rows on rings differ from the prompt launch\n", .{p.name});
        return 1;
    }

    /// Best microseconds of real launches: 64 heads, 512 selected rows of a 64K-row cache; small calls as a pass mean.
    pub fn timing(s: *Suite, io: std.Io) !void {
        for ([_]usize{ 1, 4, 16, 32, 8192 }) |rows| {
            var p = try Proj.init(s.a, s.device, s.queue, .{ .out = .bf16, .layout = if (rows == 8192) .prompt else .decode }, "timing");
            defer p.deinit();
            var f = try ref.Attn.init(s.a, sh(rows, 64, 512), .{ .layout = p.v.layout, .comp_rows = 65536, .timing = true });
            errdefer f.deinit();
            try f.fill(.random, 7);
            p.f = f;
            try p.upload(&f);
            const reps: usize = if (rows == 8192) 1 else 32;
            var best: f64 = std.math.inf(f64);
            for (0..7) |_| {
                const t0 = std.Io.Clock.awake.now(io);
                const pass = p.rig.begin();
                for (0..reps) |_| try p.encode(pass.e, 0, rows, p.v.layout);
                try p.rig.finish(pass);
                const ns: f64 = @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds());
                best = @min(best, ns / @as(f64, @floatFromInt(reps)));
            }
            const us = best / 1000.0;
            std.debug.print("TIME {s}{s}: {d} rows, 64 heads, 512 selected, window 128: {d:.1} us a launch, {d:.2} us a row\n", .{ base, @tagName(p.v.layout), rows, us, us / @as(f64, @floatFromInt(rows)) });
        }
    }
};

fn rr(r: attn.Ref) rig_.Ref {
    return .{ .buf = r.buf, .off = r.off };
}
