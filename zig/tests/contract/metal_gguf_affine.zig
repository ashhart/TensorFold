//! Native FP32 metadata reaches the shared gates directly; source faults include dropping bias and narrowing metadata.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const r = @import("rig.zig");
const affine = @import("metal_affine.zig");
const row = tf.row_projection;
const lane = tf.lane_projection;
const Fixture = contract.affine_f32.Fixture;
pub const formats = [_][2]usize{ .{ 4, 32 }, .{ 5, 32 }, .{ 6, 16 }, .{ 8, 32 } };
pub const Kind = enum { row, strided, lane, embed };
pub const Fault = enum { none, bias, narrow };
fn label(buf: []u8, kind: Kind, format: [2]usize, sk: usize, fault: Fault) []const u8 {
    return std.fmt.bufPrint(buf, "metal GGUF affine {t} q{d} g{d} sk{d}{s}", .{ kind, format[0], format[1], sk, switch (fault) {
        .none => "",
        .bias => " missing-bias control",
        .narrow => " narrowed-metadata control",
    } }) catch unreachable;
}
fn depth16(s: c.Shape) usize {
    return 128 + 2 * (s.k / 16) + 32;
}
fn point(_: c.Shape) usize {
    return 0;
}
fn edit(a: std.mem.Allocator, source: []const u8, old: []const u8, new: []const u8) ![]u8 {
    if (std.mem.count(u8, source, old) == 0) return error.SourceAnchorChanged;
    return std.mem.replaceOwned(u8, a, source, old, new);
}
pub fn rowSource(a: std.mem.Allocator, fault: Fault) ![]u8 {
    var text = try a.dupe(u8, row.source(.f32));
    errdefer a.free(text);
    if (fault == .bias) {
        const next = try edit(a, text, "out[b] += scale * accum + sum[b] * bias;", "out[b] += scale * accum;");
        a.free(text);
        text = next;
    } else if (fault == .narrow) {
        for ([_][]const u8{ "float(sc[j * KG])", "float(bi[j * KG])" }) |old| {
            const changed = try std.fmt.allocPrint(a, "float(bfloat({s}))", .{old});
            defer a.free(changed);
            const next = try edit(a, text, old, changed);
            a.free(text);
            text = next;
        }
    }
    return text;
}
pub fn laneSource(a: std.mem.Allocator, layout: lane.Layout, fault: Fault) ![]u8 {
    var text = try lane.source(a, layout);
    errdefer a.free(text);
    if (fault == .bias) {
        const next = try edit(a, text, "fma(bb[f][j], r ? xs1 : xs0, C[i])", "C[i]");
        a.free(text);
        text = next;
    } else if (fault == .narrow) {
        for ([_][]const u8{ "SBt[(size_t(g) * TF_N + col) * 2]", "SBt[(size_t(g) * TF_N + col) * 2 + 1]" }) |old| {
            const changed = try std.fmt.allocPrint(a, "float(bfloat({s}))", .{old});
            defer a.free(changed);
            const next = try edit(a, text, old, changed);
            a.free(text);
            text = next;
        }
    }
    return text;
}
pub fn embedSource(a: std.mem.Allocator, fault: Fault) ![]u8 {
    const source = tf.qwen27.runtime_sources.glue;
    const first = std.mem.indexOf(u8, source, "kernel void q27_embed_f32meta(") orelse return error.SourceAnchorChanged;
    const last = first + (std.mem.indexOf(u8, source[first..], "\n}") orelse return error.SourceAnchorChanged) + 2;
    var body = try a.dupe(u8, source[first..last]);
    defer a.free(body);
    if (fault == .bias) {
        const next = try edit(a, body, "+ q27_read(biases, meta, a.metadata)", "+ 0.0f");
        a.free(body);
        body = next;
    } else if (fault == .narrow) {
        for ([_][]const u8{ "q27_read(scales, meta, a.metadata)", "q27_read(biases, meta, a.metadata)" }) |old| {
            const changed = try std.fmt.allocPrint(a, "float(bfloat({s}))", .{old});
            defer a.free(changed);
            const next = try edit(a, body, old, changed);
            a.free(body);
            body = next;
        }
    }
    return std.mem.concat(a, u8, &.{ source[0..first], body, source[last..] });
}
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: r.Rig,
    kind: Kind,
    format: [2]usize,
    sk: usize,
    fault: Fault,
    name: [144]u8 = undefined,
    pipe: ?mtl.Pipeline = null,
    projection: ?lane.Projection = null,
    f: ?Fixture = null,
    shape: c.Shape = undefined,
    fill: g.Fill = undefined,
    stride: usize = undefined,
    x: r.Ref = undefined,
    w: r.Ref = undefined,
    s: r.Ref = undefined,
    b: r.Ref = undefined,
    ids: r.Ref = undefined,
    out: r.Ref = undefined,
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, format: [2]usize, sk: usize, fault: Fault) !Probe {
        var p = Probe{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .kind = kind, .format = format, .sk = sk, .fault = fault };
        if (kind != .lane) {
            const text = if (kind == .embed) try embedSource(a, fault) else try rowSource(a, fault);
            defer a.free(text);
            var fun: [96]u8 = undefined;
            const name = if (kind == .embed) "q27_embed_f32meta" else try std.fmt.bufPrint(&fun, "tf_row_projection_{s}q{d}_f32", .{ if (kind == .strided) "strided_" else "", format[0] });
            p.pipe = try r.pipeline(device, text, name);
        }
        return p;
    }
    pub fn extra(p: *Probe) !void {
        if (p.kind == .row and p.fault == .none) try @import("metal_gguf_native.zig").run(p.a, p.rig.device, p.rig.queue, p.format);
    }
    pub fn deinit(p: *Probe) void {
        p.drop();
        if (p.pipe) |pipe| pipe.deinit();
        p.rig.deinit();
    }
    fn drop(p: *Probe) void {
        if (p.f) |f| f.deinit();
        p.f = null;
        if (p.projection) |v| v.deinit();
        p.projection = null;
        p.rig.reset();
    }
    pub fn kernel(p: *Probe) g.Kernel {
        var spec = affine.spec(.{ .kind = if (p.kind == .lane) .lane else .row, .bits = @intCast(p.format[0]), .group = p.format[1], .sk = p.sk }, label(&p.name, p.kind, p.format, p.sk, p.fault));
        spec.served = &.{ .{ .rows = 1, .n = 64, .k = 5120 }, .{ .rows = 5, .n = 5120, .k = 128 }, .{ .rows = 16, .n = 64, .k = 512 } };
        if (p.kind == .lane and p.format[1] == 16) spec.depth = depth16;
        if (p.kind == .embed) {
            spec.accum = .none;
            spec.depth = point;
            spec.tiles = .{ .{ .size = 16 }, .{ .size = 128, .step = 32, .min = 32 }, .{} };
            spec.served = &.{ .{ .rows = 1, .n = 5120, .k = 1 }, .{ .rows = 16, .n = 128, .k = 1 } };
        }
        return .{ .spec = spec, .ctx = p, .vt = &vt, .control = if (p.fault == .none) null else .shapes };
    }
    const vt = g.hooks(Probe);
    pub fn layout(p: *Probe, shape: c.Shape) lane.Layout {
        return .{ .n = shape.n, .k = shape.k, .format = .{ .bits = @intCast(p.format[0]), .group = p.format[1], .metadata = .f32 }, .sk = p.sk };
    }
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        if (p.kind == .lane) {
            if (shape.rows > 16) return false;
            p.layout(shape).validate() catch return false;
        } else if (p.kind == .embed) {
            if (shape.n % p.format[1] != 0 or shape.n * p.format[0] % 32 != 0) return false;
        } else {
            (row.Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = shape.n, .k = shape.k, .bits = @intCast(p.format[0]), .group = p.format[1], .metadata = .f32 }).validate() catch return false;
        }
        p.drop();
        p.shape = shape;
        p.fill = fill;
        p.stride = shape.n + @as(usize, if (p.kind == .strided) 17 else 0);
        const fs = if (p.kind == .embed) c.Shape{ .rows = 1, .n = 17, .k = shape.n } else shape;
        p.f = try Fixture.init(p.a, fs, @intCast(p.format[0]), p.format[1]);
        const f = p.f.?;
        try f.fill(fill, seed);
        const words = try p.a.alloc(u32, fs.n * fs.k * p.format[0] / 32);
        defer p.a.free(words);
        f.f.words(words);
        p.x = try p.rig.input(std.mem.sliceAsBytes(f.f.x));
        if (p.kind == .lane) {
            const l = p.layout(shape);
            const pw = try p.a.alloc(u32, l.weightWords());
            defer p.a.free(pw);
            const sb = try p.a.alloc(f32, l.metadataElements());
            defer p.a.free(sb);
            try lane.packF32(l, words, f.scales, f.biases, pw, sb);
            p.w = try p.rig.input(std.mem.sliceAsBytes(pw));
            p.s = try p.rig.input(std.mem.sliceAsBytes(sb));
            p.projection = try lane.Projection.init(p.a, p.rig.device, l);
            if (p.fault != .none) {
                const body = try laneSource(p.a, l, p.fault);
                defer p.a.free(body);
                const text = try tf.frags.source(p.rig.device, p.a, body);
                defer p.a.free(text);
                const pipe = try r.pipeline(p.rig.device, text, "tf_lane");
                p.projection.?.pipe.deinit();
                p.projection.?.pipe = pipe;
            }
        } else {
            if (p.kind == .embed and fill == .nonfinite) {
                for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |v, i| f.scales[i * (fs.k / p.format[1])] = v;
            }
            p.w = try p.rig.input(std.mem.sliceAsBytes(words));
            p.s = try p.rig.input(std.mem.sliceAsBytes(f.scales));
            p.b = try p.rig.input(std.mem.sliceAsBytes(f.biases));
            if (p.kind == .embed) {
                const ids = try p.a.alloc(u32, shape.rows);
                defer p.a.free(ids);
                for (ids, 0..) |*v, i| v.* = @intCast(i % 17);
                p.ids = try p.rig.input(std.mem.sliceAsBytes(ids));
            }
        }
        p.out = try p.rig.output(2 * p.count());
        return true;
    }
    pub fn count(p: *const Probe) usize {
        return p.shape.rows * p.stride;
    }
    pub fn run(p: *Probe) !void {
        const bytes = p.rig.bytes(p.out, 2 * p.count());
        @memset(bytes, 0xff);
        for (0..p.count()) |i| if (i % p.stride >= p.shape.n) {
            std.mem.writeInt(u16, bytes[2 * i ..][0..2], c.toBf16(10), .little);
        };
        const pass = p.rig.begin();
        if (p.kind == .lane) {
            try p.projection.?.encode(pass.e, .{ .buf = p.x.buf, .off = p.x.off }, .{ .buf = p.w.buf, .off = p.w.off }, .{ .buf = p.s.buf, .off = p.s.off }, .{ .buf = p.out.buf, .off = p.out.off }, @intCast(p.shape.rows));
        } else if (p.kind == .embed) {
            const args = extern struct { width: u32, rows: u32, vocab: u32, bits: u32, group: u32, metadata: u32 };
            pass.e.setPipeline(p.pipe.?);
            for ([_]r.Ref{ p.ids, p.w, p.s, p.b }, 0..) |v, i| pass.e.setBuffer(v.buf, v.off, i);
            pass.e.setValue(args{ .width = @intCast(p.shape.n), .rows = @intCast(p.shape.rows), .vocab = 17, .bits = @intCast(p.format[0]), .group = @intCast(p.format[1]), .metadata = 2 }, 4);
            pass.e.setBuffer(p.out.buf, p.out.off, 5);
            pass.e.dispatchThreads(mtl.Size.of(p.shape.n, p.shape.rows, 1), mtl.Size.of(256, 1, 1));
        } else {
            var pipelines: row.Pipelines = undefined;
            pipelines.metadata = .f32;
            const i = std.mem.indexOfScalar(u8, &row.bit_widths, @intCast(p.format[0])).? * 2;
            if (p.kind == .strided) pipelines.strided[i] = p.pipe.? else pipelines.plain[i] = p.pipe.?;
            const weights = row.Weights{ .w = p.w.buf, .w_off = p.w.off, .scales = p.s.buf, .s_off = p.s.off, .biases = p.b.buf, .b_off = p.b.off, .n = p.shape.n, .k = p.shape.k, .bits = @intCast(p.format[0]), .group = p.format[1], .metadata = .f32 };
            const call = if (p.kind == .strided) try row.callStrided(&pipelines, weights, p.shape.rows, p.stride) else try row.call(&pipelines, weights, p.shape.rows, false);
            pass.e.setPipeline(call.pipeline);
            for ([_]r.Ref{ p.x, p.w, p.s, p.b }, 0..) |v, j| pass.e.setBuffer(v.buf, v.off, j);
            pass.e.setValue(call.dims, 4);
            pass.e.setBuffer(p.out.buf, p.out.off, 5);
            pass.e.dispatchGroups(.{ .width = call.groups }, .{ .width = call.threads });
        }
        try p.rig.finish(pass);
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const f = p.f.?;
        for (want, mag, 0..) |*w, *m, i| {
            const col = i % p.stride;
            if (col >= p.shape.n) {
                w.* = 10;
                m.* = 0;
            } else if (p.kind == .embed) {
                const token = (i / p.stride) % 17;
                const group = token * (p.shape.n / p.format[1]) + col / p.format[1];
                const value = @as(f64, f.scales[group]) * @as(f64, @floatFromInt(f.f.codes[token * p.shape.n + col])) + @as(f64, f.biases[group]);
                w.* = c.fromBf16(c.toBf16(@floatCast(value)));
                m.* = @abs(w.*);
            } else {
                const pair = f.one(i / p.stride, col);
                w.* = pair[0];
                m.* = pair[1];
            }
        }
    }
    pub fn bounds(p: *Probe, want: []const f64, mag: []const f64, allow: []f64) !void {
        const spec = p.kernel().spec;
        for (want, mag, allow, 0..) |w, m, *b, i| b.* = if (i % p.stride >= p.shape.n or p.kind == .embed) 0 else if (p.fill == .stress) c.boundExact(spec, w) else c.bound(spec, p.shape, w, m);
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const f = p.f.?;
        const compact = try p.a.alloc(f64, f.f.outputs());
        defer p.a.free(compact);
        try f.simulate(acc, order, compact);
        for (out, 0..) |*v, i| v.* = if (i % p.stride >= p.shape.n) 10 else compact[(i / p.stride) * p.shape.n + i % p.stride];
    }
};
pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Probe) = .empty,
    pub fn collect(s: *Suite, device: mtl.Device, queue: mtl.Queue, only: ?[]const u8, kernels: *std.ArrayList(g.Kernel)) !void {
        for (std.enums.values(Kind)) |kind| {
            if (kind == .lane and !device.tensorUnits()) continue;
            const splits: []const usize = if (kind == .lane) &.{ 1, 8 } else &.{1};
            for (formats) |format| for (splits) |sk| for (std.enums.values(Fault)) |fault| {
                var name: [144]u8 = undefined;
                if (only) |filter| if (std.mem.indexOf(u8, label(&name, kind, format, sk, fault), filter) == null) continue;
                const p = try s.a.create(Probe);
                errdefer s.a.destroy(p);
                p.* = try Probe.init(s.a, device, queue, kind, format, sk, fault);
                s.probes.append(s.a, p) catch |err| {
                    p.deinit();
                    return err;
                };
                try kernels.append(s.a, p.kernel());
            };
        }
    }
    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
