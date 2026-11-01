//! MOSS-Image's image decoder kernels under the contract, and md_pixels; controls each break one gate.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const mi = @import("metal_moss_image_common.zig");
const st = @import("metal_moss_image_decoder_steps.zig");

const c = mi.c;
const g = mi.g;
const Range = mi.Range;
const Ends = mi.Ends;
const dec = tf.moss_image.decoder;
const ref = st.ref;
const uniform = st.uniform;
const plant = st.plant;
const nan = st.nan;
const inf = st.inf;
const rw = @import("metal_moss_image_decoder_rows.zig");
const wn = @import("metal_moss_image_decoder_wino.zig");
const Wino = wn.Wino;
const Unpack = st.Unpack;
const Conv = st.Conv;
const Norm = st.Norm;
const conv_forms = st.conv_forms;
const GroupPart = rw.GroupPart;
const GroupFinish = rw.GroupFinish;
const Attention = rw.Attention;

// --- md_pixels: the 8-bit picture, as the pipeline's clamp, * 0.5 + 0.5, * 255 and round to even make it -----------

pub const Pixels = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    name: []const u8,
    shape: c.Shape = undefined,
    y: []f32 = &.{},
    ins: [1]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: pixels; n: the 3 channels (k unused).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = mi.noSums,
            .out = .raw,
            .nan = .{ .stated = "+Inf gives 255 and -Inf 0; a NaN's byte is unspecified, as numpy's cast of NaN to uint8 is" },
            .tiles = .{ .{ .size = 256 }, .{}, .{} },
            .served = &.{ .{ .rows = 1024 * 1024, .n = 3, .k = 1 }, .{ .rows = 1200 * 880, .n = 3, .k = 1 }, .{ .rows = 1344 * 768, .n = 3, .k = 1 } },
        };
    }

    pub fn deinit(t: *Pixels) void {
        t.a.free(t.y);
        t.rig.deinit();
    }
    pub fn kernel(t: *Pixels, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Pixels);

    /// The byte of y by the same fp32 steps.
    fn byte(y: f32) u8 {
        const v = std.math.clamp(std.math.clamp(y, -1, 1) * 0.5 + 0.5, 0, 1) * 255;
        const r = @round(v);
        const tie = r - v == 0.5 and @mod(r, 2) == 1;
        return @intFromFloat(if (tie) r - 1 else r);
    }

    /// The decoder outputs whose * 255 lands exactly halfway between two bytes, one for each byte up to 254.
    fn ties(out: *std.ArrayList(f32), a: std.mem.Allocator) !void {
        for (0..255) |k| {
            const want: f32 = @as(f32, @floatFromInt(k)) + 0.5;
            var v: f32 = want / 255;
            for (0..64) |_| {
                if (v * 255 == want) break;
                v = if (v * 255 < want) std.math.nextAfter(f32, v, 2) else std.math.nextAfter(f32, v, -2);
            }
            if (v * 255 != want) continue;
            const y = v * 2 - 1;
            if (std.math.clamp(y, -1, 1) * 0.5 + 0.5 == v) try out.append(a, y);
        }
    }

    pub fn prepare(t: *Pixels, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != 3) return false;
        t.rig.reset();
        t.a.free(t.y);
        t.shape = shape;
        t.y = try t.a.alloc(f32, shape.rows * 3);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        uniform(rng, t.y, -1.5, 1.5);
        var tie: std.ArrayList(f32) = .empty;
        defer tie.deinit(t.a);
        try ties(&tie, t.a);
        for (tie.items, 0..) |v, i| t.y[(i * 37) % t.y.len] = v;
        if (fill == .nonfinite) plant(t.y, 3);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.y));
        t.out = try t.rig.output(t.y.len);
        return true;
    }

    pub fn run(t: *Pixels) !void {
        const p = t.rig.begin();
        dec.pixels(p.e, t.pipe, ref(t.ins[0]), ref(t.out), t.y.len);
        try t.rig.finish(p);
    }

    pub fn count(t: *Pixels) usize {
        return t.y.len;
    }

    pub fn reference(t: *Pixels, want: []f64, mag: []f64) !void {
        for (want, mag, t.y) |*w, *m, y| {
            w.* = if (std.math.isNan(y)) 0 else @floatFromInt(byte(y));
            m.* = 0;
        }
    }

    /// Every byte equals the fp32 steps' (NaN's aside), ties to even included.
    pub fn check(t: *Pixels, _: g.Gate) !?[]const u8 {
        const got = t.rig.bytes(t.out, t.y.len);
        for (got, t.y) |b, y| {
            if (std.math.isNan(y)) continue;
            if (b != byte(y)) return "a byte differs from the fp32 steps' (clamp, * 0.5 + 0.5, * 255, round to even)";
        }
        return null;
    }
};

// --- the suite: the served source's pipelines, and each control's own copy of the source ------------------------

/// A control: the kernel it breaks, the gate that must catch it, and the one text change that breaks it.
const Control = struct { kernel: []const u8, gate: g.Gate, what: []const u8, from: []const u8, to: []const u8 };

const controls = [_]Control{
    .{ .kernel = "md_unpack", .gate = .shapes, .what = "swaps the patch's axes", .from = "const uint k = c * 4 + (py % 2) * 2 + px % 2;", .to = "const uint k = c * 4 + (px % 2) * 2 + py % 2;" },
    .{ .kernel = "md_conv 1x1", .gate = .shapes, .what = "stages its inputs in half", .from = "for (uint i = 0; i < 4; i++) dst[i] = src[i];", .to = "for (uint i = 0; i < 4; i++) dst[i] = float4(half4(src[i]));" },
    .{ .kernel = "md_wino 3x3", .gate = .shapes, .what = "stages its patch in half", .from = "*(threadgroup float4*)(buf + px * BK + q * 4) = v;", .to = "*(threadgroup float4*)(buf + px * BK + q * 4) = float4(half4(v));" },
    .{ .kernel = "md_wino_32 3x3", .gate = .shapes, .what = "drops a term of the output transform", .from = "s[j][1] = (m1 - m2) - m3;", .to = "s[j][1] = (m1 - m2);" },
    .{ .kernel = "md_norm", .gate = .shapes, .what = "its SiLU in half", .from = "y[i] = p.silu ? v / (1.0f + precise::exp(-v)) : v;", .to = "y[i] = p.silu ? float(half(v) / (1.0h + exp(-half(v)))) : v;" },
    .{ .kernel = "md_group_part", .gate = .sums, .what = "sums in half", .from = "s += p.centered ? d * d : v;", .to = "s = float(half(s) + half(p.centered ? d * d : v));" },
    .{ .kernel = "md_group_finish", .gate = .sums, .what = "sums in half", .from = "for (uint i = lane; i < p.chunks; i += 32) s += part[sg * p.chunks + i];", .to = "for (uint i = lane; i < p.chunks; i += 32) s = float(half(s) + half(part[sg * p.chunks + i]));" },
    .{ .kernel = "md_attention", .gate = .shapes, .what = "keeps stale maxima (no rescale)", .from = "    if (rescale) {", .to = "    if (false) {" },
    .{ .kernel = "md_pixels", .gate = .shapes, .what = "rounds halves up", .from = "rgb[i] = uchar(metal::rint(v * 255.0f));", .to = "rgb[i] = uchar(floor(v * 255.0f + 0.5f));" },
};

/// The decoder's source as the engine compiles it, with `ctl`'s change made where it occurs, exactly once.
fn compile(a: std.mem.Allocator, device: mtl.Device, ctl: ?Control) !mtl.Library {
    const served = try tf.frags.source(device, a, sources.moss_image_decoder);
    defer a.free(served);
    const text = if (ctl) |x| blk: {
        if (std.mem.count(u8, served, x.from) != 1) return error.ControlTextNotFoundOnce;
        break :blk try std.mem.replaceOwned(u8, a, served, x.from, x.to);
    } else try a.dupe(u8, served);
    defer a.free(text);
    return mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
}

pub const Suite = struct {
    a: std.mem.Allocator,
    pipes: std.ArrayList(mtl.Pipeline) = .empty,
    names: std.ArrayList([]u8) = .empty,
    libs: std.ArrayList(mtl.Library) = .empty,
    unpacks: std.ArrayList(*Unpack) = .empty,
    convs: std.ArrayList(*Conv) = .empty,
    winos: std.ArrayList(*Wino) = .empty,
    norms: std.ArrayList(*Norm) = .empty,
    parts: std.ArrayList(*GroupPart) = .empty,
    finishes: std.ArrayList(*GroupFinish) = .empty,
    attns: std.ArrayList(*Attention) = .empty,
    pixels: std.ArrayList(*Pixels) = .empty,

    fn pipeline(s: *Suite, device: mtl.Device, lib: mtl.Library, name: []const u8) !mtl.Pipeline {
        const p = try mtl.Pipeline.init(device, lib, name, false);
        errdefer p.deinit();
        try s.pipes.append(s.a, p);
        return p;
    }

    fn label(s: *Suite, comptime fmt: []const u8, args: anytype) ![]const u8 {
        const name = try std.fmt.allocPrint(s.a, fmt, args);
        errdefer s.a.free(name);
        try s.names.append(s.a, name);
        return name;
    }

    fn make(s: *Suite, comptime T: type, list: *std.ArrayList(*T), value: T) !*T {
        const t = try s.a.create(T);
        errdefer s.a.destroy(t);
        t.* = value;
        try list.append(s.a, t);
        return t;
    }

    /// Each kernel (or, with `ctl`, the one it breaks) on `lib`'s pipelines.
    fn add(s: *Suite, device: mtl.Device, queue: mtl.Queue, lib: mtl.Library, ctl: ?Control, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        const suffix = if (ctl) |x| try s.label(" control: {s}", .{x.what}) else "";
        const gate: ?g.Gate = if (ctl) |x| x.gate else null;
        const rig: rig_.Rig = .{ .a = s.a, .device = device, .queue = queue };
        const ends: Ends = .{ .a = s.a };
        const want = struct {
            fn is(x: ?Control, kernel: []const u8) bool {
                return if (x) |y| std.mem.eql(u8, y.kernel, kernel) else true;
            }
            fn keep(o: ?[]const u8, name: []const u8) bool {
                return if (o) |needle| std.mem.indexOf(u8, name, needle) != null else true;
            }
        };
        {
            const name = try s.label("metal moss image decoder md_unpack{s}", .{suffix});
            if (want.is(ctl, "md_unpack") and want.keep(only, name)) {
                const t = try s.make(Unpack, &s.unpacks, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "md_unpack"), .name = name, .ends = ends });
                try out.append(s.a, t.kernel(gate));
            }
        }
        for (conv_forms) |f| {
            const kind = try s.label("md_conv {s}", .{f.label});
            const name = try s.label("metal moss image decoder {s}{s}", .{ kind, suffix });
            if (!want.is(ctl, kind) or !want.keep(only, name)) continue;
            const t = try s.make(Conv, &s.convs, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "md_conv"), .form = f, .name = name });
            try out.append(s.a, t.kernel(gate));
        }
        for (wn.forms) |f| {
            const name = try s.label("metal moss image decoder {s}{s}", .{ f.label, suffix });
            if (!want.is(ctl, f.label) or !want.keep(only, name)) continue;
            const t = try s.make(Wino, &s.winos, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, if (f.group == 64) "md_wino" else "md_wino_32"), .form = f, .name = name });
            try out.append(s.a, t.kernel(gate));
        }
        for ([_]bool{ true, false }) |silu| {
            const name = try s.label("metal moss image decoder md_norm{s}{s}", .{ if (silu) " silu" else "", suffix });
            if (!want.is(ctl, "md_norm") or !want.keep(only, name) or (ctl != null and !silu)) continue;
            const t = try s.make(Norm, &s.norms, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "md_norm"), .silu = silu, .name = name, .ends = ends });
            try out.append(s.a, t.kernel(gate));
        }
        for ([_]bool{ false, true }) |centered| {
            const name = try s.label("metal moss image decoder md_group_part{s}{s}", .{ if (centered) " centered" else "", suffix });
            if (!want.is(ctl, "md_group_part") or !want.keep(only, name) or (ctl != null and centered)) continue;
            const t = try s.make(GroupPart, &s.parts, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "md_group_part"), .centered = centered, .name = name });
            try out.append(s.a, t.kernel(gate));
        }
        {
            const name = try s.label("metal moss image decoder md_group_finish{s}", .{suffix});
            if (want.is(ctl, "md_group_finish") and want.keep(only, name)) {
                const t = try s.make(GroupFinish, &s.finishes, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "md_group_finish"), .name = name, .ends = ends });
                try out.append(s.a, t.kernel(gate));
            }
        }
        {
            const name = try s.label("metal moss image decoder md_attention{s}", .{suffix});
            if (want.is(ctl, "md_attention") and want.keep(only, name)) {
                const t = try s.make(Attention, &s.attns, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "md_attention"), .name = name });
                try out.append(s.a, t.kernel(gate));
            }
        }
        {
            const name = try s.label("metal moss image decoder md_pixels{s}", .{suffix});
            if (want.is(ctl, "md_pixels") and want.keep(only, name)) {
                const t = try s.make(Pixels, &s.pixels, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "md_pixels"), .name = name });
                try out.append(s.a, t.kernel(gate));
            }
        }
    }

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        if (!device.tensorUnits()) {
            std.debug.print("UNSUPPORTED metal moss image decoder: the family runs on the M5 alone; its kernels unchecked here\n", .{});
            return s;
        }
        for ([_]?Control{null} ++ controls) |x| {
            const lib = try compile(a, device, x);
            try s.libs.append(a, lib);
            try s.add(device, queue, lib, x, out, only);
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        const a = s.a;
        inline for (.{ &s.unpacks, &s.convs, &s.winos, &s.norms, &s.parts, &s.finishes, &s.attns, &s.pixels }) |list| {
            for (list.items) |t| {
                t.deinit();
                a.destroy(t);
            }
            list.deinit(a);
        }
        for (s.pipes.items) |p| p.deinit();
        s.pipes.deinit(a);
        for (s.libs.items) |l| l.deinit();
        s.libs.deinit(a);
        for (s.names.items) |n| a.free(n);
        s.names.deinit(a);
        a.destroy(s);
    }
};
