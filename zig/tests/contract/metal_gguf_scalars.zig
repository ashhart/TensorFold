//! Original F32 GGUF norm gains and negative SSM coefficients meet the shared independent-reference gates.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const r = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
pub const Kind = enum { input, residual, head, prompt, pre, step };
pub fn norm(k: Kind) bool {
    return @backingInt(k) <= @backingInt(Kind.head);
}
pub fn symbol(k: Kind) []const u8 {
    return switch (k) {
        .input => "custom_kernel_q27_norm_input_bfloat16_t_bfloat16_t_floatc_int32_t_bfloat16_t_float",
        .residual => "custom_kernel_q27_norm_xs_bfloat16_t_bfloat16_t_bfloat16_t_floatc_int32_t_bfloat16_t_bfloat16_t_float",
        .head => "q27_head_norm",
        .prompt => "q27_prompt_decay",
        .pre => "tf_delta_gdn_pre",
        .step => "tf_delta_gdn_pre_step",
    };
}
pub fn source(a: std.mem.Allocator, k: Kind, width: usize, fault: bool) ![]u8 {
    const raw = switch (k) {
        .input => tf.qwen27.runtime_sources.norm_input,
        .residual => tf.qwen27.runtime_sources.norm_xs,
        .head => tf.qwen27.runtime_sources.glue,
        .prompt => tf.qwen27.runtime_sources.prompt_glue,
        .pre, .step => sources.core_deltanet,
    };
    if (k == .input or k == .residual) {
        const body = try std.mem.replaceOwned(u8, a, raw, "Q27_SPECIALIZE", try std.fmt.allocPrint(a, "constexpr int K = {d};", .{width}));
        const edited = if (fault) try std.mem.replaceOwned(u8, a, body, "float(Wt[int(t) * E + i])", "(float(Wt[int(t) * E + i]) + 1.0f)") else body;
        return std.mem.concat(a, u8, &.{ "#define TF_Q27_GAIN float\n", edited });
    }
    if (k == .head and fault) return std.mem.replaceOwned(u8, a, raw, "q27_read(W, at, a.gain)", "(q27_read(W, at, a.gain) + 1.0f)");
    return a.dupe(u8, raw);
}
fn depth(s: c.Shape) usize {
    return 16 + 5 + (s.k + 511) / 512 + 8;
}
fn point(_: c.Shape) usize {
    return 0;
}
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: r.Rig,
    kind: Kind,
    fault: bool,
    name: [96]u8 = undefined,
    pipe: ?mtl.Pipeline = null,
    shape: c.Shape = undefined,
    x: r.Ref = undefined,
    residual: r.Ref = undefined,
    gain: r.Ref = undefined,
    out: r.Ref = undefined,
    ho: r.Ref = undefined,
    sums: r.Ref = undefined,
    bound: [16]r.Ref = undefined,
    pub fn deinit(p: *Probe) void {
        if (p.pipe) |v| v.deinit();
        p.rig.deinit();
    }
    pub fn kernel(p: *Probe) g.Kernel {
        const name = std.fmt.bufPrint(&p.name, "metal GGUF scalars {t}{s}", .{ p.kind, if (p.fault) " coefficient control" else "" }) catch unreachable;
        const normal: c.Spec = .{ .name = name, .accum = .f32, .depth = depth, .ops = 8, .out = .bf16, .served = if (p.kind == .head) &.{ .{ .rows = 4, .n = 256, .k = 256 }, .{ .rows = 24, .n = 256, .k = 256 }, .{ .rows = 72, .n = 256, .k = 256 } } else &.{ .{ .rows = 1, .n = 5120, .k = 5120 }, .{ .rows = 3, .n = 5120, .k = 5120 }, .{ .rows = 16, .n = 512, .k = 512 } }, .sums = .{ .f16 = 100, .bf16 = 0, .exact = false, .why = "FP16 overflows; BF16 normalized outputs mask balanced BF16-sum errors" } };
        const decay: c.Spec = .{ .name = name, .accum = .none, .depth = point, .ops = 16, .out = .f32, .served = &.{ .{ .rows = 1, .n = 48, .k = 128 }, .{ .rows = 3, .n = 48, .k = 128 } } };
        var spec = if (norm(p.kind)) normal else decay;
        if (p.kind == .head) spec.mids = 1;
        return .{ .spec = spec, .ctx = p, .vt = &vt, .control = if (p.fault) .shapes else null };
    }
    const vt = g.hooks(Probe);
    fn bf(v: f64) f64 {
        return c.fromBf16(c.toBf16(@floatCast(v)));
    }
    fn input(p: *Probe, comptime T: type, n: usize, value: T) !r.Ref {
        const bytes = try p.a.alloc(T, n);
        defer p.a.free(bytes);
        @memset(bytes, value);
        return p.rig.input(std.mem.sliceAsBytes(bytes));
    }
    fn stride(p: *const Probe) usize {
        return p.shape.k * @as(usize, if (p.kind == .head and p.shape.rows % 24 == 0) 2 else 1);
    }
    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.rows > (if (p.kind == .head) @as(usize, 384) else 16)) return false;
        if (norm(p.kind)) {
            if (s.n != s.k or s.k == 0 or s.k > 5120) return false;
            if (s.k % (if (p.kind == .head) @as(usize, 32) else 512) != 0) return false;
        } else if (s.n != 48 or s.k != 128) return false;
        p.rig.reset();
        if (p.pipe) |v| v.deinit();
        p.pipe = null;
        var arena = std.heap.ArenaAllocator.init(p.a);
        defer arena.deinit();
        const a = arena.allocator();
        p.pipe = try r.pipeline(p.rig.device, try source(a, p.kind, s.k, p.fault), symbol(p.kind));
        p.shape = s;
        if (norm(p.kind)) {
            const x = try a.alloc(u16, s.rows * p.stride());
            for (x, 0..) |*v, i| v.* = c.toBf16(if (fill == .stress) @as(f32, 512) else @as(f32, @floatFromInt(1 + (i + seed % 97) % 31)) / 16);
            if (fill == .nonfinite) for (0..s.rows) |row| {
                x[row * p.stride()] = c.toBf16(([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) })[row % 3]);
            };
            p.x = try p.rig.input(std.mem.sliceAsBytes(x));
            p.residual = try p.input(u16, x.len, c.toBf16(0.125));
            const gains = try a.alloc(f32, s.k);
            for (gains, 0..) |*v, i| v.* = 1.67578125 + @as(f32, @floatFromInt(i % 7)) / 256;
            p.gain = try p.rig.input(std.mem.sliceAsBytes(gains));
            p.out = try p.rig.output(s.rows * s.k * 2);
            p.ho = try p.rig.scratch(x.len * 2);
            p.sums = try p.rig.scratch(s.k / 64 * std.mem.alignForward(usize, s.rows, 16) * 4);
        } else try p.prepareDecay(s, fill);
        return true;
    }
    fn prepareDecay(p: *Probe, s: c.Shape, fill: g.Fill) !void {
        const channels = 2 * 16 * 128 + 48 * 128;
        var arena = std.heap.ArenaAllocator.init(p.a);
        defer arena.deinit();
        const a = arena.allocator();
        const gate = try a.alloc(u16, s.rows * 48);
        @memset(gate, 0);
        if (fill == .nonfinite) for (0..s.rows) |row| {
            gate[row * 48] = c.toBf16(([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) })[row % 3]);
        };
        const weights = try a.alloc(f32, 48);
        for (weights, 0..) |*w, i| w.* = -0.1104574353 - @as(f32, @floatFromInt(i)) / 64;
        p.gain = try p.rig.input(std.mem.sliceAsBytes(weights));
        p.x = try p.rig.input(std.mem.sliceAsBytes(gate));
        const beta = try p.input(u16, gate.len, 0);
        const bias = try p.input(u16, 48, 0);
        p.out = try p.rig.output(gate.len * 4);
        const mixing = try p.rig.scratch(gate.len * 2);
        if (p.kind == .prompt) {
            p.bound[0..7].* = .{ p.x, beta, p.gain, bias, p.out, mixing, mixing };
            return;
        }
        const projection = try p.input(u16, s.rows * channels, 0);
        const history = try p.rig.scratch(s.rows * 3 * channels * 2);
        @memset(p.rig.bytes(history, s.rows * 3 * channels * 2), 0);
        const conv = try p.input(u16, channels * 4, 0);
        const windows = try a.alloc(u32, s.rows * 4);
        const slots = try a.alloc(u32, s.rows);
        for (0..s.rows) |row| {
            for (0..4) |tap| windows[row * 4 + tap] = @intCast(if (tap == 3) 3 + row else tap);
            slots[row] = @intCast(row);
        }
        p.bound = .{ projection, p.x, beta, history, conv, p.gain, bias, try p.rig.input(std.mem.sliceAsBytes(windows)), try p.rig.input(std.mem.sliceAsBytes(slots)), mixing, try p.rig.scratch(s.rows * 16 * 128 * 2), try p.rig.scratch(s.rows * 16 * 128 * 2), try p.rig.scratch(s.rows * 48 * 128 * 2), p.out, mixing, try p.rig.scratch(s.rows * 3 * channels * 2) };
    }
    pub fn run(p: *Probe) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe.?);
        if (p.kind == .head) {
            e.setBuffer(p.x.buf, p.x.off, 0);
            e.setBuffer(p.gain.buf, p.gain.off, 1);
            const h = tf.qwen27.glue.Head{ .width = @intCast(s.k), .rows = @intCast(s.rows), .stride = @intCast(p.stride()), .activation = .bf16, .gain = .f32, .eps = 1e-6 };
            e.setValue(h, 2);
            e.setBuffer(p.out.buf, p.out.off, 3);
            e.dispatchGroups(mtl.Size.of(s.rows, 1, 1), mtl.Size.of(try tf.qwen27.glue.headThreads(h), 1, 1));
        } else if (norm(p.kind)) {
            const refs: []const r.Ref = if (p.kind == .residual) &.{ p.x, p.residual, p.gain } else &.{ p.x, p.gain };
            for (refs, 0..) |v, i| e.setBuffer(v.buf, v.off, i);
            e.setValue(@as(f32, 1e-6), refs.len);
            e.setValue([_]i32{ @intCast(s.rows), @intCast(std.mem.alignForward(usize, s.rows, 16)) }, refs.len + 1);
            var slot = refs.len + 2;
            if (p.kind == .residual) {
                e.setBuffer(p.ho.buf, p.ho.off, slot);
                slot += 1;
            }
            e.setBuffer(p.out.buf, p.out.off, slot);
            e.setBuffer(p.sums.buf, p.sums.off, slot + 1);
            e.dispatchGroups(mtl.Size.of(1, std.mem.alignForward(usize, s.rows, 16), 1), mtl.Size.of(s.k / 16, 1, 1));
        } else if (p.kind == .prompt) {
            for (p.bound[0..4], 0..) |v, i| e.setBuffer(v.buf, v.off, i);
            e.setValue(extern struct { rows: u32, heads: u32, flags: u32, pad: u32 }{ .rows = @intCast(s.rows), .heads = 48, .flags = if (p.fault) 2 else 18, .pad = 0 }, 4);
            e.setBuffer(p.out.buf, p.out.off, 5);
            e.setBuffer(p.bound[5].buf, p.bound[5].off, 6);
            e.dispatchThreads(mtl.Size.of(s.rows * 48, 1, 1), mtl.Size.of(64, 1, 1));
        } else {
            for (p.bound, 0..) |v, i| if (i != 9) e.setBuffer(v.buf, v.off, i);
            e.setValue(tf.qwen27.gdn_contract.Params{ .rows = @intCast(s.rows), .nk = 16, .nv = 48, .dk = 128, .dv = 128, .taps = 4, .slots = @intCast(s.rows), .weight_flags = if (p.fault) 2 else 18, .eps = 1e-6 }, 9);
            e.dispatchGroups(mtl.Size.of(1, 80, s.rows), mtl.Size.of(if (p.kind == .step) 128 else 32, 1, 1));
        }
        try p.rig.finish(pass);
        if (p.kind == .input or p.kind == .residual) try p.auxiliary();
    }
    fn auxiliary(p: *Probe) !void {
        @setFloatMode(.strict);
        const s = p.shape;
        const mp = std.mem.alignForward(usize, s.rows, 16);
        const values = p.rig.bytes(p.out, s.rows * s.k * 2);
        const sums = p.rig.bytes(p.sums, s.k / 64 * mp * 4);
        // XO meets FP64 independently; XS must then equal its specified ordered FP32 sum, including padded rows.
        for (0..s.k / 64) |group| for (0..mp) |row| {
            var total: f32 = 0;
            if (row < s.rows) for (0..64) |col| {
                total += @floatCast(c.read(.bf16, values, row * s.k + group * 64 + col));
            };
            const at = (group * mp + row) * 4;
            const got: f32 = @bitCast(std.mem.readInt(u32, sums[at..][0..4], .little));
            if (c.classOf(total) != c.classOf(got)) return error.GgufScalarSumClass;
            if (std.math.isFinite(total) and @as(u32, @bitCast(total)) != @as(u32, @bitCast(got))) return error.GgufScalarSumOrder;
        };
        if (p.kind == .residual) {
            const h = p.rig.bytes(p.x, s.rows * s.k * 2);
            const residual = p.rig.bytes(p.residual, s.rows * s.k * 2);
            const ho = p.rig.bytes(p.ho, s.rows * s.k * 2);
            for (0..s.rows * s.k) |at| {
                const want = bf(c.read(.bf16, h, at) + c.read(.bf16, residual, at));
                const got = c.read(.bf16, ho, at);
                if (c.classOf(want) != c.classOf(got)) return error.GgufScalarResidualClass;
                if (std.math.isFinite(want) and want != got) return error.GgufScalarResidualRounding;
            }
        }
    }
    pub fn count(p: *Probe) usize {
        return p.shape.rows * p.shape.n;
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const s = p.shape;
        const x = p.rig.bytes(p.x, (if (norm(p.kind)) s.rows * p.stride() else s.rows * 48) * 2);
        const gains = p.rig.bytes(p.gain, (if (norm(p.kind)) s.k else 48) * 4);
        for (0..s.rows) |row| {
            var square: f64 = 0;
            if (norm(p.kind)) for (0..s.k) |col| {
                const h = c.read(.bf16, x, row * p.stride() + col) + (if (p.kind == .residual) @as(f64, 0.125) else 0);
                const v = bf(h);
                square += v * v;
            };
            const inv = 1.0 / @sqrt(square / @as(f64, @floatFromInt(s.k)) + @as(f64, @as(f32, 1e-6)));
            for (0..s.n) |col| {
                const at = row * s.n + col;
                const weight = c.read(.f32, gains, col);
                var value: f64 = undefined;
                if (norm(p.kind)) {
                    const h = bf(c.read(.bf16, x, row * p.stride() + col) + (if (p.kind == .residual) @as(f64, 0.125) else 0));
                    const normalized = h * inv;
                    value = weight * (if (p.kind == .head) bf(normalized) else normalized);
                    mag[at] = @abs(value);
                } else {
                    const v = c.read(.bf16, x, at);
                    const smooth = bf(if (std.math.isNan(v)) v else @max(v, 0) + @log(1 + @exp(-@abs(v))));
                    const exponent = weight * smooth;
                    value = @exp(exponent);
                    mag[at] = @abs(value) * (1 + @abs(exponent));
                }
                want[at] = value;
            }
        }
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const s = p.shape;
        const terms = try p.a.alloc(f64, s.k);
        defer p.a.free(terms);
        const x = p.rig.bytes(p.x, s.rows * p.stride() * 2);
        const gains = p.rig.bytes(p.gain, s.k * 4);
        for (0..s.rows) |row| {
            for (terms, 0..) |*term, col| {
                const h = bf(c.read(.bf16, x, row * p.stride() + col) + (if (p.kind == .residual) @as(f64, 0.125) else 0));
                term.* = h * h;
            }
            const square = contract.rows.sumIn(acc, order, terms);
            const inv = 1.0 / @sqrt(square / @as(f64, @floatFromInt(s.k)) + @as(f64, @as(f32, 1e-6)));
            for (0..s.k) |col| {
                const h = bf(c.read(.bf16, x, row * p.stride() + col) + (if (p.kind == .residual) @as(f64, 0.125) else 0));
                const normalized = h * inv;
                out[row * s.k + col] = bf(c.read(.f32, gains, col) * (if (p.kind == .head) bf(normalized) else normalized));
            }
        }
    }
};
pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Probe) = .empty,
    pub fn collect(s: *Suite, device: mtl.Device, queue: mtl.Queue, only: ?[]const u8, list: *std.ArrayList(g.Kernel)) !void {
        for (std.enums.values(Kind)) |kind| for ([_]bool{ false, true }) |fault| {
            var probe: Probe = .{ .a = s.a, .rig = .{ .a = s.a, .device = device, .queue = queue }, .kind = kind, .fault = fault };
            const name = probe.kernel().spec.name;
            if (only) |filter| if (std.mem.indexOf(u8, name, filter) == null) continue;
            const p = try s.a.create(Probe);
            p.* = probe;
            try s.probes.append(s.a, p);
            try list.append(s.a, p.kernel());
        };
    }
    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
