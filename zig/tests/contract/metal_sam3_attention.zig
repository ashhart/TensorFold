const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const ar = contract.attention_range;
const dim = 256;
const period = 67;
const vt = g.hooks(Probe);

const single_shapes = [_]c.Shape{
    .{ .rows = 65, .n = dim, .k = 97 },
    .{ .rows = 5184, .n = dim, .k = 5184 },
    .{ .rows = 5184, .n = dim, .k = 36544 },
    .{ .rows = 5185, .n = dim, .k = 33 },
    .{ .rows = 2047, .n = dim, .k = 97 },
    .{ .rows = 2048, .n = dim, .k = 97 },
    .{ .rows = 2049, .n = dim, .k = 97 },
    .{ .rows = 4096, .n = dim, .k = 33 },
    .{ .rows = 4097, .n = dim, .k = 33 },
};
const batch_shapes = [_]c.Shape{
    .{ .rows = 65, .n = 2 * dim, .k = 97 },
    .{ .rows = 5185, .n = 2 * dim, .k = 33 },
    .{ .rows = 2049, .n = 2 * dim, .k = 97 },
    .{ .rows = 4096, .n = 2 * dim, .k = 33 },
};
const split_shapes = single_shapes ++ [_]c.Shape{
    .{ .rows = 65, .n = dim, .k = 2047 },
    .{ .rows = 65, .n = dim, .k = 2048 },
    .{ .rows = 65, .n = dim, .k = 2049 },
    .{ .rows = 5184, .n = dim, .k = 36348 },
};
const split_batches = batch_shapes ++ [_]c.Shape{
    .{ .rows = 65, .n = 2 * dim, .k = 2049 },
};
const device_shapes = [_]c.Shape{
    .{ .rows = 65, .n = 1024, .k = 1 },
    .{ .rows = 65, .n = 1024, .k = 97 },
    .{ .rows = 63, .n = 1024, .k = 65 },
    .{ .rows = 64, .n = 1024, .k = 64 },
    .{ .rows = 5184, .n = 1024, .k = 5184 },
};
const device_batches = [_]c.Shape{
    .{ .rows = 65, .n = 1024, .k = 97 },
    .{ .rows = 576, .n = 1024, .k = 576 },
};
const device32_shapes = [_]c.Shape{
    .{ .rows = 65, .n = 256, .k = 1 },
    .{ .rows = 65, .n = 256, .k = 97 },
    .{ .rows = 63, .n = 256, .k = 65 },
    .{ .rows = 64, .n = 256, .k = 64 },
    .{ .rows = 5184, .n = 256, .k = 5184 },
    .{ .rows = 5184, .n = 256, .k = 5200 },
    .{ .rows = 65, .n = 256, .k = 36544 },
};
const device32_batches = [_]c.Shape{
    .{ .rows = 65, .n = 256, .k = 97 },
    .{ .rows = 576, .n = 256, .k = 576 },
};

fn depth(s: c.Shape) usize {
    return dim + s.k;
}
fn splitDepth(s: c.Shape) usize {
    const chunk = sources.sam3_attention_split.chunk_blocks * 32;
    return dim + @min(s.k, chunk) + 5 * ((s.k + chunk - 1) / chunk);
}
fn deviceDepth(s: c.Shape) usize {
    return 64 + s.k;
}
fn split32Depth(s: c.Shape) usize {
    const chunk = sources.sam3_attention_split32.chunk_blocks * 32;
    return 32 + @min(s.k, chunk) + 5 * ((s.k + chunk - 1) / chunk);
}

fn label(batched: bool, control: bool, two_pass: bool, replacement: bool, split: bool, device: bool, width: usize) []const u8 {
    if (device and split and width == 64) return if (control) "control: sam3 attention D64 split value head" else if (batched) "metal sam3 attention D64 split batches and heads" else "metal sam3 attention D64 split full";
    if (device and split) return if (control) "control: sam3 attention D32 split value head" else if (batched) "metal sam3 attention D32 split batches and heads" else "metal sam3 attention D32 split full";
    if (device and width == 32) return if (control) "control: sam3 attention D32 device omits fallback" else if (batched) "metal sam3 attention D32 device batches and heads" else "metal sam3 attention D32 device full";
    if (device) return if (control) "control: sam3 attention D64 device omits fallback" else if (batched) "metal sam3 attention D64 device batches and heads" else "metal sam3 attention D64 device full";
    if (split) return if (control) "control: sam3 attention D256 split value partition" else if (batched) "metal sam3 attention D256 split batches and heads" else "metal sam3 attention D256 split full";
    if (control) return if (two_pass) "control: sam3 attention D256 two-pass value partition" else "control: sam3 attention D256 value partition";
    if (replacement) return if (batched) "metal sam3 attention D256 two-pass batched replacement" else "metal sam3 attention D256 two-pass replacement";
    if (two_pass) return if (batched) "metal sam3 attention D256 two-pass batches and heads" else "metal sam3 attention D256 two-pass full";
    return if (batched) "metal sam3 attention D256 batches and heads" else "metal sam3 attention D256 full";
}

fn stressValue(j: usize, n: usize) f32 {
    const half = n / 2;
    const at = if (j < half) j else j - half;
    if (j >= half * 2 or at >= 256) return 0;
    const big: f32 = if (at % 2 == 0) 127 else 16256;
    return if (j < half) big + @as(f32, if (at < 8 and at % 2 == 0) 1 else 0) else -big;
}

pub fn text(a: std.mem.Allocator, control: bool) ![]u8 {
    const raw = try sources.tensor_library.source(a, sources.sam3_source);
    if (!control) return raw;
    defer a.free(raw);
    const from = "P[19] + value_part * VDIM;";
    if (std.mem.count(u8, raw, from) != 1) return error.ControlTextNotFound;
    return std.mem.replaceOwned(u8, a, raw, from, "P[19];");
}

pub fn twoPassText(a: std.mem.Allocator, control: bool) ![]u8 {
    const raw = try text(a, false);
    if (!control) return raw;
    defer a.free(raw);
    const from = "P[19] + part * 128;";
    if (std.mem.count(u8, raw, from) != 1) return error.ControlTextNotFound;
    return std.mem.replaceOwned(u8, a, raw, from, "P[19];");
}

pub fn splitText(a: std.mem.Allocator, control: bool) ![]u8 {
    const raw = try text(a, false);
    if (!control) return raw;
    defer a.free(raw);
    const from = "P[20] + part * (256 / SAM_SPLIT_VALUE_PARTS);";
    if (std.mem.count(u8, raw, from) != 1) return error.ControlTextNotFound;
    return std.mem.replaceOwned(u8, a, raw, from, "P[20];");
}

pub fn split32Text(a: std.mem.Allocator, control: bool) ![]u8 {
    const raw = try text(a, false);
    if (!control) return raw;
    defer a.free(raw);
    const from = "const long v_head = long(tg.y) * P[19];";
    if (std.mem.count(u8, raw, from) != 1) return error.ControlTextNotFound;
    return std.mem.replaceOwned(u8, a, raw, from, "const long v_head = 0;");
}

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    batched: bool,
    control: bool,
    two_pass: bool,
    replacement: bool,
    split: bool,
    device: bool = false,
    width: usize = dim,
    passes: [2]?mtl.Pipeline = .{ null, null },
    split_passes: [3]?mtl.Pipeline = @splat(null),
    device_passes: [6]?mtl.Pipeline = @splat(null),
    split32_passes: [6]?mtl.Pipeline = @splat(null),
    shape: c.Shape = undefined,
    stress: bool = false,
    fill: g.Fill = .random,
    q: []f32 = &.{},
    k: []f32 = &.{},
    v: []f32 = &.{},
    half: []f64 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    base: rig_.Ref = undefined,
    work: [3]rig_.Ref = undefined,

    fn spec(p: *const Probe) c.Spec {
        return .{
            .name = label(p.batched, p.control, p.two_pass, p.replacement, p.split, p.device, p.width),
            .accum = .f32,
            .depth = if (p.device and p.split) split32Depth else if (p.device) deviceDepth else if (p.split) splitDepth else depth,
            .ops = 8,
            .out = .f32,
            .replaces = if (p.replacement) "split-value SAM D256 attention" else null,
            .served = if (p.device and p.width == 32) (if (p.batched) &device32_batches else &device32_shapes) else if (p.device) (if (p.batched) &device_batches else &device_shapes) else if (p.split) (if (p.batched) &split_batches else &split_shapes) else if (p.batched) &batch_shapes else &single_shapes,
            .tiles = .{ .{ .size = if (p.device and !p.split and p.width == 64) 32 else 64 }, .{}, .{ .size = if (p.device and (if (p.split) p.width == 32 else p.width == 64)) 64 else 32 } },
        };
    }

    fn heads(p: *const Probe) usize {
        return if (p.device) (if (p.width == 32) 8 else 16) else if (p.batched) 2 else 1;
    }

    fn batches(p: *const Probe) usize {
        return if (p.device and p.batched and p.width == 64) 9 else if (p.batched) 2 else 1;
    }

    fn reset(p: *Probe) void {
        for ([_][]f32{ p.q, p.k, p.v }) |x| p.a.free(x);
        p.a.free(p.half);
        p.q = &.{};
        p.k = &.{};
        p.v = &.{};
        p.half = &.{};
        p.rig.reset();
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.k == 0 or s.n != p.heads() * p.width) return false;
        p.reset();
        p.shape = s;
        p.stress = fill == .stress;
        p.fill = fill;
        p.q = try p.a.alloc(f32, p.count());
        p.k = try p.a.alloc(f32, p.batches() * s.k * s.n);
        p.v = try p.a.alloc(f32, p.k.len);
        p.half = try p.a.alloc(f64, p.count());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..p.batches()) |b| for (0..s.rows) |r| {
            const row = p.q[(b * s.rows + r) * s.n ..][0..s.n];
            if (r < period) {
                for (row) |*x| x.* = if (p.stress) 0 else rng.float(f32) - 0.5;
            } else @memcpy(row, p.q[(b * s.rows + r % period) * s.n ..][0..s.n]);
        };
        for (0..p.batches()) |b| for (0..s.k) |j| for (0..s.n) |d| {
            const i = (b * s.k + j) * s.n + d;
            p.k[i] = if (p.stress) 0 else rng.float(f32) - 0.5;
            p.v[i] = if (p.stress) stressValue(j, s.k) else rng.float(f32) * 2 - 1;
        };
        if (fill == .nonfinite) {
            p.q[0] = std.math.nan(f32);
            p.q[s.n] = std.math.inf(f32);
            p.q[2 * s.n] = -std.math.inf(f32);
            p.q[3 * s.n] = std.math.inf(f32);
            p.q[3 * s.n + 1] = -std.math.inf(f32);
            p.v[0] = std.math.inf(f32);
            p.v[1] = -std.math.inf(f32);
            p.v[2] = std.math.inf(f32);
            p.v[(if (s.k > 1) s.n else 0) + 2] = -std.math.inf(f32);
            p.v[3] = std.math.nan(f32);
            if (p.batched) p.k[s.k * s.n + p.width] = std.math.nan(f32);
        } else if (p.device and fill == .random) {
            if (s.k == 1) {
                @memset(p.q, 0);
                @memset(p.k, 0);
                for (p.v, 0..) |*value, i| value.* = switch (i % 64) {
                    0 => @bitCast(@as(u32, 1)),
                    1 => std.math.floatMax(f32),
                    2 => -std.math.floatMax(f32),
                    3 => @bitCast(@as(u32, 0x80000001)),
                    else => value.*,
                };
            }
            if (s.k == 97) p.v[0] = @bitCast(@as(u32, 1));
            if (s.k == 65) for (0..p.batches()) |b| {
                p.k[b * s.k * s.n] = 32;
                for (0..s.k) |j| p.v[(b * s.k + j) * s.n] = if (j == 0) 0 else 0x1p36;
                for (0..s.rows) |r| if (r % period == 0) {
                    p.q[(b * s.rows + r) * s.n] = 32;
                };
            };
        }
        for ([_][]const f32{ p.q, p.k, p.v }, &p.ins) |values, *input| input.* = try p.rig.input(std.mem.sliceAsBytes(values));
        p.out = try p.rig.output(p.count() * 4);
        if (p.replacement) p.base = try p.rig.scratch(p.count() * 4);
        if (p.device) {
            p.work[0] = try p.rig.scratch((1 + (s.rows + 63) / 64 * p.heads() * p.batches()) * 4);
            if (p.split) {
                const chunk = sources.sam3_attention_split32.chunk_blocks * 32;
                const planes = p.batches() * p.heads() * s.rows;
                const splits = (s.k + chunk - 1) / chunk;
                p.work[1] = try p.rig.scratch(planes * splits * p.width * 4);
                p.work[2] = try p.rig.scratch(planes * splits * 3 * 4);
            } else p.work[1] = try p.rig.scratch(if (p.width == 64 and s.rows == 576 and s.k == 576) (p.k.len * 2 + p.batches() * p.heads() * 64) * 4 else 4);
        } else if (p.two_pass or p.split) {
            const blocks = (s.k + 31) / 32;
            const planes = p.batches() * p.heads() * s.rows;
            p.work[0] = try p.rig.scratch(planes * blocks * 32 * 4);
            if (p.split) {
                const chunk_blocks = sources.sam3_attention_split.chunk_blocks;
                const splits = (blocks + chunk_blocks - 1) / chunk_blocks;
                p.work[1] = try p.rig.scratch(planes * splits * (chunk_blocks + 2) * 4);
                p.work[2] = try p.rig.scratch(p.count() * splits * 4);
            } else p.work[1] = try p.rig.scratch(planes * (blocks + 1) * 4);
        }
        return true;
    }

    fn launch(p: *Probe, baseline: bool) !void {
        const s = p.shape;
        const nheads = p.heads();
        const nq = (s.rows + 63) / 64;
        const vals = [_]usize{
            p.batches(),  nheads,  s.rows, s.k,       1,       nq,  (s.k + 31) / 32, s.rows / 64, s.k / 32, s.rows % 64,  s.k % 32, 0,
            s.rows * s.n, p.width, s.n,    s.k * s.n, p.width, s.n, s.k * s.n,       p.width,     s.n,      s.rows * s.n, p.width,  s.n,
        };
        var params: [24]i32 = undefined;
        for (vals, &params) |x, *v| v.* = @intCast(x);
        const pass = p.rig.begin();
        if (p.device and p.split) {
            const Binding = struct { buffer: mtl.Buffer, offset: usize };
            var refs: [7]Binding = undefined;
            for ([_]rig_.Ref{ p.ins[0], p.ins[1], p.ins[2], p.work[1], p.work[2], p.out, p.work[0] }, &refs) |r, *binding|
                binding.* = .{ .buffer = r.buf, .offset = r.off };
            var split_params: [27]i32 = undefined;
            @memcpy(split_params[0..24], &params);
            const chunk = sources.sam3_attention_split32.chunk_blocks * 32;
            split_params[24] = @intCast((s.k + chunk - 1) / chunk);
            split_params[25] = sources.sam3_attention_split32.chunk_blocks;
            split_params[26] = @bitCast(1 / @sqrt(@as(f32, @floatFromInt(p.width))));
            var pipes: [6]mtl.Pipeline = undefined;
            for (p.split32_passes, &pipes) |pipe, *value| value.* = pipe.?;
            sources.sam3_attention_split32.encode(mtl, pass.e, pipes, refs, split_params);
            return p.rig.finish(pass);
        }
        if (p.device) {
            const Binding = struct { buffer: mtl.Buffer, offset: usize };
            var refs: [6]Binding = undefined;
            for ([_]rig_.Ref{ p.ins[0], p.ins[1], p.ins[2], p.out, p.work[0], p.work[1] }, &refs) |r, *binding|
                binding.* = .{ .buffer = r.buf, .offset = r.off };
            var device_params: [25]i32 = undefined;
            @memcpy(device_params[0..24], &params);
            device_params[24] = @bitCast(1 / @sqrt(@as(f32, @floatFromInt(p.width))));
            var pipes: [6]mtl.Pipeline = undefined;
            for (p.device_passes, &pipes) |pipe, *value| value.* = pipe.?;
            sources.sam3_attention_device.encode(mtl, pass.e, pipes, refs, device_params, !p.control);
            return p.rig.finish(pass);
        }
        if (p.split) {
            const Binding = struct { buffer: mtl.Buffer, offset: usize };
            var refs: [7]Binding = undefined;
            for ([_]rig_.Ref{ p.ins[0], p.ins[1], p.ins[2], p.work[0], p.work[1], p.work[2], p.out }, &refs) |r, *binding|
                binding.* = .{ .buffer = r.buf, .offset = r.off };
            var split_params: [27]i32 = undefined;
            @memcpy(split_params[0..24], &params);
            const chunk_blocks = sources.sam3_attention_split.chunk_blocks;
            split_params[24] = @intCast(((s.k + 31) / 32 + chunk_blocks - 1) / chunk_blocks);
            split_params[25] = chunk_blocks;
            split_params[26] = @bitCast(@as(f32, 0.0625));
            sources.sam3_attention_split.encode(mtl, pass.e, .{ p.split_passes[0].?, p.split_passes[1].?, p.split_passes[2].? }, refs, split_params);
            try p.rig.finish(pass);
            return;
        }
        if (p.two_pass and !baseline) {
            const Binding = struct { buffer: mtl.Buffer, offset: usize };
            var refs: [6]Binding = undefined;
            for ([_]rig_.Ref{ p.ins[0], p.ins[1], p.ins[2], p.work[0], p.work[1], p.out }, &refs) |r, *binding|
                binding.* = .{ .buffer = r.buf, .offset = r.off };
            sources.tensor_attention_passes.encode(mtl, pass.e, p.passes[0].?, p.passes[1].?, refs, params, 0.0625);
            try p.rig.finish(pass);
            return;
        }
        pass.e.setPipeline(p.pipe);
        for (p.ins, 0..) |r, i| pass.e.setBuffer(r.buf, r.off, i);
        pass.e.setValue(params, 3);
        pass.e.setValue([4]f32{ 0.0625, 0, 0, 0 }, 4);
        const output = if (baseline) p.base else p.out;
        pass.e.setBuffer(output.buf, output.off, 5);
        pass.e.dispatchThreads(mtl.Size.of(nq * 32, nheads * 4, p.batches() * 2), mtl.Size.of(32, 4, 1));
        try p.rig.finish(pass);
    }

    pub fn run(p: *Probe) !void {
        const before = try p.a.dupe(u8, try p.rig.inputBytes());
        defer p.a.free(before);
        try p.launch(false);
        if (p.shape.rows > 64) {
            const first = try p.a.dupe(u8, p.rig.bytes(p.out, p.count() * 4));
            defer p.a.free(first);
            for (1..5) |_| {
                try p.launch(false);
                if (!std.mem.eql(u8, first, p.rig.bytes(p.out, first.len))) return error.AttentionNotRepeatable;
            }
        }
        if (!std.mem.eql(u8, before, try p.rig.inputBytes())) return error.AttentionInputsChanged;
        if (!try p.rig.intact()) return error.AttentionGuardsChanged;
    }

    pub fn replaced(p: *Probe) ![]const u8 {
        try p.launch(true);
        return p.rig.bytes(p.base, p.count() * 4);
    }

    pub fn count(p: *const Probe) usize {
        return p.batches() * p.shape.rows * p.shape.n;
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const s = p.shape;
        const width = p.width;
        const kv = try p.a.alloc(f64, 2 * s.k * width + 2 * s.k);
        defer p.a.free(kv);
        const keys = kv[0 .. s.k * width];
        const values = kv[s.k * width ..][0 .. s.k * width];
        const weights = kv[2 * s.k * width ..][0..s.k];
        const tau = kv[2 * s.k * width + s.k ..];
        var query: [dim]f64 = undefined;
        const blocks: f64 = @floatFromInt((s.k + 31) / 32);
        const chunk: usize = if (p.device and p.split) sources.sam3_attention_split32.chunk_blocks else sources.sam3_attention_split.chunk_blocks;
        const chunk_blocks: f64 = @min(blocks, @as(f64, @floatFromInt(chunk)));
        const splits: f64 = @ceil(blocks / @as(f64, @floatFromInt(chunk)));
        const form = ar.Form{
            .score = if (p.device) @max(16 + 6 * 4, 4 * 6 * ar.mma_step) else @as(f64, @floatFromInt(width / 16 * 6 * ar.mma_step)),
            .score_round = 4,
            .fast_exp = false,
            .chunk_exp = false,
            .p_rel = 0,
            .num = if (p.device) @max(44, 12 * ar.mma_step + 1) * (if (p.split) chunk_blocks else blocks) + (if (p.split) 10 * splits else 0) else if (p.split) (12 * ar.mma_step + 1) * chunk_blocks + 10 * splits else (12 * ar.mma_step + 1) * blocks,
            .den = if (p.device) 80 * (if (p.split) chunk_blocks else blocks) + (if (p.split) 8 * splits else 0) else if (p.split) 3 * chunk_blocks + 3 + 8 * splits else 3 * blocks + 3,
            .div = 4,
            .exp_ulps = .{ if (p.split) 4 * chunk_blocks + 4 else 4 * blocks, 2 * std.math.log2e + 0.5 },
        };
        for (0..p.batches()) |b| for (0..p.heads()) |h| {
            for (0..s.k) |j| for (0..width) |d| {
                const src = (b * s.k + j) * s.n + h * width + d;
                keys[j * width + d] = p.k[src];
                values[j * width + d] = p.v[src];
            };
            for (0..s.rows) |r| {
                const at = (b * s.rows + r) * s.n + h * width;
                if (r >= period) {
                    const from = (b * s.rows + r % period) * s.n + h * width;
                    @memcpy(want[at..][0..width], want[from..][0..width]);
                    @memcpy(p.half[at..][0..width], p.half[from..][0..width]);
                    continue;
                }
                for (query[0..width], p.q[at..][0..width]) |*x, q| x.* = q;
                ar.head(query[0..width], keys, values, s.k, width, 1 / @sqrt(@as(f64, @floatFromInt(width))), form, weights, tau, want[at..][0..width], p.half[at..][0..width]);
            }
        };
        @memcpy(mag, p.half);
    }

    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        for (want, p.half, allow) |w, half, *v| v.* = if (p.stress) c.boundExact(p.spec(), w) else half;
    }

    pub fn intervals(p: *Probe, low: []f64, high: []f64) !void {
        @memset(low, -std.math.floatMax(f64));
        @memset(high, std.math.floatMax(f64));
        if (!p.device or p.fill != .random or p.shape.k != 1) return;
        for (0..p.batches()) |b| for (0..p.shape.rows) |r| for (0..p.shape.n) |d| {
            const at = (b * p.shape.rows + r) * p.shape.n + d;
            low[at] = p.v[b * p.shape.n + d];
            high[at] = low[at];
        };
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try p.a.alloc(f64, p.shape.k);
        defer p.a.free(terms);
        for (terms, 0..) |*x, j| x.* = stressValue(j, terms.len);
        const result = contract.rows.sumIn(acc, order, terms) / @as(f64, @floatFromInt(terms.len));
        @memset(out, result);
    }

    fn deinit(p: *Probe) void {
        p.reset();
        p.rig.deinit();
        p.pipe.deinit();
        for (p.passes) |maybe| if (maybe) |pipe| pipe.deinit();
        for (p.split_passes) |maybe| if (maybe) |pipe| pipe.deinit();
        for (p.device_passes) |maybe| if (maybe) |pipe| pipe.deinit();
        for (p.split32_passes) |maybe| if (maybe) |pipe| pipe.deinit();
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: [23]?*Probe = @splat(null),

    pub fn collect(s: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        if (!device.tensorUnits()) return;
        for (0..23) |i| {
            const device_attention = i >= 11;
            const device_width: usize = if (i >= 14 and i < 20) 32 else 64;
            const split = (i >= 8 and i < 11) or i >= 17;
            const batched = i == 1 or i == 4 or i == 6 or i == 9 or i == 12 or i == 15 or i == 18 or i == 21;
            const control = i == 2 or i == 7 or i == 10 or i == 13 or i == 16 or i == 19 or i == 22;
            const two_pass = i >= 3 and i < 8;
            const replacement = i == 5 or i == 6;
            if (only) |needle| if (std.mem.indexOf(u8, label(batched, control, two_pass, replacement, split, device_attention, device_width), needle) == null) continue;
            const body = if (device_attention and split) try split32Text(s.a, control) else if (device_attention) try text(s.a, false) else if (split) try splitText(s.a, control) else if (two_pass) try twoPassText(s.a, control) else try text(s.a, control);
            defer s.a.free(body);
            const lib = try mtl.Library.fromSource(device, body, mtl.CompileOptions.mlx());
            defer lib.deinit();
            const pipe = try mtl.Pipeline.init(device, lib, if (device_attention) (if (device_width == 32) "im_attention_f32_full32" else "im_attention_f32_full64") else "im_attention_f32_full256", false);
            errdefer pipe.deinit();
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            try kernels.ensureUnusedCapacity(s.a, 1);
            const scores = if (two_pass) try mtl.Pipeline.init(device, lib, "im_attention_scores256", false) else null;
            errdefer if (scores) |score| score.deinit();
            const values = if (two_pass) try mtl.Pipeline.init(device, lib, "im_attention_values256", false) else null;
            errdefer if (values) |value| value.deinit();
            var split_passes: [3]?mtl.Pipeline = @splat(null);
            errdefer for (split_passes) |maybe| if (maybe) |value| value.deinit();
            if (split and !device_attention) for ([_][]const u8{ "sam_attention_split_scores", "sam_attention_split_values", "sam_attention_split_merge" }, &split_passes) |name, *slot| {
                slot.* = try mtl.Pipeline.init(device, lib, name, false);
            };
            p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = device, .queue = queue }, .pipe = pipe, .batched = batched, .control = control, .two_pass = two_pass, .replacement = replacement, .split = split, .passes = .{ scores, values }, .split_passes = split_passes, .device = device_attention, .width = if (device_attention) device_width else dim };
            const device_names = if (device_width == 32) sources.sam3_attention_device.names32 else sources.sam3_attention_device.names;
            if (device_attention and !split) for (device_names, &p.device_passes) |name, *slot| {
                slot.* = try mtl.Pipeline.init(device, lib, name, false);
            };
            const split_device_names = if (device_width == 32) sources.sam3_attention_split32.names else sources.sam3_attention_split32.names64;
            if (device_attention and split) for (split_device_names, &p.split32_passes) |name, *slot| {
                slot.* = try mtl.Pipeline.init(device, lib, name, false);
            };
            s.probes[i] = p;
            kernels.appendAssumeCapacity(.{ .spec = p.spec(), .ctx = p, .vt = &vt, .control = if (control) .shapes else null });
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes) |maybe| if (maybe) |p| {
            p.deinit();
            s.a.destroy(p);
        };
    }
};
