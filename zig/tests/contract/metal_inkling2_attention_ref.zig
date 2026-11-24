//! Float64 references and frozen main baselines for Inkling 2 Mini Metal attention gates 1-4.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const controls = @import("metal_inkling2_attention_controls.zig");
const nax_src = @import("kernel_sources").nax;
const plan = @import("tensorfold").inkling2_metal_host.attention.planning;
const c = contract.spec;
const g = contract.gates;
const Which = controls.Which;

pub const D = 128;
pub const H = 24;
pub const KV = 8;
pub const R = 16;
pub const EXTENT = 1024;
pub const scale = 0.08838834764831845;
const base_attn_src = @embedFile("baselines/inkling2_attention_main.metal.txt");
const base_prompt_src = @embedFile("baselines/inkling2_prompt_attention_main.metal.txt");

pub const Sim = struct { acc: c.Accum, order: g.Order };
pub const Inputs = struct {
    which: Which,
    shape: c.Shape,
    layer: plan.Layer,
    q: []const u16,
    kc: []const u16,
    vc: []const u16,
    sink: []const u16,
    gate: []const u16,
    rel: []const u16,
    proj: []const u16,
    qs: []const u16,
    rs: []const u16,
    pos: []const usize,
    sizes: [4]usize,
};

pub fn bf(x: u16) f64 {
    return c.fromBf16(x);
}
pub fn rn(x: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(x)));
}
pub fn bits(x: f64) u16 {
    return c.toBf16(@floatCast(x));
}
fn minNum(x: f64, y: f64) f64 {
    return if (std.math.isNan(x)) y else if (std.math.isNan(y)) x else @min(x, y);
}
fn maxNan(x: f64, y: f64) f64 {
    return if (std.math.isNan(x) or std.math.isNan(y)) std.math.nan(f64) else @max(x, y);
}
fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}
pub fn tau(pos: usize) f64 {
    return 1 + 0.1 * @log(@max(@as(f64, @floatFromInt(pos + 1)) / 8192, 1));
}
pub fn periodic(p: usize, col: usize) f64 {
    const i = p % 1024;
    if (i < 512) return @floatFromInt(129 + (i * 73 + col * 31) % 127);
    const j = i - 512;
    return -@as(f64, @floatFromInt(129 + ((j + 1) * 73 + col * 31) % 127)) + @as(f64, if (j < 8) 1 else 0);
}
fn sum(v: []const f64, sim: ?Sim) f64 {
    if (sim) |s| return contract.rows.sumIn(s.acc, s.order, v);
    var total: f64 = 0;
    for (v) |x| total += x;
    return total;
}
/// Prompt P rounds to BF16 at each tile maximum; later rescaling does not round twice. The result anchors weights.
fn promptWeights(score: []const f64, weights: []f64) f64 {
    const tile_keys = 64;
    const sub_keys = 1024;
    var top = -std.math.inf(f64);
    var upper = -std.math.inf(f64);
    var first: usize = 0;
    while (first < score.len) {
        const end = @min(first + tile_keys, score.len);
        const sub_first = first - first % sub_keys;
        var tile_peak = top;
        for (score[first..end]) |sc| tile_peak = maxNan(tile_peak, sc);
        const mu = if (tile_peak == -std.math.inf(f64)) 0 else tile_peak;
        const keep = @exp(top - mu);
        for (weights[sub_first..first]) |*w| w.* *= keep;
        for (weights[first..end], score[first..end]) |*w, sc| w.* = rn(@exp(sc - mu));
        top = tile_peak;
        if (first % sub_keys == sub_keys - tile_keys or end == score.len) {
            if (top != -std.math.inf(f64)) {
                const merged = maxNan(upper, top);
                const old_scale = @exp(upper - merged);
                const new_scale = @exp(top - merged);
                for (weights[0..sub_first]) |*w| w.* *= old_scale;
                for (weights[sub_first..end]) |*w| w.* *= new_scale;
                upper = merged;
            }
            top = -std.math.inf(f64);
        }
        first = end;
    }
    return upper;
}
fn headClip(x: []const f64, threshold: f64, sim: ?Sim, t: ?f64, want: []f64, mag: []f64) void {
    var sq: [D]f64 = undefined;
    @memset(mag, 0);
    for (sq[0..x.len], x, 0..) |*v, y, i| {
        v.* = y * y;
        mag[i] = @abs(y);
    }
    const f = minNum(threshold / (@sqrt(sum(sq[0..x.len], sim)) + 1e-6), 1);
    for (want, x, mag) |*w, v, *m| {
        w.* = rn(v * f);
        if (t) |tv| w.* = rn(w.* * tv);
        m.* = @abs(v) + v * v;
    }
}
/// prep uses independent K/V and attention uses populated ring caches; q and rel never alias either one.
pub fn prepRef(in: Inputs, sim: ?Sim, want: []f64, mag: []f64) !void {
    @memset(want, 0);
    @memset(mag, 0);
    const z = in.sizes;
    var x: [D]f64 = undefined;
    for (0..in.shape.rows) |r| {
        const tv: ?f64 = if (in.layer.tau) tau(in.pos[r]) else null;
        for (0..H) |h| {
            for (&x, 0..) |*v, d| v.* = bf(in.q[(r * H + h) * D + d]) * bf(in.qs[h * D + d]);
            headClip(&x, 22.627416997969522, sim, tv, want[(r * H + h) * D ..][0..D], mag[(r * H + h) * D ..][0..D]);
            for (x[0..R], 0..) |*v, d| v.* = bf(in.rel[(r * H + h) * R + d]) * bf(in.rs[h * R + d]);
            const at = z[0] + (r * H + h) * R;
            headClip(x[0..R], 8, sim, tv, want[at..][0..R], mag[at..][0..R]);
        }
        for (0..KV) |h| {
            const slot = (h * in.layer.ring + in.pos[r] % in.layer.ring) * D;
            for (&x, 0..) |*v, d| v.* = bf(in.kc[(r * KV + h) * D + d]);
            headClip(&x, 22.627416997969522, sim, null, want[z[0] + z[1] + slot ..][0..D], mag[z[0] + z[1] + slot ..][0..D]);
            var sq: [D]f64 = undefined;
            for (&x, &sq, 0..) |*v, *ss, d| {
                v.* = bf(in.vc[(r * KV + h) * D + d]);
                ss.* = v.* * v.*;
            }
            const inv = 1 / @sqrt(sum(&sq, sim) / D + 1e-6);
            for (0..D) |d| {
                const at = z[0] + z[1] + z[2] + slot + d;
                want[at] = rn(x[d] * inv);
                mag[at] = sq[d] + @abs(x[d]);
            }
        }
    }
}
pub fn attentionRef(a: std.mem.Allocator, in: Inputs, sim: ?Sim, want: []f64, mag: []f64) !void {
    const most = in.shape.k;
    const score = try a.alloc(f64, most);
    defer a.free(score);
    const weights = try a.alloc(f64, most);
    defer a.free(weights);
    const terms = try a.alloc(f64, @max(most, D));
    defer a.free(terms);
    for (0..in.shape.rows) |r| {
        for (0..H) |h| {
            const rh = r * H + h;
            const pos = in.pos[r];
            const lo = if (in.layer.window > 0) (pos + 1) -| in.layer.window else 0;
            const n = pos + 1 - lo;
            const kvh = h / (H / KV);
            const z = bf(in.sink[rh]);
            // Prompt P is BF16 at the keys' maximum; the sink joins in the final merge using the shared maximum.
            const prompt = sim == null and controls.isPrompt(in.which);
            var peak = if (prompt) -std.math.inf(f64) else z;
            for (lo..pos + 1) |kp| {
                const j = kp - lo;
                const at = (kvh * in.layer.ring + kp % in.layer.ring) * D;
                for (0..D) |d| terms[d] = bf(in.q[rh * D + d]) * bf(in.kc[at + d]);
                const distance = pos - kp;
                var bias: f64 = 0;
                if (distance < EXTENT) {
                    var rterms: [R]f64 = undefined;
                    for (&rterms, 0..) |*v, rj| v.* = bf(in.rel[rh * R + rj]) * bf(in.proj[rj * EXTENT + distance]);
                    bias = rn(sum(&rterms, sim));
                }
                score[j] = sum(terms[0..D], sim) * scale + bias;
                peak = maxNan(peak, score[j]);
            }
            if (prompt) {
                peak = promptWeights(score[0..n], weights[0..n]);
            } else for (weights[0..n], score[0..n]) |*w, sc| w.* = @exp(sc - peak);
            const merged_peak = if (prompt) maxNan(peak, z) else peak;
            const rescale = if (prompt) @exp(peak - merged_peak) else 1;
            const den = sum(weights[0..n], sim) * rescale + @exp(z - merged_peak);
            for (0..D) |d| {
                for (lo..pos + 1) |kp| {
                    const j = kp - lo;
                    const at = (kvh * in.layer.ring + kp % in.layer.ring) * D;
                    terms[j] = weights[j] * bf(in.vc[at + d]);
                }
                const num = sum(terms[0..n], sim) * rescale;
                const y = num / den * sigmoid(bf(in.gate[rh * D + d]));
                want[rh * D + d] = if (sim != null) rn(y) else y;
                mag[rh * D + d] = (@abs(num) + den + 1) * (1 + @abs(y));
                for (terms[0..n]) |v| mag[rh * D + d] += @abs(v * rescale);
            }
        }
    }
}
pub fn encodeMerge(e: mtl.ComputeEncoder, pipe: mtl.Pipeline, po: rig_.Ref, pml: rig_.Ref, sink: rig_.Ref, gate: rig_.Ref, pos: rig_.Ref, out: rig_.Ref, layer: plan.Layer, rows: u32, stride: u32) void {
    e.setPipeline(pipe);
    e.setBuffer(po.buf, po.off, 0);
    e.setBuffer(pml.buf, pml.off, 1);
    e.setBuffer(sink.buf, sink.off, 2);
    e.setBuffer(gate.buf, gate.off, 3);
    e.setBuffer(pos.buf, pos.off, 4);
    e.setBuffer(out.buf, out.off, 5);
    e.setValue(plan.MergeArgs{ .heads = H, .window = layer.window, .chunk = layer.chunk, .c0 = 0, .chunks = stride }, 6);
    e.dispatchGroups(mtl.Size.of(rows * H, 1, 1), mtl.Size.of(D, 1, 1));
}
pub const Baseline = struct {
    scalar_lib: mtl.Library,
    prompt_lib: ?mtl.Library = null,
    rel_table: mtl.Pipeline,
    attn: mtl.Pipeline,
    merge: mtl.Pipeline,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, w: Which) !Baseline {
        var scalar_lib = try mtl.Library.fromSource(device, base_attn_src, mtl.CompileOptions.mlx());
        errdefer scalar_lib.deinit();
        var rel_table = try mtl.Pipeline.init(device, scalar_lib, "tf_ink2_rel_table", false);
        errdefer rel_table.deinit();
        var merge = try mtl.Pipeline.init(device, scalar_lib, "tf_ink2_merge", false);
        errdefer merge.deinit();
        var prompt_lib: ?mtl.Library = null;
        errdefer if (prompt_lib) |loaded| loaded.deinit();
        if (controls.isPrompt(w)) {
            const prefix = if (controls.isPreM5(w)) "#define TF_SIMD_FRAGS 1\n" else "";
            const p_src = try std.mem.concat(a, u8, &.{ prefix, nax_src, base_prompt_src });
            defer a.free(p_src);
            prompt_lib = try mtl.Library.fromSource(device, p_src, mtl.CompileOptions.mlx());
        }
        const attn = try mtl.Pipeline.init(device, prompt_lib orelse scalar_lib, if (controls.isPrompt(w)) "tf_ink2_attn_prompt" else "tf_ink2_attn_scalar", false);
        return .{ .scalar_lib = scalar_lib, .prompt_lib = prompt_lib, .rel_table = rel_table, .attn = attn, .merge = merge };
    }
    pub fn deinit(b: *Baseline) void {
        b.attn.deinit();
        b.merge.deinit();
        b.rel_table.deinit();
        if (b.prompt_lib) |loaded| loaded.deinit();
        b.scalar_lib.deinit();
    }
    pub fn run(b: *const Baseline, rig: *rig_.Rig, which: Which, layer: plan.Layer, shape: c.Shape, refs: []const rig_.Ref, meta: []const rig_.Ref, tab: rig_.Ref, po: rig_.Ref, pml: rig_.Ref, out: rig_.Ref) ![]const u8 {
        const rows: u32 = @intCast(shape.rows);
        const stride = plan.slotsFor(layer, shape.k);
        const out_bytes = shape.rows * H * D * 2;
        @memset(rig.bytes(tab, shape.rows * H * EXTENT * 2), 0xff);
        @memset(rig.bytes(out, out_bytes), 0xff);
        const pass = rig.begin();
        const e = pass.e;
        e.setPipeline(b.rel_table);
        e.setBuffer(refs[3].buf, refs[3].off, 0);
        e.setBuffer(refs[4].buf, refs[4].off, 1);
        e.setBuffer(tab.buf, tab.off, 2);
        e.dispatchGroups(mtl.Size.of(rows * H, 1, 1), mtl.Size.of(256, 1, 1));
        e.barrier();
        e.setPipeline(b.attn);
        for (refs[0..3], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setBuffer(tab.buf, tab.off, 3);
        e.setBuffer(meta[2].buf, meta[2].off, 4);
        e.setBuffer(po.buf, po.off, 5);
        e.setBuffer(pml.buf, pml.off, 6);
        const sc: f32 = 1.0 / @sqrt(@as(f32, D));
        if (controls.isPrompt(which)) {
            e.setValue(plan.PromptArgs{ .heads = H, .kv_heads = KV, .ring = layer.ring, .chunk = layer.chunk, .c0 = 0, .chunks = stride, .scale = sc }, 7);
            const blocks = (rows * (H / KV) + plan.prompt_pairs - 1) / plan.prompt_pairs;
            e.dispatchGroups(mtl.Size.of(blocks, KV, stride), mtl.Size.of(256, 1, 1));
        } else {
            e.setBuffer(meta[0].buf, meta[0].off, 7);
            e.setBuffer(meta[1].buf, meta[1].off, 8);
            e.setValue(plan.AttnArgs{ .heads = H, .kv_heads = KV, .ring = layer.ring, .window = layer.window, .chunk = layer.chunk, .c0 = 0, .chunks = stride, .scale = sc }, 9);
            e.dispatchGroups(mtl.Size.of(rows * H, stride, 1), mtl.Size.of(32, 1, 1));
        }
        e.barrier();
        encodeMerge(e, b.merge, po, pml, refs[5], refs[6], meta[0], out, layer, rows, stride);
        try rig.finish(pass);
        return rig.bytes(out, out_bytes);
    }
};
