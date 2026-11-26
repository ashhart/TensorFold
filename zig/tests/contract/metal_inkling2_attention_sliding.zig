//! Raw-byte 0-ULP parity between tf_ink2_attn_sliding_prompt and tf_ink2_attn_scalar (partials + merged output).
const std = @import("std");
const mtl = @import("metal");
const rig_ = @import("rig.zig");
const ref = @import("metal_inkling2_attention_ref.zig");
const sources = @import("inkling2_metal");
const plan = @import("tensorfold").inkling2_metal_host.attention.planning;

const D = ref.D;
const H = ref.H;
const KV = ref.KV;
const R = ref.R;
const EXTENT = ref.EXTENT;

const Case = struct { rows: usize, k: usize, window: u32, label: []const u8 };
const cases = [_]Case{
    .{ .rows = 2, .k = 1030, .window = 1024, .label = "rows=2 window-edge" },
    .{ .rows = 3, .k = 1025, .window = 1024, .label = "rows=3 odd-tail" },
    .{ .rows = 4, .k = 2049, .window = 1024, .label = "rows=4 ring-wrap" },
    .{ .rows = 8, .k = 1030, .window = 1024, .label = "rows=8 window-edge" },
    .{ .rows = 16, .k = 1156, .window = 1024, .label = "rows=16 ring-wrap" },
    .{ .rows = 17, .k = 1600, .window = 1536, .label = "rows=17 partial-rel-extent" },
    .{ .rows = 32, .k = 2070, .window = 1024, .label = "rows=32 multi-chunk" },
    .{ .rows = 64, .k = 500, .window = 1024, .label = "rows=64 pos<window" },
    .{ .rows = 65, .k = 1090, .window = 1024, .label = "rows=65 tail-row-and-wrap" },
    .{ .rows = 128, .k = 2170, .window = 1024, .label = "rows=128 multi-tg" },
    .{ .rows = 256, .k = 2256, .window = 1024, .label = "rows=256 full-prefill" },
};

const RunResult = struct {
    part_o: []u8,
    part_ml: []u8,
    out: []u8,

    fn deinit(self: RunResult, a: std.mem.Allocator) void {
        a.free(self.part_o);
        a.free(self.part_ml);
        a.free(self.out);
    }
};

fn fillRandomBf16(rng: std.Random, buf: []u16, raw_bits: bool, scale_dev: f64) void {
    for (buf, 0..) |*v, i| {
        if (raw_bits and (i % 7 == 0)) {
            // Mix arbitrary finite BF16 encodings (subnormals, negative zero, varied exponents) into random inputs.
            const sign: u16 = @intCast((rng.int(u16) & 1) << 15);
            const exp: u16 = @intCast(rng.uintLessThan(u16, 134));
            const mant: u16 = @intCast(rng.int(u16) & 0x7f);
            v.* = sign | (exp << 7) | mant;
        } else {
            v.* = ref.bits(rng.floatNorm(f64) * scale_dev);
        }
    }
}

fn runKernel(
    a: std.mem.Allocator,
    rig: *rig_.Rig,
    pipe: mtl.Pipeline,
    merge: mtl.Pipeline,
    sliding_prompt: bool,
    layer: plan.Layer,
    rows: usize,
    p0: usize,
    refs: [7]rig_.Ref,
    pos32: []const i32,
) !RunResult {
    const stride = plan.slotsFor(layer, p0 + rows);
    const rr = plan.Rows{ .rows = @intCast(rows), .heads = H, .kv_heads = KV, .last_pos = @intCast(p0 + rows - 1), .streams = 1 };
    const launch = if (sliding_prompt)
        try plan.attend(rr, layer, stride, 0)
    else
        plan.Launch{
            .path = .decode,
            .grid = .{ rr.rows * H, stride, 1 },
            .threads = plan.decode_threads,
            .merge_grid = .{ rr.rows * H, 1, 1 },
            .partials_rh = stride,
        };
    if (sliding_prompt and launch.path != .sliding_prompt) return error.ExpectedSlidingPromptPath;
    const pos_ref = try rig.input(std.mem.sliceAsBytes(pos32));
    const seg_of32 = try a.alloc(i32, rows);
    defer a.free(seg_of32);
    @memset(seg_of32, 0);
    const seg_of_ref = try rig.input(std.mem.sliceAsBytes(seg_of32));
    const seg = [1]plan.Seg{.{ .row0 = 0, .n = @intCast(rows), .p0 = @intCast(p0), .off = 0 }};
    const seg_ref = try rig.input(std.mem.asBytes(&seg));
    const po_bytes = rows * H * stride * D * 4;
    const pml_bytes = rows * H * stride * 8;
    const out_bytes = rows * H * D * 2;
    const po = try rig.scratch(po_bytes);
    const pml = try rig.scratch(pml_bytes);
    const out = try rig.output(out_bytes);
    const pass = rig.begin();
    const e = pass.e;
    e.setPipeline(pipe);
    for (refs[0..5], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setBuffer(seg_ref.buf, seg_ref.off, 5);
    e.setBuffer(po.buf, po.off, 6);
    e.setBuffer(pml.buf, pml.off, 7);
    e.setBuffer(pos_ref.buf, pos_ref.off, 8);
    e.setBuffer(seg_of_ref.buf, seg_of_ref.off, 9);
    const sc: f32 = 1.0 / @sqrt(@as(f32, D));
    e.setValue(plan.AttnArgs{ .heads = H, .kv_heads = KV, .ring = layer.ring, .window = layer.window, .chunk = layer.chunk, .c0 = launch.c0, .chunks = launch.partials_rh, .scale = sc }, 10);
    e.dispatchGroups(mtl.Size.of(launch.grid[0], launch.grid[1], launch.grid[2]), mtl.Size.of(launch.threads, 1, 1));
    e.barrier();
    ref.encodeMerge(e, merge, po, pml, refs[5], refs[6], pos_ref, out, layer, @intCast(rows), launch.partials_rh);
    try rig.finish(pass);
    return .{
        .part_o = try a.dupe(u8, rig.bytes(po, po_bytes)),
        .part_ml = try a.dupe(u8, rig.bytes(pml, pml_bytes)),
        .out = try a.dupe(u8, rig.bytes(out, out_bytes)),
    };
}

fn expectEqualBytes(label: []const u8, kind: []const u8, want: []const u8, got: []const u8) !void {
    if (!std.mem.eql(u8, want, got)) {
        const at = std.mem.indexOfDiff(u8, want, got).?;
        std.debug.print("FAIL sliding prompt 0-ULP {s} ({s}): byte {d} differs (want=0x{x:0>2}, got=0x{x:0>2})\n", .{ label, kind, at, want[at], got[at] });
        return error.SlidingParityFailed;
    }
}

pub fn run(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, seed: u64) !void {
    const lib = try mtl.Library.fromSource(device, sources.attention, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const scalar_pipe = try mtl.Pipeline.init(device, lib, "tf_ink2_attn_scalar", false);
    defer scalar_pipe.deinit();
    const slide_pipe = try mtl.Pipeline.init(device, lib, "tf_ink2_attn_sliding_prompt", false);
    defer slide_pipe.deinit();
    const merge_pipe = try mtl.Pipeline.init(device, lib, "tf_ink2_merge", false);
    defer merge_pipe.deinit();

    var prng = std.Random.DefaultPrng.init(seed ^ 0x51a7);
    const rng = prng.random();
    var rig = rig_.Rig{ .a = a, .device = device, .queue = queue };
    defer rig.deinit();

    for (cases) |cs| {
        const layer = plan.Layer.sliding(cs.window, @intCast(cs.rows));
        const rh = cs.rows * H;
        const cache = KV * layer.ring * D;
        const q = try a.alloc(u16, rh * D);
        defer a.free(q);
        const kc = try a.alloc(u16, cache);
        defer a.free(kc);
        const vc = try a.alloc(u16, cache);
        defer a.free(vc);
        const rel = try a.alloc(u16, rh * R);
        defer a.free(rel);
        const proj = try a.alloc(u16, R * EXTENT);
        defer a.free(proj);
        const sink = try a.alloc(u16, rh);
        defer a.free(sink);
        const gate = try a.alloc(u16, rh * D);
        defer a.free(gate);
        const pos32 = try a.alloc(i32, cs.rows);
        defer a.free(pos32);
        const p0 = cs.k - cs.rows;
        for (pos32, 0..) |*v, i| v.* = @intCast(p0 + i);

        for ([_]bool{ false, true }) |nonfinite| {
            rig.reset();
            fillRandomBf16(rng, q, !nonfinite, 0.5);
            fillRandomBf16(rng, kc, !nonfinite, 0.5);
            fillRandomBf16(rng, vc, !nonfinite, 1.0);
            fillRandomBf16(rng, rel, !nonfinite, 1.0);
            fillRandomBf16(rng, proj, !nonfinite, 0.5);
            for (sink) |*v| v.* = ref.bits(2 + 4 * rng.float(f64));
            fillRandomBf16(rng, gate, !nonfinite, 1.0);
            if (nonfinite) {
                const nan = ref.bits(std.math.nan(f64));
                const inf = ref.bits(std.math.inf(f64));
                q[D + 5] = nan;
                kc[7] = inf;
                vc[D + 3] = nan;
                sink[2] = inf;
                gate[D] = inf;
                gate[D + 1] = inf | 0x8000;
                rel[R + 2] = nan;
                rel[5 * R + 1] = inf;
            }
            const arrays = [_][]u16{ q, kc, vc, rel, proj, sink, gate };
            var refs: [7]rig_.Ref = undefined;
            for (arrays, 0..) |arr, i| refs[i] = try rig.input(std.mem.sliceAsBytes(arr));

            const want = try runKernel(a, &rig, scalar_pipe, merge_pipe, false, layer, cs.rows, p0, refs, pos32);
            defer want.deinit(a);
            const got = try runKernel(a, &rig, slide_pipe, merge_pipe, true, layer, cs.rows, p0, refs, pos32);
            defer got.deinit(a);

            try expectEqualBytes(cs.label, "part_ml", want.part_ml, got.part_ml);
            try expectEqualBytes(cs.label, "part_o", want.part_o, got.part_o);
            try expectEqualBytes(cs.label, "out", want.out, got.out);
        }
    }
    std.debug.print("PASS sliding prompt 0-ULP (part_ml + part_o + out raw-byte equal to scalar, random + nonfinite): {d} geometries\n", .{cases.len});
}
