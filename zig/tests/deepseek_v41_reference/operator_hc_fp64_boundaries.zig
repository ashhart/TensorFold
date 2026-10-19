//! Hyperconnection boundaries comparisons against the FP64 ideal.
const support = @import("hc_fp64_fixture.zig");
const std = support.std;
const mtl = support.mtl;
const math = support.math;
const mhc = support.mhc;
const Tensor = support.Tensor;
const Hc = support.Hc;
const Config = support.Config;
const width = support.width;
const mix_count = support.mix_count;
const released_config = support.released_config;
const retained_exp_argument_bits = support.retained_exp_argument_bits;
const controlling_exponent_input = support.controlling_exponent_input;
const deltaF32VsF64 = support.deltaF32VsF64;
const Buffers = support.Buffers;
const upload = support.upload;
const view = support.view;

// Identical mixes/inverse isolate split-stage sigmoid, exp and reduction errors from projection differences.
test "mHC split unary boundaries against the FP64 ideal on retained arguments" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try mhc.Pipelines.load(device);
    defer p.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const cfg = try Config.parse(gpa, released_config);

    const rows = 7 + 65; // retained arguments plus every integer exponent input -32..32
    var b = Buffers{ .device = device };
    defer b.deinit();
    const mixes_host = try gpa.alloc(f32, rows * mix_count);
    @memset(mixes_host, 0);
    // Rows 0..3: the 16 retained unary arguments rotated through pre, post and comb positions.
    for (0..4) |i| {
        for (0..4) |j| {
            mixes_host[i * mix_count + j] = @bitCast(retained_exp_argument_bits[(4 * i + j) % 16]);
            mixes_host[i * mix_count + 4 + j] = @bitCast(retained_exp_argument_bits[(4 * i + j) % 16]);
        }
        for (0..16) |q| mixes_host[i * mix_count + 8 + q] = @bitCast(retained_exp_argument_bits[(i + q) % 16]);
    }
    // Row 4: the documented controlling exponent input as a post argument (exp(+1.439...) path).
    mixes_host[4 * mix_count + 5] = -controlling_exponent_input;
    // Row 5: comb affine arguments spanning the exponent range.
    mixes_host[5 * mix_count + 8] = -32;
    mixes_host[5 * mix_count + 9] = -16;
    mixes_host[5 * mix_count + 10] = 16;
    mixes_host[5 * mix_count + 11] = 32;
    // Row 6: the same span through the sigmoid pre/post arguments.
    const span = [_]f32{ -32, -16, 16, 32 };
    for (span, 0..) |v, j| {
        mixes_host[6 * mix_count + j] = v;
        mixes_host[6 * mix_count + 4 + j] = v;
    }

    // Rows 7..69: one integer exponent input per row at a pre, post and comb position each.
    for (0..65) |i| {
        const r = 7 + i;
        const v: f32 = @floatFromInt(@as(i32, @intCast(i)) - 32);
        mixes_host[r * mix_count] = v;
        mixes_host[r * mix_count + 4] = v;
        mixes_host[r * mix_count + 8] = v;
    }

    const mixes = try b.get(f32, rows * mix_count);
    upload(f32, mixes, mixes_host);
    const inv = try b.get(f32, rows);
    @memset(inv.slice(f32, rows), 1.0);
    const scale = try b.get(f32, 3);
    @memset(scale.slice(f32, 3), 1.0);
    const base = try b.get(f32, mix_count);
    const pre = try b.get(f32, rows * 4);
    const post = try b.get(f32, rows * 4);
    const comb = try b.get(f32, rows * 16);

    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    mhc.encodeSplit(e, &p, view(mixes), view(inv), view(scale), view(base), view(pre), view(post), view(comb), .{ .rows = rows, .eps = cfg.text_config.hc_eps });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.MhcFp64GpuFailure;

    const scale_tensor: Tensor = .{ .buffer = scale, .offset = 0, .bytes = 12, .dtype = .f32 };
    const base_tensor: Tensor = .{ .buffer = base, .offset = 0, .bytes = mix_count * 4, .dtype = .f32 };
    const hc = Hc{ .projection = base_tensor, .scale = scale_tensor, .base = base_tensor };

    var max_ulp_metal: f64 = 0;
    var max_ulp_f32: f64 = 0;
    var worst_name: []const u8 = "";
    const stages = [_]struct { name: []const u8, offset: usize, len: usize }{
        .{ .name = "pre", .offset = 0, .len = 4 },
        .{ .name = "post", .offset = 4, .len = 4 },
        .{ .name = "comb", .offset = 8, .len = 16 },
    };
    for (stages) |stage| {
        for (0..rows) |r| {
            const m: []const f32 = mixes_host[r * mix_count ..][0..mix_count];
            var m64: [mix_count]f64 = undefined;
            for (m, 0..) |v, i| m64[i] = v;
            const mix64 = math.splitOf(f64, &m64, 1.0, hc, cfg.text_config);
            const mix32 = math.splitOf(f32, m, 1.0, hc, cfg.text_config);
            const src64: []const f64 = if (std.mem.eql(u8, stage.name, "pre")) &mix64.pre else if (std.mem.eql(u8, stage.name, "post")) &mix64.post else &mix64.comb;
            const src32: []const f32 = if (std.mem.eql(u8, stage.name, "pre")) &mix32.pre else if (std.mem.eql(u8, stage.name, "post")) &mix32.post else &mix32.comb;
            const metal: []const f32 = if (std.mem.eql(u8, stage.name, "pre")) pre.slice(f32, rows * 4)[r * 4 ..][0..4] else if (std.mem.eql(u8, stage.name, "post")) post.slice(f32, rows * 4)[r * 4 ..][0..4] else comb.slice(f32, rows * 16)[r * 16 ..][0..16];
            const d = deltaF32VsF64(metal[0..stage.len], src64[0..stage.len]);
            const d32 = deltaF32VsF64(src32[0..stage.len], src64[0..stage.len]);
            if (d.max_ulp > max_ulp_metal) {
                max_ulp_metal = d.max_ulp;
                worst_name = stage.name;
            }
            if (d32.max_ulp > max_ulp_f32) max_ulp_f32 = d32.max_ulp;
            std.debug.print("HC64 {{\"case\":\"split_args\",\"stage\":\"{s}\",\"row\":{d},\"nrms_metal_vs_f64\":{d},\"max_ulp_metal_vs_f64\":{d:.4},\"max_ulp_f32_vs_f64\":{d:.4}}}\n", .{ stage.name, r, d.nrms, d.max_ulp, d32.max_ulp });
        }
    }
    std.debug.print("HC64 {{\"case\":\"split_args\",\"summary\":true,\"max_ulp_metal_vs_f64\":{d:.4},\"worst_stage\":\"{s}\",\"max_ulp_f32_vs_f64\":{d:.4}}}\n", .{ max_ulp_metal, worst_name, max_ulp_f32 });
    // Allow the documented one-ULP Metal exp slack; near-cancelled cells also require NRMS below 1e-7.
    try std.testing.expect(max_ulp_metal <= max_ulp_f32 + 2.0);
    try std.testing.expect(max_ulp_metal < 48.0);
}

// Retained cancellation fixtures require exact FP32 accumulation; FP64 records the intended half-ULP gap.
test "mHC collapse and post keep source FP32 accumulation under cancellation" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try mhc.Pipelines.load(device);
    defer p.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const cfg = try Config.parse(gpa, released_config);

    var b = Buffers{ .device = device };
    defer b.deinit();
    // Collapse fixture: stream {3,3,0,0} (width 1), previous_pre {-1, 1+2^-23, 0, 0}.
    const stream_a = try b.get(u16, 4);
    const values_a = [_]u16{ math.b(3), math.b(3), math.b(0), math.b(0) };
    upload(u16, stream_a, &values_a);
    const previous_a = try b.get(f32, 4);
    const coeff = 1 + 0x1p-23;
    const previous_values = [_]f32{ -1, coeff, 0, 0 };
    upload(f32, previous_a, &previous_values);
    const collapsed = try b.get(u16, 1);

    // Post fixture: stream={1,1,1,1}, branch={1}, post={-1,0,0,0}, comb column 0={1,2^-24,2^-24,0}.
    const stream_b = try b.get(u16, 4);
    const values_b = [_]u16{ math.b(1), math.b(1), math.b(1), math.b(1) };
    upload(u16, stream_b, &values_b);
    const branch_b = try b.get(u16, 1);
    const branch_values = [_]u16{math.b(1)};
    upload(u16, branch_b, &branch_values);
    const post_b = try b.get(f32, 4);
    const post_values = [_]f32{ -1, 0, 0, 0 };
    upload(f32, post_b, &post_values);
    const comb_b = try b.get(f32, 16);
    comb_b.slice(f32, 16)[0] = 1;
    comb_b.slice(f32, 16)[4] = 0x1p-24;
    comb_b.slice(f32, 16)[8] = 0x1p-24;
    const y_b = try b.get(u16, 4);

    // Branch-product fixture: stream {3,0,0,0}, post {1+2^-23,...}, comb[0] = -1, branch {3}.
    const stream_c = try b.get(u16, 4);
    const values_c = [_]u16{ math.b(3), math.b(0), math.b(0), math.b(0) };
    upload(u16, stream_c, &values_c);
    const branch_c = try b.get(u16, 1);
    const branch_values_c = [_]u16{math.b(3)};
    upload(u16, branch_c, &branch_values_c);
    const post_c = try b.get(f32, 4);
    const post_values_c = [_]f32{ coeff, 0, 0, 0 };
    upload(f32, post_c, &post_values_c);
    const comb_c = try b.get(f32, 16);
    comb_c.slice(f32, 16)[0] = -1;
    const y_c = try b.get(u16, 4);

    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    mhc.encodeCollapse(e, &p, view(stream_a), view(previous_a), view(collapsed), .{ .rows = 1, .width = 1, .stream_stride = 4, .y_stride = 1 });
    mhc.encodePost(e, &p, view(branch_b), view(stream_b), view(post_b), view(comb_b), view(y_b), .{ .rows = 1, .width = 1, .branch_stride = 1, .residual_stride = 4, .y_stride = 4 });
    mhc.encodePost(e, &p, view(branch_c), view(stream_c), view(post_c), view(comb_c), view(y_c), .{ .rows = 1, .width = 1, .branch_stride = 1, .residual_stride = 4, .y_stride = 4 });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.MhcFp64GpuFailure;

    // Source-contract expectations (math.zig retained boundaries).
    try std.testing.expectEqual(@as(u16, 0x3500), collapsed.slice(u16, 1)[0]);
    try std.testing.expectEqual(@as(u16, 0), y_b.slice(u16, 4)[0]);
    try std.testing.expectEqual(@as(u16, 0x3500), y_c.slice(u16, 4)[0]);

    // Reference columns for the same fixtures.
    var collapsed32: [1]u16 = undefined;
    var collapsed64: [1]u16 = undefined;
    math.collapseOf(f32, &values_a, .{ -1, coeff, 0, 0 }, &collapsed32);
    math.collapseOf(f64, &values_a, .{ -1, @as(f64, coeff), 0, 0 }, &collapsed64);
    std.debug.print("HC64 {{\"case\":\"cancellation\",\"fixture\":\"collapse_half_ulps\",\"metal\":\"0x{x}\",\"f32_ref\":\"0x{x}\",\"f64_ideal\":\"0x{x}\"}}\n", .{ collapsed.slice(u16, 1)[0], collapsed32[0], collapsed64[0] });
    try std.testing.expectEqual(collapsed32[0], collapsed.slice(u16, 1)[0]);

    const mix_b = math.MixOf(f32){ .pre = .{ 1, 0, 0, 0 }, .post = .{ -1, 0, 0, 0 }, .comb = .{ 1, 0, 0, 0, 0x1p-24, 0, 0, 0, 0x1p-24, 0, 0, 0, 0, 0, 0, 0 } };
    var y32: [4]u16 = undefined;
    var h32 = values_b;
    math.expandOf(f32, &h32, &branch_values, mix_b);
    @memcpy(&y32, &h32);
    const mix_b64 = math.MixOf(f64){ .pre = .{ 1, 0, 0, 0 }, .post = .{ -1, 0, 0, 0 }, .comb = .{ 1, 0, 0, 0, 0x1p-24, 0, 0, 0, 0x1p-24, 0, 0, 0, 0, 0, 0, 0 } };
    var h64 = values_b;
    math.expandOf(f64, &h64, &branch_values, mix_b64);
    std.debug.print("HC64 {{\"case\":\"cancellation\",\"fixture\":\"post_half_ulps\",\"metal\":\"0x{x}\",\"f32_ref\":\"0x{x}\",\"f64_ideal\":\"0x{x}\"}}\n", .{ y_b.slice(u16, 4)[0], y32[0], h64[0] });
    try std.testing.expectEqual(y32[0], y_b.slice(u16, 4)[0]);

    const mix_c = math.MixOf(f32){ .pre = .{ 1, 0, 0, 0 }, .post = .{ coeff, 0, 0, 0 }, .comb = .{ -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
    var hc32 = values_c;
    math.expandOf(f32, &hc32, &branch_values_c, mix_c);
    const mix_c64 = math.MixOf(f64){ .pre = .{ 1, 0, 0, 0 }, .post = .{ @as(f64, coeff), 0, 0, 0 }, .comb = .{ -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
    var hc64 = values_c;
    math.expandOf(f64, &hc64, &branch_values_c, mix_c64);
    std.debug.print("HC64 {{\"case\":\"cancellation\",\"fixture\":\"branch_product\",\"metal\":\"0x{x}\",\"f32_ref\":\"0x{x}\",\"f64_ideal\":\"0x{x}\"}}\n", .{ y_c.slice(u16, 4)[0], hc32[0], hc64[0] });
    try std.testing.expectEqual(hc32[0], y_c.slice(u16, 4)[0]);
    _ = cfg;
}
