//! The stored experts' two checkpoint checks: one-hot rows read each decoded weight back, random rows the outputs.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const cs = @import("contract_stored_experts.zig");
const st = @import("stored_experts_tests.zig");

const Gpu = check.Gpu;
const grouped = cuda.grouped;
const ex_ = cuda.experts;
const mx = cuda.experts_mx;
const c = core.contract.spec;

const chunk = 16; // experts a pass of the one-hot check

/// Every weight of both matrices of all 128 experts through the fp32 down kernel: x one-hot, so out = w (x factor).
pub fn decoded(gpu: Gpu, s: st.Stored, layer: usize) !void {
    const a = gpu.gpa;
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var k4 = try ex_.Nvfp4Experts.load(gpu.d, sms);
    defer k4.unload();
    var k8 = try mx.Mxfp8Experts.load(gpu.d, sms);
    defer k8.unload();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try grouped.Router.resolve(mod);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    for ([_]st.Matrix{ s.w13, s.w2 }, [_][]const u8{ "w13", "w2" }) |m, name| {
        const n = m.n;
        const k = m.k;
        const words = (if (s.nv) ex_.Nvfp4Experts.bytes(1, n, k, 1) else mx.Mxfp8Experts.bytes(1, n, k, 1)) / 4;
        const packed_ = try a.alloc(u32, st.experts * words);
        defer a.free(packed_);
        for (0..st.experts) |e| {
            const o = packed_[e * words ..][0..words];
            if (s.nv) ex_.Nvfp4Experts.packOne(o, m.expertCodes(e), m.expertScales(e), n, k) else mx.Mxfp8Experts.packOne(o, m.expertCodes(e), m.expertScales(e), n, k);
        }
        var w = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(packed_));
        defer w.free();
        var fs = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&m.factor));
        defer fs.free();
        // x: the k x k identity; pair i * chunk + j reads row i (slots = chunk) and expert base + j
        const eye = try a.alloc(u16, k * k);
        defer a.free(eye);
        @memset(eye, 0);
        for (0..k) |i| eye[i * k + i] = 0x3F80;
        var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(eye));
        defer x.free();
        const pairs = k * chunk;
        var pb = try st.PlanBufs.init(gpu, pairs, 16);
        defer pb.deinit();
        const ids = try a.alloc(i32, pairs);
        defer a.free(ids);
        var dids = try cuda.DeviceBuffer.alloc(gpu.d, pairs * 4);
        defer dids.free();
        var out = try cuda.DeviceBuffer.alloc(gpu.d, pairs * n * 4);
        defer out.free();
        const got = try a.alloc(f32, pairs * n);
        defer a.free(got);
        var differ: usize = 0;
        var zeros: usize = 0;
        var base: usize = 0;
        while (base < st.experts) : (base += chunk) {
            for (ids, 0..) |*v, p| v.* = @intCast(base + p % chunk);
            try dids.upload(0, std.mem.sliceAsBytes(ids));
            try router.route(stream, dids.ptr, pairs, st.experts, 16, pb.plan);
            try out.fill8(0xff, stream.handle);
            const in: ex_.Rows = .{ .x = x.ptr, .stride = k, .slots = chunk };
            const nb = n / 32;
            if (s.nv) {
                // the existing fp32 down (decode form) read with token rows: Nvfp4Experts.run passes slots only to up
                var args: cuda.Args = .{};
                args.add(in.x);
                for ([_]usize{ k, chunk }) |v| args.add(@as(i32, @intCast(v)));
                args.add(w.ptr);
                args.add(fs.ptr);
                for ([_]usize{ k / 32, nb }) |v| args.add(@as(i32, @intCast(v)));
                for ([_]u64{ pb.plan.items, pb.plan.counts, pb.plan.members, out.ptr }) |v| args.add(v);
                args.add(@as(i32, @intCast(n)));
                args.add(@as(f32, 0));
                args.add(@as(i32, -1));
                const grid = @min((pb.items * nb + k4.warps - 1) / k4.warps, k4.resident[1]);
                try cuda.launch.launch(k4.fns[1], .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = @intCast(k4.warps * 32) } }, stream, &args);
            } else {
                var args = mx.argsOf(in, true, w.ptr, null, k / 32, nb, pb.plan, out.ptr, n, .{ 0, 0 }, -1);
                const grid = @min((pb.items * nb + 3) / 4, k8.resident[1]);
                try cuda.launch.launch(k8.fns[1], .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = 128 } }, stream, &args);
            }
            try stream.synchronize();
            try out.download(0, std.mem.sliceAsBytes(got));
            for (0..k) |i| for (0..chunk) |j| for (0..n) |row| {
                const want = m.f32At(s.nv, base + j, row, i);
                const v = got[(i * chunk + j) * n + row];
                if (v != want) differ += 1 else if (@as(u32, @bitCast(v)) != @as(u32, @bitCast(want))) zeros += 1;
            };
        }
        try check.expect(differ == 0, "layer {d} {s}: {d} of {d} decoded weights differ from the host decode", .{ layer, name, differ, st.experts * n * k });
        check.pass("layer {d} {s} ({s}): all {d} weights of 128 experts [{d}, {d}] equal the host decode ({d} zeros differ in sign only)", .{ layer, name, if (s.nv) "NVFP4" else "MXFP8", st.experts * n * k, n, k, zeros });
    }
}

/// Gate-up then down on 4 tokens (decode form) and 64 (prompt form), each output within the contract's fp32 bound.
pub fn outputs(gpu: Gpu, s: st.Stored, layer: usize) !void {
    const a = gpu.gpa;
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var k4 = try ex_.Nvfp4Experts.load(gpu.d, sms);
    defer k4.unload();
    var k8 = try mx.Mxfp8Experts.load(gpu.d, sms);
    defer k8.unload();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try grouped.Router.resolve(mod);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var bufs: [4]cuda.DeviceBuffer = undefined;
    const l = try s.upload(gpu, &bufs);
    defer for (&bufs) |*b| b.free();
    const gx: ex_.Grouped = .{ .nvfp4 = &k4, .mxfp8 = &k8 };
    for ([_]usize{ 4, 64 }) |tokens| {
        const prompt = tokens > 4;
        const pairs = tokens * st.top;
        const tile: usize = if (prompt) 64 else 16;
        var prng = std.Random.DefaultPrng.init(0x5eed + tokens + layer);
        const xs = try a.alloc(u16, tokens * st.hidden);
        defer a.free(xs);
        for (xs) |*v| v.* = c.toBf16((prng.random().float(f32) * 2 - 1) * 2);
        const ids = try st.picks(a, tokens, 0xc3 + tokens);
        defer a.free(ids);
        var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xs));
        defer dx.free();
        var dids = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ids));
        defer dids.free();
        var pb = try st.PlanBufs.init(gpu, pairs, tile);
        defer pb.deinit();
        try router.route(stream, dids.ptr, pairs, st.experts, tile, pb.plan);
        var act = try cuda.DeviceBuffer.alloc(gpu.d, pairs * st.width * 2);
        defer act.free();
        const yw: usize = if (prompt) 2 else 4;
        var y = try cuda.DeviceBuffer.alloc(gpu.d, pairs * st.hidden * yw);
        defer y.free();
        try act.fill8(0xff, stream.handle);
        try y.fill8(0xff, stream.handle);
        const up_in: ex_.Rows = .{ .x = dx.ptr, .stride = st.hidden, .slots = st.top };
        const down_in: ex_.Rows = .{ .x = act.ptr, .stride = st.width };
        if (prompt) {
            try gx.prompt(stream, .up, up_in, l, pb.plan, act.ptr, pb.items, pairs, tile, -1);
            try gx.prompt(stream, .down_bf16, down_in, l, pb.plan, y.ptr, pb.items, pairs, tile, -1);
        } else {
            try gx.decode(stream, .up, up_in, l, pb.plan, act.ptr, pb.items, pairs, tile, -1);
            try gx.decode(stream, .down_f32, down_in, l, pb.plan, y.ptr, pb.items, pairs, tile, -1);
        }
        try stream.synchronize();
        const ga = try a.alloc(u16, pairs * st.width);
        defer a.free(ga);
        try act.download(0, std.mem.sliceAsBytes(ga));
        const gy = try a.alloc(u8, pairs * st.hidden * yw);
        defer a.free(gy);
        try y.download(0, gy);
        const fmt = if (s.nv) cs.Format.nvfp4 else cs.Format.mxfp8;
        const up_spec = cs.spec(.{ .name = "up", .format = fmt, .kind = .up, .prompt = prompt });
        const down_spec = cs.spec(.{ .name = "down", .format = fmt, .kind = if (prompt) .down_bf16 else .down_f32, .prompt = prompt });
        var worst: [2]f64 = .{ 0, 0 };
        var bad: [2]usize = .{ 0, 0 };
        for (0..pairs) |p| {
            const e: usize = @intCast(ids[p]);
            const xr = xs[(p / st.top) * st.hidden ..][0..st.hidden];
            for (0..st.width) |j| {
                var sum: [2]f64 = .{ 0, 0 };
                var abs: [2]f64 = .{ 0, 0 };
                for (0..2) |m| for (0..st.hidden) |i| {
                    const t = c.fromBf16(xr[i]) * s.w13.f64At(s.nv, e, 2 * j + m, i);
                    sum[m] += t;
                    abs[m] += @abs(t);
                };
                const r = cs.softcap(sum[0], sum[1]);
                const mag = @abs(r[1]) * abs[0] + @abs(r[2]) * abs[1];
                const err = @abs(@as(f64, c.fromBf16(ga[p * st.width + j])) - r[0]);
                const b = c.bound(up_spec, .{ .rows = tokens, .n = st.width, .k = st.hidden }, r[0], mag);
                worst[0] = @max(worst[0], err / b);
                if (!(err <= b)) bad[0] += 1;
            }
            const ar = ga[p * st.width ..][0..st.width];
            for (0..st.hidden) |row| {
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (0..st.width) |i| {
                    const t = c.fromBf16(ar[i]) * s.w2.f64At(s.nv, e, row, i);
                    sum += t;
                    abs += @abs(t);
                }
                const o = p * st.hidden + row;
                const v: f64 = if (prompt) c.fromBf16(std.mem.bytesToValue(u16, gy[o * 2 ..][0..2])) else std.mem.bytesToValue(f32, gy[o * 4 ..][0..4]);
                const b = c.bound(down_spec, .{ .rows = tokens, .n = st.hidden, .k = st.width }, sum, abs);
                worst[1] = @max(worst[1], @abs(v - sum) / b);
                if (!(@abs(v - sum) <= b)) bad[1] += 1;
            }
        }
        try check.expect(bad[0] == 0 and bad[1] == 0, "layer {d} {d} tokens: {d} gate-up and {d} down outputs outside the fp32 bound", .{ layer, tokens, bad[0], bad[1] });
        check.pass("layer {d} {s} {d} tokens ({s}): {d} gate-up and {d} down outputs within the fp32 bound; worst err/bound {d:.3} and {d:.3}", .{ layer, if (s.nv) "NVFP4" else "MXFP8", tokens, if (prompt) "prompt form" else "decode form", pairs * st.width, pairs * st.hidden, worst[0], worst[1] });
    }
}

/// MXFP8's E8M0 edges through tf_mxfp8_down_f32: scale byte 0 is 2^-127 (not 0, not 2^-126).
pub fn e8m0Edges(gpu: Gpu) !void {
    const a = gpu.gpa;
    const n = 32;
    const k = 32;
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var k8 = try mx.Mxfp8Experts.load(gpu.d, sms);
    defer k8.unload();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try grouped.Router.resolve(mod);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const edges = [_]u8{ 0, 1, 2, 126, 127, 128, 200, 240 }; // 448 x 2^113 still fits fp32
    const values = [_]u8{ 0x38, 0x7E, 0x30, 0xB8 }; // 1, 448, 0.5, -1: each times 2^-127 stays a bf16 (>= 2^-133)
    var codes: [n * k]u8 = undefined;
    var scales: [n]u8 = undefined;
    for (0..n) |r| {
        scales[r] = edges[r % edges.len];
        for (0..k) |i| codes[r * k + i] = values[(r / edges.len + i) % values.len];
    }
    var blocks: [mx.Mxfp8Experts.words]u32 = undefined;
    mx.Mxfp8Experts.packOne(&blocks, &codes, &scales, n, k);
    var w = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&blocks));
    defer w.free();
    var eye: [k * k]u16 = @splat(0);
    for (0..k) |i| eye[i * k + i] = 0x3F80;
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&eye));
    defer x.free();
    var ids: [k]i32 = @splat(0);
    var dids = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&ids));
    defer dids.free();
    var pb = try st.PlanBufs.init(gpu, k, 16);
    defer pb.deinit();
    try router.route(stream, dids.ptr, k, st.experts, 16, pb.plan);
    var out = try cuda.DeviceBuffer.alloc(gpu.d, k * n * 4);
    defer out.free();
    try out.fill8(0xff, stream.handle);
    const l: ex_.Layer = .{ .format = .mxfp8, .up = 0, .down = w.ptr, .width = k, .dims = n, .experts = st.experts, .caps = .{ 4, 25 } };
    const gx: ex_.Grouped = .{ .mxfp8 = &k8 };
    try gx.decode(stream, .down_f32, .{ .x = x.ptr, .stride = k }, l, pb.plan, out.ptr, pb.items, k, 16, -1);
    try stream.synchronize();
    const got = try a.alloc(f32, k * n);
    defer a.free(got);
    try out.download(0, std.mem.sliceAsBytes(got));
    for (0..k) |i| for (0..n) |r| {
        const want: f32 = @floatCast(core.expert_decode.mxfp8(&codes, &scales, k, r, i));
        try check.expect(got[i * n + r] == want, "E8M0 byte {d}, code {x}: {e}, host {e}", .{ scales[r], codes[r * k + i], got[i * n + r], want });
    };
    try check.expect(core.expert_decode.e8m0(0) == 0x1p-127, "host E8M0 byte 0", .{});
    check.pass("MXFP8 E8M0 bytes {any} decode exactly on the GPU (byte 0 = 2^-127)", .{edges});
}
