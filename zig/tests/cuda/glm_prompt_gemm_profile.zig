//! Isolated GLM-5.3-Flash NVFP4 TP=2 prompt launches; no production dispatch changes.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const shapes = @import("glm_prompt_gemm_shapes.zig");
const Gpu = check.Gpu;
const Shape = shapes.Shape;
const Launch = shapes.Launch;

pub fn run(gpu: Gpu, name: []const u8) !void {
    const shape = shapes.find(name) orelse return error.UnknownPromptGemmShape;
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    switch (shape.kind) {
        .dense_up, .dense_down => try dense(gpu, stream, shape),
        .routed_up, .routed_down => try routed(gpu, stream, shape),
    }
}

fn filled(gpu: Gpu, stream: cuda.Stream, bytes: usize, value: u8) !cuda.DeviceBuffer {
    var b = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
    errdefer b.free();
    try b.fill8(value, stream.handle);
    return b;
}

fn metrics(gpu: Gpu, shape: Shape, symbol: []const u8, f: cuda.Function, launch: Launch) !void {
    const regs = try f.attribute(.num_regs);
    const static_shared = try f.attribute(.shared_size_bytes);
    const local = try f.attribute(.local_size_bytes);
    const bound = try f.attribute(.max_threads_per_block);
    const ctas = try f.occupancy(launch.threads, launch.shared);
    const sms = try gpu.ctx.attribute(.multiprocessor_count);
    const l1 = try gpu.ctx.attribute(.max_shared_memory_per_multiprocessor);
    const warps = ctas * launch.threads / 32;
    const resident = ctas * @as(u32, @intCast(sms));
    const row_label: []const u8 = if (shape.kind == .routed_up or shape.kind == .routed_down) "token_rows" else "M";
    std.debug.print(
        "PROFILE {s} symbol={s} {s}={d} N={d} K={d} grid=({d},1,1) block=({d},1,1) dynamic_shared={d} static_shared={d} registers={d} local_bytes={d} launch_bound={d} L1_shared_per_SM={d} occupancy_CTA_per_SM={d} occupancy_warps_per_SM={d} resident_CTA_capacity={d}\n",
        .{ shape.name, symbol, row_label, shape.rows, shape.n, shape.k, launch.grid_x, launch.threads, launch.shared, static_shared, regs, local, bound, l1, ctas, warps, resident },
    );
}

fn same(gpu: Gpu, a: *const cuda.DeviceBuffer, b: *const cuda.DeviceBuffer) !void {
    const got = try check.download(gpu, a.*);
    defer gpu.gpa.free(got);
    const want = try check.download(gpu, b.*);
    defer gpu.gpa.free(want);
    try check.sameBytes("target versus ordinary output", got, want);
}

fn sameRouted(gpu: Gpu, a: *const cuda.DeviceBuffer, b: *const cuda.DeviceBuffer, shape: Shape) !void {
    const got = try check.download(gpu, a.*);
    defer gpu.gpa.free(got);
    const want = try check.download(gpu, b.*);
    defer gpu.gpa.free(want);
    const row_bytes = shape.n * 2;
    for (0..shape.rows) |row| for (0..shapes.routed_slots) |slot| {
        const at = (row * shapes.slots + slot) * row_bytes;
        try check.sameBytes("target versus ordinary routed output", got[at..][0..row_bytes], want[at..][0..row_bytes]);
    };
}

fn dense(gpu: Gpu, stream: cuda.Stream, shape: Shape) !void {
    var prompt = try cuda.nvfp4.Prompt.load(gpu.d);
    defer prompt.unload();
    const npad = cuda.nvfp4.padded(shape.n);
    var x = try filled(gpu, stream, shape.rows * shape.k * 2, 0x3f);
    defer x.free();
    var codes = try filled(gpu, stream, npad * shape.k / 2, 0x11);
    defer codes.free();
    var scales = try filled(gpu, stream, npad * shape.k / 16, 0x38);
    defer scales.free();
    const out_size: usize = if (shape.kind == .dense_down) 4 else 2;
    var target = try filled(gpu, stream, shape.rows * shape.n * out_size, 0xa5);
    defer target.free();
    var ordinary = try filled(gpu, stream, shape.rows * shape.n * out_size, 0x5a);
    defer ordinary.free();
    const w = cuda.nvfp4.weight(codes.ptr, scales.ptr, 0.0625, shape.n, shape.k);
    const launch = try shape.geometry(prompt.sms);
    const up = shape.kind == .dense_up;
    const f = if (up) prompt.lean else prompt.lean_f32;
    try metrics(gpu, shape, cuda.nvfp4.Prompt.lean_symbols[@intFromBool(!up)], f, launch);

    const rows_t = (shape.rows + 127) / 128;
    const band = @max(1, @min(rows_t, (12 << 20) / (128 * shape.k * 2)));
    var args: cuda.Args = .{};
    args.add(x.ptr);
    args.add(w.codes);
    args.add(w.scales);
    args.add(w.scale);
    args.add(target.ptr);
    inline for (.{ shape.rows, shape.n, shape.k, npad, shape.k, band }) |v| args.add(@as(i32, @intCast(v)));
    for ([_]u64{ 0, 0 }) |v| args.add(v);
    args.add(@as(f32, 0));
    args.add(@as(u64, 0));
    try cuda.launch.launch(f, .{ .grid = .{ .x = launch.grid_x }, .block = .{ .x = launch.threads }, .shared = launch.shared }, stream, &args);
    try stream.synchronize();
    const kind: cuda.qmmf.Out = if (up) .bf16 else .f32;
    try prompt.matmulOut(stream, x.ptr, shape.k, shape.rows, w, ordinary.ptr, kind);
    try stream.synchronize();
    try same(gpu, &target, &ordinary);
    check.pass("glm-prompt-gemm-profile: {s} direct served symbol equals ordinary output byte for byte", .{shape.name});
}

fn routed(gpu: Gpu, stream: cuda.Stream, shape: Shape) !void {
    const up = shape.kind == .routed_up;
    const pairs = shape.rows * shapes.slots;
    const experts = shapes.routed_experts;
    const plan_experts = shapes.plan_experts;
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var ex = try cuda.experts.Nvfp4Experts.load(gpu.d, sms);
    defer ex.unload();
    var plan_mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer plan_mod.unload();
    const router = try cuda.grouped.Router.resolve(plan_mod);
    const parts: usize = if (up) 2 else 1;
    var weights = try filled(gpu, stream, cuda.experts.Nvfp4Experts.bytes(experts, shape.n, shape.k, parts), 0x11);
    defer weights.free();
    var global = try filled(gpu, stream, experts * parts * 4, 0x3f);
    defer global.free();
    var x = try filled(gpu, stream, (if (up) shape.rows else pairs) * shape.k * 2, 0x3f);
    defer x.free();
    var target = try filled(gpu, stream, pairs * shape.n * 2, 0xa5);
    defer target.free();
    var ordinary = try filled(gpu, stream, pairs * shape.n * 2, 0x5a);
    defer ordinary.free();

    const host_picks = try gpu.gpa.alloc(i32, pairs);
    defer gpu.gpa.free(host_picks);
    for (host_picks, 0..) |*pick, i| pick.* = @intCast(if (i % shapes.slots == shapes.routed_slots) shapes.plan_experts - 1 else (i / shapes.slots * shapes.routed_slots + i % shapes.slots) % shapes.routed_experts);
    var picks = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host_picks));
    defer picks.free();
    const sizes = [5]usize{ pairs * 4, shape.item_bound * 12, 8, pairs * 4, ((pairs + 1023) / 1024 * plan_experts) * 4 };
    var buffers: [5]cuda.DeviceBuffer = undefined;
    var made: usize = 0;
    defer for (buffers[0..made]) |*b| b.free();
    for (sizes, &buffers) |size, *b| {
        b.* = try filled(gpu, stream, size, 0);
        made += 1;
    }
    const plan: cuda.grouped.Plan = .{ .members = buffers[0].ptr, .items = buffers[1].ptr, .counts = buffers[2].ptr, .rank = buffers[3].ptr, .hist = buffers[4].ptr };
    try router.route(stream, picks.ptr, pairs, plan_experts, 64, plan);
    try stream.synchronize();
    const counts = try check.download(gpu, buffers[2]);
    defer gpu.gpa.free(counts);
    const planned: usize = @intCast(std.mem.readInt(i32, counts[0..4], .little));
    try check.expect(planned == shape.planned_items, "glm-prompt-gemm-profile: planned {d} items, expected {d}", .{ planned, shape.planned_items });

    const layer: cuda.experts.Layer = .{ .format = .nvfp4, .up = if (up) weights.ptr else 0, .down = if (up) 0 else weights.ptr, .up_scale = if (up) global.ptr else 0, .down_scale = if (up) 0 else global.ptr, .width = 1024, .dims = 4096, .experts = experts, .limit = 10 };
    const input: cuda.experts.Rows = .{ .x = x.ptr, .stride = shape.k, .slots = if (up) shapes.slots else 0 };
    const fi: usize = if (up) 1 else 2;
    const f = if (shape.resident) ex.resident_prompt_fns[fi - 1] else ex.prompt_fns[fi];
    const launch = try shape.geometry(sms);
    const symbol = if (shape.resident) cuda.experts.Nvfp4Experts.symbols.resident_prompt[fi - 1] else cuda.experts.Nvfp4Experts.symbols.prompt[fi];
    try metrics(gpu, shape, symbol, f, launch);
    std.debug.print("PROFILE {s} pair_rows={d} routed_pairs={d} planned_items={d} item_bound={d} skipped_shared_expert={d}\n", .{ shape.name, pairs, shape.rows * shapes.routed_slots, planned, shape.item_bound, shapes.plan_experts - 1 });

    var args: cuda.Args = .{};
    args.add(input.x);
    args.add(@as(i32, @intCast(input.stride)));
    args.add(@as(i32, @intCast(if (up) input.slots else 0)));
    args.add(if (up) layer.up else layer.down);
    args.add(if (up) layer.up_scale else layer.down_scale);
    args.add(@as(i32, @intCast((if (up) layer.dims else layer.width) / 32)));
    args.add(@as(i32, @intCast((if (up) layer.width else layer.dims) / 32)));
    for ([_]u64{ plan.items, plan.counts, plan.members, target.ptr }) |v| args.add(v);
    args.add(@as(i32, @intCast(if (up) layer.width else layer.dims)));
    args.add(if (up) layer.limit else @as(f32, 0));
    args.add(@as(i32, @intCast(shapes.plan_experts - 1)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = launch.grid_x }, .block = .{ .x = launch.threads }, .shared = launch.shared }, stream, &args);
    try stream.synchronize();
    const grouped: cuda.experts.Grouped = .{ .nvfp4 = &ex };
    try grouped.prompt(stream, if (up) .up else .down_bf16, input, layer, plan, ordinary.ptr, shape.item_bound, pairs, 64, shapes.plan_experts - 1);
    try stream.synchronize();
    try sameRouted(gpu, &target, &ordinary, shape);
    check.pass("glm-prompt-gemm-profile: {s} direct served symbol equals ordinary output byte for byte", .{shape.name});
}
