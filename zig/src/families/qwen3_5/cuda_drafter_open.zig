//! --drafter over the Qwen CUDA engine: DFlash2 or the target checkpoint's own MTP layer.
const std = @import("std");
const lanes = @import("lanes");
const native = @import("cuda_native.zig");
const Draft = @import("cuda_draft.zig").Draft;
const Mtp = @import("cuda_mtp.zig").Mtp;
const mtp_weights = @import("cuda_mtp_weights.zig");
const draft_config = @import("cuda_draft_config.zig");
const cost_adapter = native.cost_adapter;
const Lanes = native.Lanes;

/// A drafter opened over the engine; DFlash2's `prepare` times the target's windows once the pool is reserved.
pub const Opened = struct {
    drafter: lanes.drafter.Drafter,
    ctx: *anyopaque,
    close: *const fn (*anyopaque) void,
    reserve: *const fn (*anyopaque, u32) anyerror!void,
    prepare: ?struct { ctx: *anyopaque, run: *const fn (*anyopaque) anyerror!lanes.drafter.Facts, bytes: usize } = null,
};

/// The startup room callback points into cuda.open's stack; release it after drafter setup, including early returns.
pub fn finishDrafterOpen(p: *anyopaque) void {
    const l: *Lanes = @ptrCast(@alignCast(p));
    l.room = null;
}

const DrafterKind = enum { dflash, mtp };

fn selectDrafter(has_dflash: bool, has_mtp: bool, depth: ?u32) !DrafterKind {
    if (has_dflash and has_mtp) return error.AmbiguousDrafterFormat;
    if (has_dflash) {
        if (depth) |value| if (value > 0) return error.MtpDepthWithDFlash2;
        return .dflash;
    }
    if (has_mtp) {
        if (depth) |value| if (value == 0) return error.MtpDepthZero;
        return .mtp;
    }
    if (depth) |value| if (value > 0) return error.NoMtpLayer;
    return error.UnsupportedDrafterFormat;
}

/// Classify a real checkpoint directory using the same format markers as the serving path.
fn selectDrafterDirectory(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, depth: ?u32) !DrafterKind {
    return selectDrafter(draft_config.declared(gpa, io, dir), mtp_weights.hasHead(gpa, io, dir), depth);
}

/// Loop admits only its BF16 MTP head, at either accepted drafter-bits setting.
pub fn loopRoute(declared: bool, bits: u8) !void {
    if (declared) return error.SenseNovaDrafter;
    if (bits != 0 and bits != 4) return error.DrafterBitsUnsupported;
}

/// --drafter's directory over engine `p`: DFlash2 by config, else a checkpoint with a usable MTP head.
pub fn openDrafter(gpa: std.mem.Allocator, io: std.Io, p: *anyopaque, dir: []const u8, bits: u8) !Opened {
    const l: *Lanes = @ptrCast(@alignCast(p));
    if (l.e.g.loop != null) {
        try loopRoute(draft_config.declared(gpa, io, dir), bits);
        return openMtp(gpa, io, p, dir, bits);
    }
    const kind = try selectDrafterDirectory(gpa, io, dir, l.mtp_drafts);
    return switch (kind) {
        .dflash => openDFlash(gpa, io, p, dir, bits),
        .mtp => openMtp(gpa, io, p, dir, bits),
    };
}

/// The DFlash2 drafter in `dir` over engine `p`, 4-bit only, and the target's windows timed for the round loop.
fn openDFlash(gpa: std.mem.Allocator, io: std.Io, p: *anyopaque, dir: []const u8, bits: u8) !Opened {
    if (bits != 4) return error.DrafterBitsUnsupported;
    const l: *Lanes = @ptrCast(@alignCast(p));
    if (l.e.g.exl3 and l.e.g.moe()) return error.Exl3DrafterUnsupported; // a dense checkpoint drafts as an MLX one
    if (l.caches.count() != 0 or l.pool.items.len != 0) return error.TapsAfterStreams;
    const d = try Draft.init(gpa, io, l.e, dir);
    errdefer d.deinit();
    try l.e.setTaps(d.taps());
    d.target_taps = l.e.tapBytes();
    cost_adapter.identify(l, dir, .dflash);
    l.calibration = d;
    l.draft = d;
    return .{ .drafter = d.drafter(), .ctx = d, .close = closeDrafter, .reserve = reserveDrafter, .prepare = .{ .ctx = l, .run = calibrateDrafter, .bytes = if (l.capacity < l.e.window_rows) try native.cacheBytes(l.e, l.e.window_rows, l.page_test) else 0 } };
}

/// The MTP layer beside the checkpoint in `dir` as the drafter over engine `p`, its ring the window.
fn openMtp(gpa: std.mem.Allocator, io: std.Io, p: *anyopaque, dir: []const u8, bits: u8) !Opened {
    const l: *Lanes = @ptrCast(@alignCast(p));
    if (bits != 4 and !(l.e.g.loop != null and bits == 0)) return error.DrafterBitsUnsupported;
    if (l.e.g.exl3 and l.e.g.moe()) return error.Exl3DrafterUnsupported; // a dense checkpoint drafts as an MLX one
    if (l.caches.count() != 0 or l.pool.items.len != 0) return error.TapsAfterStreams;
    const calibration_cache_bytes = try native.cacheBytesWithTaps(l.e, l.e.window_rows, l.page_test, 1);
    const d = try Mtp.initConfigured(gpa, io, l.e, dir, l.capacity, l.mtp_drafts, calibration_cache_bytes, l.room);
    errdefer d.deinit();
    try installMtpTap(d);
    cost_adapter.identify(l, dir, .mtp);
    try l.timeWindows(d, null);
    l.mtp = d;
    return .{ .drafter = d.drafter(), .ctx = d, .close = closeMtp, .reserve = reserveMtp };
}

fn installMtpTap(d: *Mtp) !void {
    try d.e.setTaps(d.taps());
    // The host priced the target before attaching a drafter. Each whole-window cache also owns this tap ring.
    d.stream_bytes = try std.math.add(usize, d.stream_bytes, d.e.tapBytes());
}

fn reserveMtp(p: *anyopaque, streams: u32) anyerror!void {
    const d: *Mtp = @ptrCast(@alignCast(p));
    try d.reserve(streams);
}

fn closeMtp(p: *anyopaque) void {
    const d: *Mtp = @ptrCast(@alignCast(p));
    d.deinit();
}

fn calibrateDrafter(p: *anyopaque) !lanes.drafter.Facts {
    const l: *Lanes = @ptrCast(@alignCast(p));
    const d = l.calibration orelse return error.MissingCalibrationDrafter;
    if (d.n_costs == 0) {
        if (l.pool.items.len == 0 and !l.engine_paged) return error.MissingReservedQwenCache;
        // engine_paged: no reserved window, so the timing makes a window_rows cache of its own (null)
        try l.timeWindows(d, if (l.pool.items.len > 0) l.pool.items[0] else null);
    }
    if (d.n_costs != l.e.window_rows) return error.IncompleteWindowMeasurements;
    l.calibration = null;
    var phases = l.startup;
    phases.add(d.startup);
    phases.print();
    return d.drafter().facts();
}

fn reserveDrafter(p: *anyopaque, streams: u32) anyerror!void {
    const d: *Draft = @ptrCast(@alignCast(p));
    try d.reserve(streams);
}

fn closeDrafter(p: *anyopaque) void {
    const d: *Draft = @ptrCast(@alignCast(p));
    d.deinit();
}

test "MTP drafter selection: dense checkpoint without MTP or DFlash2 is refused before loading" {
    try std.testing.expectError(error.UnsupportedDrafterFormat, selectDrafter(false, false, null));
    try std.testing.expectError(error.UnsupportedDrafterFormat, selectDrafter(false, false, 0));
    try std.testing.expectError(error.NoMtpLayer, selectDrafter(false, false, 3));
}

test "MTP drafter selection: checkpoint type and draft depth select only compatible paths" {
    try std.testing.expectEqual(DrafterKind.dflash, try selectDrafter(true, false, null));
    try std.testing.expectEqual(DrafterKind.dflash, try selectDrafter(true, false, 0));
    try std.testing.expectEqual(DrafterKind.mtp, try selectDrafter(false, true, null));
    try std.testing.expectError(error.MtpDepthWithDFlash2, selectDrafter(true, false, 3));
    try std.testing.expectError(error.MtpDepthZero, selectDrafter(false, true, 0));
    try std.testing.expectError(error.AmbiguousDrafterFormat, selectDrafter(true, true, null));
}

test "MTP drafter selection: dense heads require indexed BF16; standalone affine-4 remains routed-only" {
    const Geometry = @import("cuda_geometry.zig").Geometry;
    const Source = mtp_weights.Source;
    var dense: Geometry = .{
        .hidden = 64,
        .intermediate = 128,
        .layers = 1,
        .vocab = 256,
        .k_heads = 1,
        .v_heads = 1,
        .q_heads = 2,
        .kv_heads = 1,
        .eps = 1e-6,
        .tied = false,
        .context = 512,
        .linear = @splat(false),
    };
    dense.kinds.modelopt = true;
    try std.testing.expectError(error.UnsupportedMtpWeightFormat, mtp_weights.sourceForTarget(dense, true, false, false));
    try std.testing.expectEqual(Source.indexed_bf16, try mtp_weights.sourceForTarget(dense, false, true, false));
    try std.testing.expectError(error.UnsupportedMtpWeightFormat, mtp_weights.sourceForTarget(dense, false, true, true));
    dense.kinds.modelopt = false;
    try std.testing.expectError(error.UnsupportedMtpWeightFormat, mtp_weights.sourceForTarget(dense, true, false, false));
    try std.testing.expectEqual(Source.indexed_bf16, try mtp_weights.sourceForTarget(dense, false, true, false));
    try std.testing.expectError(error.NoMtpLayer, mtp_weights.sourceForTarget(dense, false, false, false));
    dense.kinds.modelopt = true;
    dense.experts = 4;
    try std.testing.expectEqual(Source.standalone_affine4, try mtp_weights.sourceForTarget(dense, true, false, false));
}

const FixtureTensor = struct { name: []const u8, rows: usize, columns: usize = 0 };

fn appendFixtureTensor(a: std.mem.Allocator, json: *std.ArrayList(u8), first: bool, offset: *usize, t: FixtureTensor) !void {
    if (!first) try json.appendSlice(a, ",");
    const elements = t.rows * (if (t.columns == 0) @as(usize, 1) else t.columns);
    const end = offset.* + elements * 2;
    const entry = if (t.columns == 0)
        try std.fmt.allocPrint(a, "\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}", .{ t.name, t.rows, offset.*, end })
    else
        try std.fmt.allocPrint(a, "\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ t.name, t.rows, t.columns, offset.*, end });
    defer a.free(entry);
    try json.appendSlice(a, entry);
    offset.* = end;
}

fn denseMtpFixture(a: std.mem.Allocator, intermediate: usize) !struct { header: []u8, payload: usize } {
    const tensors = [_]FixtureTensor{
        .{ .name = "mtp.fc.weight", .rows = 256, .columns = 512 },
        .{ .name = "mtp.pre_fc_norm_embedding.weight", .rows = 256 },
        .{ .name = "mtp.pre_fc_norm_hidden.weight", .rows = 256 },
        .{ .name = "mtp.norm.weight", .rows = 256 },
        .{ .name = "mtp.layers.0.input_layernorm.weight", .rows = 256 },
        .{ .name = "mtp.layers.0.post_attention_layernorm.weight", .rows = 256 },
        .{ .name = "mtp.layers.0.self_attn.q_norm.weight", .rows = 256 },
        .{ .name = "mtp.layers.0.self_attn.k_norm.weight", .rows = 256 },
        .{ .name = "mtp.layers.0.self_attn.q_proj.weight", .rows = 1024, .columns = 256 },
        .{ .name = "mtp.layers.0.self_attn.k_proj.weight", .rows = 256, .columns = 256 },
        .{ .name = "mtp.layers.0.self_attn.v_proj.weight", .rows = 256, .columns = 256 },
        .{ .name = "mtp.layers.0.self_attn.o_proj.weight", .rows = 256, .columns = 512 },
        .{ .name = "mtp.layers.0.mlp.gate_proj.weight", .rows = intermediate, .columns = 256 },
        .{ .name = "mtp.layers.0.mlp.up_proj.weight", .rows = intermediate, .columns = 256 },
        .{ .name = "mtp.layers.0.mlp.down_proj.weight", .rows = 256, .columns = intermediate },
    };
    var json: std.ArrayList(u8) = .empty;
    errdefer json.deinit(a);
    try json.appendSlice(a, "{");
    var offset: usize = 0;
    for (tensors, 0..) |tensor, i| try appendFixtureTensor(a, &json, i == 0, &offset, tensor);
    try json.appendSlice(a, "}");
    return .{ .header = try json.toOwnedSlice(a), .payload = offset };
}

fn writeSafetensors(dir: std.Io.Dir, name: []const u8, header: []const u8, payload: usize) !void {
    const io = std.testing.io;
    const file = try dir.createFile(io, name, .{});
    defer file.close(io);
    var prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &prefix, header.len, .little);
    try file.writePositionalAll(io, &prefix, 0);
    try file.writePositionalAll(io, header, 8);
    try file.setLength(io, 8 + header.len + payload);
}

fn fixtureGeometry() @import("cuda_geometry.zig").Geometry {
    return .{
        .hidden = 256,
        .intermediate = 128,
        .layers = 1,
        .vocab = 256,
        .k_heads = 1,
        .v_heads = 1,
        .q_heads = 2,
        .kv_heads = 1,
        .eps = 1e-6,
        .tied = false,
        .context = 512,
        .linear = @splat(false),
    };
}

test "MTP drafter selection: directory distinguishes compatible dense MTP, DFlash2, and neither" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const g = fixtureGeometry();

    try tmp.dir.createDir(io, "dense", .default_dir);
    var dense_dir = try tmp.dir.openDir(io, "dense", .{});
    defer dense_dir.close(io);
    const dense_path = try std.fs.path.join(a, &.{ root, "dense" });
    defer a.free(dense_path);
    try dense_dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"qwen3_5_text\"}" });
    const fixture = try denseMtpFixture(a, g.intermediate);
    defer a.free(fixture.header);
    try writeSafetensors(dense_dir, "model.safetensors", fixture.header, fixture.payload);
    _ = try mtp_weights.allocationPlan(a, io, g, dense_path);
    try std.testing.expectEqual(DrafterKind.mtp, try selectDrafterDirectory(a, io, dense_path, null));

    try tmp.dir.createDir(io, "dflash", .default_dir);
    var dflash_dir = try tmp.dir.openDir(io, "dflash", .{});
    defer dflash_dir.close(io);
    const dflash_path = try std.fs.path.join(a, &.{ root, "dflash" });
    defer a.free(dflash_path);
    const dflash_config = "{\"model_type\":\"qwen3\",\"hidden_size\":256,\"intermediate_size\":128,\"num_hidden_layers\":1,\"vocab_size\":256,\"num_attention_heads\":2,\"num_key_value_heads\":1,\"head_dim\":128,\"sliding_window\":16,\"rms_norm_eps\":0.000001,\"use_sliding_window\":true,\"layer_types\":[\"sliding_attention\"],\"rope_parameters\":{\"rope_theta\":10000,\"rope_type\":\"default\"},\"dflash_config\":{\"block_size\":8,\"mask_token_id\":0,\"conv_group_size\":64,\"selector_rank\":16,\"selector_top_k\":2,\"conv_kernel_size\":2,\"target_layer_ids\":[0]}}";
    try dflash_dir.writeFile(io, .{ .sub_path = "config.json", .data = dflash_config });
    try std.testing.expectEqual(draft_config.Config.Kind.dflash2, (try draft_config.Config.read(a, io, dflash_path, g)).kind);
    try std.testing.expectEqual(DrafterKind.dflash, try selectDrafterDirectory(a, io, dflash_path, null));

    try tmp.dir.createDir(io, "ambiguous", .default_dir);
    var ambiguous_dir = try tmp.dir.openDir(io, "ambiguous", .{});
    defer ambiguous_dir.close(io);
    const ambiguous_path = try std.fs.path.join(a, &.{ root, "ambiguous" });
    defer a.free(ambiguous_path);
    try ambiguous_dir.writeFile(io, .{ .sub_path = "config.json", .data = dflash_config });
    try writeSafetensors(ambiguous_dir, "model.safetensors", fixture.header, fixture.payload);
    try std.testing.expectError(error.AmbiguousDrafterFormat, selectDrafterDirectory(a, io, ambiguous_path, null));

    try tmp.dir.createDir(io, "neither", .default_dir);
    var neither_dir = try tmp.dir.openDir(io, "neither", .{});
    defer neither_dir.close(io);
    const neither_path = try std.fs.path.join(a, &.{ root, "neither" });
    defer a.free(neither_path);
    try neither_dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"qwen3_5_text\"}" });
    try writeSafetensors(neither_dir, "model.safetensors", "{\"model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[1,1],\"data_offsets\":[0,2]}}", 2);
    try std.testing.expectError(error.UnsupportedDrafterFormat, selectDrafterDirectory(a, io, neither_path, null));
}

test "MTP drafter selection: dense MTP explicitly refuses the standalone affine-4 drafter format" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    try tmp.dir.createDir(io, "standalone", .default_dir);
    var dir = try tmp.dir.openDir(io, "standalone", .{});
    defer dir.close(io);
    const path = try std.fs.path.join(a, &.{ root, "standalone" });
    defer a.free(path);
    const file = try dir.createFile(io, mtp_weights.file, .{});
    file.close(io);
    try std.testing.expectEqual(DrafterKind.mtp, try selectDrafterDirectory(a, io, path, null));
    try std.testing.expectError(error.UnsupportedMtpWeightFormat, mtp_weights.allocationPlan(a, io, fixtureGeometry(), path));
}

test "MTP drafter selection: Loop admission prices BF16 shards, FC halves, padded MLP and optional main norm" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    var g = fixtureGeometry();
    g.loop = .{ .start = 0, .end = 0, .count = 1, .mhc = null };
    g.intermediate = 64; // Gate and up occupy 128 packed rows each.
    try std.testing.expectError(error.NoLoopMtpFile, mtp_weights.allocationPlan(a, io, g, root));
    try tmp.dir.createDir(io, "mtp", .default_dir);
    const path = try std.fs.path.join(a, &.{ root, "mtp" });
    defer a.free(path);
    const fixture = try denseMtpFixture(a, g.intermediate);
    defer a.free(fixture.header);
    try writeSafetensors(tmp.dir, "mtp/head.safetensors", fixture.header, fixture.payload);
    const expected: mtp_weights.AllocationPlan = .{
        .resident = 2 * (2 * 256 * 256 + 1024 * 256 + 2 * 256 * 256 + 256 * 512 +
            2 * 128 * 256 + 256 * 64) + 7 * 512,
        .staging = 1024 * 256 * 2,
    };
    try std.testing.expectEqual(expected, try mtp_weights.allocationPlan(a, io, g, path));
    const main = "{\"mtp.main_norm.weight\":{\"dtype\":\"F32\",\"shape\":[256],\"data_offsets\":[0,1024]}}";
    try writeSafetensors(tmp.dir, "mtp/norm.safetensors", main, 1024);
    const with_norm = try mtp_weights.allocationPlan(a, io, g, root);
    try std.testing.expectEqual(expected.resident + 512, with_norm.resident);
    try std.testing.expectEqual(expected.staging, with_norm.staging);
    try writeSafetensors(tmp.dir, "mtp/duplicate.safetensors", main, 1024);
    try std.testing.expectError(error.DuplicateTensor, mtp_weights.allocationPlan(a, io, g, root));
    // A root head wins over shards even when that head is incomplete or incompatible.
    try writeSafetensors(tmp.dir, "mtp.safetensors", main, 1024);
    try std.testing.expectError(error.MissingTensor, mtp_weights.allocationPlan(a, io, g, root));
    var bad_geometry = g;
    bad_geometry.hidden = 128;
    try writeSafetensors(tmp.dir, "mtp.safetensors", fixture.header, fixture.payload);
    try std.testing.expectError(error.UnexpectedTensor, mtp_weights.allocationPlan(a, io, bad_geometry, root));
    try std.testing.expectEqual(expected, try mtp_weights.allocationPlan(a, io, g, root));
}

test "MTP drafter selection: whole windows and paged runs count the target tap exactly once" {
    const E = @import("cuda_engine.zig").Engine;
    var e: E = undefined;
    e.g = fixtureGeometry();
    e.g.hidden = 4096;
    e.scratch_rows = 4096;
    e.kv8 = false;
    e.n_taps = 0;
    var d: Mtp = undefined;
    d.e = &e;
    d.tap = .{0};
    d.capacity = 512;
    d.depth = 3;
    d.step_ms = 0;
    d.trees = false;
    d.n_costs = 0;
    try d.initializeMemorySizes();
    const before = try native.cacheBytes(&e, d.capacity, 0);
    try installMtpTap(&d);
    const own = 2 * d.capacity * d.kvWidth() * 2 + 2 * e.g.hidden * 2 + Mtp.max_chain_tokens * 4;
    try std.testing.expectEqual(@as(usize, 48 << 20), e.tapBytes());
    try std.testing.expectEqual(own + e.tapBytes(), d.stream_bytes);
    var target: lanes.fake.Fake = .{ .gpa = std.testing.allocator };
    defer target.deinit();
    var wrapped = try lanes.drafted.Drafted.init(std.testing.allocator, target.backend(), d.drafter());
    defer wrapped.deinit();
    const reported = try wrapped.streamBytes(before);
    try std.testing.expectEqual(try native.cacheBytes(&e, d.capacity, 0) + own, reported);

    var l: Lanes = .{ .gpa = std.testing.allocator, .io = std.testing.io, .e = &e, .capacity = d.capacity, .target = undefined, .page_tokens = 256, .mtp = &d };
    const expected = @import("cuda_pages.zig").layout(&e, d.capacity, l.page_tokens, d.areaBytes(), 0);
    try std.testing.expectEqual(expected, native.pagedLayout(&l, reported - before).?);
}
