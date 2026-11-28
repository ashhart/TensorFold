//! EXL3 or ModelOpt NVFP4 BF16 embeddings on host; TFQ3 keeps its automatic host-mapping path.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const batched = @import("cuda_mtp_batch.zig");
const mtp_memory = @import("cuda_mtp_memory.zig");

pub const Residency = api.EmbeddingResidency;

/// Inputs for the load estimate: available device room, one context, selected drafter and residency policy.
pub const Fit = struct {
    room: ?u64,
    context: usize,
    drafter: ?[]const u8,
    mtp_drafts: ?u32 = null,
    residency: Residency = .auto,
    explicit: bool = false,
    prompt_int8: bool = false,
};

pub fn fromRoom(room: anytype, context: usize, drafts: bool, drafter_dir: ?[]const u8, mtp_drafts: ?u32, residency: Residency, explicit: bool, prompt_int8: bool) Fit {
    const available = if (room) |r| r.read(r.ctx) else null;
    return .{ .room = available, .context = context, .drafter = if (drafts) drafter_dir else null, .mtp_drafts = if (drafts) mtp_drafts else null, .residency = residency, .explicit = explicit, .prompt_int8 = prompt_int8 };
}

/// Generic CUDA driver/kernel allowance. Deterministic model and workspace allocations are accounted separately.
const generic_overhead: u64 = 96 << 20;

/// Decide host residency. Auto uses a one-stream estimate; host also covers multi-stream configurations.
pub fn shouldMap(unified: bool, residency: Residency, room: ?u64, required: u64) bool {
    if (unified) return false;
    return switch (residency) {
        .device => false,
        .host => true,
        .auto => if (room) |available| required > available else false,
    };
}

pub fn supported(g: @import("cuda_geometry.zig").Geometry) bool {
    return g.embeds and (g.exl3 or (g.kinds.modelopt and @import("cuda_modelopt.zig").hasNvfp4(g)));
}

fn requiredBytes(device: u64, stage: u64, pending: u64, part: u64, stream: u64, beside: u64, mtp_peak: u64, mtp_resident: u64) u64 {
    const load = device +| stage +| generic_overhead;
    const serve = device +| pending +| part +| generic_overhead +| stream +| beside;
    const mtp_startup = device +| pending +| part +| generic_overhead +| mtp_peak;
    const mtp_serve = device +| pending +| part +| generic_overhead +| mtp_resident +| stream;
    return @max(load, serve, mtp_startup, mtp_serve);
}

fn refusesUnsupportedOverride(supported_format: bool, fit: ?Fit) bool {
    if (supported_format) return false;
    return if (fit) |f| f.explicit else false;
}

fn autoRoomUnknown(fit: Fit) bool {
    return fit.residency == .auto and fit.room == null;
}

/// The engine's arena of weights and scratch rows, with any host memory for the embedding rows made first.
pub fn arena(e: anytype, io: std.Io, ctx: *const cuda.Context, ck: anytype, fit: ?Fit) !cuda.DeviceBuffer {
    e.host_embed = null;
    errdefer if (e.host_embed) |*h| h.free();
    const bytes = try measure(e, io, ctx, ck, fit);
    return cuda.DeviceBuffer.alloc(e.d, bytes);
}

/// The arena and the host memory `arena` made.
pub fn release(e: anytype) void {
    e.arena.free();
    if (e.host_embed) |*h| h.free();
}

/// The weights' bytes on the card (EXL3: a dry run of its loader, as trellis widths come from the tensors).
fn measure(e: anytype, io: std.Io, ctx: *const cuda.Context, ck: anytype, fit: ?Fit) !usize {
    const weights = @import("cuda_weights.zig");
    const modelopt = @import("cuda_modelopt.zig");
    const format_supported = supported(e.g);
    if (refusesUnsupportedOverride(format_supported, fit)) return error.EmbeddingResidencyUnsupported;
    if (!format_supported) {
        if (e.g.tfq3) return measureTfq3(e, io, ctx, ck, fit);
        return if (e.g.exl3) try @import("cuda_exl3_weights.zig").measure(e, ck) else weights.total(e.g);
    }

    const bytes = if (e.g.exl3) try @import("cuda_exl3_weights.zig").measure(e, ck) else weights.total(e.g);
    const f = fit orelse return bytes;
    if (f.residency == .device) return bytes;
    if (autoRoomUnknown(f)) return bytes;
    const unified = (try ctx.attribute(.integrated)) != 0;
    if (f.residency == .host and unified) return error.EmbeddingHostOffloadUnavailableOnUnifiedMemory;
    if (unified) return bytes;

    const without_embed = if (e.g.exl3)
        bytes - @import("cuda_engine.zig").aligned(e.g.vocab * e.g.hidden * 2)
    else
        weights.totalWithoutEmbedding(e.g);
    if (f.residency == .host) {
        _ = try allocateHost(e, f, e.g.vocab * e.g.hidden * 2, if (e.g.exl3) "EXL3" else "ModelOpt NVFP4");
        return without_embed;
    }

    const stream = e.streamBytes(f.context);
    const beside = if (f.drafter) |dir| try drafter(e, io, dir) else 0;
    // Both paths include the target scratch that startup commits before stream admission.
    const pending_bytes = try startupPending(e, f);

    if (e.g.exl3) {
        // Keep EXL3 auto selection compatible with its existing one-stream admission policy.
        const required = bytes +| pending_bytes +| generic_overhead +| stream +| beside;
        if (!shouldMap(unified, f.residency, f.room, required)) return bytes;
        if (!(try allocateHost(e, f, e.g.vocab * e.g.hidden * 2, "EXL3"))) return bytes;
        return without_embed;
    }

    const stage = weights.stageBytes(e.g);
    const clusters = (try ctx.attribute(.compute_capability_major)) >= 9;
    const part = modelopt.reservePartsBytes(e.g, clusters);
    const mtp = try mtpRequirements(e, io, f);
    // MTP reads one target tap ring per served stream; it is selected after the engine is loaded.
    const target_stream = e.streamBytesWithTaps(f.context, mtp.target_taps);
    const required = requiredBytes(
        @intCast(bytes),
        @intCast(stage),
        @intCast(pending_bytes),
        @intCast(part),
        @intCast(target_stream),
        beside,
        @intCast(mtp.peak),
        @intCast(mtp.resident),
    );
    if (!shouldMap(unified, f.residency, f.room, required)) return bytes;
    if (!(try allocateHost(e, f, e.g.vocab * e.g.hidden * 2, "ModelOpt NVFP4"))) return bytes;
    return without_embed;
}

/// Preserve TFQ3's existing automatic host mapping while ModelOpt/EXL3 use explicit residency planning.
fn measureTfq3(e: anytype, io: std.Io, ctx: *const cuda.Context, ck: anytype, fit: ?Fit) !usize {
    const tfq3w = @import("cuda_tfq3_weights.zig");
    const bytes = try tfq3w.measure(e, ck);
    const f = fit orelse return bytes;
    if (f.residency != .auto or autoRoomUnknown(f)) return bytes;
    const unified = (try ctx.attribute(.integrated)) != 0;
    if (unified or !tfq3w.hasBf16Embed(ck, e.g)) return bytes;
    const stream = e.streamBytes(f.context);
    const beside = if (f.drafter) |dir| try drafter(e, io, dir) else 0;
    const pending = try startupPending(e, f);
    const extra_shards = tfq3w.shardBytes(e.gpa, io, e.dir);
    const required = @as(u64, @intCast(bytes)) +| extra_shards +| @as(u64, @intCast(pending)) +| generic_overhead +| @as(u64, @intCast(stream)) +| beside;
    if (!shouldMap(unified, .auto, f.room, required)) return bytes;
    if (!(try allocateHost(e, f, e.g.vocab * e.g.hidden * 2, "TFQ3"))) return bytes;
    return tfq3w.measure(e, ck);
}

const MtpRequirements = struct { peak: usize = 0, resident: usize = 0, target_taps: usize = 0 };

/// Mirror the MTP loader's own memory check, including its actual geometry-sized scratch and session area.
fn mtpRequirements(e: anytype, io: std.Io, fit: Fit) !MtpRequirements {
    const dir = fit.drafter orelse return .{};
    if (fit.mtp_drafts) |depth| if (depth == 0) return .{};
    const mtp_weights = @import("cuda_mtp_weights.zig");
    if (!mtp_weights.hasHead(e.gpa, io, dir)) return .{};
    const sizes = try mtp_memory.sizes(e.g, fit.context, batched.most, @sizeOf(batched.RingRow));
    const page_test = try @import("cuda_cache.zig").pageTest();
    const calibration = try @import("cuda_native_plan.zig").cacheBytesWithTaps(e, e.window_rows, page_test, 1);
    const plan = try mtp_memory.startupPlan(e.gpa, io, e.g, dir, sizes.scratch_bytes, sizes.area_bytes, calibration);
    return .{ .peak = plan.peak, .resident = plan.resident, .target_taps = 1 };
}

fn allocateHost(e: anytype, fit: Fit, bytes: usize, format: []const u8) !bool {
    const Alloc = struct {
        fn mapped(d: *const cuda.Driver, n: usize) cuda.Error!cuda.HostBuffer {
            return cuda.HostBuffer.allocMapped(d, n);
        }
    };
    return allocateHostUsing(e, fit, bytes, format, Alloc.mapped, reportHostAllocationFailure);
}

fn reportHostAllocationFailure(format: []const u8, bytes: usize, reason: []const u8) void {
    std.log.warn("qwen3_5: {s} embedding offload needs {d:.2} GiB of pinned host RAM; allocation failed ({s})", .{ format, @as(f64, @floatFromInt(bytes)) / (1 << 30), reason });
}

fn allocateHostUsing(e: anytype, fit: Fit, bytes: usize, format: []const u8, allocator: anytype, report: anytype) !bool {
    const host = allocator(e.d, bytes) catch |err| {
        report(format, bytes, @errorName(err));
        if (fit.residency == .auto) {
            std.log.warn("qwen3_5: automatic {s} embedding offload is falling back to device memory", .{format});
            return false;
        }
        return error.EmbeddingHostAllocationFailed;
    };
    e.host_embed = host;
    if (fit.residency == .host) {
        std.log.info("qwen3_5: {s} embedding rows explicitly placed in mapped host memory", .{format});
    } else {
        std.log.info("qwen3_5: {s} embedding rows placed in mapped host memory to improve available device capacity for a {d}-token request", .{ format, fit.context });
    }
    return true;
}

fn startupPending(e: anytype, fit: Fit) !usize {
    const scratch = @import("cuda_scratch.zig");
    const prompt_rows = e.scratch_rows; // the context's rows, TF_QWEN35_CHUNK_SOLO's at most (cuda_solo.scratchRows)
    const main_rows = @max(prompt_rows, e.verify_rows);
    var measured = e.*;
    const main_pending = try scratch.scratch(&measured, main_rows, .measure);
    const int8_pending = if (fit.prompt_int8) blk: {
        const k = @max(e.g.intermediate, e.g.valueWidth(), e.g.attnWidth(), e.g.hidden);
        break :blk cuda.w4a8.Kernels.scratch(prompt_rows, k);
    } else 0;
    return main_pending + int8_pending;
}

/// A DFlash2 drafter's device bytes: its weights (bf16 linears made 4-bit, 5/16) and one stream's ring, taps and frame.
fn drafter(e: anytype, io: std.Io, dir: []const u8) !u64 {
    const draft = @import("cuda_draft.zig");
    const c = draft.Config.read(e.gpa, io, dir, e.g) catch return 0; // not a DFlash2 drafter: its open says why
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var total: u64 = 0;
    var it = d.iterate();
    while (try it.next(io)) |f| {
        if (std.mem.endsWith(u8, f.name, ".safetensors")) total += (try d.statFile(io, f.name, .{})).size;
    }
    return total / 16 * 5 + try draft.Draft.streamBytesFor(c, e.g.hidden, e.tapRows());
}

/// The rows copied to the pinned host memory the device reads in place.
pub fn rows(host: cuda.HostBuffer, bytes: []const u8) !u64 {
    try copyRows(host.bytes, bytes);
    return host.device();
}

/// Copy the checkpoint's BF16 payload without interpreting or converting its bits.
pub fn copyRows(destination: []u8, bytes: []const u8) !void {
    if (bytes.len != destination.len) return error.EmbeddingShape;
    @memcpy(destination, bytes);
}

test "the rows stay on the card unless a discrete card's room is short of the device bytes and one stream" {
    try std.testing.expect(!shouldMap(false, .auto, 20, 20)); // exact-fit boundary
    try std.testing.expect(shouldMap(false, .auto, 19, 20));
    try std.testing.expect(!shouldMap(true, .auto, 0, 20)); // unified memory: host memory is the same memory
    try std.testing.expect(!shouldMap(false, .auto, null, 20)); // no room reading means no automatic decision
    try std.testing.expect(!shouldMap(false, .device, 0, 20));
    try std.testing.expect(shouldMap(false, .host, null, 0));
}

test "unsupported formats reject only explicit residency choices; unknown auto room skips planning" {
    var fit: Fit = .{ .room = null, .context = 1, .drafter = null };
    try std.testing.expect(!refusesUnsupportedOverride(false, null));
    try std.testing.expect(!refusesUnsupportedOverride(false, fit)); // implicit auto remains compatible
    for (std.enums.values(Residency)) |residency| {
        fit.residency = residency;
        fit.explicit = true;
        try std.testing.expect(refusesUnsupportedOverride(false, fit));
    }
    try std.testing.expect(!refusesUnsupportedOverride(true, fit));
    fit = .{ .room = null, .context = 1, .drafter = null };
    try std.testing.expect(autoRoomUnknown(fit));
    fit.room = 0;
    try std.testing.expect(!autoRoomUnknown(fit));
    fit.residency = .host;
    fit.room = null;
    try std.testing.expect(!autoRoomUnknown(fit));
}

test "mapped embedding support includes EXL3 and NVFP4 ModelOpt/Compressed Tensors only" {
    const Geometry = @import("cuda_geometry.zig").Geometry;
    var g: Geometry = .{ .hidden = 64, .intermediate = 128, .layers = 1, .vocab = 256, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = false, .context = 128, .linear = @splat(false) };
    g.kinds = .{ .modelopt = true, .head = .bf16 };
    try std.testing.expect(!supported(g)); // plain BF16
    g.kinds.head = .fp8;
    g.kinds.proj[0][@backingInt(@import("cuda_geometry.zig").Slot.down)] = .fp8;
    try std.testing.expect(!supported(g)); // FP8-only
    g.kinds.ct = true;
    g.kinds.proj[0][@backingInt(@import("cuda_geometry.zig").Slot.down)] = .nvfp4;
    try std.testing.expect(supported(g)); // compressed-tensors exports, including Unsloth NVFP4
    g.kinds.modelopt = false;
    try std.testing.expect(!supported(g));
    g.exl3 = true;
    try std.testing.expect(supported(g));
    g.embeds = false;
    try std.testing.expect(!supported(g));
}

test "mapped BF16 row preparation preserves the exact checkpoint bits" {
    const input: [10]u16 = .{ 0, 1, 0x8000, 0x7f80, 0xff80, 0x7fc1, 0x3f80, 0x0001, 0x7f7f, 0x3555 };
    const source = std.mem.sliceAsBytes(&input);
    var destination: [@sizeOf(@TypeOf(input))]u8 = undefined;
    try copyRows(&destination, source);
    try std.testing.expectEqualSlices(u8, source, &destination);
    try std.testing.expectError(error.EmbeddingShape, copyRows(destination[0 .. destination.len - 1], source));
}

test "failed pinned allocation leaves no host embedding owner behind" {
    const Fake = struct { d: u8 = 0, host_embed: ?u8 = null };
    const Alloc = struct {
        fn fail(_: u8, _: usize) error{OutOfMemory}!u8 {
            return error.OutOfMemory;
        }
    };
    const Report = struct {
        var format: []const u8 = "";
        var bytes: usize = 0;
        var reason: []const u8 = "";

        fn capture(format_: []const u8, bytes_: usize, reason_: []const u8) void {
            format = format_;
            bytes = bytes_;
            reason = reason_;
        }
    };
    var fake: Fake = .{};
    const fit: Fit = .{ .room = null, .context = 1, .drafter = null, .residency = .host };
    try std.testing.expectError(error.EmbeddingHostAllocationFailed, allocateHostUsing(&fake, fit, 4096, "test", Alloc.fail, Report.capture));
    try std.testing.expect(fake.host_embed == null);
    try std.testing.expectEqualStrings("test", Report.format);
    try std.testing.expectEqual(@as(usize, 4096), Report.bytes);
    try std.testing.expectEqualStrings("OutOfMemory", Report.reason);
}

test "automatic pinned allocation failure falls back without claiming host rows" {
    const Fake = struct { d: u8 = 0, host_embed: ?u8 = null };
    const Alloc = struct {
        fn fail(_: u8, _: usize) error{OutOfMemory}!u8 {
            return error.OutOfMemory;
        }
    };
    const Report = struct {
        var format: []const u8 = "";
        var bytes: usize = 0;
        var reason: []const u8 = "";

        fn capture(format_: []const u8, bytes_: usize, reason_: []const u8) void {
            format = format_;
            bytes = bytes_;
            reason = reason_;
        }
    };
    var fake: Fake = .{};
    const fit: Fit = .{ .room = null, .context = 1, .drafter = null, .residency = .auto };
    try std.testing.expect(!try allocateHostUsing(&fake, fit, 4096, "test", Alloc.fail, Report.capture));
    try std.testing.expect(fake.host_embed == null);
    try std.testing.expectEqualStrings("test", Report.format);
    try std.testing.expectEqual(@as(usize, 4096), Report.bytes);
    try std.testing.expectEqualStrings("OutOfMemory", Report.reason);
}

test "automatic offload follows startup and speculative peaks even when offload cannot fit the whole request" {
    const with_workspace = requiredBytes(100, 10, 20, 30, 25, 0, 0, 0);
    try std.testing.expect(with_workspace > 100 + 10 + 20 + 25 + generic_overhead);
    try std.testing.expect(shouldMap(false, .auto, 200, with_workspace));
    const staging_peak = requiredBytes(100, 200, 10, 30, 1, 0, 0, 0);
    try std.testing.expectEqual(@as(u64, 300 + generic_overhead), staging_peak);
    try std.testing.expect(shouldMap(false, .auto, 299 + generic_overhead, staging_peak));
    const mtp_peak = requiredBytes(100, 10, 20, 0, 25, 0, 80, 60);
    try std.testing.expect(mtp_peak > 100 + 10 + 20 + 25 + generic_overhead);
    try std.testing.expect(shouldMap(false, .auto, 200, mtp_peak));
    try std.testing.expect(shouldMap(false, .auto, 1, 1000)); // host copy is useful even if this context still cannot fit
}
