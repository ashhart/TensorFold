//! A CUDA family's options from the serve flags it takes: its ranks, images and videos, NVFP4 precision and fp8 caches.
const std = @import("std");
const api = @import("engine_api");

/// Sets what `o` asks of `F` in `opts`; false with `problem` set when F cannot take a flag given (--precision).
pub fn set(comptime F: type, o: api.Open, opts: *F.Options, problem: *[]const u8) bool {
    if (@hasField(F.Options, "mtp_drafts")) opts.mtp_drafts = o.mtp_drafts;
    if (@hasField(F.Options, "mtp_confidence")) opts.mtp_confidence = o.mtp_confidence;
    if (@hasField(F.Options, "mtp_drafts") and o.mtp_drafts == 0) opts.drafts = false;
    if (@hasField(F.Options, "lanes")) opts.lanes = o.lanes;
    if (@hasField(F.Options, "lanes_fixed")) opts.lanes_fixed = o.lanes_fixed;
    if (@hasField(F.Options, "rank")) {
        opts.tp = @intCast(o.tp);
        opts.rank = @intCast(o.rank);
        opts.master = if (o.master.len > 0) o.master else null;
        opts.master_port = o.master_port;
    }
    if (comptime @hasField(F.Options, "vision")) {
        if (o.vision) |vo| {
            const own: ?u32 = comptime if (@hasDecl(F, "default_image_tokens")) F.default_image_tokens else null;
            opts.vision = .{ .image_tokens = if (vo.image_tokens_set) vo.image_tokens else own orelse vo.image_tokens, .max_images = vo.max_images };
            // a family that reads videos takes the counts its tower holds (Nemotron Omni's only a token budget)
            const L = @typeInfo(@FieldType(F.Options, "vision")).optional.child;
            if (comptime @hasField(L, "video_tokens")) opts.vision.?.video_tokens = vo.video_tokens;
            if (comptime @hasField(L, "max_videos")) opts.vision.?.max_videos = vo.max_videos;
            if (comptime @hasField(L, "max_frames")) opts.vision.?.max_frames = vo.max_frames;
        }
    }
    if (o.precision) |p| {
        if (!@hasField(F.Options, "precision")) {
            problem.* = "--precision: this CUDA family reads no NVFP4 checkpoint; serve it without --precision";
            return false;
        }
        opts.precision = std.meta.stringToEnum(@typeInfo(@TypeOf(opts.precision)).optional.child, @tagName(p)) orelse {
            problem.* = "--precision: unknown value for this CUDA family; serve without --precision or with --precision full";
            return false;
        };
    }
    if (comptime @hasField(F.Options, "drafter")) opts.drafter = o.drafter;
    if (comptime @hasField(F.Options, "drafter_bits")) opts.drafter_bits = o.drafter_bits;
    if (@hasField(F.Options, "kv8")) opts.kv8 = o.kv8;
    if (@hasField(F.Options, "prompt_int8")) opts.prompt_int8 = o.prompt_int8;
    if (@hasField(F.Options, "loops")) opts.loops = o.loops;
    return true;
}

test "mtp policy: native options forward positive controls and preserve the explicit zero switch" {
    const F = struct {
        pub const Options = struct { drafts: bool = true, mtp_drafts: ?u32 = null, mtp_confidence: ?f64 = null };
    };
    var opts: F.Options = .{};
    var why: []const u8 = "";
    try std.testing.expect(set(F, .{ .dir = "", .model_type = "nemotron_h", .mtp_drafts = 2, .mtp_confidence = 0.5 }, &opts, &why));
    try std.testing.expectEqual(@as(?u32, 2), opts.mtp_drafts);
    try std.testing.expectEqual(@as(?f64, 0.5), opts.mtp_confidence);
    try std.testing.expect(opts.drafts);
    opts = .{};
    try std.testing.expect(set(F, .{ .dir = "", .model_type = "nemotron_h", .mtp_drafts = 0 }, &opts, &why));
    try std.testing.expect(!opts.drafts);
}

test "native family options forward lane counts and fixedness only when declared" {
    const Counted = struct {
        pub const Options = struct { lanes: u32 = 1 };
    };
    const Fixed = struct {
        pub const Options = struct { lanes: u32 = 1, lanes_fixed: bool = false };
    };
    var why: []const u8 = "";
    var counted: Counted.Options = .{};
    var fixed: Fixed.Options = .{};
    const request: api.Open = .{ .dir = "", .model_type = "kolibri1", .lanes = 7, .lanes_fixed = true };
    try std.testing.expect(set(Counted, request, &counted, &why));
    try std.testing.expect(set(Fixed, request, &fixed, &why));
    try std.testing.expectEqual(@as(u32, 7), counted.lanes);
    try std.testing.expectEqual(@as(u32, 7), fixed.lanes);
    try std.testing.expect(fixed.lanes_fixed);
}

test "prompt precision: family options receive both explicit int8 and default bf16" {
    const F = struct {
        pub const Options = struct { prompt_int8: bool = false };
    };
    var why: []const u8 = "";
    var opts: F.Options = .{ .prompt_int8 = true };
    for ([_]bool{ false, true, false }) |on| {
        try std.testing.expect(set(F, .{ .dir = "d", .model_type = "qwen3_5", .prompt_int8 = on }, &opts, &why));
        try std.testing.expectEqual(on, opts.prompt_int8);
    }
}

test "full-only precision: checkpoint is unknown before a family can load or allocate" {
    const Full = struct {
        pub const Options = struct { precision: ?enum { full } = null };
    };
    var opts: Full.Options = .{};
    var why: []const u8 = "";
    try std.testing.expect(set(Full, .{ .dir = "d", .model_type = "qwen3_5", .precision = .full }, &opts, &why));
    try std.testing.expectEqual(@as(?@typeInfo(@TypeOf(opts.precision)).optional.child, .full), opts.precision);
    try std.testing.expect(!set(Full, .{ .dir = "d", .model_type = "qwen3_5", .precision = .checkpoint }, &opts, &why));
    try std.testing.expect(std.mem.indexOf(u8, why, "unknown") != null);
    const Both = struct {
        pub const Options = struct { precision: ?enum { checkpoint, full } = null };
    };
    var existing: Both.Options = .{};
    try std.testing.expect(set(Both, .{ .dir = "d", .model_type = "nemotron_h", .precision = .checkpoint }, &existing, &why));
    try std.testing.expectEqual(@as(?@typeInfo(@TypeOf(existing.precision)).optional.child, .checkpoint), existing.precision);
}
