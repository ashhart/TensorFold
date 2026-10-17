//! Diagnostic: production layer-0 attention on frozen inputs; args CHECKPOINT INPUT EXPECTED OUTPUT_DIR.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const drama = @import("drama");
const Model = drama.backbone.Model;
const H = 2048;
const R = 128;
const source_rows = 182;
const lengths = [_]usize{ 1, 7, 128, 182 };
const LD = drama.backbone.weights.gdn_in_rows;
const Capture = struct { file: []const u8, dtype: []const u8, shape: [2]usize, sha256: [64]u8 };
const View = struct { name: []const u8, ptr: u64, cols: usize, rows: usize, f32: bool = false };

fn sha(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}
fn finite(bytes: []align(16) const u8, is_f32: bool) bool {
    if (is_f32) {
        for (std.mem.bytesAsSlice(f32, bytes)) |x| if (!std.math.isFinite(x)) return false;
    } else {
        for (std.mem.bytesAsSlice(u16, bytes)) |x| if ((x & 0x7f80) == 0x7f80) return false;
    }
    return true;
}
fn download(driver: *cuda.Driver, staging: []align(16) u8, view: View) ![]align(16) const u8 {
    const bytes = view.rows * view.cols * @as(usize, if (view.f32) 4 else 2);
    const buffer: cuda.DeviceBuffer = .{ .d = driver, .ptr = view.ptr, .len = bytes, .borrowed = true };
    try buffer.download(0, staging[0..bytes]);
    if (!finite(staging[0..bytes], view.f32)) return error.NonfiniteCapture;
    return staging[0..bytes];
}
fn capture(gpa: std.mem.Allocator, io: std.Io, out: std.Io.Dir, driver: *cuda.Driver, staging: []align(16) u8, prefix: []const u8, view: View) !Capture {
    const bytes = try download(driver, staging, view);
    const file = try std.fmt.allocPrint(gpa, "{s}{s}.{s}", .{ prefix, view.name, if (view.f32) "f32" else "bf16" });
    errdefer gpa.free(file);
    try out.writeFile(io, .{ .sub_path = file, .data = bytes, .flags = .{ .exclusive = true } });
    return .{ .file = file, .dtype = if (view.f32) "f32le" else "bf16le", .shape = .{ view.rows, view.cols }, .sha256 = sha(bytes) };
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5) return error.UsageCheckpointInputExpectedOutputDirectory;
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;
    const out = try std.Io.Dir.cwd().openDir(io, args[4], .{});
    defer out.close(io);
    const total_bytes = source_rows * H * 2;
    const input_data = try std.Io.Dir.cwd().readFileAllocOptions(io, args[2], gpa, .limited(total_bytes + 1), .@"16", null);
    defer gpa.free(input_data);
    const expected = try std.Io.Dir.cwd().readFileAllocOptions(io, args[3], gpa, .limited(total_bytes + 1), .@"16", null);
    defer gpa.free(expected);
    if (input_data.len != total_bytes or expected.len != total_bytes) return error.FixtureShape;
    if (!finite(input_data, false) or !finite(expected, false)) return error.NonfiniteFixture;
    _ = try cuda.setAllocationLimit(54 * (1 << 30));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var model = try Model.init(gpa, io, &ctx, args[1], "", .{ .context = 512, .max_rows = R });
    defer model.deinit();
    if (!drama.backbone.config.linear(0) or model.hidden != H) return error.UnexpectedArchitecture;
    var input = try cuda.HostBuffer.alloc(&driver, total_bytes);
    defer input.free();
    var output = try cuda.HostBuffer.alloc(&driver, total_bytes);
    defer output.free();
    const staging = try gpa.alignedAlloc(u8, .@"16", LD * H * 2);
    defer gpa.free(staging);
    const w = &model.w.layers[0];
    const weight_views = [_]View{
        .{ .name = "weight_in", .ptr = w.gdn_in, .rows = LD, .cols = H },
        .{ .name = "weight_conv", .ptr = w.conv, .rows = 8192, .cols = 4 },
        .{ .name = "weight_a_log", .ptr = w.a_log, .rows = 32, .cols = 1, .f32 = true },
        .{ .name = "weight_dt_bias", .ptr = w.dt_bias, .rows = 32, .cols = 1, .f32 = true },
        .{ .name = "weight_norm", .ptr = w.gnorm, .rows = 1, .cols = 128 },
        .{ .name = "weight_out", .ptr = w.gdn_out, .rows = H, .cols = 4096 },
    };
    var captures: std.ArrayList(Capture) = .empty;
    defer {
        for (captures.items) |c| gpa.free(c.file);
        captures.deinit(gpa);
    }
    try captures.ensureTotalCapacity(gpa, weight_views.len + lengths.len * 6);
    for (weight_views) |view| captures.appendAssumeCapacity(try capture(gpa, io, out, &driver, staging, "", view));
    for (lengths) |rows| {
        const bytes = rows * H * 2;
        const row0: usize = if (rows == source_rows) R else 0;
        const captured_rows = rows - row0;
        const captured_at = row0 * H * 2;
        const chunks: []const usize = if (rows == source_rows) &.{ R, source_rows - R } else &.{rows};
        @memcpy(input.bytes[0..bytes], input_data[0..bytes]);
        const views = [_]View{
            .{ .name = "input", .ptr = model.s.normed, .rows = captured_rows, .cols = H },
            .{ .name = "proj", .ptr = model.s.proj, .rows = captured_rows, .cols = LD },
            .{ .name = "conv", .ptr = model.s.conv, .rows = captured_rows, .cols = 8192 },
            .{ .name = "core", .ptr = model.s.core, .rows = captured_rows, .cols = 4096 },
            .{ .name = "gated", .ptr = model.s.gated, .rows = captured_rows, .cols = 4096 },
            .{ .name = "output", .ptr = model.s.x, .rows = captured_rows, .cols = H },
        };
        var hashes: [views.len][64]u8 = undefined;
        var prefix_buffer: [32]u8 = undefined;
        const prefix = try std.fmt.bufPrint(&prefix_buffer, "rows-{d}-", .{rows});
        for (0..2) |repeat| {
            try model.reset(stream);
            @memset(output.bytes[0..bytes], 0xff);
            try model.replayOperator(stream, 0, .attention, chunks, input.bytes[0..bytes], output.bytes[0..bytes]);
            if (!std.mem.eql(u8, output.bytes[0..bytes], expected[0..bytes])) return error.NativeFrozenSelfControl;
            if (!std.mem.eql(u8, input.bytes[0..bytes], input_data[0..bytes])) return error.HostInputMutated;
            for (views, 0..) |view, i| {
                if (repeat == 0) {
                    const c = try capture(gpa, io, out, &driver, staging, prefix, view);
                    hashes[i] = c.sha256;
                    captures.appendAssumeCapacity(c);
                } else if (!std.mem.eql(u8, &sha(try download(&driver, staging, view)), &hashes[i])) {
                    return error.ScratchRepeatMismatch;
                }
            }
            if (!std.mem.eql(u8, &hashes[0], &sha(input_data[captured_at..bytes]))) return error.DeviceInputMutated;
            if (!std.mem.eql(u8, &hashes[5], &sha(output.bytes[captured_at..bytes]))) return error.OutputCaptureMismatch;
        }
    }
    for (weight_views, 0..) |view, i| {
        if (!std.mem.eql(u8, &sha(try download(&driver, staging, view)), &captures.items[i].sha256)) return error.WeightMutated;
    }
    const used = cuda.usage(false);
    const manifest = try std.json.Stringify.valueAlloc(gpa, .{
        .producer = "tf-drama-gdn-layer0-trace",
        .evidence_valid = true,
        .layer = 0,
        .rows = lengths,
        .repeats = 2,
        .native_frozen_prefix_controls = true,
        .capture_row_offsets = [_]usize{ 0, 0, 0, 128 },
        .capture_rows = [_]usize{ 1, 7, 128, 54 },
        .scratch_repeat_controls = true,
        .input_and_weight_immutability = true,
        .input_sha256 = sha(input_data),
        .expected_sha256 = sha(expected),
        .driver_source_sha256 = sha(@embedFile("drama_gdn_layer0_trace.zig")),
        .backbone_source_sha256 = sha(drama.backbone.trace_source),
        .backbone_image_sha256 = sha(cuda.kernels.drama_backbone),
        .gemm_image_sha256 = sha(cuda.kernels.drama_backbone_gemm),
        .norm_image_sha256 = sha(cuda.kernels.norms),
        .cuda_bytes = used.device,
        .cuda_peak_bytes = used.peak,
        .pinned_host_bytes = used.host,
        .captures = captures.items,
    }, .{ .whitespace = .indent_2 });
    defer gpa.free(manifest);
    try out.writeFile(io, .{ .sub_path = "manifest.json", .data = manifest, .flags = .{ .exclusive = true } });
    std.debug.print("layer=0 rows=1,7,128,182 repeats=2 frozen_native_exact=true captures={d} cuda_bytes={d}\n", .{ captures.items.len, used.device });
    return 0;
}
