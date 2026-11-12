//! Image exactness: serial rows versus shared teacher verification, partial keep, and owned snapshot resume.
const std = @import("std");
const hip = @import("hip");
const q = @import("qwen3_5");
const shared = @import("qwen_hip_vision");
const lanes = @import("lanes");
const fixture = @import("fixture.zig");
const qf = @import("qwen_fixture.zig");
const state = @import("qwen_exact_state.zig");
const Allocator = std.mem.Allocator;
const Engine = q.engine.Engine;
const Request = @typeInfo(@FieldType(q.hip_lanes.Lane, "vision")).optional.child;
const width = 16;
const kept = 3;
const streams = 2;

const Sink = struct {
    e: *Engine,
    bytes: []u8,
    rows: usize,
    first: usize = 0,
    take: usize = 0,

    fn rowBytes(s: *const Sink) usize {
        return s.e.model().spec.hidden * s.e.model().act.size();
    }
    fn put(s: *Sink, slot: usize, x: hip.ops.Tensor, count: usize) !void {
        try s.e.stream.synchronize();
        const n = s.rowBytes();
        const at = (slot * s.rows + s.first) * n;
        try hip.runtime.check(s.e.driver.api.hipMemcpyDtoH(s.bytes[at..].ptr, @ptrFromInt(x.ptr), count * n));
        try state.finite(s.bytes[at..][0 .. count * n], x.kind, true);
    }
    fn layer(ctx: *anyopaque, index: usize, x: hip.ops.Tensor, _: usize) anyerror!void {
        const s: *Sink = @ptrCast(@alignCast(ctx));
        try s.put(index, x, s.take);
    }
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4 or args.len > 5) {
        std.debug.print("usage: tf-qwen-hip-exact MODEL TEACHER16_REFERENCE_DIR OUTPUT_DIR [DEVICE]\n", .{});
        return 2;
    }
    var files = try fixture.Files.init(a, init.io, args[2], args[3]);
    defer files.output.close(init.io);
    try qf.verify(files, args[1]);
    const req = try fixture.field(files.manifest, "request");
    var cfg = try q.Config.read(a, init.io, args[1]);
    defer cfg.deinit();
    const prompt = try qf.ids(a, try fixture.field(req, "prompt_ids"), cfg.spec.vocab);
    const teacher = try qf.ids(a, try fixture.field(req, "continuation_ids"), cfg.spec.vocab);
    if (teacher.len < width or prompt.len == 0) return error.Teacher16Required;
    const tail = teacher[0..width];
    const pixels = try files.floats("processor.pixel_values");
    const grid = try qf.integers(a, files, "processor.image_grid_thw");
    if (grid.len != 3 or grid[0] != 1 or grid[1] <= 0 or grid[2] <= 0 or grid[1] > 4096 or grid[2] > 4096) return error.BadFixture;
    const gh: u32 = @intCast(grid[1]);
    const gw: u32 = @intCast(grid[2]);
    const patches = try shared.rowsOf(&.{.{ .h = gh, .w = gw }}, pixels.data.len);
    const image_tokens = patches / 4;
    const device = if (args.len == 5) try std.fmt.parseInt(c_int, args[4], 10) else 0;
    if (device < 0) return error.BadDevice;
    var preflight: q.engine.Preflight = .{};
    const e = try Engine.open(init.gpa, init.io, args[1], .{
        .capacity = prompt.len + width + 17,
        .prompt_rows = prompt.len,
        .batch_rows = streams * width,
        .streams = 8,
        .snapshots = 1,
        .device = device,
        .graphs = false,
        .policy = .{ .slots = 0, .graphs = .off },
        .preflight = &preflight,
        .vision = .{ .image_tokens = @intCast(image_tokens), .max_images = 1 },
    });
    defer e.deinit();
    const v = e.vision orelse return error.VisionNotLoaded;
    const image_at = try qf.imageRange(prompt, v.c.image_token, image_tokens);
    const images = [_]lanes.stream.Image{.{ .at = @intCast(image_at), .gh = gh, .gw = gw, .tokens = @intCast(image_tokens), .pixels = pixels.data }};
    var media = try Request.init(v, &images, prompt, e.o.capacity);
    defer media.deinit(init.gpa);
    defer e.drain();
    // A concurrent text lane has a different physical prefix and zero M-RoPE shift.
    const text = try std.mem.concat(a, u32, &.{ prompt[0 .. image_at - 1], prompt[image_at + image_tokens + 1 ..] });
    if (text.len == 0) return error.EmptyTextControl;
    var text_media = try Request.init(v, &.{}, text, e.o.capacity);
    defer text_media.deinit(init.gpa);
    defer e.drain();
    const media_ptrs = [_]*Request{ &media, &text_media };
    var alone: [streams]state.Lane = undefined;
    var together: [streams]state.Lane = undefined;
    var made: usize = 0;
    defer for (0..made) |j| {
        alone[j].deinit(e);
        together[j].deinit(e);
    };
    for (0..streams) |j| {
        alone[j] = try state.Lane.init(e);
        together[j] = state.Lane.init(e) catch |err| {
            alone[j].deinit(e);
            return err;
        };
        made += 1;
    }
    // Save inside the expanded image, on the production 64-row prompt cut grid.
    const cut = (image_at + image_tokens / 2) / q.pages.tokens * q.pages.tokens;
    if (cut <= image_at or cut >= image_at + image_tokens) return error.ImageCutRequired;
    try prefill(e, &alone[0], &media, prompt, 0, cut);
    var snapshot = try state.Snapshot.save(init.gpa, e, &alone[0]);
    defer snapshot.deinit(e);
    try snapshot.restore(e, &together[0]);
    try equalState(a, e, "snapshot restored GDN/KV", &alone[0], &together[0]);
    // Fresh image state, compared with the restored prefix completed through the remaining image rows.
    var fresh = try state.Lane.init(e);
    defer fresh.deinit(e);
    std.mem.swap(state.Lane, &alone[0], &fresh);
    try prefill(e, &alone[0], &media, prompt, 0, prompt.len);
    try prefill(e, &together[0], &media, prompt, cut, prompt.len);
    try equalState(a, e, "image snapshot resume equals fresh", &alone[0], &together[0]);
    // Keep the same image positions but remove feature insertion: target state must change.
    var control = try state.Lane.init(e);
    defer control.deinit(e);
    var no_features = media.frame();
    no_features.runs = &.{};
    e.bridge.model.frame = no_features;
    _ = try e.prefill(&control.caches, prompt, 0, control.hidden, .{ .sampling = null, .position = prompt.len }, null);
    e.bridge.model.frame = null;
    control.len = prompt.len;
    const control_sum = try state.digest(e.gpa, e, &control);
    const image_sum = try state.digest(e.gpa, e, &alone[0]);
    if (std.mem.eql(u8, &control_sum, &image_sum)) return error.ImageFeaturesHaveNoEffect;
    try prefill(e, &alone[1], &text_media, text, 0, text.len);
    try prefill(e, &together[1], &text_media, text, 0, text.len);

    const rows = streams * width;
    const nbytes = (e.model().spec.n_layers + 1) * rows * e.model().spec.hidden * e.model().act.size();
    var serial: Sink = .{ .e = e, .bytes = try a.alloc(u8, nbytes), .rows = rows };
    var wide: Sink = .{ .e = e, .bytes = try a.alloc(u8, nbytes), .rows = rows };
    var serial_tokens: [streams][width]u32 = undefined;
    var wide_tokens: [rows]u32 = undefined;
    var partial: [streams][32]u8 = undefined;
    var final: [streams][32]u8 = undefined;
    for (0..streams) |j| {
        for (tail, 0..) |token, i| {
            serial.first = j * width + i;
            const one = [_]*state.Lane{&alone[j]};
            var got: [1]u32 = undefined;
            try round(e, &one, &.{media_ptrs[j].layout.shift}, &.{&.{token}}, 1, &serial, &got);
            serial_tokens[j][i] = got[0];
            if (i + 1 == kept) partial[j] = try state.digest(e.gpa, e, &alone[j]);
        }
        final[j] = try state.digest(e.gpa, e, &alone[j]);
    }
    const both = [_]*state.Lane{ &together[0], &together[1] };
    wide.first = 0;
    try round(e, &both, &.{ media.layout.shift, text_media.layout.shift }, &.{ tail, tail }, kept, &wide, &wide_tokens);
    try equalRows(e, &serial, &wide, "serial versus shared teacher verification");
    for (0..streams) |j| {
        if (!std.mem.eql(u32, &serial_tokens[j], wide_tokens[j * width ..][0..width])) return error.SampledTokensDiffer;
        if (!std.mem.eql(u8, &partial[j], &(try state.digest(e.gpa, e, &together[j])))) return error.PartialKeepStateDiffers;
    }
    // Rejected rows must have no GDN effect, and their KV slots must be overwritten on replay.
    for (kept..width) |i| for (0..streams) |j| {
        wide.first = j * width + i;
        const one = [_]*state.Lane{&together[j]};
        var got: [1]u32 = undefined;
        try round(e, &one, &.{media_ptrs[j].layout.shift}, &.{tail[i..][0..1]}, 1, &wide, &got);
        if (got[0] != serial_tokens[j][i]) return error.ReplayedTokensDiffer;
    };
    try equalRows(e, &serial, &wide, "partial keep and rejected row replay");
    for (0..streams) |j| if (!std.mem.eql(u8, &final[j], &(try state.digest(e.gpa, e, &together[j])))) return error.FinalStateDiffers;
    const receipt = try std.json.Stringify.valueAlloc(a, .{
        .schema = "tensorfold.lighton_qwen_hip.exact.v1",
        .checkpoint = try fixture.field(files.manifest, "checkpoint"),
        .reference_directory = args[2],
        .model_directory = args[1],
        .device = device,
        .architecture = hip.kernels.arch,
        .teacher_rows = width,
        .streams = streams,
        .kept = kept,
        .snapshot_cut = cut,
        .image_shift = media.layout.shift,
        .text_shift = text_media.layout.shift,
        .results = .{ .snapshot_bytes_equal = true, .resumed_fresh_state_equal = true, .image_features_change_target_state = true, .serial_shared_layer_bytes_equal = true, .greedy_tokens_equal = true, .partial_keep_state_equal = true, .rejected_replay_layer_bytes_equal = true, .final_gdn_kv_hidden_equal = true },
        .scope = "production image prefill, static teacher draft verification and owned snapshot; internal image radix reuse and learned draft heads are not exercised",
        .final_state_sha256 = .{ std.fmt.bytesToHex(final[0], .lower), std.fmt.bytesToHex(final[1], .lower) },
    }, .{ .whitespace = .indent_2 });
    try files.output.writeFile(init.io, .{ .sub_path = "exact.json", .data = receipt });
    std.debug.print("PASS image snapshot resume, mixed image/text shared width16, partial keep3 and rejected replay: every layer, greedy token and committed GDN/KV/hidden byte equals serial\n", .{});
    return 0;
}

fn prefill(e: *Engine, lane: *state.Lane, media: *const Request, tokens: []const u32, from: usize, to: usize) !void {
    e.bridge.model.frame = media.frame();
    defer e.bridge.model.frame = null;
    if (to == tokens.len) {
        _ = try e.prefill(&lane.caches, tokens, from, lane.hidden, .{ .sampling = null, .position = to }, null);
    } else try e.advance(&lane.caches, tokens, from, to, null);
    lane.len = to;
}

fn round(e: *Engine, ls: []const *state.Lane, shifts: []const i32, tokens: []const []const u32, keep: usize, sink: *Sink, out: []u32) !void {
    var rows: [streams]Engine.Rows = undefined;
    var reqs: [streams * width]q.draw.Request = undefined;
    var total: usize = 0;
    for (ls, shifts, tokens, 0..) |lane, shift, ts, j| {
        rows[j] = .{ .caches = &lane.caches, .pos = lane.len, .tokens = ts, .shift = shift };
        for (0..ts.len) |i| reqs[total + i] = .{ .sampling = null, .position = lane.len + i + 1 };
        total += ts.len;
    }
    sink.take = total;
    e.round.trace = .{ .ctx = sink, .layer = Sink.layer };
    defer e.round.trace = null;
    const done = try e.verify(rows[0..ls.len], reqs[0..total], out[0..total]);
    try sink.put(e.model().spec.n_layers, done.hidden, total);
    for (ls, tokens, 0..) |lane, ts, j| {
        e.keep(j, @min(keep, ts.len));
        lane.len += @min(keep, ts.len);
    }
    try e.flush();
    try e.stream.synchronize();
}

fn equalRows(e: *Engine, left: *const Sink, right: *const Sink, label: []const u8) !void {
    if (std.mem.indexOfDiff(u8, left.bytes, right.bytes)) |at| {
        const n = left.rowBytes();
        std.debug.print("FAIL {s}: layer {d}, flat row {d}, column {d}\n", .{ label, at / (left.rows * n), at / n % left.rows, at % n / e.model().act.size() });
        return error.HiddenBytesDiffer;
    }
}

fn equalState(_: Allocator, e: *Engine, label: []const u8, left: *const state.Lane, right: *const state.Lane) !void {
    const want = try state.digest(e.gpa, e, left);
    const got = try state.digest(e.gpa, e, right);
    if (!std.mem.eql(u8, &want, &got)) {
        std.debug.print("FAIL {s}\n", .{label});
        return error.StateBytesDiffer;
    }
}
