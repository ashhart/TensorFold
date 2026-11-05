//! Native HIP byte parity checks over image prompts, shared forwards, partial keeps and snapshot callbacks.
const std = @import("std");
const hip = @import("hip");
const family = @import("lighton_ocr");
const lanes = @import("lanes");
const fixture = @import("fixture.zig");
const Allocator = std.mem.Allocator;
const Graph = family.native.Capture;
const Be = lanes.backend;

const Result = struct {
    name: []const u8,
    expected_bytes: usize,
    actual_bytes: usize,
    different: usize,
    first: ?usize,
    expected_sha256: [64]u8,
    actual_sha256: [64]u8,
};

const Checks = struct {
    a: Allocator,
    results: std.ArrayList(Result) = .empty,
    failed: bool = false,

    fn equal(c: *Checks, name: []const u8, expected: []const u8, actual: []const u8) !void {
        const common = @min(expected.len, actual.len);
        var different = @max(expected.len, actual.len) - common;
        var first: ?usize = if (different > 0) common else null;
        for (expected[0..common], actual[0..common], 0..) |want, got, i| if (want != got) {
            different += 1;
            first = @min(first orelse i, i);
        };
        c.failed = c.failed or different > 0;
        try c.results.append(c.a, .{
            .name = name,
            .expected_bytes = expected.len,
            .actual_bytes = actual.len,
            .different = different,
            .first = first,
            .expected_sha256 = sha(expected),
            .actual_sha256 = sha(actual),
        });
        std.debug.print("{s}: different={d}/{d} first={?d}\n", .{ name, different, expected.len, first });
    }
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4 or args.len > 5) {
        std.debug.print("usage: tf-lighton-hip-exact MODEL REFERENCE_DIR OUTPUT_DIR [DEVICE]\n", .{});
        return 2;
    }
    var files = try fixture.Files.init(a, init.io, args[2], args[3]);
    defer files.output.close(init.io);
    try files.verifyCheckpoint(args[1]);
    const tensors = try fixture.field(files.manifest, "tensors");
    if (tensors != .object) return error.BadFixture;
    var records = tensors.object.iterator();
    while (records.next()) |record| _ = try files.read(record.key_ptr.*);
    const c = try family.config.Config.read(a, init.io, args[1]);
    const request = try fixture.field(files.manifest, "request");
    if ((try fixture.list(try fixture.field(request, "images_sha256"))).len != 1) return error.OneImageRequired;
    const prompt = try ids(a, try fixture.field(request, "prompt_ids"), c.vocab);
    const teacher = try ids(a, try fixture.field(request, "continuation_ids"), c.vocab);
    if (prompt.len == 0) return error.EmptyPrompt;
    const tokens = try std.mem.concat(a, u32, &.{ prompt, teacher });
    const window = @max(tokens.len, prompt.len + 8);
    if (window > c.context) return error.PastContext;
    const pixels = try files.floats("processor.pixel_values");
    const patches = try fixture.patchify(a, pixels.array.shape[0..pixels.array.rank], pixels.data);
    const shape = try family.tower.Shape.init(c, patches.gh, patches.gw, patches.data.len);
    const image_first = try imageRange(prompt, c.image_token, shape.tokens);
    const device = if (args.len == 5) try std.fmt.parseInt(c_int, args[4], 10) else 0;
    if (device < 0) return error.BadDevice;
    const caps = hip.Device.capsOf(device) orelse return error.UnsupportedGpu;
    const dev: hip.Device = .{ .index = device, .caps = caps, .policy = .{ .slots = 2, .graphs = .off } };
    var problem: []const u8 = "";
    const loaded = family.native.open(a, init.gpa, init.io, dev, args[1], family.native.Options{
        .window = window,
        .streams = 2,
        .fixed = true,
        .cache_gib = 0.01,
        .vision = .{ .image_tokens = @intCast(shape.tokens), .max_images = 1 },
    }, &problem) catch |err| {
        std.debug.print("native HIP load failed: {s}\n", .{problem});
        return err;
    };
    defer loaded.deinit(loaded.ctx);
    const graph = try loaded.capture();
    std.debug.print("{s}\n", .{loaded.startup});
    var projected = try hip.DeviceBuffer.alloc(graph.r, shape.tokens * c.hidden * 2);
    defer projected.free();
    defer family.snapshot.drain(graph.s);
    try graph.tower.encode(patches.data, patches.gh, patches.gw, projected.base());
    const input: Input = .{ .a = a, .graph = graph, .image_first = image_first, .image_tokens = shape.tokens, .features = projected.base() };
    var checks: Checks = .{ .a = a };
    const wide = try input.sequence(tokens, 0, family.text.max_rows);
    const serial = try input.sequence(tokens, 1, 1);
    try checks.equal("per_row_vs_wide_logits", wide, serial);
    try checks.equal("per_row_vs_wide_head_major_kv", try prefix(a, graph, 0, tokens.len), try prefix(a, graph, 1, tokens.len));

    const other = try a.dupe(u32, tokens);
    const changed = for (other, 0..) |token, i| {
        if (token != c.image_token) break i;
    } else return error.NoTextToken;
    other[changed] = (other[changed] + 1) % c.vocab;
    const alone_other = try input.sequence(other, 1, family.text.max_rows);
    const alone_kv = [_][]const u8{ try prefix(a, graph, 0, tokens.len), try prefix(a, graph, 1, tokens.len) };
    const shared = try input.shared(tokens, other);
    try checks.equal("shared_span_0_vs_alone_logits", wide, shared[0]);
    try checks.equal("shared_span_1_vs_alone_logits", alone_other, shared[1]);
    for (0..2) |slot| try checks.equal(try std.fmt.allocPrint(a, "shared_span_{d}_vs_alone_kv", .{slot}), alone_kv[slot], try prefix(a, graph, slot, tokens.len));

    const image: lanes.stream.Image = .{ .pixels = patches.data, .gh = patches.gh, .gw = patches.gw, .at = @intCast(image_first), .tokens = @intCast(shape.tokens) };
    const direct_prefix = try prefix(a, graph, 0, prompt.len);
    const production = try backendChecks(a, init.gpa, loaded, graph, prompt, image, direct_prefix, &checks);
    const manifest_path = try std.fs.path.join(a, &.{ args[2], "manifest.json" });
    const manifest_bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, manifest_path, a, .limited(8 << 20));
    const receipt = try std.json.Stringify.valueAlloc(a, .{
        .schema = "tensorfold.lighton_hip.exact.v1",
        .scope = "native engine byte equality; no cross-framework tolerance or qualification claim",
        .passed = !checks.failed,
        .checkpoint = try fixture.field(files.manifest, "checkpoint"),
        .checkpoint_checksums_verified = true,
        .reference_tensor_checksums_verified = tensors.object.count(),
        .input_manifest_sha256 = sha(manifest_bytes),
        .request = request,
        .device = device,
        .architecture = hip.kernels.arch,
        .prompt_rows = prompt.len,
        .teacher_rows = teacher.len,
        .changed_other_stream_row = changed,
        .shared_widths = .{ 47, 81 },
        .checks = checks.results.items,
        .production = production,
    }, .{ .whitespace = .indent_2 });
    try files.output.writeFile(init.io, .{ .sub_path = "exact.json", .data = receipt });
    return if (checks.failed) 1 else 0;
}

const Input = struct {
    a: Allocator,
    graph: Graph,
    image_first: usize,
    image_tokens: usize,
    features: u64,

    fn imageRows(in: Input, at: usize, count: usize, flat: usize) ?family.text.ImageRows {
        const start = @max(at, in.image_first);
        const end = @min(at + count, in.image_first + in.image_tokens);
        if (start >= end) return null;
        return .{ .first = @intCast(flat + start - at), .rows = @intCast(end - start), .features = in.features + (start - in.image_first) * in.graph.c.hidden * 2 };
    }

    fn sequence(in: Input, tokens: []const u32, slot: u32, width: usize) ![]u8 {
        const row_bytes = @as(usize, in.graph.c.vocab) * 2;
        const out = try in.a.alloc(u8, tokens.len * row_bytes);
        var at: usize = 0;
        while (at < tokens.len) {
            const count = @min(width, tokens.len - at);
            var images: [1]family.text.ImageRows = undefined;
            const rows: []const family.text.ImageRows = if (in.imageRows(at, count, 0)) |r| blk: {
                images[0] = r;
                break :blk &images;
            } else &.{};
            try in.graph.text.forward(tokens[at..][0..count], &.{.{ .slot = slot, .first = 0, .rows = @intCast(count), .committed = @intCast(at) }}, rows);
            try read(in.graph, in.graph.text.logits, out[at * row_bytes ..][0 .. count * row_bytes]);
            at += count;
        }
        try finite(out);
        return out;
    }

    fn shared(in: Input, first: []const u32, second: []const u32) ![2][]u8 {
        const row_bytes = @as(usize, in.graph.c.vocab) * 2;
        const out = [2][]u8{ try in.a.alloc(u8, first.len * row_bytes), try in.a.alloc(u8, second.len * row_bytes) };
        const sources = [2][]const u32{ first, second };
        var at = [2]usize{ 0, 0 };
        while (at[0] < first.len or at[1] < second.len) {
            var tokens: [family.text.max_rows]u32 = undefined;
            var spans: [2]family.text.Span = undefined;
            var images: [2]family.text.ImageRows = undefined;
            var starts: [2]usize = undefined;
            var counts: [2]usize = undefined;
            var total: usize = 0;
            var nspans: usize = 0;
            var nimages: usize = 0;
            for (sources, [_]usize{ 47, 81 }, 0..) |source, width, slot| {
                const count = @min(width, source.len - at[slot]);
                starts[slot] = total;
                counts[slot] = count;
                if (count == 0) continue;
                @memcpy(tokens[total..][0..count], source[at[slot]..][0..count]);
                spans[nspans] = .{ .slot = @intCast(slot), .first = @intCast(total), .rows = @intCast(count), .committed = @intCast(at[slot]) };
                nspans += 1;
                if (in.imageRows(at[slot], count, total)) |r| {
                    images[nimages] = r;
                    nimages += 1;
                }
                total += count;
            }
            try in.graph.text.forward(tokens[0..total], spans[0..nspans], images[0..nimages]);
            for (0..2) |slot| {
                try read(in.graph, in.graph.text.logits + starts[slot] * row_bytes, out[slot][at[slot] * row_bytes ..][0 .. counts[slot] * row_bytes]);
                at[slot] += counts[slot];
            }
        }
        for (out) |values| try finite(values);
        return out;
    }
};

const Production = struct { pending: u32, accepted_draft: u32, rejected_draft: u32, correction: u32, kept_rows: usize };

fn backendChecks(a: Allocator, gpa: Allocator, loaded: family.native.Loaded, graph: Graph, prompt: []const u32, image: lanes.stream.Image, direct_prefix: []const u8, checks: *Checks) !Production {
    var s0 = try lanes.Stream.init(gpa, .{ .id = "partial-keep", .prompt = prompt, .max_new = 8, .images = &.{image} });
    defer s0.deinit(gpa);
    var s1 = try lanes.Stream.init(gpa, .{ .id = "sequential", .prompt = prompt, .max_new = 8, .images = &.{image} });
    defer s1.deinit(gpa);
    const b = loaded.backend;
    defer b.release(&s0);
    defer b.release(&s1);
    try b.prefill(&s0);
    try b.prefill(&s1);
    const snapshots = (loaded.offer.cache orelse return error.NoSnapshotCallbacks).family;
    const saved = try snapshots.vtable.save(snapshots.ptr, &s0, @intCast(prompt.len));
    defer snapshots.vtable.drop(snapshots.ptr, saved);
    const host: *const family.snapshot.Saved = @ptrCast(@alignCast(saved));
    const expected_prefix = try prefix(a, graph, 0, prompt.len);
    try checks.equal("production_image_prefill_vs_direct_graph", direct_prefix, expected_prefix);
    try checks.equal("snapshot_saved_vs_independent_head_major_download", expected_prefix, host.host.bytes);
    try checks.equal("production_two_prompt_slots", expected_prefix, try prefix(a, graph, 1, prompt.len));
    const pending = try b.read(try b.first(&s0, prompt.len));
    const pending1 = try b.read(try b.first(&s1, prompt.len));
    try checks.equal("production_first_tokens", std.mem.asBytes(&pending), std.mem.asBytes(&pending1));
    const next = try b.read(try b.queue(&s1, .{ .value = pending }, prompt.len + 1));
    const next_logits = try logits(a, graph);
    const correction = try b.read(try b.queue(&s1, .{ .value = next }, prompt.len + 2));
    try snapshots.vtable.restore(snapshots.ptr, &s1, saved);
    const wrong = (correction + 1) % graph.c.vocab;
    const positions = [_]u64{ prompt.len + 1, prompt.len + 2, prompt.len + 3, prompt.len + 4 };
    const windows = [_]Be.Window{
        .{ .stream = &s0, .pending = pending, .held = 0, .tokens = &.{ next, wrong, next }, .parents = null, .positions = &positions },
        .{ .stream = &s1, .pending = pending, .held = 0, .tokens = &.{next}, .parents = null, .positions = positions[0..2] },
    };
    var sampled0: [4]u32 = undefined;
    var sampled1: [2]u32 = undefined;
    var drafts0: [3]u32 = undefined;
    var drafts1: [1]u32 = undefined;
    var verified = [_]Be.Verified{ .{ .sampled = &sampled0, .drafts = &drafts0 }, .{ .sampled = &sampled1, .drafts = &drafts1 } };
    try b.verify(&windows, &verified);
    try checks.equal("production_shared_verify_expected_picks", std.mem.sliceAsBytes(&[_]u32{ next, correction }), std.mem.sliceAsBytes(sampled0[0..2]));
    try checks.equal("production_shared_verify_equal_prefix", std.mem.sliceAsBytes(&sampled1), std.mem.sliceAsBytes(sampled0[0..2]));
    const parents = try lanes.accept.rowParents(a, 4, null);
    const path = try lanes.accept.acceptPath(a, &.{ pending, next, wrong, next }, parents, &sampled0);
    try checks.equal("partial_acceptance_path", std.mem.sliceAsBytes(&[_]u32{ 0, 1 }), std.mem.sliceAsBytes(path));
    if (path.len != 2) return .{ .pending = pending, .accepted_draft = next, .rejected_draft = wrong, .correction = correction, .kept_rows = path.len };
    try b.keep(&windows, &.{ path, &.{ 0, 1 } });
    const resumed0 = try b.read(try b.queue(&s0, .{ .value = correction }, prompt.len + 3));
    const resumed_logits0 = try logits(a, graph);
    const resumed1 = try b.read(try b.queue(&s1, .{ .value = correction }, prompt.len + 3));
    try checks.equal("partial_keep_continuation_token", std.mem.asBytes(&resumed1), std.mem.asBytes(&resumed0));
    try checks.equal("partial_keep_continuation_logits", try logits(a, graph), resumed_logits0);
    try checks.equal("partial_keep_continuation_kv", try prefix(a, graph, 1, prompt.len + 3), try prefix(a, graph, 0, prompt.len + 3));

    try corruptSlot(graph, 0);
    const damaged = try prefix(a, graph, 0, prompt.len);
    if (std.mem.eql(u8, expected_prefix, damaged)) return error.CorruptionControlFailed;
    try snapshots.vtable.restore(snapshots.ptr, &s0, saved);
    try checks.equal("snapshot_restore_after_cache_corruption", expected_prefix, try prefix(a, graph, 0, prompt.len));
    const untouched = try tail(a, graph, 0, prompt.len);
    const sentinel = try a.alloc(u8, untouched.len);
    @memset(sentinel, 0xa5);
    try checks.equal("snapshot_restore_preserves_uncommitted_tail", sentinel, untouched);
    const restored_next = try b.read(try b.queue(&s0, .{ .value = pending }, prompt.len + 1));
    try checks.equal("snapshot_restore_continuation_token", std.mem.asBytes(&next), std.mem.asBytes(&restored_next));
    try checks.equal("snapshot_restore_continuation_logits", next_logits, try logits(a, graph));
    return .{ .pending = pending, .accepted_draft = next, .rejected_draft = wrong, .correction = correction, .kept_rows = path.len };
}

fn read(graph: Graph, address: u64, out: []u8) !void {
    try graph.s.synchronize();
    if (out.len == 0) return;
    try hip.runtime.check(graph.r.api.hipMemcpyDtoH(out.ptr, @ptrFromInt(address), out.len));
}

fn logits(a: Allocator, graph: Graph) ![]u8 {
    const out = try a.alloc(u8, @as(usize, graph.c.vocab) * 2);
    try read(graph, graph.text.logits, out);
    try finite(out);
    return out;
}

fn finite(values: []const u8) !void {
    if (values.len % 2 != 0) return error.ParityShape;
    var at: usize = 0;
    while (at < values.len) : (at += 2) {
        const word = std.mem.readInt(u16, values[at..][0..2], .little);
        if (word & 0x7f80 == 0x7f80) return error.NonfiniteNativeLogits;
    }
}

/// This address walk is independent of the production snapshot implementation and checks every head's prefix.
fn prefix(a: Allocator, graph: Graph, slot: usize, at: usize) ![]u8 {
    const head_bytes = at * graph.c.head_dim * 2;
    const out = try a.alloc(u8, graph.c.layers * 2 * graph.c.kv_heads * head_bytes);
    var offset: usize = 0;
    for (0..graph.c.layers) |layer| for ([_]u64{ graph.text.key(slot, layer), graph.text.value(slot, layer) }) |base| {
        for (0..graph.c.kv_heads) |head| {
            try read(graph, base + head * graph.text.context * graph.c.head_dim * 2, out[offset..][0..head_bytes]);
            offset += head_bytes;
        }
    };
    return out;
}

fn corruptSlot(graph: Graph, slot: usize) !void {
    try graph.s.synchronize();
    const size = graph.text.context * graph.c.kvWidth() * 2;
    for (0..graph.c.layers) |layer| for ([_]u64{ graph.text.key(slot, layer), graph.text.value(slot, layer) }) |base| {
        try hip.runtime.check(graph.r.api.hipMemset(@ptrFromInt(base), 0xa5, size));
    };
    try graph.s.synchronize();
}

fn tail(a: Allocator, graph: Graph, slot: usize, at: usize) ![]u8 {
    const out = try a.alloc(u8, graph.c.layers * 2 * graph.c.kv_heads * 2);
    var offset: usize = 0;
    for (0..graph.c.layers) |layer| for ([_]u64{ graph.text.key(slot, layer), graph.text.value(slot, layer) }) |base| {
        for (0..graph.c.kv_heads) |head| {
            try read(graph, base + (head * graph.text.context + at) * graph.c.head_dim * 2, out[offset..][0..2]);
            offset += 2;
        }
    };
    return out;
}

fn sha(bytes: []const u8) [64]u8 {
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
    return std.fmt.bytesToHex(hash, .lower);
}

fn ids(a: Allocator, value: std.json.Value, vocab: u32) ![]u32 {
    const values = try fixture.list(value);
    const out = try a.alloc(u32, values.len);
    for (out, values) |*dst, v| {
        const id = try fixture.integer(v);
        if (id >= vocab) return error.TextToken;
        dst.* = @intCast(id);
    }
    return out;
}

fn imageRange(prompt: []const u32, token: u32, count: usize) !usize {
    const first = std.mem.indexOfScalar(u32, prompt, token) orelse return error.ImageRows;
    if (count == 0 or count > prompt.len - first) return error.ImageRows;
    for (prompt[first..][0..count]) |id| if (id != token) return error.ImageRows;
    if (std.mem.indexOfScalar(u32, prompt[first + count ..], token) != null) return error.ImageRows;
    return first;
}
