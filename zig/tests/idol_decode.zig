//! tf-idol-decode: Idol's octree and Gaussian decoders on a case from tools/zig/idol_case.py decode, gated and timed.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const id = tf.idol;
const dc = id.decoder_config;
const oc = id.octree;
const gs = id.gaussian;
const dr = id.decode_reference;
const ref = id.reference;

const usage =
    \\tf-idol-decode SNAPSHOT_DIR CASE_DIR
    \\tf-idol-decode SNAPSHOT_DIR CASE_DIR time GAUSSIANS [OUT.ply]
    \\SNAPSHOT_DIR is a speridlabs/idol snapshot (config.json, decoder.safetensors, decoder_gs.safetensors). CASE_DIR
    \\holds decode.safetensors, decode.json and reference.ply from `idol_case.py decode` (--ruler: float32 rulers).
    \\Gates, each on the SDK's own inputs: every level's counts from the SDK's probabilities and draws identical; its
    \\probabilities against float32 within twice the worst of the SDK's own distance, the fp16 floor's (float32, its
    \\last rows rounded once) and the SDK's reruns with 1% of a first block's inputs an ulp off; levels 1 to 3 block
    \\by block, rows and updates, within twice the SDK's; the Gaussian decoder's rows as the levels; fields; the PLY's
    \\header, rows and columns; the camera; byte-identical repeats. A free run on the SDK's draws reports where its
    \\topology leaves the SDK's. Writes CASE_DIR/tf-idol-decode.json, .ply and camera.json; exit 1 on a failed gate.
    \\With `time`, decodes case.safetensors's sample at GAUSSIANS with the native generator, twice, timed.
;

fn tensor(f: *const tf.safetensors.File, name: []const u8) !tf.safetensors.Tensor {
    return f.get(name) orelse {
        std.debug.print("the case has no {s}\n", .{name});
        return error.MissingTensor;
    };
}

fn floats(a: std.mem.Allocator, f: *const tf.safetensors.File, name: []const u8) ![]f32 {
    const t = try tensor(f, name);
    if (t.dtype != .f32) return error.BadCaseTensor;
    const out = try a.alloc(f32, t.numel());
    @memcpy(std.mem.sliceAsBytes(out), t.bytes);
    return out;
}

fn ints(a: std.mem.Allocator, f: *const tf.safetensors.File, name: []const u8) ![]u32 {
    const t = try tensor(f, name);
    if (t.dtype != .i32) return error.BadCaseTensor;
    const out = try a.alloc(u32, t.numel());
    @memcpy(std.mem.sliceAsBytes(out), t.bytes);
    return out;
}

fn coords(a: std.mem.Allocator, f: *const tf.safetensors.File, name: []const u8) ![]const [3]u32 {
    const v = try ints(a, f, name);
    return @as([*]const [3]u32, @ptrCast(v.ptr))[0 .. v.len / 3];
}

fn print(list: *std.ArrayList(u8), a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
    try list.appendSlice(a, try std.fmt.allocPrint(a, fmt, args));
}

fn verdict(ok: bool) []const u8 {
    return if (ok) "pass" else "FAIL";
}

/// The SDK's run of the case: levels, draws and jitter.
const Case = struct {
    f: *const tf.safetensors.File,
    parents: [dc.max_level][]const [3]u32,
    draws: [dc.max_level][]f32,
    replay: oc.Replay,
};

/// Each level teacher-forced on the SDK's parents: probabilities, the level modulation, and counts.
fn levels(a: std.mem.Allocator, d: *id.Decoder, c: *const Case, out: *std.ArrayList(u8)) !bool {
    var pass = true;
    try print(out, a, ", \"levels\": [", .{});
    for (1..dc.max_level + 1) |l| {
        var name: [32]u8 = undefined;
        const parents = c.parents[l - 1];
        const rows = parents.len;
        const mods = try floats(a, c.f, try std.fmt.bufPrint(&name, "mods.{d}", .{l}));
        // the case holds adaLN's fp16 output; the blocks read the scales (parts 1 and 4) as fp16(1 + scale)
        var mod_diff: usize = 0;
        for (d.levelMods(l), mods, 0..) |x, y, i| {
            const part = i / dc.hidden;
            const want: f16 = if (part == 1 or part == 4) @floatCast(1 + y) else @floatCast(y);
            mod_diff += @intFromBool(x != want);
        }
        const centres = try a.alloc(f32, rows * 3);
        oc.centres(parents, @intCast(l), centres);
        const ours = try a.alloc(f32, rows * 8);
        _ = try d.octreeLogits(centres, l, ours);
        oc.softmax(ours);
        const sdk = try floats(a, c.f, try std.fmt.bufPrint(&name, "probs.{d}", .{l}));
        const m = ref.match(ours, sdk, 8);
        const w = dr.worst(ours, sdk, 8);
        std.debug.print("level {d}: {d} parents; probabilities against the SDK: max |diff| {e:.3}, cosine {d:.9}, worst row {d:.5} (row {d})\n", .{ l, rows, m.max_abs, m.cosine, w.err, w.row });
        try print(out, a, "{s}{{\"level\": {d}, \"parents\": {d}, \"sdk\": {{\"max_abs\": {e:.4}, \"cosine\": {d:.10}, \"worst_row\": {d:.6}, \"worst_at\": {d}}}", .{ if (l == 1) "" else ", ", l, rows, m.max_abs, m.cosine, w.err, w.row });
        const ruler = try std.fmt.bufPrint(&name, "probs32.{d}", .{l});
        if (c.f.get(ruler) != null) {
            const exact = try floats(a, c.f, ruler);
            const floor = try floats(a, c.f, try std.fmt.bufPrint(&name, "probs32h.{d}", .{l}));
            pass = (try rulerGate(a, out, ours, sdk, exact, floor, try nudges(a, c.f, "probs16n", try std.fmt.allocPrint(a, ".{d}", .{l})), 8)) and pass;
        }
        const counts_in = try ints(a, c.f, try std.fmt.bufPrint(&name, "counts_in.{d}", .{l}));
        const want = try ints(a, c.f, try std.fmt.bufPrint(&name, "counts.{d}", .{l}));
        const level: oc.Level = .{ .coords = parents, .counts = counts_in };
        const theirs = try a.alloc(u32, rows * 8);
        const mine = try a.alloc(u32, rows * 8);
        var replay: oc.Draws = .{ .replay = &c.replay };
        _ = try oc.descend(a, level, sdk, @intCast(l), &replay, theirs);
        _ = try oc.descend(a, level, ours, @intCast(l), &replay, mine);
        const same = dr.counts(theirs, want, 8);
        const drift = dr.counts(mine, want, 8);
        pass = pass and same.rows == 0;
        std.debug.print("         counts from the SDK's probabilities and draws: {d} rows differ: {s}; from ours: {d} rows differ", .{ same.rows, verdict(same.rows == 0), drift.rows });
        if (drift.first) |r| std.debug.print(" (first at parent {d}, cell {any}: ours {any}, the SDK's {any})", .{ r, parents[r], mine[r * 8 ..][0..8], want[r * 8 ..][0..8] });
        std.debug.print("; level modulation: {d} of 6144 fp16 values differ from the SDK's\n", .{mod_diff});
        try print(out, a, ", \"counts_sdk_probs_rows_differ\": {d}, \"counts_our_probs_rows_differ\": {d}, \"first_differing_parent\": {d}, \"mods_differ\": {d}}}", .{ same.rows, drift.rows, if (drift.first) |r| @as(i64, @intCast(r)) else -1, mod_diff });
    }
    try print(out, a, "]", .{});
    return pass;
}

/// The full decode on the SDK's draws: where its topology first leaves the SDK's, and whether the anchors match.
fn freeRun(a: std.mem.Allocator, d: *id.Decoder, c: *const Case, latent: []const f32, n: usize, out: *std.ArrayList(u8)) !id.decode.Result {
    var trace: id.decode.Trace = .{};
    var draws: oc.Draws = .{ .replay = &c.replay };
    const r = try id.decode.decode(a, d, latent, n, &draws, &trace);
    var first: ?usize = null;
    try print(out, a, ", \"free_run\": {{\"parents_equal\": [", .{});
    for (1..dc.max_level + 1) |l| {
        const ours = trace.parents[l - 1];
        const theirs = c.parents[l - 1];
        const equal = ours.len == theirs.len and std.mem.eql(u8, std.mem.sliceAsBytes(ours), std.mem.sliceAsBytes(theirs));
        if (!equal and first == null) first = l;
        try print(out, a, "{s}{}", .{ if (l == 1) "" else ", ", equal });
        if (!equal) std.debug.print("free run: level {d} has {d} parents, the SDK's {d}\n", .{ l, ours.len, theirs.len });
    }
    const sdk_anchors = try floats(a, c.f, "anchors");
    const equal = std.mem.eql(u8, std.mem.sliceAsBytes(r.anchors), std.mem.sliceAsBytes(sdk_anchors));
    std.debug.print("free run on the SDK's draws: topology {s}; anchors {s} the SDK's\n", .{ if (first) |l| try std.fmt.allocPrint(a, "first differs at level {d}", .{l}) else "identical at every level", if (equal) "bit-equal to" else "differ from" });
    try print(out, a, "], \"first_divergent_level\": {d}, \"anchors_equal\": {}", .{ if (first) |l| @as(i64, @intCast(l)) else -1, equal });
    try print(out, a, ", \"octree_s\": {d:.4}, \"gaussian_s\": {d:.4}, \"activation_s\": {d:.4}}}", .{ r.time.octree, r.time.gaussian, r.time.activation });
    return r;
}

/// Ours against float32 within twice the worst of the SDK's own distance, the fp16 floor's and its nudged reruns'.
fn rulerGate(a: std.mem.Allocator, out: *std.ArrayList(u8), ours: []const f32, sdk: []const f32, exact: []const f32, floor: []const f32, nudged: []const []const f32, width: usize) !bool {
    const m = ref.match(ours, exact, width);
    const r = ref.match(sdk, exact, width);
    const f = ref.match(floor, exact, width);
    var all: [8]ref.Match = undefined;
    all[0] = r;
    all[1] = f;
    var worst_nudge: ref.Match = .{ .max_abs = 0, .cosine = 1, .worst_row = 0 };
    for (nudged, 0..) |n, i| {
        all[2 + i] = ref.match(n, exact, width);
        worst_nudge = dr.reference(&.{ worst_nudge, all[2 + i] });
    }
    const w = dr.worst(ours, exact, width);
    const ok = ref.within(m, dr.reference(all[0 .. 2 + nudged.len]));
    std.debug.print("         against float32: worst row {d:.5} (row {d}), 1 - cosine {e:.3}; the SDK's {d:.5}, {e:.3}; nudged, worst {d:.5}, {e:.3}; floor {d:.5}, {e:.3}: {s}\n", .{ m.worst_row, w.row, 1 - m.cosine, r.worst_row, 1 - r.cosine, worst_nudge.worst_row, 1 - worst_nudge.cosine, f.worst_row, 1 - f.cosine, verdict(ok) });
    try print(out, a, ", \"fp32\": {{\"worst_row\": {d:.6}, \"worst_at\": {d}, \"cosine\": {d:.10}, \"sdk_worst_row\": {d:.6}, \"sdk_cosine\": {d:.10}, \"nudged_worst_row\": {d:.6}, \"nudged_cosine\": {d:.10}, \"floor_worst_row\": {d:.6}, \"floor_cosine\": {d:.10}, \"pass\": {}}}", .{ m.worst_row, w.row, m.cosine, r.worst_row, r.cosine, worst_nudge.worst_row, worst_nudge.cosine, f.worst_row, f.cosine, ok });
    return ok;
}

/// The SDK's nudged reruns the case holds: `prefix`, the seed, then `suffix`.
fn nudges(a: std.mem.Allocator, f: *const tf.safetensors.File, prefix: []const u8, suffix: []const u8) ![]const []const f32 {
    var list: std.ArrayList([]const f32) = .empty;
    for (1..4) |seed| {
        const key = try std.fmt.allocPrint(a, "{s}{d}{s}", .{ prefix, seed, suffix });
        if (f.get(key) != null) try list.append(a, try floats(a, f, key));
    }
    return list.items;
}

fn fieldGate(a: std.mem.Allocator, out: *std.ArrayList(u8), name: []const u8, ours: []const f32, theirs: []const f32) !bool {
    const f = dr.field(ours, theirs, 1e-5, 1e-6);
    std.debug.print("activated {s}: max |diff| {e:.3} at {d}, {d} of {d} bit-equal, {d} past 1e-5 rel + 1e-6: {s}\n", .{ name, f.max_abs, f.at, f.exact, ours.len, f.bad, verdict(f.bad == 0) });
    try print(out, a, ", \"{s}\": {{\"max_abs\": {e:.4}, \"at\": {d}, \"exact\": {d}, \"bad\": {d}}}", .{ name, f.max_abs, f.at, f.exact, f.bad });
    return f.bad == 0;
}

/// The Gaussian decoder, the activation and the PLY, each on the SDK's own inputs.
fn gaussians(a: std.mem.Allocator, io: std.Io, d: *id.Decoder, c: *const Case, dir: []const u8, out: *std.ArrayList(u8)) !bool {
    var pass = true;
    const anchors = try floats(a, c.f, "anchors");
    const sdk = try floats(a, c.f, "raw");
    const raw = try a.alloc(f32, sdk.len);
    _ = try d.gaussianRows(anchors, raw);
    const m = ref.match(raw, sdk, dc.raw_width);
    const w = dr.worst(raw, sdk, dc.raw_width);
    std.debug.print("Gaussian decoder on the SDK's {d} anchors against the SDK: max |diff| {e:.3}, cosine {d:.9}, worst row {d:.5} (anchor {d})\n", .{ anchors.len / 3, m.max_abs, m.cosine, w.err, w.row });
    try print(out, a, ", \"gaussian_decoder\": {{\"sdk\": {{\"max_abs\": {e:.4}, \"cosine\": {d:.10}, \"worst_row\": {d:.6}, \"worst_at\": {d}}}", .{ m.max_abs, m.cosine, w.err, w.row });
    if (c.f.get("raw32") != null) {
        const nudged = try nudges(a, c.f, "raw16n", "");
        pass = (try rulerGate(a, out, raw, sdk, try floats(a, c.f, "raw32"), try floats(a, c.f, "raw32h"), nudged, dc.raw_width)) and pass;
    }
    try print(out, a, "}}, \"activated\": {{", .{});
    const n = anchors.len / 3 * dc.per_anchor;
    const g = try gs.Gaussians.alloc(a, n);
    id.decode.activateParallel(d.params(), sdk, anchors, g);
    const theirs: gs.Gaussians = .{ .xyz = try floats(a, c.f, "xyz"), .features_dc = try floats(a, c.f, "features_dc"), .scaling = try floats(a, c.f, "scaling"), .rotation = try floats(a, c.f, "rotation"), .opacity = try floats(a, c.f, "opacity") };
    try print(out, a, "\"rows\": {d}", .{n});
    inline for (.{ "xyz", "features_dc", "scaling", "rotation", "opacity" }) |name| {
        pass = (try fieldGate(a, out, name, @field(g, name), @field(theirs, name))) and pass;
    }
    const reference = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/reference.ply", .{dir}), a, .limited(1 << 31));
    const mine = try gs.ply(a, theirs);
    const p = dr.ply(mine, reference);
    const ok = dr.plyPass(p, 1e-5);
    pass = pass and ok;
    std.debug.print("PLY of the SDK's Gaussians against its reference.ply: header {s}, {d} rows {s}; columns differing:", .{ if (p.header_equal) "equal" else "DIFFERENT", p.rows, if (p.rows_equal) "equal" else "DIFFERENT" });
    for (gs.properties, p.differing, p.max_abs) |name, k, mx| if (k > 0) std.debug.print(" {s} {d} (max {e:.2})", .{ name, k, mx });
    std.debug.print(": {s}\n", .{verdict(ok)});
    try print(out, a, "}}, \"ply\": {{\"header_equal\": {}, \"rows_equal\": {}, \"differing\": {any}, \"max_abs\": [", .{ p.header_equal, p.rows_equal, p.differing });
    for (p.max_abs, 0..) |mx, i| try print(out, a, "{s}{e:.3}", .{ if (i == 0) "" else ", ", mx });
    try print(out, a, "], \"pass\": {}}}", .{ok});
    return pass;
}

fn camera(a: std.mem.Allocator, io: std.Io, c: *const Case, dir: []const u8, out: *std.ArrayList(u8)) !bool {
    const v = try floats(a, c.f, "camera");
    const cam = gs.camera(v[0..5].*);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/decode.json", .{dir}), a, .limited(1 << 20));
    const doc = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    const sdk = doc.value.object.get("camera") orelse return error.BadCase;
    var most: f64 = 0;
    inline for (.{ "yaw", "pitch", "radius", "fov" }) |k| {
        const s = sdk.object.get(k) orelse return error.BadCase;
        const want: f64 = switch (s) {
            .float => |x| x,
            .integer => |x| @floatFromInt(x),
            else => return error.BadCase,
        };
        most = @max(most, @abs(@field(cam, k) - want));
    }
    const json = try std.fmt.allocPrint(a, "{{\"yaw\": {d}, \"pitch\": {d}, \"radius\": {d}, \"fov\": {d}}}\n", .{ cam.yaw, cam.pitch, cam.radius, cam.fov });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/camera.json", .{dir}), .data = json });
    const ok = most <= 1e-6;
    std.debug.print("camera {s}against the SDK's: max |diff| {e:.3}: {s}\n", .{ json, most, verdict(ok) });
    try print(out, a, ", \"camera_max_abs\": {e:.3}", .{most});
    return ok;
}

/// The whole decode twice for each generator: identical PLY bytes, or not.
fn repeats(a: std.mem.Allocator, d: *id.Decoder, c: *const Case, latent: []const f32, n: usize, first: []const u8) !bool {
    var draws: oc.Draws = .{ .replay = &c.replay };
    var again = try id.decode.decode(a, d, latent, n, &draws, null);
    const same_replay = std.mem.eql(u8, first, try gs.ply(a, again.gaussians));
    var bytes: [2][]u8 = undefined;
    for (&bytes) |*b| {
        var native: oc.Draws = .{ .native = std.Random.DefaultPrng.init(42) };
        again = try id.decode.decode(a, d, latent, n, &native, null);
        b.* = try gs.ply(a, again.gaussians);
    }
    const same_native = std.mem.eql(u8, bytes[0], bytes[1]);
    std.debug.print("the decode again on the SDK's draws: {s}; twice on the native generator: {s}\n", .{ if (same_replay) "identical bytes" else "DIFFERENT bytes", if (same_native) "identical bytes" else "DIFFERENT bytes" });
    return same_replay and same_native;
}

fn timed(a: std.mem.Allocator, io: std.Io, device: mtl.Device, args: []const [:0]const u8) !u8 {
    const n = try std.fmt.parseInt(usize, args[4], 10);
    var case = try tf.safetensors.File.open(a, io, try std.fmt.allocPrint(a, "{s}/case.safetensors", .{args[2]}));
    defer case.close(io);
    const latent = try floats(a, &case, "sample");
    const loading = mtl.clock.seconds();
    const d = try id.Decoder.load(std.heap.page_allocator, io, device, args[1], try dc.anchors(n));
    defer d.deinit();
    const load_s = mtl.clock.seconds() - loading;
    var line: std.ArrayList(u8) = .empty;
    try print(&line, a, "{{\"device\": \"{s}\", \"gaussians\": {d}, \"load_s\": {d:.2}", .{ device.name(), n, load_s });
    for ([_][]const u8{ "cold", "warm" }) |label| {
        var draws: oc.Draws = .{ .native = std.Random.DefaultPrng.init(42) };
        const started = mtl.clock.seconds();
        var r = try id.decode.decode(a, d, latent, n, &draws, null);
        const decoded = mtl.clock.seconds() - started;
        const writing = mtl.clock.seconds();
        const bytes = try gs.ply(a, r.gaussians);
        if (args.len == 6) try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[5], .data = bytes });
        const ply_s = mtl.clock.seconds() - writing;
        const t = r.time;
        std.debug.print("{s}: decode {d:.3} s (octree {d:.3}, {d:.3} on the GPU; Gaussian decoder {d:.3}, {d:.3} on the GPU; activation {d:.3}); PLY {d:.3} s\n", .{ label, decoded, t.octree, t.octree_gpu, t.gaussian, t.gaussian_gpu, t.activation, ply_s });
        try print(&line, a, ", \"{s}\": {{\"decode_s\": {d:.4}, \"octree_s\": {d:.4}, \"octree_gpu_s\": {d:.4}, \"gaussian_s\": {d:.4}, \"gaussian_gpu_s\": {d:.4}, \"activation_s\": {d:.4}, \"ply_s\": {d:.4}}}", .{ label, decoded, t.octree, t.octree_gpu, t.gaussian, t.gaussian_gpu, t.activation, ply_s });
        r.deinit(a);
        a.free(bytes);
    }
    try print(&line, a, ", \"metal_allocated_bytes\": {d}}}\n", .{device.allocated()});
    std.debug.print("{s}", .{line.items});
    return 0;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    const time_mode = args.len >= 5 and std.mem.eql(u8, args[3], "time");
    if (!(args.len == 3 or (time_mode and args.len <= 6))) {
        std.debug.print("{s}\n", .{usage});
        return 2;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    if (time_mode) return timed(std.heap.page_allocator, io, device, args);
    const dir = args[2];
    var file = try tf.safetensors.File.open(a, io, try std.fmt.allocPrint(a, "{s}/decode.safetensors", .{dir}));
    defer file.close(io);
    var c: Case = .{ .f = &file, .parents = undefined, .draws = undefined, .replay = undefined };
    var name: [32]u8 = undefined;
    for (0..dc.max_level) |i| {
        c.parents[i] = try coords(a, &file, try std.fmt.bufPrint(&name, "parents.{d}", .{i + 1}));
        c.draws[i] = try floats(a, &file, try std.fmt.bufPrint(&name, "draws.{d}", .{i + 1}));
    }
    c.replay = .{ .parents = c.parents, .draws = undefined, .jitter = try floats(a, &file, "jitter") };
    for (&c.replay.draws, c.draws) |*o, v| o.* = v;
    const latent = try floats(a, &file, "latent");
    const n = (try tensor(&file, "anchors")).numel() / 3 * dc.per_anchor;
    const d = try id.Decoder.load(std.heap.page_allocator, io, device, args[1], try dc.anchors(n));
    defer d.deinit();
    try d.setLatent(latent);
    var out: std.ArrayList(u8) = .empty;
    try print(&out, a, "{{\"device\": \"{s}\", \"gaussians\": {d}", .{ device.name(), n });
    const sdk_freqs = try floats(a, &file, "freqs");
    var freq_diff: usize = 0;
    for (d.pointFreqs(), sdk_freqs) |x, y| freq_diff += @intFromBool(x != y);
    std.debug.print("point embedding frequencies: {d} of {d} differ from torch's by its CPU pow's rounding\n", .{ freq_diff, sdk_freqs.len });
    try print(&out, a, ", \"freqs_differ\": {d}", .{freq_diff});
    var pass = try levels(a, d, &c, &out);
    if (file.get("probs32.1") != null) pass = (try blockGates(a, d, &c, &out)) and pass;
    var run = try freeRun(a, d, &c, latent, n, &out);
    const ours = try gs.ply(a, run.gaussians);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/tf-idol-decode.ply", .{dir}), .data = ours });
    const reference = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/reference.ply", .{dir}), a, .limited(1 << 31));
    const whole = dr.ply(ours, reference);
    const shape = whole.header_equal and whole.rows_equal;
    pass = pass and shape;
    std.debug.print("our own PLY: header {s} the SDK's, rows {s}: {s}\n", .{ if (whole.header_equal) "equal to" else "DIFFERENT from", if (whole.rows_equal) "equal" else "DIFFERENT", verdict(shape) });
    try print(&out, a, ", \"own_ply_header_equal\": {}, \"own_ply_rows_equal\": {}", .{ whole.header_equal, whole.rows_equal });
    pass = (try gaussians(a, io, d, &c, dir, &out)) and pass;
    pass = (try camera(a, io, &c, dir, &out)) and pass;
    const repeat = try repeats(a, d, &c, latent, n, ours);
    run.deinit(a);
    pass = pass and repeat;
    try print(&out, a, ", \"repeat_identical\": {}, \"gates_pass\": {}}}\n", .{ repeat, pass });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/tf-idol-decode.json", .{dir}), .data = out.items });
    std.debug.print("gates: {s}\n", .{if (pass) "pass" else "FAIL"});
    return if (pass) 0 else 1;
}

/// Levels 1 to 3 block by block, each block alone on the SDK's own input rows against that block in float32.
fn blockGates(a: std.mem.Allocator, d: *id.Decoder, c: *const Case, out: *std.ArrayList(u8)) !bool {
    var pass = true;
    var name: [32]u8 = undefined;
    try print(out, a, ", \"blocks\": [", .{});
    for (1..4) |l| {
        const parents = c.parents[l - 1];
        const centres = try a.alloc(f32, parents.len * 3);
        oc.centres(parents, @intCast(l), centres);
        var input = try floats(a, c.f, try std.fmt.bufPrint(&name, "stem.{d}", .{l}));
        const zero = try a.alloc(f16, input.len);
        @memset(zero, 0);
        const stem = try a.alloc(f32, input.len);
        for (stem, try d.octreeRows(centres, zero, l, null)) |*o, v| o.* = v;
        const ms = ref.match(stem, input, dc.hidden);
        std.debug.print("level {d} stem against the SDK's: max |diff| {e:.3}, worst row {d:.6}\n", .{ l, ms.max_abs, ms.worst_row });
        try print(out, a, "{s}{{\"level\": {d}, \"stem_max_abs\": {e:.3}, \"stem_worst_row\": {d:.6}, \"each\": [", .{ if (l == 1) "" else ", ", l, ms.max_abs, ms.worst_row });
        for (0..dc.octree_blocks) |b| {
            const in16 = try a.alloc(f16, input.len);
            for (in16, input) |*o, v| o.* = @floatCast(v);
            const ours = try a.alloc(f32, input.len);
            for (ours, try d.octreeRows(centres, in16, l, b)) |*o, v| o.* = v;
            const sdk = try floats(a, c.f, try std.fmt.bufPrint(&name, "block{d}.{d}", .{ b, l }));
            const exact = try floats(a, c.f, try std.fmt.bufPrint(&name, "block{d}_f32.{d}", .{ b, l }));
            const m = ref.match(ours, exact, dc.hidden);
            const r = ref.match(sdk, exact, dc.hidden);
            // the residual update (output less the fp16 input) too: the output's own rounding hides it in the rows
            const deltas = try a.alloc(f32, 3 * input.len);
            for (0..input.len) |i| {
                const base: f32 = in16[i];
                deltas[i] = ours[i] - base;
                deltas[input.len + i] = sdk[i] - base;
                deltas[2 * input.len + i] = exact[i] - base;
            }
            const dm = ref.match(deltas[0..input.len], deltas[2 * input.len ..], dc.hidden);
            const dsdk = ref.match(deltas[input.len .. 2 * input.len], deltas[2 * input.len ..], dc.hidden);
            const ok = ref.within(m, r) and ref.within(dm, dsdk);
            pass = pass and ok;
            std.debug.print("  block {d} against float32: worst row {d:.6}, 1 - cosine {e:.3}; the SDK's own {d:.6}, {e:.3}; its update {d:.5}, {e:.3}; the SDK's {d:.5}, {e:.3}: {s}\n", .{ b, m.worst_row, 1 - m.cosine, r.worst_row, 1 - r.cosine, dm.worst_row, 1 - dm.cosine, dsdk.worst_row, 1 - dsdk.cosine, verdict(ok) });
            try print(out, a, "{s}{{\"worst_row\": {d:.6}, \"cosine\": {d:.10}, \"sdk_worst_row\": {d:.6}, \"sdk_cosine\": {d:.10}, \"update_worst_row\": {d:.6}, \"update_cosine\": {d:.10}, \"sdk_update_worst_row\": {d:.6}, \"sdk_update_cosine\": {d:.10}, \"pass\": {}}}", .{ if (b == 0) "" else ", ", m.worst_row, m.cosine, r.worst_row, r.cosine, dm.worst_row, dm.cosine, dsdk.worst_row, dsdk.cosine, ok });
            input = sdk;
        }
        try print(out, a, "]}}", .{});
    }
    try print(out, a, "]", .{});
    return pass;
}
