//! zonos3-generate --steer-check: a dumped request's controls resolved here against the reference's own deltas.
const std = @import("std");
const z3 = @import("zonos3");

const D = z3.config.dim;
const E = z3.config.speaker_dim;
const bundle = z3.steer_bundle;

fn doses(a: std.mem.Allocator, v: ?std.json.Value) ![]const bundle.Dose {
    const o = (v orelse return &.{}).object;
    const out = try a.alloc(bundle.Dose, o.count());
    for (out, o.keys(), o.values()) |*d, id, x| d.* = .{ .id = id, .value = number(x) };
    return out;
}

fn number(v: std.json.Value) f64 {
    return if (v == .float) v.float else @floatFromInt(v.integer);
}

/// The steering request a dump's steer.json holds; `base` takes its speaker embedding, when it has one.
pub fn dumped(a: std.mem.Allocator, side: std.json.ObjectMap, base: *[E]f32) !bundle.Request {
    const sent = side.get("request").?.object;
    var r: bundle.Request = .{ .controls = try doses(a, sent.get("controls")), .chars = try std.unicode.utf8CountCodepoints(sent.get("text").?.string) };
    if (sent.get("control_spans")) |list| {
        const spans = try a.alloc(bundle.SpanRequest, list.array.items.len);
        for (spans, list.array.items) |*s, v| s.* = .{ .start = v.object.get("start").?.integer, .end = v.object.get("end").?.integer, .controls = try doses(a, v.object.get("controls")) };
        r.spans = spans;
    }
    if (sent.get("control_span_rate")) |v| r.rate = number(v);
    if (sent.get("steer_hold")) |v| r.hold = number(v);
    if (sent.get("steer_onset_seconds")) |v| r.onset_seconds = number(v);
    if (sent.get("steer_ramp_seconds")) |v| r.ramp_seconds = number(v);
    if (sent.get("speaker_embedding")) |v| {
        for (base, v.array.items) |*x, y| x.* = @floatCast(number(y));
        r.embedding = base;
    }
    return r;
}

/// GEN_DIR/steer.json's request resolved with the bundle at `path`: the embedding and every frame's delta, bit for bit.
pub fn run(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, path: []const u8) !u8 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    var b = try bundle.load(gpa, io, path);
    defer b.deinit();
    const side = (try std.json.parseFromSliceLeaky(std.json.Value, a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "steer.json" }), a, .limited(1 << 22)), .{})).object;
    var base: [E]f32 = undefined;
    const r = try dumped(a, side, &base);
    var message: []const u8 = "";
    const got = bundle.resolve(a, &b, r, &message) catch |e| {
        std.debug.print("RESULT zonos3 steer FAIL {s}: {s}\n", .{ @errorName(e), message });
        return 1;
    };
    // the embedding the reference put in the speaker row
    var embedding_differs: usize = 0;
    const cond = (try std.json.parseFromSliceLeaky(std.json.Value, a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "input_ids.cond.json" }), a, .limited(1 << 22)), .{})).object;
    if (r.embedding != null) {
        const mine = got.embedding orelse base;
        for (mine, cond.get("speaker_embedding").?.array.items) |x, y| embedding_differs += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(@as(f32, @floatCast(number(y))))));
    }
    const arr = try z3.npy.parse(try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "steer.npy" }), a, .limited(1 << 28)));
    const want = try arr.floats(a);
    const sources = side.get("sources").?.array.items;
    const active = side.get("active").?.array.items;
    var sources_same = sources.len == got.plan.sources.len and arr.shape[1] == sources.len;
    if (sources_same) for (sources, got.plan.sources) |v, s| {
        sources_same = sources_same and v.integer == s;
    };
    if (!sources_same) {
        std.debug.print("RESULT zonos3 steer FAIL the bundle's sources are not the dump's\n", .{});
        return 1;
    }
    const n = sources.len * D;
    const out = try a.alloc(f32, n);
    const sum = try a.alloc(f32, n);
    var frames_differ: usize = 0;
    var values_differ: usize = 0;
    var steered: usize = 0;
    for (active, 0..) |flag, f| {
        const have = f > 0 and got.plan.at(f, out, sum);
        if (have != flag.bool) {
            frames_differ += 1;
            continue;
        }
        if (!have) continue;
        steered += 1;
        var bad: usize = 0;
        for (out, want[f * n ..][0..n]) |x, y| bad += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
        values_differ += bad;
        frames_differ += @intFromBool(bad != 0);
    }
    const pass = embedding_differs == 0 and frames_differ == 0;
    std.debug.print("RESULT zonos3 steer {s} frames={d} steered={d} frames_differ={d} values_differ={d} embedding_moved={} embedding_values_differ={d} spans={d} released={}\n", .{ if (pass) "PASS" else "FAIL", active.len - 1, steered, frames_differ, values_differ, got.embedding != null, embedding_differs, got.plan.spans.len, got.plan.released.len != 0 });
    return if (pass) 0 else 1;
}
