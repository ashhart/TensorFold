//! tf-cuda-test glm-weights <converted dir> <rank> <capture dir>: the native loader vs capture_ref.py's weights.json.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// Python tensors the native forward does not read (the latent path attends through absorb.wk/wv, the same bytes).
fn unused(name: []const u8) bool {
    return std.mem.endsWith(u8, name, ".kv_k.weight") or std.mem.endsWith(u8, name, ".kv_v.weight") or std.mem.startsWith(u8, name, "draft_head.");
}

pub fn run(gpu: Gpu, dir: []const u8, rank_s: []const u8, ref: []const u8) !void {
    const gw = @import("glm_weights");
    const rank = try std.fmt.parseInt(u8, rank_s, 10);
    const gpa = gpu.gpa;
    const t0 = std.Io.Timestamp.now(gpu.io, .awake);
    const planned = try gw.deviceBytes(gpu.io, dir);
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const ctx: gw.api.Ctx = .{ .d = gpu.d, .ctx = gpu.ctx, .stream = stream, .nccl = undefined, .comm = undefined, .rank = rank };
    var store = try gw.load(gpa, gpu.io, &ctx, dir, rank);
    defer store.deinit();
    const secs = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(gpu.io, .awake)).toNanoseconds())) / 1e9;
    std.debug.print("loaded {d} tensors, {d:.2} GiB (planned {d:.2} GiB) in {d:.1} s\n", .{ store.map.count(), @as(f64, @floatFromInt(store.ints.get("device_bytes").?)) / (1 << 30), @as(f64, @floatFromInt(planned)) / (1 << 30), secs });
    const jp = try std.fs.path.join(gpa, &.{ ref, "weights.json" });
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, jp, gpa, .limited(1 << 27));
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    var equal: usize = 0;
    var bad: usize = 0;
    var skipped: usize = 0;
    var seen = std.StringHashMap(void).init(gpa);
    var have: std.ArrayList(u8) = .empty;
    defer have.deinit(gpa);
    for (parsed.value.object.get("tensors").?.array.items) |tv| {
        const t = tv.object;
        const sha = t.get("sha256") orelse continue; // config values and host tensors
        const name = t.get("name").?.string;
        try seen.put(name, {});
        const n: usize = @intCast(t.get("bytes").?.integer);
        const got = store.map.get(name) orelse {
            if (unused(name)) {
                skipped += 1;
                continue;
            }
            bad += 1;
            std.debug.print("MISSING {s} ({d} bytes)\n", .{ name, n });
            continue;
        };
        if (got.bytes() != n) {
            bad += 1;
            std.debug.print("SIZE {s}: {d} bytes, Python {d}\n", .{ name, got.bytes(), n });
            continue;
        }
        try have.resize(gpa, n);
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(have.items.ptr, got.ptr, n), "download");
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(have.items, &digest, .{});
        const hex = std.fmt.bytesToHex(digest, .lower);
        if (std.mem.eql(u8, &hex, sha.string)) {
            equal += 1;
        } else {
            bad += 1;
            std.debug.print("DIFFERS {s} {s} [{d} bytes]\n", .{ name, @tagName(got.dtype), n });
        }
    }
    var extra: usize = 0;
    var it = store.map.keyIterator();
    while (it.next()) |k| if (!seen.contains(k.*)) {
        extra += 1;
        std.debug.print("EXTRA {s} (not in Python's tree)\n", .{k.*});
    };
    std.debug.print("{d} equal, {d} not, {d} unused skipped, {d} extra\n", .{ equal, bad, skipped, extra });
    try check.expect(bad == 0 and extra == 0, "{d} tensors equal, {d} not, {d} extra", .{ equal, bad, extra });
}
