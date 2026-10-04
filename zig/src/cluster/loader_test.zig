//! Loading on two fake nodes: what each rank reads, local files mapped and hashed once, missing shards pulled and checked.
const std = @import("std");
const checkpoint = @import("checkpoint.zig");
const model = @import("model.zig");
const node = @import("node.zig");
const plan = @import("plan.zig");
const loader = @import("loader.zig");

const io = std.testing.io;

const names = [_][]const u8{ "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors" };

/// Two shard files in `dir`; their bytes and the checkpoint the planner sees.
fn shards(a: std.mem.Allocator, dir: std.Io.Dir) !struct { files: [2][]u8, ck: checkpoint.Checkpoint } {
    const one = try checkpoint.image(a, &.{
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .dtype = "BF16", .shape = &.{ 8, 4 }, .fill = 1 },
        .{ .name = "model.layers.0.self_attn.o_proj.weight", .dtype = "BF16", .shape = &.{ 4, 8 }, .fill = 2 },
        .{ .name = "model.layers.0.input_layernorm.weight", .dtype = "BF16", .shape = &.{4}, .fill = 3 },
        .{ .name = "model.embed_tokens.weight", .dtype = "BF16", .shape = &.{ 16, 4 }, .fill = 4 },
        .{ .name = "lm_head.weight", .dtype = "BF16", .shape = &.{ 16, 4 }, .fill = 5 },
    });
    var specs: [9]struct { name: []const u8, dtype: []const u8, shape: []const u64, fill: u8 } = undefined;
    for (0..8) |e| specs[e] = .{ .name = try std.fmt.allocPrint(a, "model.layers.0.mlp.experts.{d}.down_proj.weight", .{e}), .dtype = "U8", .shape = &.{ 50, 800 }, .fill = @intCast(10 + e) };
    specs[8] = .{ .name = "model.layers.0.mlp.gate.weight", .dtype = "BF16", .shape = &.{ 8, 4 }, .fill = 9 };
    const two = try checkpoint.image(a, @ptrCast(&specs));
    var files: [2]checkpoint.File = undefined;
    var tensors: std.ArrayList(checkpoint.Tensor) = .empty;
    for ([_][]u8{ one, two }, 0..) |bytes, i| {
        try dir.writeFile(io, .{ .sub_path = names[i], .data = bytes });
        var sha: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &sha, .{});
        files[i] = .{ .name = names[i], .size = bytes.len, .sha256 = sha };
        try tensors.appendSlice(a, try checkpoint.parseHeader(a, @intCast(i), bytes));
    }
    return .{ .files = .{ one, two }, .ck = .{ .files = try a.dupe(checkpoint.File, &files), .tensors = tensors.items } };
}

fn tiny() model.Shape {
    var s: model.Shape = .{ .hidden = 4, .layers = 1, .vocab = 16, .heads = 8, .kv_heads = 8, .head_dim = 1, .experts = 8, .top_k = 2, .moe_inter = 4, .latent = 4 };
    s.kinds[0] = .gqa;
    return s;
}

const Seen = struct {
    a: std.mem.Allocator,
    files: [2][]const u8,
    bytes: u64 = 0,
    fn sink(s: *Seen) loader.Sink {
        return .{ .ptr = s, .run = run };
    }
    fn run(ptr: *anyopaque, r: loader.Range, view_offset: u64, view: []align(loader.page) const u8) anyerror!void {
        const s: *Seen = @ptrCast(@alignCast(ptr));
        try std.testing.expect(view_offset % loader.page == 0 and view.len % loader.page == 0);
        const want = s.files[r.file][@intCast(r.offset)..][0..@intCast(r.len)];
        try std.testing.expectEqualSlices(u8, want, view[@intCast(r.offset - view_offset)..][0..@intCast(r.len)]);
        s.bytes += r.len;
    }
};

/// A peer that reads its own copy; `corrupt` flips a byte after hashing, as a bad link would.
const Peer = struct {
    files: [2][]const u8,
    corrupt: bool = false,
    calls: u32 = 0,
    fn source(p: *Peer) loader.Source {
        return .{ .ptr = p, .pull = pull };
    }
    fn pull(ptr: *anyopaque, file: u32, offset: u64, out: []u8) anyerror![32]u8 {
        const p: *Peer = @ptrCast(@alignCast(ptr));
        p.calls += 1;
        @memcpy(out, p.files[file][@intCast(offset)..][0..out.len]);
        var sha: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(out, &sha, .{});
        if (p.corrupt) out[0] ^= 0xff;
        return sha;
    }
};

test "each rank reads its column rows, whole rows-split tensors and its experts, merged and sorted" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const sh = try shards(a, tmp.dir);
    const s = tiny();
    const nodes = [_]node.Inventory{ .{ .id = 1, .gpu_limit = node.gib }, .{ .id = 2, .gpu_limit = node.gib } };
    const p = try plan.plan(a, sh.ck, &s, &nodes, .{ .context = 16 });
    var total: [2]u64 = .{ 0, 0 };
    for (0..2) |r| {
        const rs = try loader.needs(a, &p, sh.ck, @intCast(r));
        for (rs[1..], 0..) |x, k| try std.testing.expect(x.file > rs[k].file or x.offset > rs[k].offset + rs[k].len);
        for (rs) |x| total[r] += x.len;
        for (sh.ck.tensors, 0..) |t, i| {
            if (p.bytesOn(t, i, @intCast(r)) == 0 or p.place[i].kind == .column) continue;
            var inside = false;
            for (rs) |x| inside = inside or (x.file == t.file and x.offset <= t.start and t.start + t.bytes <= x.offset + x.len);
            try std.testing.expect(inside);
        }
    }
    try std.testing.expectEqual(total[0], total[1]);
}

test "local files map zero-copy and hash once; a wrong manifest hash refuses the file" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const sh = try shards(a, tmp.dir);
    const s = tiny();
    const nodes = [_]node.Inventory{ .{ .id = 1, .gpu_limit = node.gib }, .{ .id = 2, .gpu_limit = node.gib } };
    const p = try plan.plan(a, sh.ck, &s, &nodes, .{ .context = 16 });
    const ranges = try loader.needs(a, &p, sh.ck, 1);
    var seen: Seen = .{ .a = a, .files = .{ sh.files[0], sh.files[1] } };
    var rep = try loader.load(io, a, tmp.dir, .{ .files = sh.ck.files }, ranges, null, seen.sink(), null);
    defer rep.release(a);
    try std.testing.expect(rep.local == 2 and rep.hashed == 2 and rep.pulled == 0 and seen.bytes > 0);
    var again = try loader.load(io, a, tmp.dir, .{ .files = sh.ck.files }, ranges, null, seen.sink(), null);
    defer again.release(a);
    try std.testing.expectEqual(@as(u32, 0), again.hashed);
    var wrong = try a.dupe(checkpoint.File, sh.ck.files);
    wrong[1].sha256.?[0] ^= 1;
    try tmp.dir.deleteTree(io, ".tensorfold");
    try std.testing.expectError(error.HashMismatch, loader.load(io, a, tmp.dir, .{ .files = wrong }, ranges, null, seen.sink(), null));
}

test "a missing shard is pulled from a peer, checked chunk by chunk, and not pulled again after a restart" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var src_dir = std.testing.tmpDir(.{});
    defer src_dir.cleanup();
    var dst = std.testing.tmpDir(.{});
    defer dst.cleanup();
    const sh = try shards(a, src_dir.dir);
    try dst.dir.writeFile(io, .{ .sub_path = names[0], .data = sh.files[0] });
    const s = tiny();
    const nodes = [_]node.Inventory{ .{ .id = 1, .gpu_limit = node.gib }, .{ .id = 2, .gpu_limit = node.gib } };
    const p = try plan.plan(a, sh.ck, &s, &nodes, .{ .context = 16 });
    const ranges = try loader.needs(a, &p, sh.ck, 1);
    var seen: Seen = .{ .a = a, .files = .{ sh.files[0], sh.files[1] } };
    try std.testing.expectError(error.MissingShard, loader.load(io, a, dst.dir, .{ .files = sh.ck.files }, ranges, null, seen.sink(), null));
    var bad: Peer = .{ .files = .{ sh.files[0], sh.files[1] }, .corrupt = true };
    try std.testing.expectError(error.HashMismatch, loader.load(io, a, dst.dir, .{ .files = sh.ck.files, .chunk = 4096 }, ranges, bad.source(), seen.sink(), null));
    var peer: Peer = .{ .files = .{ sh.files[0], sh.files[1] } };
    var rep = try loader.load(io, a, dst.dir, .{ .files = sh.ck.files, .chunk = 4096 }, ranges, peer.source(), seen.sink(), null);
    defer rep.release(a);
    try std.testing.expect(rep.pulled == 1 and rep.local == 1 and rep.pulled_bytes > 0 and rep.pulled_bytes < sh.files[1].len);
    const calls = peer.calls;
    var again = try loader.load(io, a, dst.dir, .{ .files = sh.ck.files, .chunk = 4096 }, ranges, peer.source(), seen.sink(), null);
    defer again.release(a);
    try std.testing.expectEqual(calls, peer.calls);
    try std.testing.expectEqual(@as(u64, 0), again.pulled_bytes);
}
