//! Compare every teacher-forced logit byte on original weights and an untaught stream of a portable taught folder.
const std = @import("std");
const tf = @import("tensorfold");
const q = tf.qwen27;
const mtl = @import("metal");
const Row = q.train.Row;
const Hash = std.crypto.hash.sha2.Sha256;

fn noShare(_: u64) bool {
    return false;
}

fn hashes(a: std.mem.Allocator, io: std.Io, dir: []const u8, cases: []const []const u32) ![][32]u8 {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = try q.model.Model.load(a, io, dir, 128, noShare);
    defer model.deinit();
    var count: usize = 0;
    var capacity: usize = 1;
    for (cases) |ids| {
        if (ids.len == 0 or ids.len > 8192) return error.BadUntaughtCase;
        count += @min(ids.len, 512) + 2;
        capacity = @max(capacity, ids.len + 1);
    }
    var runner = try q.decode_round.Runner.init(a, model, 1, @intCast(capacity));
    defer runner.deinit();
    const out = try a.alloc([32]u8, count);
    errdefer a.free(out);
    var index: usize = 0;
    for (cases) |ids| {
        try runner.reset(0);
        runner.untaught[0] = true;
        for (ids[0..@min(ids.len, 512)], 0..) |id, at| {
            var row: Row = .{};
            const round = try row.view(a, id, 0, @intCast(at), runner.capacity, @intCast(model.config.conv_kernel), 1, @intCast(model.config.vocab));
            try runner.advance(&round, .all);
            const ref = model.frame.get(.logits);
            Hash.hash(ref.buffer.contents()[ref.offset..][0 .. model.config.vocab * 2], &out[index], .{});
            index += 1;
        }
        for ([_]usize{ 128, model.frame.capacity }) |chunk| {
            try runner.reset(0);
            runner.untaught[0] = true;
            const session = q.session.Session{ .runner = &runner };
            try session.prefill(ids, chunk);
            out[index] = try session.fingerprint();
            index += 1;
        }
    }
    return out;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 4) return error.NeedOriginalTaughtAndTokenCases;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[3], a, .limited(8 << 20));
    var parsed = try std.json.parseFromSlice([]const []const u32, a, bytes, .{});
    defer parsed.deinit();
    if (parsed.value.len == 0) return error.BadUntaughtCase;
    const base = try hashes(a, init.io, args[1], parsed.value);
    const taught = try hashes(a, init.io, args[2], parsed.value);
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(base), std.mem.sliceAsBytes(taught))) return error.UntaughtModelLogitsDiffer;
    std.debug.print("Q27_UNTAUGHT digests={d} logits_equal=true prompt_state_equal=true max_teacher_rows=512 models_sequential=true\n", .{base.len});
}
