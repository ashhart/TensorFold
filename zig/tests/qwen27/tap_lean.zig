//! Fresh-process prompt tap, target, drafter cache and token digests for opt-in A/B on real weights.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const Sha = std.crypto.hash.sha2.Sha256;
fn digest(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    Sha.hash(bytes, &out, .{});
    return out;
}
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 5) return error.Usage; // tf-qwen27-tap-lean TARGET DRAFTER IDS.json CHUNK
    const cut = try std.fmt.parseInt(u32, args[4], 10);
    if (cut == 0 or cut > 2048) return error.BadChunk;
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, args[3], a, .limited(16 << 20));
    const prompt = (try std.json.parseFromSlice([]u32, a, text, .{})).value;
    if (prompt.len == 0 or prompt.len > 65536) return error.BadPrompt;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const target = try q.model.Model.load(init.gpa, init.io, args[1], 2048, null);
    defer target.deinit();
    var runner = try q.decode_round.Runner.init(init.gpa, target, 1, @intCast(prompt.len + 128));
    defer runner.deinit();
    const draft = try q.dflash.runtime_model.Model.load(init.gpa, init.io, target, args[2], .prepared_q4_reference);
    defer draft.deinit();
    var gen = try q.dflash.generation.Generation.init(init.gpa, &runner, draft);
    defer gen.deinit();
    const start = std.Io.Clock.awake.now(init.io);
    try gen.prefill(prompt, cut);
    const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
    const session = q.session.Session{ .runner = &runner };
    const target_hash = try session.fingerprint();
    const ring = draft.ring();
    const ring_hash = digest(ring.buf.contents()[0..ring.bytes(draft.committed_end)]);
    try draft.flush();
    var state = Sha.init(.{});
    const stride = @as(usize, draft.graph.config.kvWidth()) * 2;
    for (draft.backend.caches) |cache| for ([_]mtl.Buffer{ cache.keys, cache.values }) |buf| {
        var at = draft.session.context.begin;
        while (at < draft.session.context.end) : (at += 1) {
            const slot: usize = @intCast(at % draft.graph.config.window);
            state.update(buf.contents()[slot * stride ..][0..stride]);
        }
    };
    var kv: [32]u8 = undefined;
    state.final(&kv);
    const result = try gen.run(64, true, 15);
    defer init.gpa.free(result.tokens);
    const tokens = digest(std.mem.sliceAsBytes(result.tokens));
    const final = try session.fingerprint();
    std.debug.print("IDENTITY chunk={d} end={d} target={x} ring={x} draft_kv={x} tokens={x} final={x}\n", .{ cut, prompt.len, target_hash, ring_hash, kv, tokens, final });
    std.debug.print("TIME prefill_ns={d} rounds={d} verified={d} matched={d}\n", .{ elapsed, result.rounds, result.verified, result.matched });
}
