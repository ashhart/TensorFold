//! Actual Qwen prompt marks versus truncated chunks, with owned snapshot copies and exact restored continuations.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const host = @import("qwen27_served");
const Bank = q.gdn_pool_gpu.PromptMarks;

fn read(a: std.mem.Allocator, r: *const q.decode_round.Runner, st: *q.snapshot.State) ![]u8 {
    try st.landed(); // a save's blit is not waited for: it lands before the copy reads it
    const buffer = try r.model.device.buffer(st.bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer buffer.deinit();
    const cb = r.model.queue.commandBuffer();
    const e = cb.blit();
    e.copy(st.buf, 0, buffer, 0, st.bytes);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.SnapshotGpuFailure;
    return a.dupe(u8, buffer.contents()[0..st.bytes]);
}

fn logits(a: std.mem.Allocator, r: *const q.decode_round.Runner) ![]u8 {
    const ref = r.model.frame.get(.logits);
    return a.dupe(u8, ref.buffer.contents()[ref.offset..][0 .. r.model.config.vocab * 2]);
}

fn initial(r: *q.decode_round.Runner, ids: []const u32, start: u32) !void {
    try r.reset(0);
    if (start > 0) try (q.session.Session{ .runner = r }).promptChunk(ids[0..start], false);
}

fn check(a: std.mem.Allocator, r: *q.decode_round.Runner, ids: []const u32, start: u32) !void {
    const end = start + 65;
    const finish = end + 17;
    const session = q.session.Session{ .runner = r };
    try initial(r, ids, start);
    var bank = try Bank.init(r.model.device, r.gdn.budget, r.gdn.kernels, 0, start, end, .{ 1, 2, 31, 64 }, r.residency);
    defer bank.deinit();
    r.gdn.marks = &bank;
    defer r.gdn.marks = null; // clear the borrowed pointer before bank.deinit on every return
    try session.promptChunk(ids[start..end], false);
    // A cancelled or failed caller never seals the bank; even completed GPU words must not publish yet.
    if (q.snapshot.saveSlot(a, r, 0, null, start + 1, null)) |unexpected| {
        q.snapshot.drop(a, unexpected);
        return error.PublishedUnsealedMark;
    } else |err| if (err != error.NotAtMark) return err;
    try bank.complete(0, r.offsets[0]);
    var kept: [q.prompt_marks.limit]*q.snapshot.State = undefined;
    var count: usize = 0;
    defer for (kept[0..count]) |st| q.snapshot.drop(a, st);
    for (bank.position.rows) |row| {
        kept[count] = try q.snapshot.saveSlot(a, r, 0, null, start + row, null);
        count += 1;
    }
    r.gdn.marks = null;
    for (kept[0..count]) |st| try st.landed(); // the saves' blits land before the scratch is reused below
    // The cache copy owns its bytes: reusing the scratch cannot alter any captured prefix.
    @memset(bank.state.contents()[0..bank.state.length()], 0xa5);
    @memset(bank.conv.contents()[0..bank.conv.length()], 0x5a);
    try initial(r, ids, 0);
    try session.promptChunk(ids[0..finish], true);
    const full = try q.snapshot.saveSlot(a, r, 0, null, finish, null);
    defer q.snapshot.drop(a, full);
    const want_final = try read(a, r, full);
    defer a.free(want_final);
    const want_logits = try logits(a, r);
    defer a.free(want_logits);
    for (kept) |st| {
        try initial(r, ids, start);
        try session.promptChunk(ids[start..st.at], false);
        const truncated = try q.snapshot.saveSlot(a, r, 0, null, st.at, null);
        defer q.snapshot.drop(a, truncated);
        const want = try read(a, r, truncated);
        defer a.free(want);
        const got = try read(a, r, st);
        defer a.free(got);
        if (!std.mem.eql(u8, got, want)) return error.MarkStateOrConvDiffers;
        try q.snapshot.restoreSlot(a, r, 0, null, st);
        try session.promptChunk(ids[st.at..finish], true);
        const final = try q.snapshot.saveSlot(a, r, 0, null, finish, null);
        defer q.snapshot.drop(a, final);
        const actual = try read(a, r, final);
        defer a.free(actual);
        const actual_logits = try logits(a, r);
        defer a.free(actual_logits);
        if (!std.mem.eql(u8, actual, want_final) or !std.mem.eql(u8, actual_logits, want_logits)) return error.MarkContinuationDiffers;
        std.debug.print("PASS Qwen mark start={d} at={d}: full snapshot bytes and restored logits equal\n", .{ start, st.at });
    }
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) return error.Usage; // tf-qwen27-marks MODEL IDS.json; native model bytes, no drafter
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(1 << 20));
    const ids = (try std.json.parseFromSlice([]u32, a, text, .{})).value;
    if (ids.len < 96) return error.PromptTooShort;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var why: []const u8 = "";
    const h = try host.open(init.gpa, init.io, args[1], .{ .context = 512, .streams = 1, .cache_gib = 0 }, a, &why);
    defer host.close(h);
    for (ids) |id| if (id >= h.model.config.vocab) return error.BadToken;
    try check(init.gpa, &h.runner, ids, 0);
    try check(init.gpa, &h.runner, ids, 7);
}
