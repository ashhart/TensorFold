//! Gemma 4 E4B prompt seams: a pass resumed from a kept state against the fresh pass, compared bit for bit.
const std = @import("std");
const tf = @import("tensorfold");

const g4 = tf.gemma4;
const lanes = tf.lanes;
const M = g4.backend.Metal;
const vocab = g4.config.vocab;

const Kept = struct {
    metal: *M,
    at: [16]u32 = undefined,
    snap: [16]*anyopaque = undefined,
    n: usize = 0,

    fn hook(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
        const k: *Kept = @ptrCast(@alignCast(ptr));
        const saved = M.snapSave(k.metal, s, at) catch return;
        k.at[k.n] = at;
        k.snap[k.n] = saved;
        k.n += 1;
    }

    fn find(k: *const Kept, at: u32) ?*anyopaque {
        for (k.at[0..k.n], k.snap[0..k.n]) |a, s| if (a == at) return s;
        return null;
    }
};

const Result = struct {
    logits: []u16,
    tokens: [8]u32,
    final: []u16,
};

fn argmax(row: []const u16) u32 {
    var best: u32 = 0;
    var bv: f32 = -std.math.inf(f32);
    for (row, 0..) |b, i| {
        const v: f32 = @bitCast(@as(u32, b) << 16);
        if (v > bv) {
            bv = v;
            best = @intCast(i);
        }
    }
    return best;
}

/// One pass: prompt `ids`, chunk starts `chunks`, marks `marks` (kept into `kept`), resumed from `from` (null: fresh).
fn pass(gpa: std.mem.Allocator, metal: *M, ids: []const u32, chunks: []const u32, marks: []const u32, kept: *Kept, from: ?u32, source: *const Kept) !Result {
    var reuse: lanes.stream.Reuse = .{ .marks = marks, .hook = .{ .ptr = kept, .at = Kept.hook } };
    if (from) |m| {
        reuse.saved = source.find(m) orelse return error.NoState;
        reuse.at = m;
    }
    var s = try lanes.Stream.init(gpa, .{ .id = "seam", .prompt = ids, .max_new = 16, .chunks = chunks, .reuse = reuse });
    defer s.deinit(gpa);
    const b = metal.backend();
    try b.vtable.prefill(b.ptr, &s);
    if (from) |m| if (s.cached != m) return error.NotResumed;
    const cache = metal.caches.get(&s).?;
    const logits = try gpa.dupe(u16, cache.logits.slice(u16, vocab));
    var tokens: [8]u32 = undefined;
    var row = logits;
    var final: []u16 = logits;
    for (&tokens) |*t| {
        t.* = argmax(row);
        try g4.forward.window(metal.model, &metal.scratch, cache, &.{t.*}, cache.len);
        row = metal.scratch.logits.slice(u16, g4.state.prompt_rows * vocab)[0..vocab];
        final = row;
    }
    const kept_final = try gpa.dupe(u16, final);
    b.vtable.release(b.ptr, &s);
    return .{ .logits = logits, .tokens = tokens, .final = kept_final };
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const model_dir = args.next() orelse return error.Usage;
    const ids_file = args.next() orelse return error.Usage;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, ids_file, gpa, .limited(1 << 24));
    defer gpa.free(bytes);
    const all = try std.json.parseFromSlice([]u32, gpa, bytes, .{});
    defer all.deinit();
    const model = try g4.Model.load(gpa, io, model_dir);
    defer model.deinit();
    const metal = try M.init(gpa, io, model, .{ .capacity = 4096, .streams = 2 });
    defer metal.deinit();

    const resumes = [_]u32{ 40, 100, 127, 128, 129, 511, 512, 513, 640, 1000, 1024, 1025 };
    // lengths: one with tile chunks past the resume, and (for resumes under 128) a short prompt of windows only
    const lens = [_]u32{ 1300, 0 };
    var failures: usize = 0;
    var total: usize = 0;
    for (lens) |len_cfg| for (resumes) |m| {
        const len: u32 = if (len_cfg == 0) m + 3 else len_cfg;
        if (len_cfg == 0 and m >= 128) continue;
        if (len <= m) continue;
        const ids = all.value[0..len];
        // the host's plan: chunk starts at multiples of 512 (planned) are cut in every pass; marks add the resume point
        var planned: [4]u32 = undefined;
        var np: usize = 0;
        var p: u32 = 512;
        while (p < len) : (p += 512) {
            planned[np] = p;
            np += 1;
        }
        var kept = Kept{ .metal = metal };
        var none = Kept{ .metal = metal };
        // fresh with the mark (saves the state at m), fresh without it, resumed from the state
        const marked = try pass(gpa, metal, ids, planned[0..np], &.{m}, &kept, null, &none);
        const plain = try pass(gpa, metal, ids, planned[0..np], &.{}, &none, null, &none);
        const resumed = try pass(gpa, metal, ids, planned[0..np], &.{}, &none, m, &kept);
        const eq_mark = std.mem.eql(u16, marked.logits, resumed.logits) and std.mem.eql(u16, marked.final, resumed.final) and std.mem.eql(u32, &marked.tokens, &resumed.tokens);
        const eq_plain = std.mem.eql(u16, plain.logits, resumed.logits) and std.mem.eql(u16, plain.final, resumed.final) and std.mem.eql(u32, &plain.tokens, &resumed.tokens);
        const eq_cut = std.mem.eql(u16, plain.logits, marked.logits);
        std.debug.print("prompt {d:>4} resumed at {d:>4}: resumed==fresh-with-mark {s}, resumed==fresh-without-mark {s} (prompt logits {s}), first token {d}, decode tokens {any}\n", .{ len, m, if (eq_mark) "yes" else "NO", if (eq_plain) "yes" else "NO", if (eq_cut) "same" else "differ", argmax(resumed.logits), resumed.tokens });
        total += 1;
        if (!eq_mark or !eq_plain) failures += 1;
        for (kept.snap[0..kept.n]) |sn| M.snapDrop(metal, sn);
        for ([_]Result{ marked, plain, resumed }) |r| {
            gpa.free(r.logits);
            gpa.free(r.final);
        }
    };
    std.debug.print("{d} cases, {d} not equal\n", .{ total, failures });
    if (failures != 0) return error.SeamMismatch;
}
