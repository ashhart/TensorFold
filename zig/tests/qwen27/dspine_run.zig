//! Greedy decoding speed on one stream: plain, DFlash2 trees and DSpine chains through the same verify rounds.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const help = "tf-qwen27-dspine-run --model DIR (--ref DIR | --prompts FILE) [--dspine DIR] [--dflash DIR] [--mode q4|q4_body|bf16] [--tokens 256] [--nodes 15] [--context 4096] [--class agent] [--no-plain]";
const prompts = [_][]const u8{ "poem", "primes", "code", "agent" };

/// A prompt's chat-templated ids: the committed ids of the reference's first round.
fn promptIds(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/{s}-r0.safetensors", .{ dir, name }), a, .limited(1 << 31));
    const n = std.mem.readInt(u64, bytes[0..8], .little);
    const header = try std.json.parseFromSlice(std.json.Value, a, bytes[8..][0..n], .{});
    const offsets = header.value.object.get("committed").?.object.get("data_offsets").?.array.items;
    const raw = bytes[8 + n ..][@intCast(offsets[0].integer)..@intCast(offsets[1].integer)];
    const ids = try a.alloc(u32, raw.len / 4);
    for (ids, 0..) |*id, i| id.* = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
    return ids;
}

/// A prompt set's entries (a JSON list, or an object with "prompts"): "ids" pre-tokenized, "class", optional "name".
fn promptFile(a: std.mem.Allocator, io: std.Io, path: []const u8, class: ?[]const u8, most: usize, ids: *std.ArrayList([]u32), names: *std.ArrayList([]const u8)) !void {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 28));
    const doc = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    const list = switch (doc.value) {
        .array => |l| l.items,
        .object => |o| (o.get("prompts") orelse return error.BadPrompts).array.items,
        else => return error.BadPrompts,
    };
    for (list, 0..) |item, k| {
        const c = if (item.object.get("class")) |v| v.string else "";
        if (class) |want| if (!std.mem.eql(u8, c, want)) continue;
        const raw = item.object.get("ids").?.array.items;
        if (raw.len + 300 > most) continue; // room for the reply's tokens and one verify block
        const p = try a.alloc(u32, raw.len);
        for (p, raw) |*d, v| d.* = @intCast(v.integer);
        try ids.append(a, p);
        const name = if (item.object.get("id")) |v| (if (v == .string) v.string else try std.fmt.allocPrint(a, "{f}", .{std.json.fmt(v, .{})})) else try std.fmt.allocPrint(a, "{s}{d}", .{ c, k });
        try names.append(a, try std.fmt.allocPrint(a, "{s}/{d}", .{ name, raw.len }));
    }
}

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn report(kind: []const u8, name: []const u8, tokens: usize, rounds: usize, ns: i96) void {
    const s = @as(f64, @floatFromInt(ns)) / 1e9;
    const per = if (rounds == 0) 0 else @as(f64, @floatFromInt(tokens)) / @as(f64, @floatFromInt(rounds));
    std.debug.print("{s:<8} {s:<14} {d:>4} tokens {d:>4} rounds {d:>5.2} tok/round {d:>6.2} tok/s\n", .{ kind, name, tokens, rounds, per, @as(f64, @floatFromInt(tokens)) / s });
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    var model_dir: ?[]const u8 = null;
    var ref_dir: ?[]const u8 = null;
    var dspine_dir: ?[]const u8 = null;
    var dflash_dir: ?[]const u8 = null;
    var mode: q.dspine.runtime.Mode = .q4;
    var count: usize = 256;
    var nodes: usize = 15;
    var context: u32 = 4096;
    var prompts_path: ?[]const u8 = null;
    var class: ?[]const u8 = null;
    var plain = true;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--help")) return std.debug.print("{s}\n", .{help});
        if (std.mem.eql(u8, arg, "--no-plain")) {
            plain = false;
            continue;
        }
        if (i + 1 >= args.len) return error.MissingValue;
        i += 1;
        const v = args[i];
        if (std.mem.eql(u8, arg, "--model")) model_dir = v else if (std.mem.eql(u8, arg, "--ref")) ref_dir = v else if (std.mem.eql(u8, arg, "--dspine")) dspine_dir = v else if (std.mem.eql(u8, arg, "--dflash")) dflash_dir = v else if (std.mem.eql(u8, arg, "--mode")) mode = std.meta.stringToEnum(q.dspine.runtime.Mode, v) orelse return error.BadMode else if (std.mem.eql(u8, arg, "--tokens")) count = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, arg, "--nodes")) nodes = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, arg, "--context")) context = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, arg, "--prompts")) prompts_path = v else if (std.mem.eql(u8, arg, "--class")) class = v else return error.UnknownFlag;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const dir = model_dir orelse return error.MissingModel;
    const model = try q.model.Model.load(init.gpa, io, dir, 128, null);
    defer model.deinit();
    var runner = try q.decode_round.Runner.init(init.gpa, model, 1, context);
    defer runner.deinit();
    var id_list: std.ArrayList([]u32) = .empty;
    var name_list: std.ArrayList([]const u8) = .empty;
    if (prompts_path) |path| {
        try promptFile(a, io, path, class, @min(context, 8100), &id_list, &name_list);
    } else for (prompts) |name| {
        try id_list.append(a, try promptIds(a, io, ref_dir orelse return error.MissingRef, name));
        try name_list.append(a, name);
    }
    const ids = id_list.items;
    const names = name_list.items;
    const session = q.session.Session{ .runner = &runner, .slot = 0 };
    if (plain) for (ids, names) |p, name| {
        try runner.reset(0);
        try session.prefill(p, 128);
        const t0 = now(io);
        var n: usize = 0;
        while (n < count) : (n += 1) {
            const token = try session.greedy();
            if (std.mem.indexOfScalar(u32, model.config.eos[0..model.config.eos_count], token) != null) break;
            try session.step(token);
        }
        report("plain", name, n, n, now(io) - t0);
    };
    if (dflash_dir) |d| {
        const draft = try q.dflash.runtime_model.Model.load(init.gpa, io, model, d, .prepared_q4_reference);
        defer draft.deinit();
        var g = try q.dflash.generation.Generation.init(init.gpa, &runner, draft);
        defer g.deinit();
        for (ids, names) |p, name| {
            try runner.reset(0);
            try draft.reset();
            try g.prefill(p, 128);
            const t0 = now(io);
            const r = try g.run(count, false, nodes);
            report("dflash2", name, r.tokens.len, r.rounds, now(io) - t0);
            init.gpa.free(r.tokens);
        }
    }
    if (dspine_dir) |d| {
        const draft = try q.dspine.runtime.Model.load(init.gpa, io, model, d, mode, dir);
        defer draft.deinit();
        var g = try q.dflash.generation.GenerationOf(q.dspine.runtime.Model).init(init.gpa, &runner, draft);
        defer g.deinit();
        for (ids, names) |p, name| {
            try runner.reset(0);
            draft.reset();
            try g.prefill(p, 128);
            const t0 = now(io);
            const r = try g.run(count, false, nodes);
            report("dspine", name, r.tokens.len, r.rounds, now(io) - t0);
            init.gpa.free(r.tokens);
        }
    }
}
