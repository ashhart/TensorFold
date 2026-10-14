//! tf-cuda-test glm-model: the whole RED-SNOW model at TP=2, a process a rank, vs the Python engine's greedy replies.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const check = @import("check.zig");

const cap = 32768; // cache slots: the reference's longest case (16,480 tokens) and the Python engine's pool buckets

const Case = struct { name: []const u8, prompt: []u32, ids: []u32 };

/// `reference`: {"cases": [{"name", "prompt_ids", "ids"}], "eos_ids"}; rank 0 prints the PASS/FAIL/RESULT lines.
pub fn run(gpu: check.Gpu, rank_arg: []const u8, addr: []const u8, model: []const u8, reference: []const u8) !void {
    const rank = try std.fmt.parseInt(u8, rank_arg, 10);
    if (rank > 1) return error.BadRank;
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const ref = try readReference(a, gpu.io, reference);
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try cuda.nccl_id.pair(&lib, gpu.io, rank, addr);
    defer _ = lib.api.ncclCommDestroy(comm);

    const t0 = check.now(gpu.io);
    var w = glm.Weights.load(gpu.gpa, gpu.io, gpu.d, model, rank) catch |e| {
        if (rank == 0) std.debug.print("FAIL glm.model.load: {t}\n", .{e});
        return e;
    };
    defer w.deinit();
    var e = try glm.Engine.init(gpu.gpa, gpu.d, &w, &lib, comm, cap);
    defer e.deinit();
    if (rank == 0) {
        std.debug.print("RESULT glm.model.load: {d:.1} GiB a rank in {d:.0} s\n", .{ @as(f64, @floatFromInt(w.pool.bytes)) / (1 << 30), secs(gpu.io, t0) });
        check.pass("glm.model.load", .{});
    }

    var first_ok = true;
    var prompt_tokens: usize = 0;
    var prefill_s: f64 = 0;
    var decoded: usize = 0;
    var decode_s: f64 = 0;
    for (ref.cases) |c| {
        const tp = check.now(gpu.io);
        var out: std.ArrayList(u32) = .empty;
        try out.append(a, try e.prefill(c.prompt));
        prefill_s += secs(gpu.io, tp);
        prompt_tokens += c.prompt.len;
        if (out.items[0] != c.ids[0]) first_ok = false;
        const td = check.now(gpu.io);
        // both ranks pick the same tokens, so both stop together: at the reply's length, an EOS or the first difference
        while (out.items.len < c.ids.len and out.items[out.items.len - 1] == c.ids[out.items.len - 1] and !isEos(ref.eos, out.items[out.items.len - 1])) {
            try out.append(a, try e.step(out.items[out.items.len - 1]));
        }
        decode_s += secs(gpu.io, td);
        decoded += out.items.len - 1;
        if (rank != 0) continue;
        const id = try std.fmt.allocPrint(a, "glm.model.greedy.{s}", .{c.name});
        if (std.mem.eql(u32, out.items, c.ids)) {
            check.pass("{s}", .{id});
            continue;
        }
        const n = @min(out.items.len, c.ids.len);
        var at: usize = 0;
        while (at < n and out.items[at] == c.ids[at]) at += 1;
        if (at < n) {
            std.debug.print("FAIL {s}: token {d} of {d} is {d}, the Python engine's {d}\n", .{ id, at, c.ids.len, out.items[at], c.ids[at] });
        } else {
            std.debug.print("FAIL {s}: {d} tokens, the Python engine's {d}\n", .{ id, out.items.len, c.ids.len });
        }
    }
    if (rank != 0) return;
    if (first_ok) check.pass("glm.model.first_token", .{}) else std.debug.print("FAIL glm.model.first_token: a case's first token differs\n", .{});
    std.debug.print("RESULT glm.model.prefill_tps {d:.1}\n", .{@as(f64, @floatFromInt(prompt_tokens)) / prefill_s});
    std.debug.print("RESULT glm.model.decode_tps {d:.2}\n", .{@as(f64, @floatFromInt(decoded)) / decode_s});
}

fn secs(io: std.Io, t0: i96) f64 {
    return @as(f64, @floatFromInt(check.now(io) - t0)) / 1e9;
}

fn isEos(eos: []const u32, tok: u32) bool {
    return std.mem.indexOfScalar(u32, eos, tok) != null;
}

fn ints(a: std.mem.Allocator, v: std.json.Value) ![]u32 {
    const items = v.array.items;
    const out = try a.alloc(u32, items.len);
    for (items, out) |x, *o| o.* = @intCast(x.integer);
    return out;
}

fn readReference(a: std.mem.Allocator, io: std.Io, path: []const u8) !struct { cases: []Case, eos: []u32 } {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 26));
    const root = (try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{})).object;
    const items = root.get("cases").?.array.items;
    const cases = try a.alloc(Case, items.len);
    for (items, cases) |v, *c| {
        const o = v.object;
        c.* = .{ .name = o.get("name").?.string, .prompt = try ints(a, o.get("prompt_ids").?), .ids = try ints(a, o.get("ids").?) };
        if (c.prompt.len == 0 or c.ids.len == 0) return error.BadReference;
    }
    return .{ .cases = cases, .eos = try ints(a, root.get("eos_ids").?) };
}
