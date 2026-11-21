//! zonos3-generate --edit-check: a dumped /v2/edit's fills generated here on the edit adapter, against the reference's.
const std = @import("std");
const z3 = @import("zonos3");
const gen = @import("generate.zig");
const Dumped = @import("check_laci.zig").Dumped;

const cfg = z3.config;
const W = cfg.frame_width;
const C = cfg.codebooks;

/// Reference text ids as the port numbers them: the reference has marker rows before the extra rows, the port after.
pub fn mapText(w: *const z3.weights.Weights, rows: [][W]i32) void {
    const first: i32 = @intCast(w.text_rows);
    const markers: i32 = @intCast(w.marker_count);
    for (rows) |*r| {
        const id = r[C];
        if (id < first) continue;
        r[C] = if (id < first + markers) @as(i32, @intCast(w.markers)) + (id - first) else id - markers;
    }
}

fn number(v: std.json.Value) f64 {
    return if (v == .float) v.float else @floatFromInt(v.integer);
}

fn planOf(a: std.mem.Allocator, req: std.json.ObjectMap, frames: usize, max_spans: usize) !z3.edit_plan.Plan {
    const ep = z3.edit_plan;
    var r: ep.Request = .{ .target_text = std.mem.trim(u8, req.get("target_text").?.string, " \t\r\n"), .frames = frames, .max_spans = max_spans };
    if (req.get("original_words")) |list| {
        const given = try a.alloc(ep.Word, list.array.items.len);
        for (given, list.array.items) |*w, v| {
            const word = v.object.get("word").?.string;
            w.* = .{ .word = word, .start = number(v.object.get("start").?), .end = number(v.object.get("end").?), .tokens = try ep.tokens(a, word) };
        }
        var at: usize = 0;
        try ep.checkWords(given, try ep.tokens(a, req.get("original_text").?.string), @as(f64, @floatFromInt(frames)) / 15.0, &at);
        r.original = given;
    }
    if (req.get("spans")) |list| {
        const pairs = try a.alloc([2]f64, list.array.items.len);
        for (pairs, list.array.items) |*pair, v| pair.* = .{ number(v.array.items[0]), number(v.array.items[1]) };
        r.spans = pairs;
    }
    inline for (.{ "margin_seconds", "max_span_seconds", "max_context_seconds", "context_seconds", "min_fill_ratio" }) |field| {
        if (req.get(field)) |v| @field(r, field) = number(v);
    }
    return ep.plan(a, r);
}

/// DIR's edit replayed: its plan, each fill's prompt and greedy frames on the adapter, the splice; as the reference.
pub fn fills(gpa: std.mem.Allocator, io: std.Io, e: *z3.Engine, dir: []const u8) !u8 {
    if (e.adapter == null) return error.NoAdapter;
    const cwd = std.Io.Dir.cwd();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const req = (try std.json.parseFromSliceLeaky(std.json.Value, a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "request.json" }), a, .limited(1 << 24)), .{})).object;
    const resp = (try std.json.parseFromSliceLeaky(std.json.Value, a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "response.json" }), a, .limited(1 << 24)), .{})).object;
    const clip_arr = try z3.npy.parse(try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "codes.npy" }), a, .limited(1 << 28)));
    const clip: []const [C]i32 = @ptrCast(try clip_arr.ints(a));
    const plan = try planOf(a, req, clip.len, e.w.marker_count);
    // the plan against the response
    var plan_differs: usize = 0;
    const want_spans = resp.get("spans").?.array.items;
    const want_expected = resp.get("expected_seconds").?.array.items;
    plan_differs += @intFromBool(want_spans.len != plan.spans.len);
    if (plan_differs == 0) for (plan.spans, plan.expected, want_spans, want_expected) |sp, ex, ws, we| {
        const start = @as(f64, @floatFromInt(sp.start)) / 15.0;
        const end = @as(f64, @floatFromInt(sp.start + sp.len)) / 15.0;
        plan_differs += @intFromBool(start != number(ws.array.items[0]) or end != number(ws.array.items[1]) or @abs(ex - number(we)) > 1e-12);
    };
    const contexts = resp.get("contexts").?;
    plan_differs += @intFromBool((contexts != .null) != plan.windowed);
    if (contexts != .null) {
        plan_differs += @intFromBool(contexts.array.items.len != plan.windows.len);
        if (contexts.array.items.len == plan.windows.len) for (plan.windows, contexts.array.items) |w, c| {
            plan_differs += @intFromBool(@as(f64, @floatFromInt(w.start)) / 15.0 != number(c.array.items[0]) or @as(f64, @floatFromInt(w.end)) / 15.0 != number(c.array.items[1]));
        };
    }
    var n: usize = 0;
    var failed: usize = 0;
    var compared: usize = 0;
    var frames: usize = 0;
    var seconds: f64 = 0;
    const made = try a.alloc([]const [C]i32, plan.spans.len);
    for (plan.windows) |window| {
        const codes = clip[window.start..window.end];
        const spans = try a.alloc(z3.edit.Span, window.regions.len);
        for (spans, window.regions) |*sp, k| sp.* = .{ .start = plan.spans[k].start - window.start, .len = plan.spans[k].len };
        var priors: std.ArrayList([]const [C]i32) = .empty;
        for (window.regions) |region| {
            defer n += 1;
            const fill = try std.fmt.allocPrint(a, "{s}/fill_{d}", .{ dir, n });
            const meta = (try std.json.parseFromSliceLeaky(std.json.Value, a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ fill, "fill.json" }), a, .limited(1 << 22)), .{})).object;
            const greedy = number(meta.get("sampling").?.object.get("temperature").?) == 0;
            var rows: [][W]i32 = &.{};
            var replaced: std.ArrayList(z3.Replace) = .empty;
            defer replaced.deinit(gpa);
            _ = try gen.loadBranch(gpa, io, e, fill, "input_ids", &rows, &replaced);
            defer gpa.free(rows);
            mapText(e.w, rows);
            // what the plan gives this fill, and the rows it lays out after the text
            const tail = try z3.edit.layout(a, codes, spans, priors.items, @intCast(e.w.markers));
            var same_plan = std.mem.eql(u8, window.text, meta.get("text").?.string) and meta.get("priors").?.integer == priors.items.len;
            same_plan = same_plan and meta.get("min_frames").?.integer == plan.min_frames[region] and meta.get("spans").?.array.items.len == spans.len;
            same_plan = same_plan and tail.len < rows.len and std.mem.eql([W]i32, tail, rows[rows.len - tail.len ..]);
            plan_differs += @intFromBool(!same_plan);
            if (!same_plan) std.debug.print("fill {d}: the plan or the layout differs (text '{s}', min_frames {d}, {d} layout rows)\n", .{ n, window.text, plan.min_frames[region], tail.len });
            const want_arr = try z3.npy.parse(try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ fill, "rows.npy" }), a, .limited(1 << 28)));
            const want: []const [W]i32 = @ptrCast(try want_arr.ints(a));
            if (!greedy) {
                // a sampled fill is drawn by another generator: the reference's rows carry the replay on
                std.debug.print("fill {d}: sampled, its rows not compared\n", .{n});
                const taken = try z3.edit.extract(a, want, if (meta.get("eos_frame").? == .null) null else @intCast(meta.get("eos_frame").?.integer));
                made[region] = taken.frames;
                try priors.append(a, taken.frames);
                continue;
            }
            var dumped: Dumped = .{};
            defer dumped.deinit(gpa);
            const steered = try dumped.load(gpa, io, fill);
            try e.reset();
            e.adapt = true;
            defer e.adapt = false;
            const t0 = gen.nowNs(io);
            var run = try z3.generation.Run.begin(gpa, e, rows, replaced.items, &.{}, replaced.items, 1, .{ .temperature = 0 }, @min(plan.max_rows, e.o.max_ctx - rows.len));
            defer run.deinit();
            try run.asEdit(e, plan.min_frames[region]);
            if (steered) run.steer = .{ .ctx = &dumped, .at = Dumped.at, .twin = false };
            while (!try run.sample()) try run.advance(e);
            seconds += @as(f64, @floatFromInt(gen.nowNs(io) - t0)) / 1e9;
            const mine = run.rows.items;
            var same: usize = 0;
            while (same < @min(mine.len, want.len) and std.mem.eql(i32, &mine[same], &want[same])) same += 1;
            const ok = same == want.len and mine.len == want.len;
            failed += @intFromBool(!ok);
            compared += 1;
            frames += mine.len;
            if (!ok and same < @min(mine.len, want.len)) std.debug.print("fill {d}: first difference at row {d}: mine {any} reference {any}\n", .{ n, same, mine[same], want[same] });
            const taken = try z3.edit.extract(a, mine, run.eos_frame);
            made[region] = taken.frames;
            try priors.append(a, taken.frames);
            std.debug.print("fill {d}: {s} identical={d} mine={d} reference={d} end_frame={?d} prompt_rows={d} min_frames={d} steered_frames={d} fill_frames={d} stopped={}\n", .{ n, if (ok) "ok" else "DIFFERS", same, mine.len, want.len, run.eos_frame, rows.len, plan.min_frames[region], dumped.count, taken.frames.len, taken.stopped });
        }
    }
    const spliced = try z3.edit.splice(a, clip, plan.spans, made);
    const want_arr = try z3.npy.parse(try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "spliced.npy" }), a, .limited(1 << 28)));
    const want_spliced: []const [C]i32 = @ptrCast(try want_arr.ints(a));
    const splice_same = spliced.len == want_spliced.len and std.mem.eql([C]i32, spliced, want_spliced);
    const pass = n > 0 and failed == 0 and plan_differs == 0 and splice_same and n == resp.get("fills").?.integer;
    std.debug.print("RESULT zonos3 edit {s} fills={d} compared={d} differ={d} plan_differs={d} windows={d} windowed={} splice_identical={} clip_frames={d} edited_frames={d} rows={d} seconds={d:.3}\n", .{ if (pass) "PASS" else "FAIL", n, compared, failed, plan_differs, plan.windows.len, plan.windowed, splice_same, clip.len, spliced.len, frames, seconds });
    return if (pass) 0 else 1;
}
