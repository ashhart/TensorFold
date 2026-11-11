const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const Model = tf.deepseek_v41.Model;
const capture = @import("stagediff_capture.zig");
const Capture = capture.Capture;
const attention_elements = capture.attention_elements;
const layer_count = capture.layer_count;
const coarse_stride = capture.coarse_stride;
const max_rows = capture.max_rows;
const logits_width = capture.logits_width;
const argmax = capture.argmax;
const secondBest = capture.secondBest;
const compareF32 = capture.compareF32;
const firstCaptureDifference = capture.firstCaptureDifference;

fn seed(model: *Model, ids: []const u32) ![]const f32 {
    try model.reset();
    var row: []const f32 = &.{};
    var start: usize = 0;
    while (start < ids.len) {
        const end = @min(start + model.workspace.plan.rows, ids.len);
        row = try model.forward(ids[start..end], if (end == ids.len) .last else .none);
        start = end;
    }
    return row;
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, model: *Model, ids: []const u32, dir: []const u8, steps_arg: ?u32, rows_arg: ?u32) !void {
    if (ids.len == 0 or ids.len > 4096) return error.InvalidPrompt;
    const steps = steps_arg orelse 8;
    const rows: u32 = rows_arg orelse 2;
    if (steps == 0 or steps > 64) return error.InvalidStepCount;
    if (rows < 2 or rows > max_rows) return error.InvalidWindowRows;
    const lib = try mtl.Library.fromSource(model.device, "#include <metal_stdlib>\nusing namespace metal;\nkernel void sd_copy(const device uint* x [[buffer(0)]], device uint* y [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if(i<n) y[i]=x[i]; }", .{});
    defer lib.deinit();
    const pipeline = try mtl.Pipeline.init(model.device, lib, "sd_copy", false);
    defer pipeline.deinit();
    const attn_bytes = @as(usize, rows) * layer_count * attention_elements * 2;
    const coarse_bytes = @as(usize, rows) * coarse_stride * 80;
    const seq_attn_buffer = try model.device.buffer(attn_bytes, mtl.ResourceOptions.shared);
    defer seq_attn_buffer.deinit();
    const win_attn_buffer = try model.device.buffer(attn_bytes, mtl.ResourceOptions.shared);
    defer win_attn_buffer.deinit();
    const seq_coarse_buffer = try model.device.buffer(coarse_bytes, mtl.ResourceOptions.shared);
    defer seq_coarse_buffer.deinit();
    const win_coarse_buffer = try model.device.buffer(coarse_bytes, mtl.ResourceOptions.shared);
    defer win_coarse_buffer.deinit();
    model.residency.add(seq_attn_buffer);
    model.residency.add(win_attn_buffer);
    model.residency.add(seq_coarse_buffer);
    model.residency.add(win_coarse_buffer);
    model.residency.commit();
    model.residency.requestResidency();
    defer {
        model.residency.remove(seq_attn_buffer);
        model.residency.remove(win_attn_buffer);
        model.residency.remove(seq_coarse_buffer);
        model.residency.remove(win_coarse_buffer);
        model.residency.commit();
    }
    var seq_capture = Capture{ .attention_buffer = seq_attn_buffer, .coarse_buffer = seq_coarse_buffer, .pipeline = pipeline, .rows = rows };
    var win_capture = Capture{ .attention_buffer = win_attn_buffer, .coarse_buffer = win_coarse_buffer, .pipeline = pipeline, .rows = rows };
    const saved_model_observer = model.observer;
    const saved_attention_observer = model.attention_observer;
    defer {
        model.observer = saved_model_observer;
        model.attention_observer = saved_attention_observer;
    }
    const seq = try gpa.alloc(f32, @as(usize, rows) * logits_width);
    defer gpa.free(seq);
    const window = try gpa.alloc(f32, @as(usize, rows) * logits_width);
    defer gpa.free(window);
    var context: std.ArrayList(u32) = .empty;
    defer context.deinit(gpa);
    try context.appendSlice(gpa, ids);
    var row_bad: [max_rows]usize = @splat(0);
    var report: std.ArrayList(u8) = .empty;
    defer report.deinit(gpa);
    try report.print(gpa, "{{\"mode\":\"stagediff\",\"prefix_tokens\":{d},\"steps\":{d},\"rows\":{d},\"records\":[\n", .{ ids.len, steps, rows });
    for (0..steps) |step| {
        const position = context.items.len;
        const prefill_row = try seed(model, context.items);
        const pending = argmax(prefill_row);
        var tokens: [max_rows]u32 = @splat(0);
        tokens[0] = pending;
        const second = std.c.getenv("TF_DSV41_STAGEDIFF_SECOND") != null;
        // The sequential chain: one 1-row forward a row, each captured into its own slot.
        @memset(seq_attn_buffer.contents()[0..attn_bytes], 0);
        @memset(seq_coarse_buffer.contents()[0..coarse_bytes], 0);
        model.observer = .{ .ctx = &seq_capture, .encode = Capture.coarseGpu };
        model.attention_observer = .{ .ctx = &seq_capture, .capture = Capture.attentionGpu };
        _ = try seed(model, context.items);
        for (0..rows) |i| {
            seq_capture.row_base = @intCast(i);
            seq_capture.active = 1;
            const r = try model.forward(&.{tokens[i]}, .all);
            if (r.len < logits_width) return error.InvalidLogitsShape;
            @memcpy(seq[i * logits_width ..][0..logits_width], r[0..logits_width]);
            if (i + 1 < rows) tokens[i + 1] = if (second) secondBest(r, argmax(r)) else argmax(r);
        }
        // The N-row window on the same tokens, every row captured.
        @memset(win_attn_buffer.contents()[0..attn_bytes], 0);
        @memset(win_coarse_buffer.contents()[0..coarse_bytes], 0);
        model.observer = .{ .ctx = &win_capture, .encode = Capture.coarseGpu };
        model.attention_observer = .{ .ctx = &win_capture, .capture = Capture.attentionGpu };
        _ = try seed(model, context.items);
        win_capture.row_base = 0;
        win_capture.active = rows;
        const w = try model.forward(tokens[0..rows], .all);
        if (w.len < @as(usize, rows) * logits_width) return error.InvalidLogitsShape;
        @memcpy(window[0 .. @as(usize, rows) * logits_width], w[0 .. @as(usize, rows) * logits_width]);
        const diff = firstCaptureDifference(seq_attn_buffer.slice(u16, @as(usize, rows) * layer_count * attention_elements), win_attn_buffer.slice(u16, @as(usize, rows) * layer_count * attention_elements), seq_coarse_buffer.contents()[0..coarse_bytes], win_coarse_buffer.contents()[0..coarse_bytes], rows);
        try report.print(gpa, "{s}{{\"step\":{d},\"position\":{d},\"pending\":{d},\"tokens\":[{d}", .{ if (step == 0) "" else ",\n", step, position, pending, tokens[0] });
        var i: usize = 1;
        while (i < rows) : (i += 1) try report.print(gpa, ",{d}", .{tokens[i]});
        try report.appendSlice(gpa, "],\"row_equal\":[");
        for (0..rows) |r| {
            if (r != 0) try report.append(gpa, ',');
            const metric = compareF32(seq[r * logits_width ..][0..logits_width], window[r * logits_width ..][0..logits_width]);
            if (metric.unequal != 0) row_bad[r] += 1;
            try report.print(gpa, "{s}", .{if (metric.unequal == 0) "true" else "false"});
        }
        try report.appendSlice(gpa, "],\"first_diff\":");
        if (diff) |d| {
            try report.print(gpa, "{{\"row\":{d},\"layer\":{d},\"item\":\"{s}\",\"unequal\":{d},\"elements\":{d},\"nrms\":{d},\"max_abs\":{d}}}", .{ d.row, d.layer, d.item, d.metric.unequal, d.metric.elements, d.metric.nrms, d.metric.max_abs });
        } else try report.appendSlice(gpa, "null");
        try report.append(gpa, '}');
        std.debug.print("stagediff step {d} pos {d} rows {d}:", .{ step, position, rows });
        for (0..rows) |r| {
            const metric = compareF32(seq[r * logits_width ..][0..logits_width], window[r * logits_width ..][0..logits_width]);
            std.debug.print(" row{d}={s}", .{ r, if (metric.unequal == 0) "=" else "DIFF" });
            if (metric.unequal != 0) std.debug.print("(nrms {d})", .{metric.nrms});
        }
        if (diff) |d| std.debug.print("  first: row {d} layer {d} {s} {d}/{d} nrms {d}", .{ d.row, d.layer, d.item, d.metric.unequal, d.metric.elements, d.metric.nrms });
        std.debug.print("\n", .{});
        try context.append(gpa, pending);
    }
    try report.appendSlice(gpa, "\n],\"row_bad\":[");
    for (0..rows) |r| {
        if (r != 0) try report.append(gpa, ',');
        try report.print(gpa, "{d}", .{row_bad[r]});
    }
    try report.appendSlice(gpa, "],\"execution\":\"window_rows_vs_sequential_chain\"}\n");
    const path = try std.fs.path.join(gpa, &.{ dir, "stagediff.json" });
    defer gpa.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = report.items });
    std.debug.print("stagediff rows={d}:", .{rows});
    for (0..rows) |r| std.debug.print(" row{d} {d}/{d} bad", .{ r, row_bad[r], steps });
    std.debug.print("\n", .{});
}

/// The full drafted round loop in the runner: prefill, then rounds of [pending, drafts...].
fn stateDiff(model: *Model, round: usize) void {
    const a = model.caches[0];
    const b = model.caches[1];
    const ring_end = @as(usize, a.plan.layers) * 128 * 528;
    const partial_end = ring_end + a.plan.partialBytes();
    const capture_end = partial_end + 4 * 8 * 2 * 512 * 4;
    const n = a.plan.mutableBytes();
    var first: ?usize = null;
    var words: usize = 0;
    var i: usize = 0;
    while (i + 3 < n) : (i += 4) {
        const x = std.mem.readInt(u32, a.live.contents()[i..][0..4], .little);
        const y = std.mem.readInt(u32, b.live.contents()[i..][0..4], .little);
        if (x != y) {
            if (first == null) first = i;
            words += 1;
        }
    }
    if (first) |off| {
        if (off < ring_end) {
            const layer = off / (128 * 528);
            const within = off % (128 * 528);
            std.debug.print("state diff round {d}: RING layer {d} slot {d} word {d} (first {d}, {d} words differ of {d})\n", .{ round, layer, within / 528, (within % 528) / 4, off, words, n / 4 });
        } else if (off < partial_end) {
            std.debug.print("state diff round {d}: PARTIAL +{d} (first {d}, {d} words differ)\n", .{ round, off - ring_end, off, words });
        } else if (off < capture_end) {
            std.debug.print("state diff round {d}: CAPTURE +{d} (first {d}, {d} words differ)\n", .{ round, off - partial_end, off, words });
        } else std.debug.print("state diff round {d}: OTHER +{d} (first {d}, {d} words differ)\n", .{ round, off, off, words });
    } else {
        std.debug.print("state diff round {d}: live identical ({d} words)\n", .{ round, n / 4 });
    }
    // The compressed rings: capacity/ratio rows of main_stride+index_stride bytes per owner.
    const s0 = a.storage.contents();
    const s1 = b.storage.contents();
    var so: ?usize = null;
    var sw: usize = 0;
    i = 0;
    while (i + 3 < a.plan.packedBytes()) : (i += 4) {
        const x = std.mem.readInt(u32, s0[i..][0..4], .little);
        const y = std.mem.readInt(u32, s1[i..][0..4], .little);
        if (x != y) {
            if (so == null) so = i;
            sw += 1;
        }
    }
    if (so) |off| std.debug.print("state diff round {d}: STORAGE +{d} (first {d}, {d} words differ)\n", .{ round, off, off, sw });
}

pub fn loop(gpa: std.mem.Allocator, io: std.Io, model: *Model, ids: []const u32, dir: []const u8, rounds_arg: ?u32, rows_arg: ?u32) !void {
    if (ids.len == 0 or ids.len > 4096) return error.InvalidPrompt;
    const rounds = rounds_arg orelse 16;
    const rows: u32 = rows_arg orelse 4;
    if (rounds == 0 or rounds > 128) return error.InvalidStepCount;
    if (rows < 2 or rows > max_rows) return error.InvalidWindowRows;
    if (model.lanes < 2) return error.NeedTwoLanes;
    const ref = try gpa.alloc(f32, @as(usize, rows + 1) * logits_width);
    defer gpa.free(ref);
    const window = try gpa.alloc(f32, @as(usize, rows) * logits_width);
    defer gpa.free(window);
    var ctx: std.ArrayList(u32) = .empty;
    defer ctx.deinit(gpa);
    try ctx.appendSlice(gpa, ids);
    var report: std.ArrayList(u8) = .empty;
    defer report.deinit(gpa);
    try report.print(gpa, "{{\"mode\":\"draftloop\",\"prefix_tokens\":{d},\"rounds\":{d},\"rows\":{d},\"records\":[\n", .{ ids.len, rounds, rows });
    try model.resetSlot(0);
    const pre = try model.forwardOn(0, ctx.items, .last);
    var pending = argmax(pre);
    // The plain reference is a DECODE chain, not a re-prefill.
    try model.resetSlot(1);
    const pre_ref = try model.forwardOn(1, ctx.items, .last);
    var pending_ref = argmax(pre_ref);
    var diverged: ?usize = null;
    var first_detail: ?[]const u8 = null;
    var kept_hist: [max_rows]usize = @splat(0);
    // The drafted path's committed stream and the plain chain's own committed stream, as token ids.
    var committed = std.ArrayList(u32).empty;
    defer committed.deinit(gpa);
    var ref_committed = std.ArrayList(u32).empty;
    defer ref_committed.deinit(gpa);
    for (0..rounds) |round| {
        // Probe the plain chain rows tokens deep.
        try model.beginSlot(1);
        var r = try model.forwardOn(1, &.{pending_ref}, .all);
        @memcpy(ref[1 * logits_width ..][0..logits_width], r[0..logits_width]);
        for (1..rows) |j| {
            const tok = argmax(ref[j * logits_width ..][0..logits_width]);
            r = try model.forwardOn(1, &.{tok}, .all);
            @memcpy(ref[(j + 1) * logits_width ..][0..logits_width], r[0..logits_width]);
        }
        const ref0 = pending_ref;
        // The reject point cycles so kept = 1, 2, ... rows-1 across the run (unless the proposer arm).
        const k: u32 = 1 + @as(u32, @intCast(round % (rows - 1)));
        var feed: [max_rows]u32 = @splat(0);
        feed[0] = pending;
        if (std.c.getenv("TF_DSV41_DRAFTLOOP_PROPOSE") != null) {
            // The engine's own drafter between rounds: the MTP proposer's tokens ARE the window's drafts.
            const drafts = model.proposeTokens(0, pending, rows - 1) catch |err| blk: {
                std.debug.print("draftloop: proposer unavailable ({s})\n", .{@errorName(err)});
                break :blk [4]u32{ 0, 0, 0, 0 };
            };
            for (1..rows) |j| feed[j] = drafts[j - 1];
        } else for (1..rows) |j| {
            feed[j] = if (j < k) argmax(ref[j * logits_width ..][0..logits_width]) else if (j == k) (argmax(ref[j * logits_width ..][0..logits_width]) + 1) % 129280 else argmax(ref[j * logits_width ..][0..logits_width]);
        }
        try model.beginSlot(0);
        const w = try model.forwardOn(0, feed[0..rows], .all);
        if (w.len < @as(usize, rows) * logits_width) return error.InvalidLogitsShape;
        @memcpy(window[0 .. @as(usize, rows) * logits_width], w[0 .. @as(usize, rows) * logits_width]);
        var kept: u32 = 1;
        var accepted: u32 = 0;
        for (1..rows) |j| {
            const sampled_prev = argmax(window[(j - 1) * logits_width ..][0..logits_width]);
            if (sampled_prev == feed[j]) {
                accepted += 1;
            } else break;
        }
        kept = 1 + accepted;
        kept_hist[@intCast(kept)] += 1;
        // Only the rows BEFORE the rejected draft are comparable: from the row kept onward.
        var row_mismatch: ?usize = null;
        for (0..kept) |j| {
            const got = argmax(window[j * logits_width ..][0..logits_width]);
            const want = argmax(ref[(j + 1) * logits_width ..][0..logits_width]);
            if (got != want and row_mismatch == null) row_mismatch = j;
        }
        const pending_mismatch = pending != ref0;
        if (diverged == null and (pending_mismatch or row_mismatch != null)) {
            diverged = round;
            first_detail = if (pending_mismatch) "pending" else "row_argmax";
        }
        std.debug.print("draftloop round {d} pos {d} pending {d} ref0 {d} kept {d} accepted {d} row_mismatch={?} pending_mismatch={}\n", .{ round, ctx.items.len, pending, ref0, kept, accepted, row_mismatch, pending_mismatch });
        try report.print(gpa, "{s}{{\"round\":{d},\"position\":{d},\"pending\":{d},\"ref0\":{d},\"kept\":{d},\"accepted\":{d},\"row_mismatch\":", .{ if (round == 0) "" else ",\n", round, ctx.items.len, pending, ref0, kept, accepted });
        if (row_mismatch) |rm| try report.print(gpa, "{d}", .{rm}) else try report.appendSlice(gpa, "null");
        try report.print(gpa, ",\"pending_mismatch\":{s}}}", .{if (pending_mismatch) "true" else "false"});
        // The keep path, as the engine's keep() runs it.
        if (kept < rows) {
            try model.rollbackSlot(0);
            try model.restoreRing(0, kept, rows);
            try model.portKeep(0, kept);
            model.commitSlot(0) catch {};
        } else model.commitSlot(0) catch {};
        // The committed tokens are the kept feed rows; the next pending is the last kept row's argmax.
        try ctx.appendSlice(gpa, feed[0..kept]);
        try committed.appendSlice(gpa, feed[0..kept]);
        try ref_committed.append(gpa, pending_ref);
        for (1..kept) |j| try ref_committed.append(gpa, argmax(ref[j * logits_width ..][0..logits_width]));
        pending = argmax(window[(kept - 1) * logits_width ..][0..logits_width]);
        // Roll the reference probe back and advance the plain chain over exactly the kept tokens.
        try model.rollbackSlot(1);
        // The probe's own untouched rows must not linger in the reference lane's ring either.
        try model.restoreRing(1, kept, rows);
        for (0..kept) |j| {
            const tok: u32 = if (j == 0) pending_ref else argmax(ref[j * logits_width ..][0..logits_width]);
            _ = try model.forwardOn(1, &.{tok}, .all);
        }
        pending_ref = argmax(ref[kept * logits_width ..][0..logits_width]);
        if (std.c.getenv("TF_DSV41_DRAFTLOOP_STATEDIFF") != null) stateDiff(model, round);
    }
    try report.appendSlice(gpa, "\n],\"kept_hist\":[");
    for (0..rows) |j| {
        if (j != 0) try report.append(gpa, ',');
        try report.print(gpa, "{d}", .{kept_hist[j]});
    }
    try report.print(gpa, "],\"first_diverged_round\":{s},\"detail\":\"{s}\"", .{ if (diverged) |d| std.fmt.allocPrint(gpa, "{d}", .{d}) catch "?" else "null", first_detail orelse "none" });
    try report.print(gpa, ",\"committed\":[", .{});
    for (committed.items, 0..) |t, i| {
        if (i != 0) try report.append(gpa, ',');
        try report.print(gpa, "{d}", .{t});
    }
    try report.print(gpa, "],\"ref_committed\":[", .{});
    for (ref_committed.items, 0..) |t, i| {
        if (i != 0) try report.append(gpa, ',');
        try report.print(gpa, "{d}", .{t});
    }
    try report.print(gpa, "]}}\n", .{});
    // The committed stream as a plain ids list: the FP64 chain replays exactly this history.
    {
        const cpath = try std.fs.path.join(gpa, &.{ dir, "committed-ids.json" });
        defer gpa.free(cpath);
        var buf = std.ArrayList(u8).empty;
        defer buf.deinit(gpa);
        try buf.append(gpa, '[');
        for (committed.items, 0..) |t, i| {
            if (i != 0) try buf.append(gpa, ',');
            try buf.print(gpa, "{d}", .{t});
        }
        try buf.appendSlice(gpa, "]");
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = cpath, .data = buf.items });
        const rpath = try std.fs.path.join(gpa, &.{ dir, "ref-committed-ids.json" });
        defer gpa.free(rpath);
        var rbuf = std.ArrayList(u8).empty;
        defer rbuf.deinit(gpa);
        try rbuf.append(gpa, '[');
        for (ref_committed.items, 0..) |t, i| {
            if (i != 0) try rbuf.append(gpa, ',');
            try rbuf.print(gpa, "{d}", .{t});
        }
        try rbuf.appendSlice(gpa, "]");
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = rpath, .data = rbuf.items });
    }
    const path = try std.fs.path.join(gpa, &.{ dir, "draftloop.json" });
    defer gpa.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = report.items });
    std.debug.print("draftloop: {d} rounds, kept={any}, first divergence {?d} ({s})\n", .{ rounds, kept_hist, diverged, first_detail orelse "none" });
}
