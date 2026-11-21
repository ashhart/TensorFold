//! Actual checkpoint teacher comparison: independent CPU arithmetic versus the resident GPU decode path.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const Model = tf.deepseek_v41.Model;
const reference_module = @import("reference.zig");
const Reference = reference_module.Reference;
const engram_rows = tf.deepseek_v41_engram_rows;
const nrms_limit: f64 = 0.01;
/// The maintainer's floor-relative band: <= 1.05x the measured FP32-vs-FP64 floor clears, <= 1.10x is review.
const band_clear_limit: f64 = 1.05;
const band_review_limit: f64 = 1.10;

pub const Result = struct {
    tokens: usize,
    max_absolute_error: f64,
    rms_error: f64,
    normalized_rms_error: f64,
    /// KL divergence of the teacher's next-token distribution from the engine's, per row: mean and worst.
    mean_kl: f64 = 0,
    max_kl: f64 = 0,
    greedy_disagreements: usize,
    gpu_seconds: f64,
    cpu_seconds: f64,
    normalized_rms_limit: f64 = nrms_limit,
    /// The operative numeric gate: the floor-relative band on the FP64 run, the absolute trip-wire on the FP32 capture.
    numeric_gate_passed: bool = false,
    /// The literal 1% trip-wire, reported beside the band and never the arbiter.
    absolute_gate_passed: bool = false,
    /// The FP64 argmax is reported, not thresholded; the zero-argmax clause is the engine's own path invariants.
    greedy_gate_passed: bool = false,
    /// The FP32 contract capture's own engine-vs-contract NRMS and argmax count, carried into the FP64 receipt.
    contract_normalized_rms_error: f64 = 0,
    contract_greedy_disagreements: usize = 0,
    contract_measured: bool = false,
    /// The FP32 contract chain's own NRMS against the same FP64 teacher, when its hashed logits sit beside this run.
    floor_normalized_rms_error: f64 = 0,
    /// Engine NRMS over the contract's floor; the operative band is 1.05 clear / 1.10 review.
    band_ratio: f64 = 0,
    clear_ratio_limit: f64 = band_clear_limit,
    review_ratio_limit: f64 = band_review_limit,
    band_gate_passed: bool = false,
    band_measured: bool = false,
    band_verdict: []const u8 = "unmeasured",
    /// Which rule `numeric_gate_passed` reports: the floor-relative band, or the FP32 contract capture.
    gate_basis: []const u8 = "floor_relative_band",
    /// The checkpoint metadata identity the floor is bound to, as 64 lower-case hex characters.
    checkpoint_metadata_identity_hex: [64]u8 = @splat('0'),
};

/// Indexer-selection boundary per window: contract margin (BF16-rounded FP32 scores), FP64-ideal margin, support flips.
const MarginLog = struct {
    records: [1024]MarginRecord = undefined,
    len: usize = 0,

    const MarginRecord = struct {
        layer: u32,
        position: u32,
        count: u32,
        selected: u32,
        margin: f32,
        margin_ideal: f32,
        flips: u32,
        lowest_selected: f32,
        highest_unselected: f32,
    };

    fn capture(ctx: *anyopaque, layer: usize, position: usize, count: usize, selected_len: usize, margin: f32, margin_ideal: f32, flips: usize, lowest: f32, highest_unselected: f32) void {
        const self: *MarginLog = @ptrCast(@alignCast(ctx));
        if (self.len == self.records.len) return;
        self.records[self.len] = .{
            .layer = @intCast(layer),
            .position = @intCast(position),
            .count = @intCast(count),
            .selected = @intCast(selected_len),
            .margin = margin,
            .margin_ideal = margin_ideal,
            .flips = @intCast(flips),
            .lowest_selected = lowest,
            .highest_unselected = highest_unselected,
        };
        self.len += 1;
    }

    fn write(self: *MarginLog, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(gpa);
        var flips: usize = 0;
        var tight: usize = 0;
        var total: usize = 0;
        for (self.records[0..self.len]) |r| {
            total += 1;
            flips += r.flips;
            if (r.margin <= 0) tight += 1;
            var buf: [320]u8 = undefined;
            const line = try std.fmt.bufPrint(&buf, "{{\"layer\":{d},\"position\":{d},\"count\":{d},\"selected\":{d},\"margin\":{d},\"margin_ideal\":{d},\"flips\":{d},\"lowest_selected\":{d},\"highest_unselected\":{d}}}\n", .{ r.layer, r.position, r.count, r.selected, r.margin, r.margin_ideal, r.flips, r.lowest_selected, r.highest_unselected });
            try text.appendSlice(gpa, line);
        }
        var head_buf: [192]u8 = undefined;
        const head = try std.fmt.bufPrint(&head_buf, "{{\"records\":{d},\"total_flips\":{d},\"zero_or_negative_margin\":{d}}}\n", .{ total, flips, tight });
        const path = try std.fs.path.join(gpa, &.{ dir, "indexer-margins.jsonl" });
        defer gpa.free(path);
        var file = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer file.close(io);
        try file.writeStreamingAll(io, head);
        try file.writeStreamingAll(io, text.items);
        std.debug.print("indexer margins: {d} windows, {d} flips contract-vs-ideal, {d} zero-or-negative margins\n", .{ total, flips, tight });
    }
};

fn maximum(row: []const f32) !u32 {
    var id: u32 = 0;
    var best = -std.math.inf(f32);
    for (row, 0..) |v, i| {
        if (!std.math.isFinite(v)) return error.NonfiniteTeacherLogits;
        if (v > best) {
            best = v;
            id = @intCast(i);
        }
    }
    return id;
}

/// The same argmax over the FP64 chain's own logits.
fn maximumWide(row: []const f64) !u32 {
    var id: u32 = 0;
    var best = -std.math.inf(f64);
    for (row, 0..) |v, i| {
        if (!std.math.isFinite(v)) return error.NonfiniteTeacherLogits;
        if (v > best) {
            best = v;
            id = @intCast(i);
        }
    }
    return id;
}

fn streamRow(p: *anyopaque, layer: usize, row: u32, out: []u16) anyerror!void {
    const r: *engram_rows.Rows = @ptrCast(@alignCast(p));
    // `layer` arrives as the reference's own index (0 for layer 1, 1 for layer 14), which is the reader's order.
    try r.rowsU32(layer, &.{row}, 256, out);
    if (std.c.getenv("TF_DSV41_DUMP_ROW") != null) {
        // One line per row fetched: enough to check these exact bytes against a direct read of the shard.
        var fnv: u64 = 0xcbf29ce484222325;
        for (std.mem.sliceAsBytes(out)) |b| {
            fnv ^= b;
            fnv *%= 0x100000001b3;
        }
        var hex: [32]u8 = undefined;
        std.debug.print("ROWDUMP layer={d} row={d} fnv={s} first={x},{x},{x},{x}\n", .{ layer, row, std.fmt.bufPrint(&hex, "{x}", .{fnv}) catch "?", out[0], out[1], out[2], out[3] });
    }
}
pub fn run(gpa: std.mem.Allocator, io: std.Io, model: *Model, ids: []const u32, token_map: []const u8, dir: []const u8, identity: [32]u8) !Result {
    return runOf(f32, gpa, io, model, ids, token_map, dir, identity);
}

/// The same oracle against the FP64 chain -- the reference chain at its ideal contract (every reduction class widened).
pub fn runF64(gpa: std.mem.Allocator, io: std.Io, model: *Model, ids: []const u32, token_map: []const u8, dir: []const u8, identity: [32]u8) !Result {
    return runOf(f64, gpa, io, model, ids, token_map, dir, identity);
}

fn runOf(comptime T: type, gpa: std.mem.Allocator, io: std.Io, model: *Model, ids: []const u32, token_map: []const u8, dir: []const u8, identity: [32]u8) !Result {
    if (ids.len == 0 or ids.len > 256) return error.InvalidOracleWindow;
    var reference = try reference_module.ReferenceOf(T).init(gpa, model.cfg, &model.weights, &model.checkpoint, token_map, @intCast(ids.len));
    defer reference.deinit();
    // A streamed model keeps its Engram tables on disk; give the reference a row source so the comparison can happen.
    var stream_rows: ?*engram_rows.Rows = null;
    if (std.c.getenv("TF_DSV41_ENGRAM_STREAM_DIR")) |d| {
        stream_rows = try engram_rows.Rows.open(gpa, io, std.mem.span(d), 0, 1, &.{ 1, 14 }, 4096);
        reference.attachRows(.{ .ptr = stream_rows.?, .rowFn = streamRow });
    }
    defer if (stream_rows) |r| r.close();
    var margins = MarginLog{};
    reference.attention_state.score_probe = .{ .ctx = &margins, .record = &MarginLog.capture };
    const cpu = try gpa.alloc(T, ids.len * 129280);
    defer gpa.free(cpu);
    const gpu = try gpa.alloc(f32, cpu.len);
    defer gpa.free(gpu);
    try model.reset();
    var result = Result{ .tokens = ids.len, .max_absolute_error = 0, .rms_error = 0, .normalized_rms_error = 0, .greedy_disagreements = 0, .gpu_seconds = 0, .cpu_seconds = 0, .checkpoint_metadata_identity_hex = std.fmt.bytesToHex(identity, .lower) };
    var squared_error: f64 = 0;
    var squared_reference: f64 = 0;
    // The engine's logits only.
    if (std.c.getenv("TF_DSV41_ORACLE_GPU_ONLY") != null) {
        for (ids, 0..) |id, i| {
            const actual = try model.forward(&.{id}, .all);
            result.gpu_seconds += model.last_gpu_seconds;
            @memcpy(gpu[i * 129280 ..][0..129280], actual);
        }
        const gp = try std.fs.path.join(gpa, &.{ dir, "gpu-logits.f32" });
        defer gpa.free(gp);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = gp, .data = std.mem.sliceAsBytes(gpu) });
        const report = try std.fs.path.join(gpa, &.{ dir, "oracle.json" });
        defer gpa.free(report);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = report, .data = "{\"mode\":\"oracle-gpu-only\"}" });
        std.debug.print("oracle-gpu-only: {d} tokens, {d:.3} gpu seconds\n", .{ ids.len, result.gpu_seconds });
        return result;
    }
    for (ids, 0..) |id, i| {
        const actual = try model.forward(&.{id}, .all);
        result.gpu_seconds += model.last_gpu_seconds;
        @memcpy(gpu[i * 129280 ..][0..129280], actual);
        const start = mtl.clock.seconds();
        const expected = try reference.step(id);
        result.cpu_seconds += mtl.clock.seconds() - start;
        @memcpy(cpu[i * 129280 ..][0..129280], expected);
        const kl = try klRow(actual, expected);
        result.mean_kl += kl;
        result.max_kl = @max(result.max_kl, kl);
        const expected_token = if (T == f32) try maximum(expected) else try maximumWide(expected);
        if (try maximum(actual) != expected_token) result.greedy_disagreements += 1;
        for (actual, expected) |a, b| {
            if (!std.math.isFinite(a) or !std.math.isFinite(b)) return error.NonfiniteTeacherLogits;
            const wide: f64 = if (T == f32) @as(f64, b) else b;
            const error_: f64 = @as(f64, a) - wide;
            result.max_absolute_error = @max(result.max_absolute_error, @abs(error_));
            squared_error += error_ * error_;
            squared_reference += wide * wide;
        }
        std.debug.print("oracle teacher {d}/{d}: GPU token={d}, CPU token={d}\n", .{ i + 1, ids.len, try maximum(actual), expected_token });
    }
    if (ids.len > 0) result.mean_kl /= @as(f64, @floatFromInt(ids.len));
    result.rms_error = @sqrt(squared_error / @as(f64, @floatFromInt(cpu.len)));
    result.normalized_rms_error = if (squared_reference == 0) result.rms_error else @sqrt(squared_error / squared_reference);
    result.absolute_gate_passed = result.normalized_rms_error <= result.normalized_rms_limit;
    result.greedy_gate_passed = result.greedy_disagreements == 0;
    // The operative whole-model clause is floor-relative: the FP32 contract chain's own FP64 gap is the floor.
    if (T == f64) {
        if (try readFloor(gpa, io, dir, ids, cpu.len, identity)) |floor| {
            defer gpa.free(floor);
            var floor_squared: f64 = 0;
            for (floor, cpu) |f, c| {
                const delta = @as(f64, f) - c;
                floor_squared += delta * delta;
            }
            const floor_nrms = if (squared_reference == 0) 0 else @sqrt(floor_squared / squared_reference);
            const band = bandVerdict(result.normalized_rms_error, floor_nrms);
            result.floor_normalized_rms_error = floor_nrms;
            result.band_ratio = band.ratio;
            result.band_verdict = band.verdict;
            result.band_gate_passed = band.passed;
            result.band_measured = band.measured;
        }
        if (try readContract(gpa, io, dir)) |contract| {
            result.contract_normalized_rms_error = contract.nrms;
            result.contract_greedy_disagreements = contract.greedy;
            result.contract_measured = true;
        }
    }
    result.numeric_gate_passed = if (T == f64) result.band_gate_passed else result.absolute_gate_passed;
    result.gate_basis = if (T == f64) "floor_relative_band" else "contract_capture";
    // A capture that cannot reproduce the contract is not a floor; this is a sanity bound, not the band.
    if (T == f32 and (!std.math.isFinite(result.normalized_rms_error) or result.normalized_rms_error > 0.6)) return error.ContractReferenceMismatch;
    // The FP32 chain's teacher keeps the shipped file name; the FP64 chain's teacher writes its own FP64 file.
    const cp = try std.fs.path.join(gpa, &.{ dir, if (T == f32) "oracle-logits.f32" else "oracle-logits.f64" });
    defer gpa.free(cp);
    const gp = try std.fs.path.join(gpa, &.{ dir, "gpu-logits.f32" });
    defer gpa.free(gp);
    const report = try std.fs.path.join(gpa, &.{ dir, "oracle.json" });
    defer gpa.free(report);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = cp, .data = std.mem.sliceAsBytes(cpu) });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = gp, .data = std.mem.sliceAsBytes(gpu) });
    if (T == f32) {
        try writeFloor(gpa, io, dir, ids, cpu, identity);
        try writeContract(gpa, io, dir, result.normalized_rms_error, result.greedy_disagreements);
    }
    const json = try std.json.Stringify.valueAlloc(gpa, result, .{});
    defer gpa.free(json);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = report, .data = json });
    try margins.write(gpa, io, dir);
    std.debug.print("{s}\n", .{json});
    if (T == f64) try enforceGate(&result);
    return result;
}

/// The teacher's distribution against the engine's on one logits row: KL(teacher || engine), f64.
fn klRow(actual: []const f32, expected: anytype) !f64 {
    var max_a: f64 = -std.math.inf(f64);
    var max_e: f64 = -std.math.inf(f64);
    for (actual, expected) |a, b| {
        if (!std.math.isFinite(a)) return error.NonfiniteTeacherLogits;
        const w: f64 = if (@TypeOf(expected[0]) == f32) @as(f64, b) else b;
        if (!std.math.isFinite(w)) return error.NonfiniteTeacherLogits;
        max_a = @max(max_a, @as(f64, a));
        max_e = @max(max_e, w);
    }
    var sum_a: f64 = 0;
    var sum_e: f64 = 0;
    for (actual, expected) |a, b| {
        const w: f64 = if (@TypeOf(expected[0]) == f32) @as(f64, b) else b;
        sum_a += @exp(@as(f64, a) - max_a);
        sum_e += @exp(w - max_e);
    }
    const lse_a = max_a + @log(sum_a);
    const lse_e = max_e + @log(sum_e);
    var kl: f64 = 0;
    for (actual, expected) |a, b| {
        const w: f64 = if (@TypeOf(expected[0]) == f32) @as(f64, b) else b;
        const p = @exp(w - lse_e);
        kl += p * ((w - lse_e) - (@as(f64, a) - lse_a));
    }
    return @max(kl, 0);
}

const Band = struct { ratio: f64 = 0, verdict: []const u8 = "unmeasured", measured: bool = false, passed: bool = false };

/// The engine's FP64 gap over the FP32 contract chain's own FP64 gap: 1.05 clears, 1.10 is review.
fn bandVerdict(engine_nrms: f64, floor_nrms: f64) Band {
    if (!std.math.isFinite(engine_nrms) or !std.math.isFinite(floor_nrms) or floor_nrms <= 0) return .{};
    const ratio = engine_nrms / floor_nrms;
    const verdict: []const u8 = if (ratio <= band_clear_limit) "clear" else if (ratio <= band_review_limit) "review" else "fail";
    return .{ .ratio = ratio, .verdict = verdict, .measured = true, .passed = ratio <= band_review_limit };
}

/// The floor manifest binds the window, its hash, the contract logits hash and the checkpoint metadata identity.
fn fnv1a(bytes: []const u8) u64 {
    var hash: u64 = 0xcbf29ce484222325;
    for (bytes) |b| {
        hash ^= b;
        hash *%= 0x100000001b3;
    }
    return hash;
}

/// Written by the FP32 contract capture; readFloor refuses any other window, bytes or metadata identity.
fn writeFloor(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, ids: []const u32, cpu: []const f32, identity: [32]u8) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, "floor.txt" });
    defer gpa.free(path);
    var buf: [192]u8 = undefined;
    const hex = std.fmt.bytesToHex(identity, .lower);
    const text = try std.fmt.bufPrint(&buf, "{d}\n{x}\n{x}\n{d}\n{s}\n", .{ ids.len, fnv1a(std.mem.sliceAsBytes(ids)), fnv1a(std.mem.sliceAsBytes(cpu)), cpu.len, &hex });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = text });
}

/// The FP32 contract chain's logits from the same directory, verified against the manifest the capture wrote.
fn readFloor(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, ids: []const u32, cells: usize, identity: [32]u8) !?[]f32 {
    const manifest_path = try std.fs.path.join(gpa, &.{ dir, "floor.txt" });
    defer gpa.free(manifest_path);
    const manifest = std.Io.Dir.cwd().readFileAlloc(io, manifest_path, gpa, .limited(512)) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer gpa.free(manifest);
    var lines = std.mem.splitScalar(u8, manifest, '\n');
    const tokens = std.fmt.parseInt(usize, lines.next() orelse return error.FloorReferenceMismatch, 10) catch return error.FloorReferenceMismatch;
    const ids_hash = std.fmt.parseInt(u64, lines.next() orelse return error.FloorReferenceMismatch, 16) catch return error.FloorReferenceMismatch;
    const logits_hash = std.fmt.parseInt(u64, lines.next() orelse return error.FloorReferenceMismatch, 16) catch return error.FloorReferenceMismatch;
    const manifest_cells = std.fmt.parseInt(usize, lines.next() orelse return error.FloorReferenceMismatch, 10) catch return error.FloorReferenceMismatch;
    const identity_line = std.mem.trim(u8, lines.next() orelse return error.FloorReferenceMismatch, " \r\n");
    var manifest_identity: [32]u8 = undefined;
    if (identity_line.len != 64) return error.FloorReferenceMismatch;
    _ = std.fmt.hexToBytes(&manifest_identity, identity_line) catch return error.FloorReferenceMismatch;
    if (tokens != ids.len or manifest_cells != cells or ids_hash != fnv1a(std.mem.sliceAsBytes(ids))) return error.FloorReferenceMismatch;
    if (!std.mem.eql(u8, &manifest_identity, &identity)) return error.FloorReferenceMismatch;
    const path = try std.fs.path.join(gpa, &.{ dir, "oracle-logits.f32" });
    defer gpa.free(path);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(cells * @sizeOf(f32) + 1)) catch |err| switch (err) {
        error.FileNotFound => return error.FloorReferenceMismatch,
        else => return err,
    };
    defer gpa.free(bytes);
    if (bytes.len != cells * @sizeOf(f32) or fnv1a(bytes) != logits_hash) return error.FloorReferenceMismatch;
    const out = try gpa.alloc(f32, cells);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

const Contract = struct { nrms: f64 = 0, greedy: usize = 0 };

/// The FP32 capture's own engine-vs-contract figures, carried beside the floor so the FP64 receipt retains them.
fn writeContract(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, nrms: f64, greedy: usize) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, "contract.txt" });
    defer gpa.free(path);
    var buf: [128]u8 = undefined;
    const text = try std.fmt.bufPrint(&buf, "{d}\n{d}\n", .{ nrms, greedy });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = text });
}

fn readContract(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !?Contract {
    const path = try std.fs.path.join(gpa, &.{ dir, "contract.txt" });
    defer gpa.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(128)) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer gpa.free(text);
    var lines = std.mem.splitScalar(u8, text, '\n');
    const nrms = std.fmt.parseFloat(f64, lines.next() orelse return error.ContractReferenceMismatch) catch return error.ContractReferenceMismatch;
    const greedy = std.fmt.parseInt(usize, lines.next() orelse return error.ContractReferenceMismatch, 10) catch return error.ContractReferenceMismatch;
    return .{ .nrms = nrms, .greedy = greedy };
}

/// The FP64 teacher's verdict is the floor-relative band; the absolute 1% remains a reported trip-wire.
fn enforceGate(result: *const Result) !void {
    if (!std.math.isFinite(result.normalized_rms_error)) return error.NonfiniteTeacherLogits;
    if (!result.band_measured) return error.FloorReferenceMissing;
    if (std.mem.eql(u8, result.band_verdict, "review")) std.debug.print(
        "oracle band: review, {d:.4}x the floor (<= {d:.2} clears, <= {d:.2} reviews)\n",
        .{ result.band_ratio, result.clear_ratio_limit, result.review_ratio_limit },
    );
    if (!result.absolute_gate_passed) std.debug.print(
        "oracle band: the absolute 1% trip-wire is reported false ({d:.6} > {d:.6}); it is a report, not the arbiter\n",
        .{ result.normalized_rms_error, result.normalized_rms_limit },
    );
    if (!result.band_gate_passed) return error.TeacherBandMismatch;
}

test "the enforced band accepts review and rejects a missing or failed floor" {
    var r = Result{ .tokens = 1, .max_absolute_error = 0, .rms_error = 0, .normalized_rms_error = 0.107, .greedy_disagreements = 4, .gpu_seconds = 0, .cpu_seconds = 0 };
    try std.testing.expectError(error.FloorReferenceMissing, enforceGate(&r));
    r.floor_normalized_rms_error = 0.1;
    r.band_ratio = 1.07;
    r.band_verdict = "review";
    r.band_gate_passed = true;
    r.band_measured = true;
    try enforceGate(&r);
    r.band_ratio = 1.11;
    r.band_verdict = "fail";
    r.band_gate_passed = false;
    try std.testing.expectError(error.TeacherBandMismatch, enforceGate(&r));
}

test "the FP64 argmax is reported, not the own-contract gate" {
    var r = Result{ .tokens = 1, .max_absolute_error = 0, .rms_error = 0, .normalized_rms_error = 0.1, .greedy_disagreements = 4, .gpu_seconds = 0, .cpu_seconds = 0 };
    r.band_measured = true;
    r.band_gate_passed = true;
    r.band_verdict = "clear";
    try enforceGate(&r);
    try std.testing.expect(!r.greedy_gate_passed);
}

test "the floor manifest refuses a mismatched window and accepts its own" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const a = std.testing.allocator;
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    defer a.free(dir);
    const ids = [_]u32{ 1, 2, 3 };
    const cpu = [_]f32{ 1.0, -2.0, 3.5 };
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "oracle-logits.f32", .data = std.mem.sliceAsBytes(&cpu) });
    const identity: [32]u8 = @splat(7);
    try writeFloor(a, std.testing.io, dir, &ids, &cpu, identity);
    const got = (try readFloor(a, std.testing.io, dir, &ids, cpu.len, identity)).?;
    defer a.free(got);
    try std.testing.expectEqualSlices(f32, &cpu, got);
    const wrong = [_]u32{ 1, 2, 4 };
    try std.testing.expectError(error.FloorReferenceMismatch, readFloor(a, std.testing.io, dir, &wrong, cpu.len, identity));
    var other = identity;
    other[0] ^= 1;
    try std.testing.expectError(error.FloorReferenceMismatch, readFloor(a, std.testing.io, dir, &ids, cpu.len, other));
}

test "the release-quality capture is the fixed 128-token input" {
    // The release-weights KL/top-1 window; its first 16 tokens are the 16-token window fixed below.
    const text = @embedFile("release-quality-128.json");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const ids = try std.json.parseFromSliceLeaky([]const u32, arena.allocator(), text, .{});
    try std.testing.expectEqual(@as(usize, 128), ids.len);
    try std.testing.expectEqual(@as(u64, 0x067323ba89384ac3), fnv1a(std.mem.sliceAsBytes(ids)));
    for (ids) |id| try std.testing.expect(id < 129280);
}

test "the release-quality 16-token window is fixed in the tree" {
    const text = @embedFile("release-quality-16.json");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const ids = try std.json.parseFromSliceLeaky([]const u32, arena.allocator(), text, .{});
    try std.testing.expectEqual(@as(usize, 16), ids.len);
    try std.testing.expectEqual(@as(u64, 0xfccc25db656b1c80), fnv1a(std.mem.sliceAsBytes(ids)));
    for (ids) |id| try std.testing.expect(id < 129280);
    // It is the first 16 tokens of the full capture, not a separate sequence.
    const full = try std.json.parseFromSliceLeaky([]const u32, arena.allocator(), @embedFile("release-quality-128.json"), .{});
    try std.testing.expectEqualSlices(u32, ids, full[0..16]);
}

test "the band is floor-relative and keeps its inclusive 1.05/1.10 boundaries" {
    try std.testing.expectEqual(@as(f64, 0), bandVerdict(1, 0).ratio);
    try std.testing.expectEqualStrings("unmeasured", bandVerdict(1, 0).verdict);
    try std.testing.expect(!bandVerdict(1, 0).measured);
    try std.testing.expectEqualStrings("clear", bandVerdict(1.05, 1.0).verdict);
    try std.testing.expect(bandVerdict(1.05, 1.0).passed);
    try std.testing.expectEqualStrings("review", bandVerdict(1.10, 1.0).verdict);
    try std.testing.expect(bandVerdict(1.10, 1.0).passed);
    try std.testing.expectEqualStrings("fail", bandVerdict(1.1000001, 1.0).verdict);
    try std.testing.expect(!bandVerdict(1.1000001, 1.0).passed);
}
