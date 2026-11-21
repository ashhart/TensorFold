//! DSpine's one-round gate: the MLX reference's committed taps and anchor in, Metal drafts and top-16 compared.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const dspine = q.dspine;
const help = "tf-qwen27-dspine-gate --model DIR --drafter DIR --ref DIR [--mode q4|bf16] [--fit ROWS] [--control]";

/// One tensor of a small safetensors file read whole.
const File = struct {
    bytes: []const u8,
    header: std.json.Parsed(std.json.Value),
    data: []const u8,

    fn open(a: std.mem.Allocator, io: std.Io, path: []const u8) !File {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 31));
        const n = std.mem.readInt(u64, bytes[0..8], .little);
        return .{ .bytes = bytes, .header = try std.json.parseFromSlice(std.json.Value, a, bytes[8..][0..n], .{}), .data = bytes[8 + n ..] };
    }
    fn get(f: File, name: []const u8) ![]const u8 {
        const e = f.header.value.object.get(name) orelse return error.MissingTensor;
        const offsets = e.object.get("data_offsets").?.array.items;
        return f.data[@intCast(offsets[0].integer)..@intCast(offsets[1].integer)];
    }
    fn words(f: File, name: []const u8) ![]align(1) const u32 {
        return std.mem.bytesAsSlice(u32, try f.get(name));
    }
};

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    var model_dir: ?[]const u8 = null;
    var drafter_dir: ?[]const u8 = null;
    var ref_dir: ?[]const u8 = null;
    var mode: dspine.runtime.Mode = .q4;
    var fit: ?u64 = null;
    var control = false; // the planted control: the second pass off, each row's own top logit
    dspine.forward.setProfile(std.c.getenv("TF_DSPINE_PROFILE") != null);
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--help")) return std.debug.print("{s}\n", .{help});
        if (std.mem.eql(u8, arg, "--control")) {
            control = true;
            continue;
        }
        if (i + 1 >= args.len) return error.MissingValue;
        i += 1;
        if (std.mem.eql(u8, arg, "--model")) model_dir = args[i] else if (std.mem.eql(u8, arg, "--drafter")) drafter_dir = args[i] else if (std.mem.eql(u8, arg, "--ref")) ref_dir = args[i] else if (std.mem.eql(u8, arg, "--mode")) mode = std.meta.stringToEnum(dspine.runtime.Mode, args[i]) orelse return error.BadMode else if (std.mem.eql(u8, arg, "--fit")) fit = try std.fmt.parseInt(u64, args[i], 10) else return error.UnknownFlag;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const target = try q.model.Model.load(init.gpa, io, model_dir orelse return error.MissingModel, 128, null);
    defer target.deinit();
    var t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    const m = try dspine.runtime.Model.load(init.gpa, io, target, drafter_dir orelse return error.MissingDrafter, mode, model_dir);
    defer m.deinit();
    if (fit) |rows| try m.fit(rows); // ring and K/V sized for streams that reach `rows`, as a served context sizes them
    std.debug.print("dspine: loaded ({s}, tensor units {}, head copy {}) in {d:.1} s\n", .{ @tagName(mode), target.device.tensorUnits(), m.head_rows != null, @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e9 });
    const dir = ref_dir orelse return error.MissingRef;
    const index_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "index.json" }), a, .limited(1 << 20));
    const index = try std.json.parseFromSlice(std.json.Value, a, index_text, .{});
    const width = dspine.config.tapWidth(m.shape());
    const taps = try target.device.buffer(128 * width * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer taps.deinit();
    var rounds: usize = 0;
    var chains: usize = 0;
    var cand_rows: usize = 0;
    var cand_same: usize = 0;
    var draft_same: usize = 0;
    // a second stream's context: the previous prompt, proposed beside this one in one batched pass
    const other = try m.newContext(init.gpa, true);
    defer other.deinit(init.gpa);
    var previous: ?struct { anchor: u32, p: dspine.forward.Proposal } = null;
    var batched: usize = 0;
    var batched_same: usize = 0;
    var batched_drift: f32 = 0;
    var batched_ms: f64 = 0;
    var solo_ms: f64 = 0;
    for (index.value.array.items) |item| {
        const tag = item.object.get("tag").?.string;
        const f = try File.open(a, io, try std.fmt.allocPrint(a, "{s}/{s}.safetensors", .{ dir, tag }));
        const rows_bytes = try f.get("taps");
        const s_rows = rows_bytes.len / (width * 2);
        m.reset();
        var first: usize = 0;
        while (first < s_rows) : (first += 128) {
            const n = @min(s_rows - first, 128);
            @memcpy(taps.contents()[0 .. n * width * 2], rows_bytes[first * width * 2 ..][0 .. n * width * 2]);
            try m.absorb(.{ .buf = taps }, @intCast(n));
        }
        const anchor = (try f.words("anchor"))[0];
        t0 = std.Io.Clock.awake.now(io).toNanoseconds();
        var solo: [1]dspine.forward.Proposal = undefined;
        try dspine.forward.propose(m, &.{m.state()}, &.{anchor}, &solo);
        var p = solo[0];
        const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e6;
        if (previous) |prev| {
            var two: [2]dspine.forward.Proposal = undefined;
            const b0 = std.Io.Clock.awake.now(io).toNanoseconds();
            try dspine.forward.propose(m, &.{ m.state(), try m.stateOf(other) }, &.{ anchor, prev.anchor }, &two);
            batched_ms += @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - b0)) / 1e6;
            solo_ms += ms;
            for ([_]dspine.forward.Proposal{ p, prev.p }, two) |want, got| {
                batched += 1;
                batched_same += @intFromBool(std.mem.eql(u32, &want.drafts, &got.drafts) and std.mem.eql(u8, std.mem.asBytes(&want.candidates), std.mem.asBytes(&got.candidates)));
                const x: *const [15 * 16 * 16]f32 = @ptrCast(&want.table);
                const y: *const [15 * 16 * 16]f32 = @ptrCast(&got.table);
                for (x, y) |u, w| batched_drift = @max(batched_drift, @abs(u - w));
            }
        }
        // the other stream takes this prompt's taps for the next one's batch
        try other.reset();
        first = 0;
        while (first < s_rows) : (first += 128) {
            const n = @min(s_rows - first, 128);
            @memcpy(taps.contents()[0 .. n * width * 2], rows_bytes[first * width * 2 ..][0 .. n * width * 2]);
            try other.ring().absorb(taps, 0, @intCast(n));
        }
        previous = .{ .anchor = anchor, .p = p };
        if (control) p.drafts = p.tops;
        const want_drafts = try f.words("drafts");
        const want_cand = try f.words("candidates");
        var same_rows: usize = 0;
        for (0..15) |r| {
            var row_same = true;
            for (0..16) |j| row_same = row_same and p.candidates[r][j] == want_cand[r * 16 + j];
            same_rows += @intFromBool(row_same);
        }
        var prefix: usize = 0;
        while (prefix < 15 and p.drafts[prefix] == want_drafts[prefix]) prefix += 1;
        rounds += 1;
        chains += @intFromBool(prefix == 15);
        cand_rows += 15;
        cand_same += same_rows;
        draft_same += prefix;
        std.debug.print("{s} ctx {d} anchor {d}: drafts {d}/15 same prefix, candidate rows {d}/15 same, {d:.1} ms", .{ tag, s_rows, anchor, prefix, same_rows, ms });
        if (prefix < 15) {
            // the Metal table's margin at the first divergence: its pick against the reference's pick on that row
            const r = prefix;
            var sender: usize = 0;
            if (r > 0) for (0..16) |j| {
                if (p.candidates[r - 1][j] == p.drafts[r - 1]) sender = j;
            };
            var mine: f32 = 0;
            var theirs: f32 = std.math.nan(f32);
            for (0..16) |j| {
                if (p.candidates[r][j] == p.drafts[r]) mine = p.table[r][sender][j];
                if (p.candidates[r][j] == want_drafts[r]) theirs = p.table[r][sender][j];
            }
            std.debug.print("; first divergence row {d}: metal {d} vs ref {d}, metal score margin {d:.4}", .{ r, p.drafts[r], want_drafts[r], mine - theirs });
        }
        std.debug.print("\n", .{});
    }
    std.debug.print("dspine batch of 2: {d}/{d} proposals identical to their own pass (drafts and candidates), table drift {d:.6}; {d:.1} ms a pass of two vs {d:.1} ms alone\n", .{ batched_same, batched, batched_drift, batched_ms / @as(f64, @floatFromInt(@max(batched / 2, 1))), solo_ms / @as(f64, @floatFromInt(@max(batched / 2, 1))) });
    std.debug.print("dspine gate ({s}{s}): {d}/{d} chains identical, {d}/{d} draft positions in the same prefix, {d}/{d} candidate rows identical\n", .{ @tagName(mode), if (control) ", control" else "", chains, rounds, draft_same, rounds * 15, cand_same, cand_rows });
}
