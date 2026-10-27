//! Derives the control vectors from the model itself: the prompt sets through the native forward, then the GLP file.
const std = @import("std");
const q = @import("qwen27");
const mtl = @import("metal");
const core = @import("core");
const derive = @import("steer_derive");
const glp = @import("glp");
const prompts = @import("steer_prompts");

const help =
    \\tf-qwen27-steer-derive --model DIR --out FILE [--alpha 1.0] [--repo ID] [--revision REV] [--prompts N]
    \\                            [--capture mean|last]
    \\  Derives one control direction per layer from the model itself, then writes the GLP file.
    \\  The refusal set and the form-matched harmless set each run through the native forward.
    \\  --model DIR is the MLX checkpoint directory; --out FILE is the GLP file to write.
    \\  --repo ID and --revision REV are the checkpoint's identity in the file's metadata.
    \\  --prompts N uses only the first N pairs of each set.
    \\  --capture mean reads every row of a prompt and captures its mean; --capture last
    \\  takes the last row alone, where the model decides what to say next.
    \\  Layer 0 is not steerable, so the file holds the other layers.
;

const Hooks = struct {
    runner: *q.decode_round.Runner,
    capture: *derive.Capture,

    fn gdn(ptr: *anyopaque, e: mtl.ComputeEncoder, block: usize, weights: q.gpu_weights.Gdn, frame: *q.gpu_frame.Frame, rows: u32) !void {
        const h: *Hooks = @ptrCast(@alignCast(ptr));
        const r = h.runner;
        var hook = q.gdn_hook.Hook{ .pool = &r.gdn, .quant = &r.model.kernels, .glue = r.model.glue, .plan = r.current_gdn.?, .linear_index = r.linear_index, .accepted_chain = true };
        try q.gdn_hook.Hook.forward(&hook, e, block, weights, frame, rows);
    }

    fn attention(ptr: *anyopaque, e: mtl.ComputeEncoder, block: usize, weights: q.gpu_weights.Attention, f: *q.gpu_frame.Frame, rows: u32) !void {
        const h: *Hooks = @ptrCast(@alignCast(ptr));
        const r = h.runner;
        const plan = r.current_attention.?.*;
        const layer = r.attention_index[block];
        const cache = r.caches[layer];
        try r.bindings[layer].upload(plan, &.{cache});
        try r.model.kernels.quant(e, weights.q, f.get(.input), q.forward.sums(f, .sums, rows), f.get(.dims), f.get(.q_gate), rows);
        try r.model.kernels.quant(e, weights.kv, f.get(.input), q.forward.sums(f, .sums, rows), f.get(.dims), f.get(.mixed), rows);
        e.barrier();
        const kv: u32 = @intCast(r.model.config.kvDim());
        try r.model.glue.unstack(e, f.get(.mixed), f.get(.key), .{ .rows = rows, .width = kv, .stride = 2 * kv, .offset = 0 });
        try r.model.glue.unstack(e, f.get(.mixed), f.get(.value), .{ .rows = rows, .width = kv, .stride = 2 * kv, .offset = kv });
        e.barrier();
        try r.attention.preprocess(e, .{ .qg = f.get(.q_gate), .key = f.get(.key), .q_gain = q.gpu_weights.ref(weights.q_norm), .k_gain = q.gpu_weights.ref(weights.k_norm), .norm_q = f.get(.norm_query), .norm_k = f.get(.norm_key), .query = f.get(.query), .rotated_key = f.get(.rotated_key), .positions = f.get(.positions), .rows = rows, .eps = r.model.config.eps, .theta = r.model.config.rope_theta, .h_end = r.current_placed.h_end, .w_end = r.current_placed.w_end, .plane = f.capacity, .geometry = r.geometry });
        try r.attention.encode(e, &r.scratch, &r.bindings[layer], plan, f.get(.query), f.get(.rotated_key), f.get(.value), f.get(.mixer_out), &.{cache}, f.get(.q_gate), r.model.kernels.prompt);
    }

    /// The capture hook, in the tap's slot: the post-block residual, exactly where a projection would act.
    fn tap(ptr: *anyopaque, e: mtl.ComputeEncoder, layer: usize, hidden: q.projection.Ref, _: u32) !void {
        const h: *Hooks = @ptrCast(@alignCast(ptr));
        try h.capture.record(e, layer, hidden);
    }
};

/// One prompt through the native forward, its layers' residual rows summed on the way.
fn drive(runner: *q.decode_round.Runner, ids: []const u32, capture: *derive.Capture) !void {
    const a = runner.allocator;
    var round = try q.round_plan.Round.init(a, &.{.{ .slot = 0, .start = runner.offsets[0], .capacity = runner.capacity, .ids = ids }}, @intCast(runner.model.config.conv_kernel), runner.slots, @intCast(runner.model.config.vocab));
    defer round.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    // The accepted chain an ordinary prompt takes: every row's parent is the row before it.
    const path = try a.alloc(u32, ids.len);
    defer a.free(path);
    for (path, 0..) |*v, i| v.* = @intCast(i);
    const params = try runner.gdn.budget.shape.params(ids.len, runner.slots, .{ .conv = .bf16, .a_log = .bf16, .dt = .bf16, .norm = .bf16 });
    var recurrent = try round.recurrent(params, &.{path});
    defer recurrent.deinit();
    var windows: [q.attention.planning.max_streams]q.attention.planning.Window = undefined;
    for (round.windows, 0..) |window, i| windows[i] = .{ .parents = window.parents, .start = window.start, .capacity = runner.capacity };
    var attention = try q.attention.Plan.init(a, windows[0..round.windows.len], runner.geometry);
    defer attention.deinit();
    try runner.gdn.prepare(runner.model.device, recurrent, true);
    try runner.model.frame.setRows(round.ids, round.feed_positions);
    try runner.scratch.upload(attention);
    runner.current_round = &round;
    runner.current_gdn = &recurrent;
    runner.current_attention = &attention;
    defer {
        runner.current_round = null;
        runner.current_gdn = null;
        runner.current_attention = null;
    }
    capture.reset();
    var hooks = Hooks{ .runner = runner, .capture = capture };
    var stages = q.forward.Stages{ .model = runner.model, .rows = @intCast(ids.len), .hooks = .{ .ptr = &hooks, .gdn = Hooks.gdn, .attention = Hooks.attention, .tap = Hooks.tap }, .head = .none };
    _ = try core.segments.runCommitting(runner.model.device, &.{runner.model.queue}, runner.model.config.layers, .concurrent, &stages, 4);
}

/// One prompt set's running mean: every prompt's own mean, counted once.
fn captureSet(runner: *q.decode_round.Runner, capture: *derive.Capture, tok: *core.tokenizer.Tokenizer, texts: []const []const u8, mean: *derive.Mean, limit: usize) !void {
    for (texts, 0..) |text, i| {
        if (i >= limit) break;
        const ids = try tok.encode(runner.allocator, text);
        defer runner.allocator.free(ids);
        if (ids.len == 0 or ids.len > runner.model.frame.capacity) return error.BadPrompt;
        try runner.reset(0);
        try drive(runner, ids, capture);
        try mean.addPrompt(capture.sums(), capture.divisor());
    }
}

/// The checkpoint's revision, when its directory is a cache snapshot named for the commit.
fn snapshotRevision(dir: []const u8) ?[]const u8 {
    const name = std.fs.path.basename(dir);
    const parent = std.fs.path.basename(std.fs.path.dirname(dir) orelse "");
    if (!std.mem.eql(u8, parent, "snapshots")) return null;
    if (name.len != 40 and name.len != 64) return null;
    for (name) |ch| if (!std.ascii.isDigit(ch) and !(ch >= 'a' and ch <= 'f')) return null;
    return name;
}

/// The UTC time of the derivation, as the file's derived_at.
fn stamp(io: std.Io, buf: *[32]u8) []const u8 {
    const ns = std.Io.Clock.real.now(io).nanoseconds;
    const secs: u64 = @intCast(@max(0, @divFloor(ns, std.time.ns_per_s)));
    const es: std.time.epoch.EpochSeconds = .{ .secs = secs };
    const yd = es.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    const ds = es.getDaySeconds();
    return std.fmt.bufPrint(buf, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}Z", .{
        yd.year, md.month.numeric(), md.day_index + 1, ds.getHoursIntoDay(), ds.getMinutesIntoHour(), ds.getSecondsIntoMinute(),
    }) catch "unknown";
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 1 or (args.len == 2 and std.mem.eql(u8, args[1], "--help"))) {
        try std.Io.File.stdout().writeStreamingAll(io, help ++ "\n");
        return;
    }
    var model_dir: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var repo: []const u8 = "";
    var revision: []const u8 = "";
    var alpha: f32 = 1.0;
    var limit: usize = std.math.maxInt(usize);
    var site: derive.Site = .mean_rows;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (i + 1 >= args.len) return error.MissingValue;
        i += 1;
        const value = args[i];
        if (std.mem.eql(u8, arg, "--model")) model_dir = value else if (std.mem.eql(u8, arg, "--out")) out_path = value else if (std.mem.eql(u8, arg, "--repo")) repo = value else if (std.mem.eql(u8, arg, "--revision")) revision = value else if (std.mem.eql(u8, arg, "--alpha")) alpha = try std.fmt.parseFloat(f32, value) else if (std.mem.eql(u8, arg, "--prompts")) limit = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, arg, "--capture")) site = if (std.mem.eql(u8, value, "last")) .last_row else if (std.mem.eql(u8, value, "mean")) .mean_rows else return error.UnknownOption else return error.UnknownOption;
    }
    if (model_dir == null or out_path == null) return error.BadOptions;
    if (revision.len == 0) revision = snapshotRevision(model_dir.?) orelse "";

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try q.model.Model.load(init.gpa, io, model_dir.?, 128, null);
    defer m.deinit();
    const tokenizer_path = try std.fs.path.join(a, &.{ model_dir.?, "tokenizer.json" });
    var tok = try core.tokenizer.loadTokenizer(io, init.gpa, tokenizer_path);
    defer tok.deinit();
    var runner = try q.decode_round.Runner.init(init.gpa, m, 1, 512);
    defer runner.deinit();
    // The blocks are one to `layers`, and the layer after the last one is the head's input.
    const sites = m.config.layers + 1;
    var capture = try derive.Capture.init(m.device, sites, m.config.hidden, m.frame.get(.dims).buffer, site);
    defer capture.deinit();

    var harmful = try derive.Mean.init(init.gpa, sites, m.config.hidden);
    defer harmful.deinit(init.gpa);
    var harmless = try derive.Mean.init(init.gpa, sites, m.config.hidden);
    defer harmless.deinit(init.gpa);
    runner.model.kernels.prompt = true;
    defer runner.model.kernels.prompt = false;
    const requests = try prompts.requests(a);
    const others = try prompts.harmless(a);
    try captureSet(&runner, &capture, &tok, requests, &harmful, limit);
    try captureSet(&runner, &capture, &tok, others, &harmless, limit);
    if (harmful.prompts == 0 or harmless.prompts == 0) return error.NoPrompts;

    var derived = try derive.derive(init.gpa, harmful, harmless, sites, m.config.hidden);
    defer derived.deinit(init.gpa);
    var layers: std.ArrayList(glp.Layer) = .empty;
    defer layers.deinit(init.gpa);
    for (derived.layers) |l| {
        // Layer 0 is not steerable, so the file holds the other layers only.
        if (l.id == 0) continue;
        try layers.append(init.gpa, .{ .id = l.id, .dir = l.dir });
    }
    if (layers.items.len == 0) return error.NoDirections;

    var time_buf: [32]u8 = undefined;
    const bytes = try glp.write(init.gpa, .{
        .mode = .project,
        .alpha = alpha,
        .hook_point = glp.residual_site,
        .derived_at = stamp(io, &time_buf),
        .base_model = repo,
        .base_revision = revision,
        .method = site.methodName(),
        .contrast = derive.contrast,
        .layers = layers.items,
    });
    defer init.gpa.free(bytes);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path.?, .data = bytes });

    const receipt = try std.json.Stringify.valueAlloc(a, .{
        .kind = "qwen27_steer_derive",
        .model = model_dir.?,
        .out = out_path.?,
        .repo = repo,
        .revision = revision,
        .alpha = alpha,
        .method = site.methodName(),
        .contrast = derive.contrast,
        .hook_point = glp.residual_site,
        .width = m.config.hidden,
        .layers = sites - 1,
        .prompts = harmful.prompts,
        .capture = @tagName(site),
        .directions = layers.items.len,
        .dropped = derived.dropped,
        .bytes = bytes.len,
    }, .{});
    try std.Io.File.stdout().writeStreamingAll(io, receipt);
    try std.Io.File.stdout().writeStreamingAll(io, "\n");
}
