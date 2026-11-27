//! Attributed shared/replayed launches resolve to real CUDA definitions and reviewed helper bodies.
const std = @import("std");
const kernel = @import("pdl_kernel_scan");

const Rule = struct {
    source: []const u8,
    names: []const []const u8,
    wait: []const u8,
    trigger: []const u8 = "",
    delegates: []const []const u8 = &.{},
    helper: []const u8 = "",
    generator: []const u8 = "",
    from: []const u8 = "",
    to: []const u8 = "",
    weights: []const u8 = "",
    inputs: []const u8 = "",
};

fn laneRule() Rule {
    return .{ .source = "zig/kernels/cuda/lane_gemv.cu", .names = &.{ "gemv_kernel", "tf_lane_gemv_many", "tf_lane_gemv_split" }, .wait = "grid_wait();", .delegates = &.{ "gemv_items<", "gemv_split<", "many_body" }, .from = "for (int s = 0; s < STAGES - 1; ++s)", .to = "for (int it", .weights = "load_w", .inputs = "load_x" };
}
fn affineRule() Rule {
    return .{ .source = "zig/kernels/cuda/affine8.cu", .names = &.{ "tf_affine8_gemv", "tf_affine##B##_gemv" }, .wait = "grid_wait();", .delegates = &.{"tf_affine8::gemv<"}, .from = "for (int s = 0; s < STAGES - 1; ++s)", .to = "for (int it", .weights = "load_w", .inputs = "load_x" };
}
fn qmmGroupRule() Rule {
    return .{ .source = "zig/kernels/cuda/qmm_group.cu", .names = &.{"group_kernel"}, .wait = "grid_wait();", .from = "for (int s = 0; s < STAGES - 1; ++s)", .to = "for (int it", .weights = "load_w", .inputs = "load_x" };
}
fn qmmfRule() Rule {
    return .{ .source = "zig/kernels/cuda/qmmf.cu", .names = &.{"qmmf_kernel"}, .wait = "grid_wait();", .delegates = &.{"TF_QMMF_WFIRST"}, .helper = "zig/kernels/cuda/qmmf_body.cuh", .from = "#ifdef TF_QMMF_WFIRST", .to = "#else", .weights = "load(s, g0 + s, 2)", .inputs = "load(s, g0 + s, 1)" };
}
fn reduceRule() Rule {
    return .{ .source = "zig/kernels/cuda/qmmf.cu", .names = &.{"reduce_kernel"}, .wait = "qmm_frag::grid_wait();", .trigger = "qmm_frag::grid_launch();" };
}
fn qmmfGroupRule() Rule {
    return .{ .source = "zig/kernels/cuda/qmmf.cu", .names = &.{"tf_qmmf_group_"}, .wait = "grid_wait();", .delegates = &.{"qmmf_group_body"}, .generator = "TF_QMMF_GROUP_ENTRY(", .helper = "zig/kernels/cuda/qmmf_body.cuh", .from = "#ifdef TF_QMMF_WFIRST", .to = "#else", .weights = "load(s, g0 + s, 2)", .inputs = "load(s, g0 + s, 1)" };
}
fn exlRotRule() Rule {
    return .{ .source = "zig/kernels/cuda/exl3_linear.cu", .names = &.{ "tf_exl3_rot_in", "tf_exl3_rot_group" }, .wait = "pdl_wait();", .trigger = "pdl_trigger();", .delegates = &.{"rot_in_body"}, .from = "void rot_in_body", .inputs = "load4(" };
}
fn exlLinearRule() Rule {
    return .{ .source = "zig/kernels/cuda/exl3_linear.cu", .names = &.{ "tf_exl3_linear_", "tf_exl3_group_" }, .wait = "pdl_wait();", .delegates = &.{"linear_body"}, .from = "void linear_body", .inputs = "load_step" };
}
fn x3segRotRule() Rule {
    return .{ .source = "zig/kernels/cuda/exl3_block/x3seg.cu", .names = &.{"seg_rot_in_kernel"}, .wait = "pdl_wait();" };
}
fn x3segLinearRule() Rule {
    return .{ .source = "zig/kernels/cuda/exl3_block/x3seg.cu", .names = &.{"seg_linear_kernel"}, .wait = "pdl_wait();", .from = "for (int m0", .weights = "load_step", .inputs = "__ldg" };
}
fn dense3Rule() Rule {
    return .{ .source = "zig/kernels/cuda/exl3_block/dense3.cu", .names = &.{"lanes_linear_kernel"}, .wait = "pdl_wait();", .from = "for (int m0", .weights = "load_group", .inputs = "load_a" };
}
fn lin2Rule() Rule {
    return .{ .source = "zig/kernels/cuda/exl3_block/lin2.cu", .names = &.{ "lanes2_kernel", "lanesc_kernel" }, .wait = "pdl_wait();", .weights = "load_group", .inputs = "load_a" };
}
fn x3ldRule() Rule {
    return .{ .source = "zig/kernels/cuda/exl3_block/x3ld.cu", .names = &.{"ld_kernel"}, .wait = "pdl_wait();" };
}

const Binding = struct {
    path: []const u8,
    handle: []const u8,
    rules: []const Rule,
    predicate: []const u8 = "",
    prefix: bool = false,
};

const bindings = [_]Binding{
    .{ .path = "zig/src/cuda/affine4_kernels.zig", .handle = "f", .rules = &.{qmmGroupRule()} },
    .{ .path = "zig/src/cuda/affine4_kernels.zig", .handle = "a.many_fn", .rules = &.{laneRule()} },
    .{ .path = "zig/src/cuda/affine4_kernels.zig", .handle = "a.gemv_fn", .rules = &.{laneRule()} },
    .{ .path = "zig/src/cuda/affine4_kernels.zig", .handle = "if (tuned)", .rules = &.{qmmGroupRule()}, .prefix = true },
    .{ .path = "zig/src/cuda/qlinear.zig", .handle = "a.gemv_fns", .rules = &.{affineRule()}, .prefix = true },
    .{ .path = "zig/src/cuda/qmmf.zig", .handle = "f", .rules = &.{qmmfRule()}, .predicate = "!prompt" },
    .{ .path = "zig/src/cuda/qmmf.zig", .handle = "l.reduce", .rules = &.{reduceRule()}, .prefix = true },
    .{ .path = "zig/src/cuda/qmmf_group.zig", .handle = "g.fns", .rules = &.{qmmfGroupRule()}, .prefix = true },
    .{ .path = "zig/src/cuda/exl3.zig", .handle = "l.rot_in", .rules = &.{exlRotRule()} },
    .{ .path = "zig/src/cuda/exl3.zig", .handle = "l.rot_group", .rules = &.{exlRotRule()} },
    .{ .path = "zig/src/cuda/exl3.zig", .handle = "l.fns", .rules = &.{exlLinearRule()}, .prefix = true },
    .{ .path = "zig/src/cuda/exl3.zig", .handle = "l.group", .rules = &.{exlLinearRule()}, .prefix = true },
    .{ .path = "zig/src/cuda/exl3_block_dense.zig", .handle = "f", .rules = &.{ x3segRotRule(), x3segLinearRule(), dense3Rule(), lin2Rule() } },
    .{ .path = "zig/src/cuda/exl3_block_dense.zig", .handle = "l2.lanesc", .rules = &.{lin2Rule()}, .prefix = true },
    .{ .path = "zig/src/cuda/exl3_block_exl3.zig", .handle = "f", .rules = &.{x3ldRule()} },
};

fn selected(name: []const u8, patterns: []const []const u8) bool {
    for (patterns) |p| if (std.mem.startsWith(u8, name, p)) return true;
    return false;
}

fn ordered(src: []const u8, rule: Rule) bool {
    const start = if (rule.from.len == 0) 0 else std.mem.indexOf(u8, src, rule.from) orelse return false;
    const end = if (rule.to.len == 0) src.len else std.mem.indexOfPos(u8, src, start, rule.to) orelse return false;
    const body = src[start..end];
    const wait = std.mem.indexOf(u8, body, rule.wait) orelse return false;
    if (rule.trigger.len != 0) {
        const trigger = std.mem.indexOf(u8, body, rule.trigger) orelse return false;
        if (trigger > wait) return false;
    }
    if (rule.weights.len != 0) {
        const weights = std.mem.indexOf(u8, body, rule.weights) orelse return false;
        if (weights > wait) return false;
    }
    return std.mem.indexOf(u8, body[wait..], rule.inputs) != null and std.mem.indexOf(u8, body[0..wait], rule.inputs) == null;
}

fn entry(opening: []const u8, rule: Rule) bool {
    if (std.mem.startsWith(u8, opening, rule.wait)) return true;
    return rule.trigger.len != 0 and std.mem.startsWith(u8, opening, rule.trigger) and
        std.mem.startsWith(u8, std.mem.trimStart(u8, opening[rule.trigger.len..], " \t\r\n"), rule.wait);
}

fn delegated(opening: []const u8, delegates: []const []const u8) bool {
    for (delegates) |d| if (std.mem.indexOf(u8, opening, d) != null) return true;
    return false;
}

fn expandGenerated(a: std.mem.Allocator, src: []const u8, generator: []const u8, found: *std.ArrayList(kernel.Definition)) !void {
    var opening: ?[]const u8 = null;
    for (found.items) |d| {
        if (std.mem.eql(u8, d.name, "NAME")) opening = d.opening;
    }
    const body = opening orelse return error.MissingMacroDefinition;
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, src, at, generator)) |g| {
        const start = g + generator.len;
        const end = std.mem.indexOfScalarPos(u8, src, start, ',') orelse return error.Unparsed;
        try found.append(a, .{ .name = std.mem.trim(u8, src[start..end], " \t\r\n"), .opening = body });
        at = end + 1;
    }
}

fn checkText(a: std.mem.Allocator, rule: Rule, src: []const u8, helper_src: []const u8, diagnose: bool, problems: *usize) !void {
    var found: std.ArrayList(kernel.Definition) = .empty;
    try kernel.definitions(a, src, &found);
    if (rule.generator.len != 0) try expandGenerated(a, src, rule.generator, &found);
    const helper = if (rule.from.len != 0) try kernel.normalize(a, helper_src) else helper_src;
    var seen: usize = 0;
    for (found.items) |d| {
        if (!selected(d.name, rule.names)) continue;
        seen += 1;
        const body = if (rule.delegates.len == 0) d.opening else helper;
        const ok = (rule.delegates.len == 0 or delegated(d.opening, rule.delegates)) and
            (if (rule.weights.len != 0 or rule.from.len != 0) ordered(body, rule) else entry(d.opening, rule));
        if (!ok) {
            if (diagnose) std.debug.print("{s}: {s} does not satisfy its scanned PDL wait rule; opening={s}\n", .{ rule.source, d.name, d.opening });
            problems.* += 1;
        }
    }
    if (seen == 0) {
        if (diagnose) std.debug.print("{s}: no real definition matches an attributed launch\n", .{rule.source});
        problems.* += 1;
    }
}

fn checkRule(a: std.mem.Allocator, rule: Rule, problems: *usize) !void {
    const src = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, rule.source, a, .limited(16 << 20));
    const helper_src = if (rule.helper.len == 0) src else try std.Io.Dir.cwd().readFileAlloc(std.testing.io, rule.helper, a, .limited(16 << 20));
    try checkText(a, rule, src, helper_src, true, problems);
}

const Observed = struct { handle: []const u8, predicate: []const u8 };

fn expressionEnd(src: []const u8, start: usize) struct { end: usize, terminator: u8 } {
    var depth: usize = 0;
    for (src[start..], start..) |c, i| switch (c) {
        '(', '[' => depth += 1,
        ')', ']' => {
            if (depth > 0) depth -= 1;
        },
        ',', ';', '}' => {
            if (depth == 0) return .{ .end = i, .terminator = c };
        },
        else => {},
    };
    return .{ .end = src.len, .terminator = 0 };
}

fn attributed(a: std.mem.Allocator, src: []const u8, out: *std.ArrayList(Observed)) !void {
    const normalized = try kernel.normalize(a, src);
    const end = std.mem.indexOf(u8, normalized, "test ") orelse normalized.len;
    const text = normalized[0..end];
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, text, at, ".pdl = ")) |p| {
        const value_start = p + ".pdl = ".len;
        const boundary = expressionEnd(text, value_start);
        const predicate = std.mem.trim(u8, text[value_start..boundary.end], " \t\r\n");
        at = boundary.end + 1;
        if (boundary.terminator == ';' or std.mem.eql(u8, predicate, "false")) continue;
        const semicolon = std.mem.lastIndexOfScalar(u8, text[0..p], ';') orelse 0;
        const before = std.mem.lastIndexOf(u8, text[semicolon..p], "launch(");
        const launch = if (before) |b| semicolon + b else std.mem.indexOfPos(u8, text, boundary.end, "launch(") orelse return error.UnparsedPdlLaunch;
        const handle_start = launch + "launch(".len;
        const boundary_handle = expressionEnd(text, handle_start);
        if (boundary_handle.terminator != ',') return error.UnparsedPdlLaunch;
        const handle = std.mem.trim(u8, text[handle_start..boundary_handle.end], " \t\r\n");
        try out.append(a, .{ .handle = handle, .predicate = predicate });
    }
}

fn bindingFor(path: []const u8, observed: Observed) ?usize {
    for (bindings, 0..) |b, i| {
        const handle_matches = std.mem.eql(u8, observed.handle, b.handle) or
            (b.prefix and std.mem.startsWith(u8, observed.handle, b.handle));
        if (!std.mem.eql(u8, path, b.path) or !handle_matches) continue;
        if (b.predicate.len != 0 and std.mem.indexOf(u8, observed.predicate, b.predicate) == null) return null;
        return i;
    }
    return null;
}

pub fn scan(a: std.mem.Allocator, problems: *usize) !void {
    const paths = [_][]const u8{
        "zig/src/cuda/affine4_kernels.zig",
        "zig/src/cuda/qlinear.zig",
        "zig/src/cuda/qmmf.zig",
        "zig/src/cuda/qmmf_group.zig",
        "zig/src/cuda/exl3.zig",
        "zig/src/cuda/exl3_block_dense.zig",
        "zig/src/cuda/exl3_block_exl3.zig",
    };
    var used: [bindings.len]bool = @splat(false);
    for (paths) |path| {
        const src = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(16 << 20));
        var found: std.ArrayList(Observed) = .empty;
        try attributed(a, src, &found);
        for (found.items) |site| {
            const index = bindingFor(path, site) orelse {
                std.debug.print("{s}: unknown attributed launch handle {s} ({s})\n", .{ path, site.handle, site.predicate });
                problems.* += 1;
                continue;
            };
            used[index] = true;
            for (bindings[index].rules) |rule| try checkRule(a, rule, problems);
        }
    }
    for (used, 0..) |ok, i| if (!ok) {
        std.debug.print("{s}: attributed launch {s} was not discovered\n", .{ bindings[i].path, bindings[i].handle });
        problems.* += 1;
    };
}

test "weight-first ordering rejects early inputs and missing weights" {
    const r = Rule{ .source = "", .names = &.{}, .wait = "grid_wait();", .weights = "load_w", .inputs = "load_x" };
    try std.testing.expect(ordered("load_w(); grid_wait(); load_x();", r));
    try std.testing.expect(!ordered("load_x(); grid_wait(); load_w(); load_x();", r));
    try std.testing.expect(!ordered("grid_wait(); load_w(); load_x();", r));
}

test "an attributed unknown handle fails closed" {
    try std.testing.expect(bindingFor("zig/src/cuda/exl3_block_dense.zig", .{ .handle = "fake", .predicate = "pdl" }) == null);
    try std.testing.expect(bindingFor("zig/src/cuda/qlinear.zig", .{ .handle = "a.new_unknown", .predicate = "a.pdl" }) == null);
}

test "a dependent trigger immediately followed by a wait is a valid reduce entry" {
    const opening = "qmm_frag::grid_launch();\n    qmm_frag::grid_wait();\n    const int i = 0;";
    try std.testing.expect(entry(opening, reduceRule()));
}

fn withoutFirst(a: std.mem.Allocator, src: []const u8, needle: []const u8) ![]u8 {
    const at = std.mem.indexOf(u8, src, needle) orelse return error.MissingPlant;
    return std.fmt.allocPrint(a, "{s}{s}", .{ src[0..at], src[at + needle.len ..] });
}

test "a planted missing Lane GEMV input wait fails the shared Qwen source rule" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const src = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/kernels/cuda/lane_gemv.cu", a, .limited(2 << 20));
    const planted = try withoutFirst(a, src, "grid_wait();");
    var problems: usize = 0;
    try checkText(a, laneRule(), planted, planted, false, &problems);
    try std.testing.expect(problems > 0);
}

test "a planted missing x3seg entry wait fails the recording-driven DeepSeek source rule" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const src = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/kernels/cuda/exl3_block/x3seg.cu", a, .limited(2 << 20));
    const planted = try withoutFirst(a, src, "pdl_wait();");
    var problems: usize = 0;
    try checkText(a, x3segRotRule(), planted, planted, false, &problems);
    try std.testing.expect(problems > 0);
}
