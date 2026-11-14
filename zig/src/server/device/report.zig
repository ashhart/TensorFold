//! The words: a two-line summary for bare `tensorfold` and `--help`, and `tensorfold device`'s short report.
const std = @import("std");
const machine = @import("machine.zig");
const support = @import("support.zig");
const profile = @import("profile.zig");
const Writer = std.Io.Writer;
const Machine = machine.Machine;
const Support = support.Support;

const gib: f64 = 1 << 30;

/// "This machine: ...", then what runs on it or why nothing does.
pub fn summary(w: *Writer, m: Machine, s: Support) !void {
    try w.writeAll("This machine: ");
    try describe(w, m, s);
    try w.writeAll(".\n");
    if (s.backend) |b| {
        try w.print("Supported: {s}{s}. Formats: ", .{ backendName(b), if (s.certain) "" else " (likely)" });
        try formatNames(w, s.formats);
        try w.writeAll(". Details: tensorfold device\n");
    } else {
        try w.print("Not supported: {s}. Details: tensorfold device\n", .{s.reason});
    }
}

/// The GPU a person names the machine by, its memory and the OS: "RTX 5090, 32 GiB (sm_120), Windows 11 (build 26100)".
pub fn describe(w: *Writer, m: Machine, s: Support) !void {
    var buf: [128]u8 = undefined;
    const os = machine.osName(&buf, m);
    if (machine.appleChip(m.cpu) != null) return w.print("{s}, {d:.0} GiB unified memory, {s}", .{ m.cpu, gibs(m.ram), os });
    const at = s.gpu orelse for (m.gpus, 0..) |g, i| {
        if (g.vendor != .other) break i;
    } else null;
    const g = m.gpus[at orelse return w.print("{s}, {d:.0} GiB RAM, no GPU listed, {s}", .{ if (m.cpu.len > 0) m.cpu else "unknown CPU", gibs(m.ram), os })];
    const name = machine.platform(m) orelse machine.shortName(g.name);
    if (g.unified) {
        try w.print("{s}", .{name});
        if (g.sm > 0) try w.print(", sm_{d}", .{g.sm});
        return w.print(", {d:.0} GiB unified memory, {s}", .{ gibs(m.ram), os });
    }
    try w.print("{s}", .{name});
    if (g.memory > 0) try w.print(" {d:.0} GiB", .{gibs(g.memory)});
    if (g.sm > 0) try w.print(" (sm_{d})", .{g.sm});
    if (g.gfx.len > 0) try w.print(" ({s})", .{g.gfx});
    try w.print(", {s}", .{os});
}

/// `tensorfold device`'s report: the machine, what runs here, the memory for weights and the profile it came from.
pub fn full(w: *Writer, doc: profile.Doc, s: Support, version: []const u8, fits: []const Fit) !void {
    const m = doc.machine;
    try w.writeAll("This machine: ");
    try describe(w, m, s);
    try w.writeAll("\n");
    try w.print("  CPU: {s}", .{if (m.cpu.len > 0) m.cpu else "unknown"});
    if (m.cores > 0) try w.print(", {d} cores", .{m.cores});
    if (m.performance_cores > 0) try w.print(" ({d} performance, {d} efficiency)", .{ m.performance_cores, m.efficiency_cores });
    try w.print("\n  RAM: {d:.0} GiB\n", .{gibs(m.ram)});
    for (m.gpus, 0..) |g, i| {
        try w.print("  GPU {d}: {s}", .{ i, g.name });
        if (g.unified) try w.writeAll(", unified memory") else if (g.memory > 0) try w.print(", {d:.1} GiB", .{gibs(g.memory)});
        if (g.sm > 0) try w.print(", sm_{d}", .{g.sm});
        if (g.gfx.len > 0) try w.print(", {s}", .{g.gfx});
        try w.print(" (from {s})\n", .{switch (g.source) {
            .os => "the OS",
            .table => "its name",
            .driver => "the driver",
        }});
    }
    if (m.gpus.len == 0) try w.writeAll("  GPU: none listed\n");
    if (s.backend) |b| {
        try w.print("Supported: {s}{s}, tensorfold-native {s}\n  Formats: ", .{ backendName(b), if (s.certain) "" else " (likely)", version });
        try formatNames(w, s.formats);
        try w.writeAll("\n  Format names: ");
        for (s.formats, 0..) |f, i| try w.print("{s}{s}", .{ if (i > 0) ", " else "", f });
        try w.writeAll("\n");
        for (s.unavailable) |u| try w.print("  Not here: {s}: {s}\n", .{ u.family, u.reason });
        for (s.notes) |n| try w.print("  Note: {s}\n", .{n});
        if (s.budget > 0) try w.print("  Memory for model weights: {d:.1} GiB ({s})\n", .{ gibs(s.budget), s.budget_rule }) else try w.print("  Memory for model weights: {s}\n", .{s.budget_rule});
    } else try w.print("Not supported by tensorfold-native {s}: {s}\n", .{ version, s.reason });
    try w.writeAll(if (fits.len > 0) "Models on disk:\n" else "Models on disk: none in the Hugging Face cache that this build serves\n");
    for (fits) |f| {
        try w.print("  {s}: {s} ({s}), {d:.1} GiB: ", .{ f.repo, f.title, f.model_type, gibs(f.bytes) });
        try w.writeAll(switch (f.verdict) {
            .fits => "fits\n",
            .too_large => "larger than the memory for weights\n",
            .not_here => "its family does not run on this machine\n",
            .not_served => "this build does not serve its family\n",
        });
    }
    try w.print("Profile: ~/.tensorfold/device.json, {s} scan at {s}; rescanned when the hardware or this binary changes, or with --rescan\n", .{ @tagName(doc.scan), doc.scanned_at });
}

/// One checkpoint on disk and whether it serves here.
pub const Fit = struct { repo: []const u8, title: []const u8, model_type: []const u8, bytes: u64, verdict: support.Verdict };

pub fn backendName(b: []const u8) []const u8 {
    const names = [_][2][]const u8{ .{ "metal", "Metal" }, .{ "cuda", "CUDA" }, .{ "hip", "HIP" }, .{ "xpu", "XPU" }, .{ "vulkan", "Vulkan" } };
    for (names) |n| if (std.mem.eql(u8, n[0], b)) return n[1];
    return b;
}

/// The formats as people name them: "MLX 2/3/4/5/6/8-bit, NVFP4, FP8, EXL3, BF16"; names it does not know as given.
pub fn formatNames(w: *Writer, formats: []const []const u8) !void {
    var bits: u16 = 0;
    var mlx = false;
    for (formats) |f| {
        if (std.mem.eql(u8, f, "mlx")) mlx = true;
        if (!std.mem.startsWith(u8, f, "mlx-q")) continue;
        const end = std.mem.indexOfScalarPos(u8, f, 5, 'g') orelse continue;
        const b = std.fmt.parseInt(u4, f[5..end], 10) catch continue;
        bits |= @as(u16, 1) << b;
    }
    var first = true;
    if (bits != 0) {
        try w.writeAll("MLX ");
        var sep = false;
        for (1..16) |b| if (bits & (@as(u16, 1) << @intCast(b)) != 0) {
            try w.print("{s}{d}", .{ if (sep) "/" else "", b });
            sep = true;
        };
        try w.writeAll("-bit");
        first = false;
    } else if (mlx) {
        try w.writeAll("MLX");
        first = false;
    }
    for (groups) |g| {
        const any = for (formats) |f| {
            if (g.matches(f)) break true;
        } else false;
        if (!any) continue;
        try w.print("{s}{s}", .{ if (first) "" else ", ", g.name });
        first = false;
    }
    for (formats) |f| {
        if (std.mem.startsWith(u8, f, "mlx")) continue;
        const known = for (groups) |g| {
            if (g.matches(f)) break true;
        } else false;
        if (known) continue;
        try w.print("{s}{s}", .{ if (first) "" else ", ", f });
        first = false;
    }
    if (first) try w.writeAll("none");
}

const Group = struct {
    name: []const u8,
    members: []const []const u8,
    fn matches(g: Group, f: []const u8) bool {
        for (g.members) |x| if (std.mem.eql(u8, x, f)) return true;
        return false;
    }
};
const groups = [_]Group{
    .{ .name = "NVFP4", .members = &.{ "nvfp4", "modelopt-nvfp4", "modelopt-mixed" } },
    .{ .name = "MXFP4", .members = &.{"mxfp4"} },
    .{ .name = "FP8", .members = &.{ "fp8", "fp8-block", "fp8-block128", "modelopt-fp8", "modelopt-mixed" } },
    .{ .name = "EXL3", .members = &.{"exl3"} },
    .{ .name = "EXLR8", .members = &.{"exlr8"} },
    .{ .name = "GPTQ int4", .members = &.{"gptq-int4-g128"} },
    .{ .name = "int4 (compressed-tensors)", .members = &.{"compressed-tensors-int4-g32"} },
    .{ .name = "GGML (ModelOpt)", .members = &.{"modelopt-ggml"} },
    .{ .name = "BF16", .members = &.{"bf16"} },
    .{ .name = "FP32", .members = &.{"fp32"} },
};

fn gibs(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / gib;
}
