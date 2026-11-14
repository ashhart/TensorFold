//! What runs here: this binary's backends and families joined with the machine, and in words why not when nothing does.
const std = @import("std");
const machine = @import("machine.zig");
const Allocator = std.mem.Allocator;
const Machine = machine.Machine;
const Gpu = machine.Gpu;

/// A family this binary serves and the checkpoint formats it reads, as `capabilities --json` lists them.
pub const Family = struct { model_type: []const u8, formats: []const []const u8 };

/// This binary's side of the join: what `capabilities --json` reports, and the backend's own memory rules.
pub const Offer = struct {
    version: []const u8,
    backends: []const []const u8,
    families: []const Family = &.{},
    /// The SMs this build's CUDA kernels are for (none for a build without kernels); null when the build does not say.
    kernel_sms: ?[]const u32 = null,
    /// CUDA's default reserve for a device of `total` bytes (cuda_memory.reserveBytes), in a CUDA build.
    cuda_reserve: ?*const fn (total: u64, unified: bool) u64 = null,
    /// The Metal load limit's share of RAM in percent (metal/residency.zig's share_percent).
    metal_share: u64 = 70,
    /// The reserve kept inside a Metal load limit (memory_policy.margin).
    margin: u64 = 2 << 30,
};

pub const Unavailable = struct { family: []const u8, reason: []const u8 };

pub const Support = struct {
    /// The backend that runs here ("metal", "cuda", "hip", "xpu"); null when none does.
    backend: ?[]const u8 = null,
    /// The GPU it runs on, an index into the machine's GPUs.
    gpu: ?usize = null,
    /// Why nothing runs here, when nothing does.
    reason: []const u8 = "",
    /// The verdict rests on the OS's or the driver's own facts and a known kernel list; serve refuses only on these.
    certain: bool = true,
    formats: []const []const u8 = &.{},
    unavailable: []const Unavailable = &.{},
    notes: []const []const u8 = &.{},
    /// Bytes for model weights under the backend's admission rule; 0 when unknown.
    budget: u64 = 0,
    budget_rule: []const u8 = "",
};

/// The release fatbins' SMs (zig/build/cuda.zig's release_sms), assumed for a build that does not record its own.
pub const release_sms = [_]u32{ 80, 86, 120, 121 };

/// Whether SASS for one of `sms` runs on an `sm` GPU: the same SM, or below sm_90 a lower minor of the same major.
pub fn covers(sms: []const u32, sm: u32) bool {
    for (sms) |k| {
        if (k == sm) return true;
        if (k < 90 and k / 10 == sm / 10 and k % 10 <= sm % 10) return true;
    }
    return false;
}

pub fn derive(a: Allocator, m: Machine, o: Offer) !Support {
    // a build fact, not the machine's: serve keeps its own refusal for it
    if (o.backends.len == 0) return .{ .certain = false, .reason = "this build has no engines (an engine-less CLI build): use a release build" };
    var first: ?Support = null;
    for (o.backends) |b| {
        const s = try backend(a, m, o, b);
        if (s.backend != null) return s;
        if (first == null) first = s;
    }
    return first.?;
}

fn backend(a: Allocator, m: Machine, o: Offer, name: []const u8) !Support {
    if (eql(name, "metal")) return metal(a, m, o);
    if (eql(name, "cuda")) return cuda(a, m, o);
    if (eql(name, "hip")) return vendor(a, m, o, .amd, "hip", "an AMD GPU");
    if (eql(name, "xpu")) return vendor(a, m, o, .intel, "xpu", "an Intel GPU");
    return .{ .certain = false, .reason = try std.fmt.allocPrint(a, "this build's {s} backend is one `tensorfold device` does not judge yet", .{name}) };
}

fn metal(a: Allocator, m: Machine, o: Offer) !Support {
    if (m.os != .macos) return .{ .reason = "this build's engines are Metal's, which run on macOS only" };
    const chip = machine.appleChip(m.cpu) orelse
        return .{ .reason = try std.fmt.allocPrint(a, "this Mac's CPU is {s}: the Metal engines need Apple silicon (M1 or later)", .{if (m.cpu.len > 0) m.cpu else "not Apple silicon"}) };
    var out: std.ArrayList(Unavailable) = .empty;
    var notes: std.ArrayList([]const u8) = .empty;
    if (chip.generation < 5) {
        if (has(o.families, "llama")) try out.append(a, .{ .family = "llama", .reason = "the native Llama engine needs Metal 4 tensor units (an M5 or later)" });
        if (has(o.families, "gemma4")) try notes.append(a, "Gemma 4 runs on the row path before the M5; its assistant and DFlash drafters need M5 tensor units");
    }
    const working = if (m.gpu_wired_limit > 0) m.gpu_wired_limit else defaultWorkingSet(m.ram);
    const limit = @min(m.ram, @min(working, m.ram / 100 * o.metal_share));
    return .{
        .backend = "metal",
        .gpu = 0,
        .formats = try formats(a, o.families, out.items),
        .unavailable = out.items,
        .notes = notes.items,
        .budget = limit -| o.margin,
        .budget_rule = try std.fmt.allocPrint(a, "the Metal load limit ({d}% of RAM, at most the GPU working set) less a {d} GiB reserve", .{ o.metal_share, o.margin >> 30 }),
    };
}

/// macOS's default GPU working set without an iogpu.wired_limit_mb setting (cluster/node.zig's defaultGpuLimit).
fn defaultWorkingSet(ram: u64) u64 {
    return @max(ram / 3 * 2, ram / 10 * 9 -| 768 * (1 << 30) / 100);
}

fn cuda(a: Allocator, m: Machine, o: Offer) !Support {
    const sms = o.kernel_sms orelse &release_sms;
    if (o.kernel_sms) |k| if (k.len == 0) return .{ .certain = false, .reason = "this build has no CUDA kernels (built without nvcc or prebuilt fatbins)" };
    var nvidia: usize = 0;
    var unknown = false;
    var from_driver = true;
    var pick: ?usize = null;
    var refused: ?usize = null;
    for (m.gpus, 0..) |g, i| {
        if (g.vendor != .nvidia) continue;
        nvidia += 1;
        from_driver = from_driver and g.source == .driver;
        if (g.sm == 0) {
            unknown = true;
        } else if (covers(sms, g.sm)) {
            if (pick == null) pick = i;
        } else if (refused == null) refused = i;
    }
    const known = o.kernel_sms != null;
    if (pick == null) {
        if (refused) |i| return .{ .certain = from_driver and known and !unknown, .reason = try smRefusal(a, m.gpus[i], sms, known) };
        if (unknown) return .{ .certain = false, .reason = "this NVIDIA GPU's compute capability is not known yet: `tensorfold device` asks the driver" };
        return .{ .certain = false, .reason = try missing(a, m, "this build's engines are CUDA's and no NVIDIA GPU is listed") };
    }
    const g = m.gpus[pick.?];
    var out: std.ArrayList(Unavailable) = .empty;
    var notes: std.ArrayList([]const u8) = .empty;
    if (g.sm != 121 and has(o.families, "minimax_m3_vl"))
        try out.append(a, .{ .family = "minimax_m3_vl", .reason = "MiniMax-M3.1-Flash on CUDA is served on GB10 (sm_121) only so far" });
    if (g.sm < 120) try notes.append(a, "NVFP4 and FP8 weights run by decoding to BF16 here; the FP4 tensor-core paths need Blackwell (sm_12x)");
    if (g.sm < 90) try notes.append(a, "no NVFP4 or FP8 model run is recorded on sm_80 to sm_89 yet");
    if (g.sm < 120 and has(o.families, "nemotron_h")) try notes.append(a, "Nemotron's --precision checkpoint (its FP4 math) needs Blackwell (sm_12x)");
    if (g.sm != 121) try notes.append(a, "--slide learning runs on GB10 (sm_121) only so far");
    if (!known) try notes.append(a, "this build does not record its kernel SMs: the release set (sm_80, sm_86, sm_120, sm_121) is assumed");
    if (g.source != .driver) try notes.append(a, "the compute capability comes from the card's name: `tensorfold device` asks the driver");
    if (nvidia > 1) try notes.append(a, try std.fmt.allocPrint(a, "{d} NVIDIA GPUs: serve uses GPU 0 unless --device or TF_CUDA_DEVICE picks another", .{nvidia}));
    const memory = if (g.unified) m.ram else g.memory;
    const reserve = if (o.cuda_reserve) |r| r(memory, g.unified) else 0;
    return .{
        .backend = "cuda",
        .gpu = pick,
        .certain = g.source == .driver,
        .formats = try formats(a, o.families, out.items),
        .unavailable = out.items,
        .notes = notes.items,
        .budget = memory -| reserve,
        .budget_rule = if (g.unified) "system RAM less the CUDA reserve (a tenth, at least 4 GiB); startup grants what is free then" else "device memory less the CUDA reserve (a 32nd, at least 1 GiB)",
    };
}

/// Why no NVIDIA GPU of `g`'s compute capability runs this build's kernels.
fn smRefusal(a: Allocator, g: Gpu, sms: []const u32, known: bool) ![]const u8 {
    const name = machine.shortName(g.name);
    if (g.sm == 90) return std.fmt.allocPrint(a, "{s} is a Hopper GPU (sm_90): TensorFold has no Hopper kernels now", .{name});
    if (g.sm == 100 or g.sm == 103) return std.fmt.allocPrint(a, "{s} is data-center Blackwell (sm_{d}): TensorFold has no kernels for it now", .{ name, g.sm });
    if (g.sm < 80) return std.fmt.allocPrint(a, "{s} is sm_{d}: TensorFold's CUDA kernels need sm_80 (Ampere) or newer", .{ name, g.sm });
    var list: std.Io.Writer.Allocating = .init(a);
    for (sms, 0..) |k, i| try list.writer.print("{s}sm_{d}", .{ if (i > 0) ", " else "", k });
    return std.fmt.allocPrint(a, "{s} is sm_{d}: {s} kernels for {s}, not for it", .{ name, g.sm, if (known) "this build has" else "the release has", list.written() });
}

/// A GPU backend that runs on any GPU of one vendor (HIP on AMD, XPU on Intel).
fn vendor(a: Allocator, m: Machine, o: Offer, v: machine.Vendor, name: []const u8, needs: []const u8) !Support {
    for (m.gpus, 0..) |g, i| {
        if (g.vendor != v) continue;
        var notes: std.ArrayList([]const u8) = .empty;
        if (g.gfx.len > 0) try notes.append(a, try std.fmt.allocPrint(a, "gfx target {s}", .{g.gfx}));
        return .{
            .backend = name,
            .gpu = i,
            .formats = try formats(a, o.families, &.{}),
            .notes = notes.items,
            .budget = g.memory -| o.margin,
            .budget_rule = if (g.memory > 0) "device memory less a 2 GiB reserve" else "unknown: the OS lists no device memory for it",
        };
    }
    const what = try std.fmt.allocPrint(a, "this build's engines are {s}'s and need {s}, and none is listed", .{ name, needs });
    return .{ .certain = false, .reason = try missing(a, m, what) };
}

/// `what`, with the GPU this machine does have and the build that would run on it.
fn missing(a: Allocator, m: Machine, what: []const u8) ![]const u8 {
    if (m.wsl) return std.fmt.allocPrint(a, "{s}: under WSL 2 GPUs show only to their driver, so run `tensorfold device`", .{what});
    for (m.gpus) |g| {
        const other: []const u8 = switch (g.vendor) {
            .nvidia => "a CUDA build (the Linux release, or the Windows source build)",
            .amd => "a HIP build (-Dhip=true)",
            .intel => "an XPU build",
            .apple => "the macOS release (Metal)",
            .other => continue,
        };
        return std.fmt.allocPrint(a, "{s}; this machine has {s}: {s} runs on it", .{ what, machine.shortName(g.name), other });
    }
    return std.fmt.allocPrint(a, "{s}; this machine lists no GPU TensorFold runs on", .{what});
}

/// The formats of every family that runs here, sorted, each once.
fn formats(a: Allocator, families: []const Family, out: []const Unavailable) ![]const []const u8 {
    var list: std.ArrayList([]const u8) = .empty;
    for (families) |f| {
        const skip = for (out) |u| {
            if (eql(u.family, f.model_type)) break true;
        } else false;
        if (skip) continue;
        for (f.formats) |fmt| {
            const seen = for (list.items) |x| {
                if (eql(x, fmt)) break true;
            } else false;
            if (!seen) try list.append(a, fmt);
        }
    }
    std.mem.sort([]const u8, list.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.order(u8, x, y) == .lt;
        }
    }.less);
    return list.items;
}

/// Why serve should stop before loading `model_type` here, from a current profile's certain facts; null to go on.
pub fn refusal(a: Allocator, s: Support, model_type: []const u8) !?[]const u8 {
    if (!s.certain) return null;
    if (s.backend == null) return try std.fmt.allocPrint(a, "this machine cannot run this build's engines: {s}", .{s.reason});
    for (s.unavailable) |u| if (eql(u.family, model_type))
        return try std.fmt.allocPrint(a, "{s} does not run on this machine: {s}", .{ model_type, u.reason });
    return null;
}

pub const Verdict = enum { fits, too_large, not_here, not_served };

/// A checkpoint of `model_type` with `bytes` on disk: served by this build, its family runs here, it fits the budget.
pub fn fit(s: Support, families: []const Family, model_type: []const u8, bytes: u64) Verdict {
    if (!has(families, model_type)) return .not_served;
    if (s.backend == null) return .not_here;
    for (s.unavailable) |u| if (eql(u.family, model_type)) return .not_here;
    if (s.budget > 0 and bytes > s.budget) return .too_large;
    return .fits;
}

fn has(families: []const Family, model_type: []const u8) bool {
    for (families) |f| if (eql(f.model_type, model_type)) return true;
    return false;
}

fn eql(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}
