//! Gate 5 on HIP: the records against every chip, this GPU's own limits, and the code objects' metadata.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const controls = @import("hip_controls");

const contract = core.contract;
const limits = contract.limits;
const records = contract.hip_records;
const hsaco = contract.hsaco;

/// Every record the audits check: the registered launches, then the planted controls.
const all = records.records ++ records.controls;

fn isControl(r: records.Record) bool {
    return std.mem.startsWith(u8, r.launch.name, "control");
}

/// The embedded code object a record names, built for this binary's -Dhip-arch.
fn image(name: []const u8) ![]align(8) const u8 {
    if (std.mem.eql(u8, name, "contract_controls")) return &controls.bytes;
    const group = std.meta.stringToEnum(hip.kernels.Group, name) orelse return error.UnknownImage;
    return hip.kernels.images[@backingInt(group)];
}

/// The file of a record's code object for one chip in a `hip-kernel-contract-objects` directory.
fn file(a: std.mem.Allocator, dir: []const u8, chip: []const u8, name: []const u8) ![]u8 {
    if (std.mem.eql(u8, name, "contract_controls")) {
        return std.fmt.allocPrint(a, "{s}/{s}/{s}.hsaco", .{ dir, chip, name });
    }
    const group = std.meta.stringToEnum(hip.kernels.Group, name) orelse return error.UnknownImage;
    return std.fmt.allocPrint(a, "{s}/{s}/group{d}.hsaco", .{ dir, chip, @backingInt(group) });
}

/// Gate 5 from the records alone: every launch on every chip, and each control red; returns the problems.
pub fn table(a: std.mem.Allocator) !usize {
    var n = try records.check(a, &records.records, false);
    const word = if (n == 0) "PASS" else "FAIL";
    const chips = "gfx1030/1100/1151/1201";
    const say = "{s} limits: {d} registered HIP launches against {s}, {d} problems\n";
    std.debug.print(say, .{ word, records.records.len, chips, n });
    for (records.controls) |r| {
        const missed = try records.controlProblems(a, r, false);
        n += missed;
        const red = missed == 0;
        const what = if (red) "red on every chip" else "not red on every chip";
        std.debug.print("{s} limits {s}: {s}\n", .{ if (red) "RED " else "MISSED", r.launch.name, what });
    }
    return n;
}

/// Gate 5 on this GPU: the device's limits against its chip's row, each loaded kernel against its record.
pub fn live(a: std.mem.Allocator, r: *const hip.Runtime, ctx: hip.Context, arch: []const u8) !usize {
    const at = for (limits.rdna, 0..) |t, i| {
        if (std.mem.eql(u8, t.name, arch)) break i;
    } else {
        std.debug.print("N/A  limits live: {s} is not a chip the records cover\n", .{arch});
        return 0;
    };
    const t = limits.rdna[at];
    var n: usize = 0;
    const dev = [_]struct { []const u8, hip.abi.DeviceAttribute, u32 }{
        .{ "threads a workgroup", .max_threads_per_block, t.threads },
        .{ "LDS bytes a workgroup", .max_shared_memory_per_block, t.lds },
        .{ "wave size", .warp_size, t.wave },
    };
    for (dev) |d| {
        const got: u32 = @intCast(try ctx.attribute(d[1]));
        if (got != d[2]) n += 1;
        const word = if (got == d[2]) "PASS" else "FAIL";
        const say = "{s} limits live {s} {s}: the device says {d}, the table {d}\n";
        std.debug.print(say, .{ word, arch, d[0], got, d[2] });
    }
    for (all) |rec| {
        const bytes = try image(rec.launch.image);
        const listed = try hsaco.kernels(a, bytes);
        defer a.free(listed);
        const meta = (hsaco.find(listed, rec.launch.symbolOn(at)) orelse return error.SymbolNotInImage).usage;
        var mod = try hip.Module.load(r, bytes);
        defer mod.unload();
        const sym = try a.dupeSentinel(u8, rec.launch.symbolOn(at), 0);
        defer a.free(sym);
        const f = try mod.function(sym);
        const regs: u32 = @intCast(try f.attribute(.num_regs));
        const lds: u32 = @intCast(try f.attribute(.shared_size_bytes));
        const most: u32 = @intCast(try f.attribute(.max_threads_per_block));
        const fits = rec.launch.threads <= most and lds + rec.launch.dynamic <= t.lds;
        const current = std.meta.eql(meta, rec.usage[at]) and regs == meta.vgprs and lds == meta.lds;
        const control = isControl(rec);
        if (!control and !(fits and current)) n += 1;
        if (control and (!current or fits or meta.vgprs == 0)) n += 1;
        const red = if (fits) "MISSED" else "RED ";
        const word = if (control) (if (!current or meta.vgprs == 0) "FAIL" else red) else if (fits and current) "PASS" else "FAIL";
        const want = rec.usage[at];
        const what = .{ rec.launch.threads, most, regs, lds, rec.launch.dynamic, want.vgprs, want.lds };
        std.debug.print("{s} limits live {s}: {d} threads launched, the runtime allows {d}; " ++
            "{d} VGPRs, {d} + {d} LDS bytes (recorded {d}, {d})\n", .{ word, rec.launch.name } ++ what);
    }
    return n;
}

/// tf-hip-kernel-contract limits DIR: each record against DIR/<chip>/'s code objects, every chip we serve.
pub fn objects(a: std.mem.Allocator, io: std.Io, dir: []const u8) !usize {
    var n: usize = 0;
    for (all) |rec| {
        const control = isControl(rec);
        var red: usize = 0;
        var line: std.ArrayList(u8) = .empty;
        defer line.deinit(a);
        for (limits.rdna, 0..) |t, i| {
            const path = try file(a, dir, t.name, rec.launch.image);
            defer a.free(path);
            const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 28)) catch |err| {
                const say = "FAIL limits {s}: {s}: {t} (build hip-kernel-contract-objects)\n";
                std.debug.print(say, .{ rec.launch.name, path, err });
                n += 1;
                continue;
            };
            defer a.free(bytes);
            const listed = try hsaco.kernels(a, bytes);
            defer a.free(listed);
            const k = hsaco.find(listed, rec.launch.symbolOn(i)) orelse return error.SymbolNotInImage;
            var out: std.ArrayList(u8) = .empty;
            defer out.deinit(a);
            const problems = try limits.hip(rec.launch, k.usage, t, &out, a);
            if (control and (k.usage.vgprs == 0 or problems == 0)) n += 1;
            if (!control) red += problems;
            const current = std.meta.eql(k.usage, rec.usage[i]);
            if (!current) n += 1;
            const u = k.usage;
            const word = if (!current) "FAIL" else if (control) (if (k.usage.vgprs != 0 and problems != 0) "RED " else "MISSED") else if (problems != 0) "FAIL" else "PASS";
            const what = .{ u.vgprs, u.sgprs, u.lds, u.private, u.bound, u.wave, if (current) "current" else "stale" };
            const say = "{s} limits {s} {s}: {d} VGPRs, {d} SGPRs, {d} LDS bytes, {d} scratch bytes, bound {d}, " ++
                "wave{d} (record {s}){s}";
            const end = if (out.items.len > 0) ": " else "\n";
            std.debug.print(say, .{ word, t.name, rec.launch.name } ++ what ++ .{end});
            if (out.items.len > 0) std.debug.print("{s}", .{out.items});
            const field = ".{{ .vgprs = {d}, .sgprs = {d}, .lds = {d}, .private = {d}, " ++
                ".bound = {d}, .wave = {d} }}, ";
            try line.print(a, field, .{
                u.vgprs, u.sgprs, u.lds, u.private, u.bound, u.wave,
            });
        }
        std.debug.print("RECORD {s}: .usage = .{{ {s}}}\n", .{ rec.launch.name, line.items });
        if (!control) n += red;
    }
    return n;
}
