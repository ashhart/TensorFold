//! Gate 5 on XPU: the records against the BMG card, and `limits DIR` against the compiled .ze_info of every image.
const std = @import("std");
const core = @import("core");

const contract = core;
const limits = contract.xpu_limits;
const records = contract.xpu_records;
const zebin = contract.zebin;
const zeinfo = contract.zeinfo;

/// Every record the audit checks: the registered launches, then the planted controls.
const all = records.records ++ records.controls;

fn isControl(r: records.Record) bool {
    return std.mem.startsWith(u8, r.launch.name, "control");
}

/// Gate 5 from the records alone: every launch on every card class, and each control red; returns the problems.
pub fn table(a: std.mem.Allocator) !usize {
    var n = try records.check(a, &records.records, false);
    const word = if (n == 0) "PASS" else "FAIL";
    const names = limits.xe[0].name;
    const say = "{s} limits: {d} registered XPU launches against {s}, {d} problems\n";
    std.debug.print(say, .{ word, records.records.len, names, n });
    for (records.controls) |r| {
        const missed = try records.controlProblems(a, r, false);
        n += missed;
        const red = missed == 0;
        const what = if (red) "red on every card class" else "not red on every card class";
        std.debug.print("{s} limits {s}: {s}\n", .{ if (red) "RED " else "MISSED", r.launch.name, what });
    }
    return n;
}

/// The file of a record's compiled image for one card class in an `xpu-kernel-contract-objects` directory.
fn file(a: std.mem.Allocator, dir: []const u8, device: []const u8, name: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "{s}/{s}/{s}.bin", .{ dir, device, name });
}

/// tf-xpu-kernel-contract limits DIR: each record against DIR/<device>/'s images, every card class we serve.
pub fn objects(a: std.mem.Allocator, io: std.Io, dir: []const u8) !usize {
    var n: usize = 0;
    const field = ".{{ .threads = .{{ {d}, {d}, {d} }}, .slm = {d}, .sub_group = {d}, .grf = {d}, " ++
        ".spill = {d}, .private = {d} }}, ";
    for (all) |rec| {
        const control = isControl(rec);
        var red: usize = 0;
        var line: std.ArrayList(u8) = .empty;
        defer line.deinit(a);
        for (limits.xe, 0..) |t, i| {
            var out: std.ArrayList(u8) = .empty;
            defer out.deinit(a);
            if (!rec.compiled) {
                const fits = try limits.fits(rec.launch, t, &out, a);
                if (control and fits == 0) n += 1;
                if (!control) red += fits;
                const word = if (control) (if (fits != 0) "RED " else "MISSED") else if (fits != 0) "FAIL" else "PASS";
                const end = if (out.items.len > 0) ": " else "\n";
                std.debug.print("{s} limits {s} {s}: declared only, ocloc does not build it for this device{s}", .{ word, t.name, rec.launch.name, end });
                if (out.items.len > 0) std.debug.print("{s}", .{out.items});
                const u = rec.usage[i];
                try line.print(a, field, .{ u.threads[0], u.threads[1], u.threads[2], u.slm, u.sub_group, u.grf, u.spill, u.private });
                continue;
            }
            const path = try file(a, dir, t.ocloc_device, rec.launch.image);
            defer a.free(path);
            const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 28)) catch |err| {
                std.debug.print("FAIL limits {s}: {s}: {t} (build xpu-kernel-contract-objects)\n", .{ rec.launch.name, path, err });
                n += 1;
                continue;
            };
            defer a.free(bytes);
            const info = zebin.zeInfo(a, bytes) catch |err| {
                std.debug.print("FAIL limits {s}: {s}: {t}\n", .{ rec.launch.name, path, err });
                n += 1;
                continue;
            };
            defer a.free(info);
            const listed = try zeinfo.kernels(a, info);
            defer a.free(listed);
            const u = zeinfo.find(listed, rec.launch.symbol) orelse {
                std.debug.print("FAIL limits {s}: {s} is not in {s}\n", .{ rec.launch.name, rec.launch.symbol, path });
                n += 1;
                continue;
            };
            const fits = try limits.fits(rec.launch, t, &out, a);
            if (control and (u.grf == 0 or fits == 0)) n += 1;
            if (!control) red += fits;
            const current = std.meta.eql(u, rec.usage[i]);
            if (!current) n += 1;
            const word = if (!current) "FAIL" else if (control) (if (u.grf != 0 and fits != 0) "RED " else "MISSED") else if (fits != 0) "FAIL" else "PASS";
            const what = .{ u.threads[0], u.slm, u.sub_group, u.grf, u.spill, u.private, if (current) "current" else "stale" };
            const say = "{s} limits {s} {s}: {d} work-items, {d} SLM bytes, sub-group {d}, {d} GRF, {d} spill bytes, " ++
                "{d} private bytes (record {s}){s}";
            const end = if (out.items.len > 0) ": " else "\n";
            std.debug.print(say, .{ word, t.name, rec.launch.name } ++ what ++ .{end});
            if (out.items.len > 0) std.debug.print("{s}", .{out.items});
            try line.print(a, field, .{ u.threads[0], u.threads[1], u.threads[2], u.slm, u.sub_group, u.grf, u.spill, u.private });
        }
        std.debug.print("RECORD {s}: .usage = .{{ {s}}}\n", .{ rec.launch.name, line.items });
        if (!control) n += red;
    }
    return n;
}
