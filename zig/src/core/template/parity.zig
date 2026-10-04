//! Renders a tools/template_parity.py corpus and writes every result for the byte comparison.
const std = @import("std");
const template = @import("template.zig");
const V = std.json.Value;

const Result = struct { ok: bool, text: []const u8 = "", @"error": []const u8 = "", raised: bool = false };

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) {
        std.debug.print("usage: parity <corpus.json> <results.json>\n", .{});
        return error.InvalidArguments;
    }
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(1 << 31));
    const corpus = (try std.json.parseFromSlice(V, a, bytes, .{})).value.object;
    const sources = corpus.get("templates").?.object;
    const tables = [_][]const V{ corpus.get("conversations").?.array.items, corpus.get("tools").?.array.items, corpus.get("contexts").?.array.items };
    var compiled: std.StringHashMapUnmanaged(template.Compiled) = .empty;
    var compile_errors: std.StringHashMapUnmanaged([]const u8) = .empty;
    var it = sources.iterator();
    while (it.next()) |e| {
        var diag = template.Diag{};
        if (template.compile(std.heap.page_allocator, e.value_ptr.string, &diag)) |t| {
            try compiled.put(a, e.key_ptr.*, t);
        } else |err| {
            if (err == error.OutOfMemory) return err;
            try compile_errors.put(a, e.key_ptr.*, diag.msg);
        }
    }
    const cases = corpus.get("cases").?.array.items;
    const results = try a.alloc(Result, cases.len);
    var mismatches: usize = 0;
    for (cases, results) |case, *r| {
        const c = case.object;
        const id = c.get("t").?.string;
        if (compile_errors.get(id)) |msg| {
            r.* = .{ .ok = false, .@"error" = msg };
        } else {
            const t = compiled.getPtr(id).?;
            const tools = if (c.get("tl").? == .integer) tables[1][@intCast(c.get("tl").?.integer)] else V.null;
            var diag = template.Diag{};
            if (template.render(a, t, tables[0][@intCast(c.get("m").?.integer)], tools, tables[2][@intCast(c.get("c").?.integer)], c.get("g").?.bool, &diag)) |text| {
                r.* = .{ .ok = true, .text = text };
            } else |err| {
                if (err == error.OutOfMemory) return err;
                r.* = .{ .ok = false, .@"error" = diag.msg, .raised = diag.raised };
            }
        }
        const want_ok = c.get("ok").?.bool;
        const same = if (std.mem.startsWith(u8, id, "unsupported-")) !r.ok else if (want_ok) r.ok and std.mem.eql(u8, r.text, c.get("text").?.string) else !r.ok and (!c.get("raised").?.bool or (r.raised and std.mem.eql(u8, r.@"error", c.get("error").?.string)));
        mismatches += @intFromBool(!same);
    }
    const out = try std.json.Stringify.valueAlloc(a, results, .{});
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[2], .data = out });
    std.debug.print("{d} cases, {d} mismatches\n", .{ cases.len, mismatches });
}
