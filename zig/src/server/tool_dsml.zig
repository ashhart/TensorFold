//! DeepSeek's DSML tool-call blocks, ``<｜DSML｜tool_calls>`` and V4.1's spaced ``<｜DSML｜ calls>``, as calls.
const std = @import("std");
const json = @import("json");
const reply_text = @import("reply_text.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const Error = error{ Invalid, OutOfMemory };
pub const Call = struct { name: []const u8, arguments: Value };

/// One spelling's tags: every one ``<｜DSML｜``, its separator, then its name.
pub const Spelling = struct {
    open: []const u8,
    close: []const u8,
    invoke_open: []const u8,
    invoke_close: []const u8,
    param_open: []const u8,
    param_close: []const u8,
};

fn spelled(comptime block: [2][]const u8, comptime sep: []const u8) Spelling {
    const mark = "\u{ff5c}DSML\u{ff5c}" ++ sep;
    return .{
        .open = block[0],
        .close = block[1],
        .invoke_open = "<" ++ mark ++ "invoke name=\"",
        .invoke_close = "</" ++ mark ++ "invoke>",
        .param_open = "<" ++ mark ++ "parameter name=\"",
        .param_close = "</" ++ mark ++ "parameter>",
    };
}

/// ``tool_calls``, then V4.1's: its release template writes a space before each tag's name and the block ``calls``.
pub const spellings = [_]Spelling{ spelled(reply_text.calls[2], ""), spelled(reply_text.calls[4], " ") };
/// The longest opener: a search resumed this far back finds one the last search saw only part of.
pub const open_max = @max(spellings[0].open.len, spellings[1].open.len);

pub const Opener = struct { at: usize, spelling: *const Spelling };

/// The first block opener of either spelling at or after ``from``.
pub fn findOpen(text: []const u8, from: usize) ?Opener {
    var first: ?Opener = null;
    for (&spellings) |*sp| if (from <= text.len) if (std.mem.indexOfPos(u8, text, from, sp.open)) |at| {
        if (first == null or at < first.?.at) first = .{ .at = at, .spelling = sp };
    };
    return first;
}

/// Where the block ``o`` opens ends, past its own spelling's closer; null while it is not closed.
pub fn blockEnd(text: []const u8, o: Opener) ?usize {
    return (find(text, o.spelling.close, o.at + o.spelling.open.len) orelse return null) + o.spelling.close.len;
}

/// The spelling whose block opener ``text`` starts with.
pub fn openedBy(text: []const u8) ?*const Spelling {
    for (&spellings) |*sp| if (std.mem.startsWith(u8, text, sp.open)) return sp;
    return null;
}

pub const Block = struct { inner: []const u8, spelling: *const Spelling };

/// ``payload`` as one whole block, its spelling's opener and closer around it; null for any other payload.
pub fn whole(payload: []const u8) ?Block {
    const sp = openedBy(payload) orelse return null;
    if (payload.len < sp.open.len + sp.close.len or !std.mem.endsWith(u8, payload, sp.close)) return null;
    return .{ .inner = payload[sp.open.len .. payload.len - sp.close.len], .spelling = sp };
}

fn find(hay: []const u8, needle: []const u8, from: usize) ?usize {
    if (from > hay.len) return null;
    return std.mem.indexOfPos(u8, hay, from, needle);
}

fn allSpace(s: []const u8) bool {
    return reply_text.pyStrip(s).len == 0;
}

/// Every invoke of a block's inside in spelling ``sp``, or Invalid when anything in it is not a well-formed invoke.
pub fn calls(a: Allocator, block: []const u8, sp: *const Spelling) Error![]Call {
    var out: std.ArrayList(Call) = .empty;
    var at: usize = 0;
    var pos: usize = 0;
    while (find(block, sp.invoke_open, pos)) |s| {
        pos = s + 1;
        const name_start = s + sp.invoke_open.len;
        const q = std.mem.indexOfScalarPos(u8, block, name_start, '"') orelse continue;
        if (!std.mem.startsWith(u8, block[q..], "\">")) continue;
        const e = find(block, sp.invoke_close, q + 2) orelse continue;
        if (!allSpace(block[at..s])) return error.Invalid;
        at = e + sp.invoke_close.len;
        pos = at;
        const body = block[q + 2 .. e];
        const arguments = try json.newObject(a);
        var p: usize = 0;
        var kept: usize = 0;
        var residue: std.ArrayList(u8) = .empty;
        while (find(body, sp.param_open, p)) |ps| {
            p = ps + 1;
            const ns = ps + sp.param_open.len;
            const nq = std.mem.indexOfScalarPos(u8, body, ns, '"') orelse continue;
            const flag = body[nq..];
            const is_string = std.mem.startsWith(u8, flag, "\" string=\"true\">");
            if (!is_string and !std.mem.startsWith(u8, flag, "\" string=\"false\">")) continue;
            const v_start = nq + (if (is_string) "\" string=\"true\">".len else "\" string=\"false\">".len);
            const pe = find(body, sp.param_close, v_start) orelse continue;
            const value = body[v_start..pe];
            try residue.appendSlice(a, body[kept..ps]);
            kept = pe + sp.param_close.len;
            p = kept;
            if (is_string) {
                try arguments.put(a, body[ns..nq], .{ .string = value });
            } else switch (try json.parseText(a, value)) {
                .ok => |v| try arguments.put(a, body[ns..nq], v),
                .err => return error.Invalid,
            }
        }
        try residue.appendSlice(a, body[kept..]);
        if (!allSpace(residue.items)) return error.Invalid;
        try out.append(a, .{ .name = reply_text.pyStrip(block[name_start..q]), .arguments = .{ .object = arguments } });
    }
    if (out.items.len == 0 or !allSpace(block[at..])) return error.Invalid;
    return out.items;
}
