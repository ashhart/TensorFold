//! wire.normal: the reply normalization the parity goldens compare through, shared by the checker.
const std = @import("std");

pub const Header = struct { name: []const u8, value: []const u8 };

pub const Reply = struct {
    status: []const u8,
    headers: std.ArrayList(Header),
    body: []const u8,
    interim: []const u8,
};

/// One open connection to fake_serve; reads with a receive timeout the closed probe sets when it runs.

fn hexAt(body: []const u8, at: usize, want: usize) bool {
    if (at + want > body.len) return false;
    for (body[at..][0..want]) |c| {
        if (!std.ascii.isHex(c) or std.ascii.isUpper(c)) return false;
    }
    return true;
}

const id_prefixes = [_][]const u8{ "chatcmpl-", "cmpl-", "resp_", "msg_", "rs_", "fc_", "call_" };

/// `(?:chatcmpl-|cmpl-|resp_|msg_|rs_|fc_|call_)[0-9a-f]{32}|call_[0-9a-f]{24}`, first seen to <idN>.
fn subIds(a: std.mem.Allocator, body: []const u8, seen: *std.ArrayList([]const u8)) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var at: usize = 0;
    while (at < body.len) {
        var take: usize = 0;
        for (id_prefixes, 0..) |p, pi| {
            if (!std.mem.startsWith(u8, body[at..], p)) continue;
            const hex = at + p.len;
            if (pi == 6) {
                if (hexAt(body, hex, 32)) take = p.len + 32 else if (hexAt(body, hex, 24)) take = p.len + 24;
            } else if (hexAt(body, hex, 32)) take = p.len + 32;
            if (take != 0) break;
        }
        if (take != 0) {
            var known = false;
            for (seen.items, 0..) |s, i| {
                if (std.mem.eql(u8, s, body[at..][0..take])) {
                    try out.print(a, "<id{d}>", .{i + 1});
                    known = true;
                }
            }
            if (!known) {
                try seen.append(a, try a.dupe(u8, body[at..][0..take]));
                try out.print(a, "<id{d}>", .{seen.items.len});
            }
            at += take;
            continue;
        }
        try out.append(a, body[at]);
        at += 1;
    }
    return out.toOwnedSlice(a);
}

fn digitsAt(body: []const u8, at: usize) ?usize {
    var end = at;
    while (end < body.len and std.ascii.isDigit(body[end])) end += 1;
    return if (end > at) end else null;
}

/// `"(created_at|completed_at|created)": \d+` to `"<name>": <ts>`.
fn subStamps(a: std.mem.Allocator, body: []const u8) ![]u8 {
    const names = [_][]const u8{ "created_at", "completed_at", "created" };
    var out: std.ArrayList(u8) = .empty;
    var at: usize = 0;
    while (at < body.len) {
        var matched = false;
        for (names) |n| {
            if (body[at] != '"' or !std.mem.startsWith(u8, body[at + 1 ..], n)) continue;
            const after = at + 1 + n.len;
            if (!std.mem.startsWith(u8, body[after..], "\": ")) continue;
            const end = digitsAt(body, after + 3) orelse continue;
            try out.appendSlice(a, body[at..after]);
            try out.appendSlice(a, "\": <ts>");
            at = end;
            matched = true;
            break;
        }
        if (!matched) {
            try out.append(a, body[at]);
            at += 1;
        }
    }
    return out.toOwnedSlice(a);
}

const time_names = [_][]const u8{ "time_to_first_token", "prefill_seconds", "tokens_per_second", "seconds" };

fn matchNumber(body: []const u8, at: usize) ?usize {
    var end = at;
    if (end < body.len and body[end] == '-') end += 1;
    var digits = false;
    while (end < body.len and (std.ascii.isDigit(body[end]) or body[end] == '.')) {
        end += 1;
        digits = true;
    }
    if (!digits) return null;
    if (end < body.len and (body[end] == 'e' or body[end] == 'E')) {
        var e = end + 1;
        if (e < body.len and (body[e] == '+' or body[e] == '-')) e += 1;
        var ed = false;
        while (e < body.len and std.ascii.isDigit(body[e])) {
            e += 1;
            ed = true;
        }
        if (ed) return e;
    }
    return end;
}

fn findTimes(body: []const u8) ?usize {
    var at: usize = 0;
    while (at < body.len) : (at += 1) {
        if (body[at] != '"') continue;
        for (time_names) |n| {
            if (!std.mem.startsWith(u8, body[at + 1 ..], n)) continue;
            const after = at + 1 + n.len;
            if (!std.mem.startsWith(u8, body[after..], "\": ")) break;
            if (matchNumber(body, after + 3)) |_| return at;
        }
    }
    return null;
}

/// `"(tokens_per_second|seconds|prefill_seconds|time_to_first_token)": -?[0-9.]+(e[-+]?\d+)?` to `<t>`.
fn subTimes(a: std.mem.Allocator, body: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var at: usize = 0;
    while (at < body.len) {
        const rel = findTimes(body[at..]) orelse {
            try out.appendSlice(a, body[at..]);
            break;
        };
        const hit = at + rel;
        try out.appendSlice(a, body[at..hit]);
        var name_end = hit + 1;
        for (time_names) |n| {
            if (std.mem.startsWith(u8, body[hit + 1 ..], n)) {
                name_end = hit + 1 + n.len;
                break;
            }
        }
        try out.appendSlice(a, body[hit..name_end]);
        try out.appendSlice(a, "\": <t>");
        at = matchNumber(body, name_end + 3).?;
    }
    return out.toOwnedSlice(a);
}

fn wordAt(body: []const u8, at: usize) ?usize {
    var end = at;
    while (end < body.len and (std.ascii.isAlphanumeric(body[end]) or body[end] == '_')) end += 1;
    return if (end > at) end else null;
}

const Bucket = struct { hit: bool, drop: bool };

fn bucketLine(line: []const u8) Bucket {
    if (!std.mem.startsWith(u8, line, "tensorfold:")) return .{ .hit = false, .drop = false };
    const ws = wordAt(line, "tensorfold:".len) orelse return .{ .hit = false, .drop = false };
    if (!std.mem.startsWith(u8, line[ws..], "_bucket{le=\"")) return .{ .hit = false, .drop = false };
    const vs = ws + "_bucket{le=\"".len;
    const ve = std.mem.indexOfScalarPos(u8, line, vs, '"') orelse return .{ .hit = false, .drop = false };
    if (!std.mem.startsWith(u8, line[ve + 1 ..], "} ")) return .{ .hit = false, .drop = false };
    if (digitsAt(line, ve + 3) == null) return .{ .hit = false, .drop = false };
    return .{ .hit = true, .drop = !std.mem.startsWith(u8, line[vs..], "+Inf") };
}

/// `^tensorfold:\w+_bucket\{le="(?!+Inf)[^"]*"\} \d+$` lines drop entirely.
fn subBucket(a: std.mem.Allocator, body: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var it = std.mem.splitScalar(u8, body, '\n');
    var first = true;
    while (it.next()) |line| {
        if (!first) try out.append(a, '\n');
        first = false;
        if (!bucketLine(line).drop) try out.appendSlice(a, line);
    }
    return out.toOwnedSlice(a);
}

fn sumLine(line: []const u8) bool {
    if (!std.mem.startsWith(u8, line, "tensorfold:")) return false;
    const ws = wordAt(line, "tensorfold:".len) orelse return false;
    if (!std.mem.startsWith(u8, line[ws..], "_sum ")) return false;
    return line[ws + "_sum ".len..].len > 0;
}

/// `^(tensorfold:\w+_sum) \S+$` to `\1 <sum>`.
fn subSums(a: std.mem.Allocator, body: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var it = std.mem.splitScalar(u8, body, '\n');
    var first = true;
    while (it.next()) |line| {
        if (!first) try out.append(a, '\n');
        first = false;
        if (sumLine(line)) {
            const ws = wordAt(line, "tensorfold:".len).?;
            try out.print(a, "{s} <sum>", .{line[0 .. ws + "_sum".len]});
        } else try out.appendSlice(a, line);
    }
    return out.toOwnedSlice(a);
}

fn footprintLine(line: []const u8) bool {
    const head = "tensorfold:process_footprint_bytes ";
    if (!std.mem.startsWith(u8, line, head)) return false;
    const end = digitsAt(line, head.len) orelse return false;
    return end == line.len;
}

/// `^(tensorfold:process_footprint_bytes) \d+$` to `\1 <n>`.
fn subFootprint(a: std.mem.Allocator, body: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var it = std.mem.splitScalar(u8, body, '\n');
    var first = true;
    while (it.next()) |line| {
        if (!first) try out.append(a, '\n');
        first = false;
        if (footprintLine(line)) try out.appendSlice(a, "tensorfold:process_footprint_bytes <n>") else try out.appendSlice(a, line);
    }
    return out.toOwnedSlice(a);
}

fn matchBucket(body: []const u8) bool {
    var it = std.mem.splitScalar(u8, body, '\n');
    while (it.next()) |line| if (bucketLine(line).hit) return true;
    return false;
}

fn matchSums(body: []const u8) bool {
    var it = std.mem.splitScalar(u8, body, '\n');
    while (it.next()) |line| if (sumLine(line)) return true;
    return false;
}

fn matchFootprint(body: []const u8) bool {
    var it = std.mem.splitScalar(u8, body, '\n');
    while (it.next()) |line| if (footprintLine(line)) return true;
    return false;
}

/// The body as text with invalid utf-8 sequences replaced, as Python's decode("utf-8", "replace") does.
fn utf8Replace(a: std.mem.Allocator, body: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < body.len) {
        const n = std.unicode.utf8ByteSequenceLength(body[i]) catch {
            try out.appendSlice(a, "\xef\xbf\xbd");
            i += 1;
            continue;
        };
        const decoded: ?u21 = if (i + n > body.len) null else (std.unicode.utf8Decode(body[i..][0..n]) catch null);
        if (decoded == null) {
            try out.appendSlice(a, "\xef\xbf\xbd");
            i += 1;
            continue;
        }
        try out.appendSlice(a, body[i..][0..n]);
        i += n;
    }
    return out.toOwnedSlice(a);
}

/// Latin-1 bytes as text: every byte its own code point.
fn latin1(a: std.mem.Allocator, bytes: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    for (bytes) |b| {
        var buf: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(@as(u21, b), &buf) catch unreachable;
        try out.appendSlice(a, buf[0..n]);
    }
    return out.toOwnedSlice(a);
}

/// wire.normal: ids first-seen to <idN>, stamps and timings replaced, Server and Date dropped.
pub const Normal = struct { status: []const u8, headers: []const u8, body: []const u8, interim: []const u8 };

pub fn normalize(a: std.mem.Allocator, r: *const Reply) !Normal {
    var seen: std.ArrayList([]const u8) = .empty;
    var body = try subStamps(a, r.body);
    body = try subIds(a, body, &seen);
    const varied = findTimes(body) != null or matchBucket(body) or matchSums(body) or matchFootprint(body);
    body = try subTimes(a, body);
    body = try subBucket(a, body);
    body = try subSums(a, body);
    body = try subFootprint(a, body);
    var head: std.ArrayList(u8) = .empty;
    for (r.headers.items) |h| {
        if (std.ascii.eqlIgnoreCase(h.name, "server") or std.ascii.eqlIgnoreCase(h.name, "date")) continue;
        try head.appendSlice(a, h.name);
        try head.appendSlice(a, ": ");
        if (std.ascii.eqlIgnoreCase(h.name, "content-length") and varied) try head.appendSlice(a, "<len>") else try head.appendSlice(a, h.value);
        try head.append(a, '\n');
    }
    return .{
        .status = r.status,
        .headers = try head.toOwnedSlice(a),
        .body = try utf8Replace(a, body),
        .interim = try latin1(a, r.interim),
    };
}

/// framed(): a fixed-length reply's Content-Length matches its body.
pub fn framed(r: *const Reply) bool {
    for (r.headers.items) |h| {
        if (std.ascii.eqlIgnoreCase(h.name, "content-length")) {
            const n = std.fmt.parseInt(usize, h.value, 10) catch return false;
            return n == r.body.len;
        }
    }
    return true;
}

pub fn replyId(a: std.mem.Allocator, r: *const Reply) ![]const u8 {
    // the id field of a JSON reply body: "id": "<value>"
    const at = std.mem.indexOf(u8, r.body, "\"id\"") orelse {
        std.debug.print("  replyId: no id in body: {s}\n", .{r.body[0..@min(r.body.len, 120)]});
        return error.NoId;
    };
    const colon = std.mem.indexOfScalarPos(u8, r.body, at + 3, ':') orelse return error.NoId;
    const open = std.mem.indexOfScalarPos(u8, r.body, colon, '"') orelse return error.NoId;
    const close = std.mem.indexOfScalarPos(u8, r.body, open + 1, '"') orelse return error.NoId;
    return a.dupe(u8, r.body[open + 1 .. close]);
}
