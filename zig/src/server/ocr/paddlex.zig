//! POST /ocr in the PaddleX OCR serving contract: the JSON request, its file and knobs, then the page loop.
const std = @import("std");
const Allocator = std.mem.Allocator;
const json = @import("json");
const errors = @import("../errors.zig");
const images = @import("../images.zig");
const http_body = @import("../http_body.zig");
const ids = @import("../ids.zig");
const pdf = @import("../pdf.zig");
const Conn = @import("../http_conn.zig").Conn;
const Server = @import("../server.zig").Server;
const types = @import("types.zig");
const run = @import("run.zig");
const fetch = @import("fetch.zig");
const query = @import("../query.zig");
const lines = @import("lines.zig");
const reply = @import("paddlex_reply.zig");
const Value = json.Value;
const Fault = types.Fault;
const Step = run.Step;

pub const Options = reply.Options;

/// What decoding needs from the server: the PDF switch and the URL fetcher.
pub const Facts = struct {
    pdf: bool,
    fetch: ?images.Fetch,

    pub fn of(srv: *const Server) Facts {
        return .{ .pdf = srv.config.vision_pdf, .fetch = if (srv.config.vision_urls) images.Fetch.network(srv.io) else null };
    }
};

/// A decoded request: the reply's options and its one file.
pub const Request = struct { options: Options, source: types.Source };

/// Answers one PaddleX POST /ocr whose body has been read; a client that leaves gets nothing.
pub fn post(srv: *Server, conn: *Conn, a: Allocator, body: []const u8, gone: anytype) void {
    var fault: Fault = .{ .class = .internal, .message = "" };
    var log_id: []const u8 = "";
    const out = handle(srv, a, body, gone, &fault, &log_id) catch |e| {
        if (log_id.len == 0) log_id = fresh(srv, a);
        switch (e) {
            error.Faulted => return reply.sendFault(conn, a, log_id, fault),
            error.OutOfMemory => return reply.sendFault(conn, a, log_id, .{ .class = .internal, .message = "out of memory" }),
            error.Cancelled => {
                conn.close = true;
                return;
            },
        }
    };
    conn.sendJson(200, out);
}

/// A new UUID v4 for a request whose logId is absent or empty.
fn fresh(srv: *const Server, a: Allocator) []const u8 {
    const id = ids.uuid4(srv.io);
    return a.dupe(u8, &id) catch "";
}

fn handle(srv: *Server, a: Allocator, body: []const u8, gone: anytype, fault: *Fault, log_id: *[]const u8) Step![]const u8 {
    const raw = switch (try json.parse(a, body)) {
        .ok => |v| v,
        .err => |m| return schema(a, fault, &.{.{ .type = "json_invalid", .loc = "body", .msg = m }}),
    };
    // the request's own logId is echoed by every reply once the body parsed
    if (raw.field("logId")) |v| if (v == .string and v.string.len > 0) {
        log_id.* = v.string;
    };
    try run.gate(srv, a, fault);
    const req = try decode(a, raw, Facts.of(srv), fault);
    var enc: reply.Encoder = .{ .o = req.options, .geometry = srv.info.ocr.?.geometry, .gpa = srv.gpa };
    try run.run(srv, a, &.{req.source}, enc.sink(), fault, gone);
    if (log_id.*.len == 0) log_id.* = fresh(srv, a);
    return enc.reply(a, log_id.*, req.source.kind == .pdf);
}

/// One validation error as PaddleX's 422 lists it; ``loc`` is dot-joined.
pub const Issue = struct { type: []const u8, loc: []const u8, msg: []const u8 };

/// Faults as bad input with ``errorMsg`` the JSON list of ``issues``, as ``json.dumps`` writes it.
fn schema(a: Allocator, fault: *Fault, issues: []const Issue) Step {
    const list = try a.alloc(Value, issues.len);
    for (issues, list) |i, *v| {
        const o = try json.newObject(a);
        try o.put(a, "type", .{ .string = i.type });
        try o.put(a, "loc", .{ .string = i.loc });
        try o.put(a, "msg", .{ .string = i.msg });
        v.* = .{ .object = o };
    }
    fault.* = .{ .class = .bad_input, .message = try json.stringify(a, .{ .array = list }, .{}) };
    return error.Faulted;
}

fn bad(a: Allocator, fault: *Fault, comptime fmt: []const u8, args: anytype) Step {
    return fault.fail(a, .bad_input, fmt, args);
}

/// The fields as PaddleX's schema types them, each read as pydantic's lax mode reads it.
pub const Kind = enum { boolean, integer, string, number };
const typed = [_]struct { []const u8, Kind }{
    .{ "logId", .string },                   .{ "useDocOrientationClassify", .boolean }, .{ "useDocUnwarping", .boolean },
    .{ "useTextlineOrientation", .boolean }, .{ "textDetLimitSideLen", .integer },       .{ "textDetLimitType", .string },
    .{ "textDetThresh", .number },           .{ "textDetBoxThresh", .number },           .{ "textDetUnclipRatio", .number },
    .{ "textRecScoreThresh", .number },      .{ "returnWordBox", .boolean },             .{ "visualize", .boolean },
};

/// Knobs that change which text a detector returns: refused unless null (or false, for the switches).
const switches = [_][]const u8{ "useDocOrientationClassify", "useDocUnwarping", "useTextlineOrientation" };
const detector = [_][]const u8{ "textDetLimitSideLen", "textDetLimitType", "textDetThresh", "textDetBoxThresh", "textDetUnclipRatio" };

/// The typed fields after conversion: absent or null fields stay null.
const Fields = struct {
    values: [typed.len]?Value = @splat(null),

    fn get(f: *const Fields, comptime name: []const u8) ?Value {
        inline for (typed, 0..) |x, i| if (comptime std.mem.eql(u8, x[0], name)) return f.values[i];
        @compileError("no field " ++ name);
    }
};

/// The request's fields checked as design 6.4 states, and its file's bytes; ``null`` means absent everywhere.
pub fn decode(a: Allocator, body: Value, facts: Facts, fault: *Fault) Step!Request {
    if (body != .object) return schema(a, fault, &.{.{ .type = "model_attributes_type", .loc = "body", .msg = "Input should be a valid dictionary or object to extract fields from" }});
    var issues: std.ArrayList(Issue) = .empty;
    const file = body.get("file");
    if (file == null) try issues.append(a, .{ .type = "missing", .loc = "body.file", .msg = "Field required" });
    if (file != null and file.? != .string) try issues.append(a, .{ .type = "string_type", .loc = "body.file", .msg = "Input should be a valid string" });
    // Literal[0, 1] in lax mode: a bool or a number equal to 0 or 1 is that literal; a string is not
    var file_type: ?bool = null; // true: 1 (an image), false: 0 (a PDF)
    if (body.field("fileType")) |v| {
        file_type = switch (v) {
            .bool => |b| b,
            .int => |n| if (std.mem.eql(u8, n, "0")) false else if (std.mem.eql(u8, n, "1")) true else null,
            .float => |f| if (f == 0) false else if (f == 1) true else null,
            else => null,
        };
        if (file_type == null) try issues.append(a, .{ .type = "literal_error", .loc = "body.fileType", .msg = "Input should be 0 or 1" });
    }
    var fields: Fields = .{};
    for (typed, &fields.values) |f, *slot| if (body.field(f[0])) |v| switch (lax(v, f[1])) {
        .ok => |c| slot.* = c,
        .bad => |b| try issues.append(a, .{ .type = b[0], .loc = try loc(a, f[0]), .msg = b[1] }),
    };
    if (issues.items.len > 0) return schema(a, fault, issues.items);
    inline for (switches) |name| if (fields.get(name)) |v| if (v.bool)
        try issues.append(a, .{ .type = "unsupported", .loc = "body." ++ name, .msg = "TensorFold reads the page as sent, with no preprocessing; send null or false" });
    inline for (detector) |name| if (fields.get(name) != null)
        try issues.append(a, .{ .type = "unsupported", .loc = "body." ++ name, .msg = "TensorFold has no text detector to apply this to; send null" });
    if (issues.items.len > 0) return schema(a, fault, issues.items);
    const o: Options = .{
        .thresh = if (fields.get("textRecScoreThresh")) |v| v.float else 0,
        .word_box = if (fields.get("returnWordBox")) |v| v.bool else false,
        .visualize = if (fields.get("visualize")) |v| v.bool else true,
    };
    const bytes = try fileBytes(a, fault, file.?.string, facts);
    // the image signature first: a PNG may hold "%PDF-" in an early text chunk
    const is_image = switch (images.sniff(bytes)) {
        .png, .jpeg, .webp => true,
        else => false,
    };
    const is_pdf = !is_image and std.mem.indexOf(u8, bytes[0..@min(bytes.len, 1024)], "%PDF-") != null;
    const want_pdf: ?bool = if (file_type) |image| !image else null;
    if (want_pdf) |w| if (w != is_pdf) return bad(a, fault, "Invalid input file: fileType {d} names {s}, and the file is not one", .{ @intFromBool(!w), if (w) "a PDF" else "an image" });
    if (is_pdf) {
        if (!facts.pdf) return bad(a, fault, "{s}", .{pdf.off_message});
        // truncated to the request's page budget, as PaddleX truncates to its max_num_input_imgs
        return .{ .options = o, .source = .{ .name = "file", .bytes = bytes, .kind = .pdf, .sel = .{ .first = std.math.maxInt(u32) } } };
    }
    if (!is_image) return bad(a, fault, "Unsupported file type; send a PNG, JPEG or WebP image, or a PDF", .{});
    return .{ .options = o, .source = .{ .name = "file", .bytes = bytes, .kind = .image, .sel = .{ .all_within = 1 } } };
}

/// A converted value (.bool, .int, .string, or a finite .float), or a pydantic error type and message.
pub const Lax = union(enum) { ok: Value, bad: [2][]const u8 };

/// ``v`` read as pydantic's lax mode reads ``k``; a number must also be finite.
pub fn lax(v: Value, k: Kind) Lax {
    const finite: Lax = .{ .bad = .{ "finite_number", "Input should be a finite number" } };
    switch (k) {
        .string => return if (v == .string) .{ .ok = v } else .{ .bad = .{ "string_type", "Input should be a valid string" } },
        .boolean => {
            const no: Lax = .{ .bad = .{ "bool_parsing", "Input should be a valid boolean, unable to interpret input" } };
            return switch (v) {
                .bool => .{ .ok = v },
                .int => |n| if (std.mem.eql(u8, n, "0") or std.mem.eql(u8, n, "1")) .{ .ok = .{ .bool = n[0] == '1' } } else no,
                .float => |f| if (f == 0 or f == 1) .{ .ok = .{ .bool = f == 1 } } else if (std.math.isFinite(f) and f == @trunc(f)) no else .{ .bad = .{ "bool_type", "Input should be a valid boolean" } },
                .string => |s| if (query.boolean(s)) |b| .{ .ok = .{ .bool = b } } else no,
                else => .{ .bad = .{ "bool_type", "Input should be a valid boolean" } },
            };
        },
        .integer => return switch (v) {
            .int => .{ .ok = v },
            .bool => |b| .{ .ok = .{ .int = if (b) "1" else "0" } },
            .float => |f| if (!std.math.isFinite(f)) finite else if (f == @trunc(f)) .{ .ok = v } else .{ .bad = .{ "int_from_float", "Input should be a valid integer, got a number with a fractional part" } },
            // past i64 an integer string stays text: it is never the default, so the knob check refuses it
            .string => |s| if (intText(lines.trimSpace(s))) .{ .ok = v } else .{ .bad = .{ "int_parsing", "Input should be a valid integer, unable to parse string as an integer" } },
            else => .{ .bad = .{ "int_type", "Input should be a valid integer" } },
        },
        .number => {
            const f: f64 = switch (v) {
                .int => |n| std.fmt.parseFloat(f64, n) catch return finite,
                .float => |f| f,
                .bool => |b| if (b) 1 else 0,
                .string => |s| floatText(lines.trimSpace(s)) orelse return .{ .bad = .{ "float_parsing", "Input should be a valid number, unable to parse string as a number" } },
                else => return .{ .bad = .{ "float_type", "Input should be a valid number" } },
            };
            return if (std.math.isFinite(f)) .{ .ok = .{ .float = f } } else finite;
        },
    }
}

/// An integer as pydantic reads a string: a sign, digits, then optionally "." and zeros only; any length.
fn intText(s: []const u8) bool {
    var i: usize = if (s.len > 0 and (s[0] == '+' or s[0] == '-')) 1 else 0;
    const digits = i;
    while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
    if (i == digits) return false;
    if (i < s.len and s[i] == '.') {
        i += 1;
        while (i < s.len and s[i] == '0') i += 1;
    }
    return i == s.len;
}

/// A float in decimal syntax only (sign, digits, ".", digits, exponent; or inf, infinity, nan); null otherwise.
fn floatText(s: []const u8) ?f64 {
    var i: usize = if (s.len > 0 and (s[0] == '+' or s[0] == '-')) 1 else 0;
    for ([_][]const u8{ "inf", "infinity", "nan" }) |word| if (std.ascii.eqlIgnoreCase(s[i..], word)) return std.math.inf(f64);
    var digits: usize = 0;
    while (i < s.len and std.ascii.isDigit(s[i])) : (i += 1) digits += 1;
    if (i < s.len and s[i] == '.') {
        i += 1;
        while (i < s.len and std.ascii.isDigit(s[i])) : (i += 1) digits += 1;
    }
    if (digits == 0) return null;
    if (i < s.len and (s[i] == 'e' or s[i] == 'E')) {
        i += 1;
        if (i < s.len and (s[i] == '+' or s[i] == '-')) i += 1;
        const exp = i;
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
        if (i == exp) return null;
    }
    if (i != s.len) return null;
    return std.fmt.parseFloat(f64, s) catch null;
}

fn loc(a: Allocator, field: []const u8) Allocator.Error![]const u8 {
    return std.mem.concat(a, u8, &.{ "body.", field });
}

/// The file's bytes: base64 (decoded leniently), a data URL, or an https URL fetched when --vision-urls is on.
fn fileBytes(a: Allocator, fault: *Fault, raw: []const u8, facts: Facts) Step![]const u8 {
    const text = std.mem.trim(u8, raw, " \t\r\n");
    if (std.ascii.startsWithIgnoreCase(text, "data:")) {
        var cx: errors.Cx = .{ .a = a };
        return images.dataBytesOf(&cx, text, http_body.limit, "file") catch |e| switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            error.Refused => bad(a, fault, "{s}", .{cx.message}),
        };
    }
    if (std.ascii.startsWithIgnoreCase(text, "http://") or std.ascii.startsWithIgnoreCase(text, "https://")) {
        return switch (try fetch.https(a, facts.fetch, text, facts.pdf)) {
            .bytes => |b| b,
            .refused => |m| bad(a, fault, "{s}", .{m}),
        };
    }
    return try base64(a, text) orelse bad(a, fault, "Invalid input file", .{});
}

/// Base64 as Python's ``b64decode`` reads it: bytes outside the alphabet skipped, the padding repaired; null if none.
pub fn base64(a: Allocator, text: []const u8) Allocator.Error!?[]const u8 {
    const clean = try a.alloc(u8, text.len + 3);
    var n: usize = 0;
    for (text) |c| if (std.ascii.isAlphanumeric(c) or c == '+' or c == '/') {
        clean[n] = c;
        n += 1;
    };
    if (n == 0 or n % 4 == 1) return null;
    // the last character's bits past the data are ignored, as Python ignores them: 4 before "==", 2 before "="
    const tail = n % 4;
    if (tail != 0) {
        const alphabet = std.base64.standard_alphabet_chars;
        const at = std.mem.indexOfScalar(u8, &alphabet, clean[n - 1]).?;
        clean[n - 1] = alphabet[at & @as(usize, if (tail == 2) 0b110000 else 0b111100)];
    }
    while (n % 4 != 0) : (n += 1) clean[n] = '=';
    const d = std.base64.standard.Decoder;
    const size = d.calcSizeForSlice(clean[0..n]) catch return null;
    const out = try a.alloc(u8, size);
    d.decode(out, clean[0..n]) catch return null;
    return if (out.len > 0) out else null;
}
