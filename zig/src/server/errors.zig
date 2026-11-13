//! A request's refusal: Python's RequestError family, or another exception's bare message.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Kind = enum {
    /// RequestError: 400, ``{"message", "type": "invalid_request_error"}``.
    request,
    /// ContextLengthError: as request, plus ``param`` and ``code: context_length_exceeded``.
    context_length,
    /// Missing optional runtime library for an enabled feature: 501.
    unimplemented,
    /// CapacityError: 503, retry shortly.
    capacity,
    /// Any other exception (ValueError, KeyError...): 400 with ``{"message"}`` only.
    other,
    /// An exception while the reply runs: 500 with ``{"message"}``, or a stream's server_error event.
    server,
};

pub const context_limit = "This server's maximum context length is"; // OpenAI's wording, which clients match to compact

pub const Refused = error{ Refused, OutOfMemory };

/// The request's allocator and the refusal it ended with.
pub const Cx = struct {
    a: Allocator,
    kind: Kind = .request,
    message: []const u8 = "",
    /// The bounded wait code a retryable refusal carries (wait_reason.zig); null on a request refusal.
    code: ?[]const u8 = null,
    /// The request field a refusal names (a Responses socket's error event reads it; no HTTP reply does).
    param: ?[]const u8 = null,
    /// The request's ``chat_template_kwargs.clear_thinking``, over the server's (GLM-5.3 templates only).
    clear_thinking: ?bool = null,
    /// The effort the request itself named (no server default): a thinking-off prompt renders only this one.
    named_effort: ?[]const u8 = null,
    /// The request controls a native prompt renderer must retain in every prefix probe.
    template_fields: @import("json").Value = .null,
    /// The request prompt-cache salt, retained by server-created compaction requests; 0: none.
    cache_salt: u64 = 0,

    pub fn fail(cx: *Cx, kind: Kind, comptime fmt: []const u8, args: anytype) Refused {
        cx.kind = kind;
        cx.message = try std.fmt.allocPrint(cx.a, fmt, args);
        return error.Refused;
    }

    pub fn refuse(cx: *Cx, message: []const u8) Refused {
        cx.kind = .request;
        cx.message = message;
        return error.Refused;
    }

    /// A capacity refusal that names its wait: the engine had no room, so the request can be sent again.
    pub fn wait(cx: *Cx, code: []const u8, comptime fmt: []const u8, args: anytype) Refused {
        cx.code = code;
        return cx.fail(.capacity, fmt, args);
    }

    pub fn other(cx: *Cx, message: []const u8) Refused {
        cx.kind = .other;
        cx.message = message;
        return error.Refused;
    }

    pub fn status(cx: *const Cx) u16 {
        return switch (cx.kind) {
            .unimplemented => 501,
            .capacity => 503,
            .server => 500,
            else => 400,
        };
    }
};
