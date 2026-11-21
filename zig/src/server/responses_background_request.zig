const json = @import("json");
const errors = @import("errors.zig");
const Cx = errors.Cx;
const Value = json.Value;

/// What a background request needs: a boolean, a stored response to poll, and no stream on the POST.
pub fn validate(cx: *Cx, body: Value, store: bool) errors.Refused!void {
    if (body.get("background").? != .bool) return cx.refuse("background must be a boolean");
    if (!store) return cx.refuse("background: true needs store: true (the default): GET /v1/responses/{id} polls the stored response");
    if (json.truthyField(body, "stream")) return cx.refuse("background: true with stream: true is not supported: send stream: false and poll GET /v1/responses/{id}");
}
