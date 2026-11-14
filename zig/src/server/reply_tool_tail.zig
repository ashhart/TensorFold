//! The first byte of a suffix made only of whole call envelopes and whitespace.
const reply = @import("reply_text.zig");

pub fn start(text: []const u8, openers: reply.Openers) ?usize {
    var pos: usize = 0;
    var tail: ?usize = null;
    while (true) {
        var next: ?usize = null;
        var which: usize = 0;
        for (openers.calls, 0..) |c, i| if (reply.findTag(text, pos, c[0])) |at| {
            if (next == null or at < next.?) {
                next = at;
                which = i;
            }
        };
        const at = next orelse return if (reply.pyStrip(text[pos..]).len == 0) tail else null;
        const tags = openers.calls[which];
        const payload = at + tags[0].len;
        if (reply.payloadOf(text[payload..], openers, which, true) != .call) {
            tail = null;
            pos = payload;
            continue;
        }
        const end = reply.findTag(text, payload, tags[1]) orelse return null;
        if (tail == null or reply.pyStrip(text[pos..at]).len > 0) tail = at;
        pos = end + tags[1].len;
    }
}
