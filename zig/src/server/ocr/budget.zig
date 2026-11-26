//! A request's page budget and image-encode budget (docs/api.md OCR: Pages and budgets).
const pdf = @import("../pdf.zig");
const types = @import("types.zig");
const Fault = types.Fault;

/// The most images one request may encode.
pub const max_count = 256;
/// The encoded bytes every request may hold, and what each selected page adds.
pub const base_bytes = 16 << 20;
pub const page_bytes = 2 << 20;
/// Encoded pixels may reach this many times the request's page pixels.
pub const pixel_factor = 4;
/// The unreadable-page reason for any encode limit.
pub const over_message = "too many image regions";

/// Pages a request may still read: images count 1 each, PDF pages as selected.
pub const Pages = struct {
    left: u32,

    /// ``TENSORFOLD_MAX_PDF_PAGES`` (default and ceiling 16); ``--vision-max-images`` does not apply.
    pub fn init() Pages {
        return .{ .left = pdf.maxPagesLimit(0) };
    }

    /// Takes ``n`` pages; false, and nothing taken, when fewer than ``n`` are left.
    pub fn take(p: *Pages, n: u32) bool {
        if (n > p.left) return false;
        p.left -= n;
        return true;
    }
};

/// The crops and visualisations a request has encoded, against its limits.
pub const Encodes = struct {
    count: u32 = 0,
    pixels: u64 = 0,
    bytes: u64 = 0,
    limit_pixels: u64,
    max_bytes: u64,

    /// The budget of a request of ``pages`` pages holding ``page_pixels`` pixels in all.
    pub fn init(page_pixels: u64, pages: usize) Encodes {
        return .{ .limit_pixels = page_pixels *| pixel_factor, .max_bytes = base_bytes +| @as(u64, pages) *| page_bytes };
    }

    /// Reserves one ``w`` x ``h`` encode; faults unreadable when it would pass the count or pixel limit.
    pub fn before(e: *Encodes, fault: *Fault, w: u32, h: u32) types.Faulted!void {
        const px = @as(u64, w) * h;
        if (e.count + 1 > max_count or e.pixels +| px > e.limit_pixels) return over(fault);
        e.count += 1;
        e.pixels += px;
    }

    /// Counts one finished encode of ``bytes_len`` bytes; faults unreadable once the total passes ``max_bytes``.
    pub fn after(e: *Encodes, fault: *Fault, bytes_len: usize) types.Faulted!void {
        e.bytes += bytes_len;
        if (e.bytes > e.max_bytes) return over(fault);
    }
};

fn over(fault: *Fault) types.Faulted {
    fault.* = .{ .class = .unreadable, .message = over_message };
    return error.Faulted;
}
