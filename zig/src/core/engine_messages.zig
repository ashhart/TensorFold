//! What the server says when an engine refuses a request or ends it for a reason the client can act on.

/// What a request refused under critical memory pressure says (a 503).
pub const pressure_message = "this Mac is under critical memory pressure: the server takes no new requests until it eases; retry shortly";

/// What a reply refused or ended for `Stale` says, a submit's error and a queued job's failure alike.
pub const stale_message = "a Living Weights delete could not be put back on the device: send the delete again to retry it, or restart the server";
