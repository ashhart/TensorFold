# Usability implementation

This private candidate uses original TensorFold requirements and code, with synthetic
profiles and fake services for qualification; it does not publish an application.

KaTeX 0.19.0 (MIT) is the only added runtime dependency: bounded local equation layout
and accessible MathML are preferable to a bespoke TeX parser. Its fonts and CSS are
bundled, and its unmodified MIT notice ships in third-party-notices; remote content, trusted commands and shared macros are disabled. The renderer
keeps malformed or oversized equations readable as source.

Search queries and excerpts are ephemeral. Global conversation search excludes
incognito, phone, Google and bot histories; host-bound current-chat search is explicit.
Results navigate by exact message identity, with saved-history-only matches labelled
when there is no display anchor. Scan limits are reported rather than called complete.

Document indexing is local and encrypted with the existing profile storage protection.
Original files remain the authority; digest and passage coordinates are checked again
before evidence opens. PDF extraction cannot reliably identify pages, so it uses an
explicit unmapped extraction label. File failures and unsupported inputs remain visible.
