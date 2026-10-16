# Portable bot recipes

Open **Bots**, then choose **Import recipe** or **Export recipe**. Review the full JSON and acknowledge the instructions before importing or saving a file. Export uses a new filename and does not overwrite an existing file.

A version 1 recipe contains only a name, responsibility, instructions and responsibility type. It excludes model and account settings, saved credentials, folder grants, VM identities, transcripts, tool selections and saved approvals. User-authored instruction text is included and can still contain private information. Recipes are JSON text, at most 64 KiB; unsupported versions and additional fields are rejected.

```json
{
  "format": "tensorfold.bot-recipe",
  "version": 1,
  "bot": {
    "name": "Evidence helper",
    "role": "Check the evidence for a conclusion",
    "instructions": "Cite sources and describe uncertainty.",
    "lane": "research"
  }
}
```

Choose an available model and account on this Mac and the unassigned workspace when importing. TensorFold creates a new paused, read-only bot, with no project or folder grant and no VM setup. No chat or routine starts. Review the bot's local setup separately before resuming it; existing bots and chats are not modified.

Instructions are text and can contain personal or confidential information. Common credential and private home-folder patterns are rejected, but these checks do not detect every secret, encoded value or personal detail. Review every line yourself and keep the exported file private until you intend to share it. Importing a recipe does not make its instructions trustworthy.

Recipe management is local to the desktop host. Paired remote clients cannot import or export recipes; browser demo mode previews/imports synthetic state but does not save native files.
