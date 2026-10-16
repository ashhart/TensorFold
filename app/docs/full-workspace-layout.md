# Full workspace layouts

Full has a Chat layout and a Complete workspace layout. Chat layout keeps New chat, recent chats, Models, Bots, Settings and recorded work activity in reach. Choose **Complete workspace** to return to projects, developer views and your existing work panes.

This preference is saved on this device, separately from model settings. Existing installations without the new preference keep the complete workspace. A fresh Full welcome flow opens an idle chat and selects Chat layout unless you already chose a layout; reopening the welcome guide preserves your choice. If the preference cannot be saved, the current session still changes and TensorFold explains that it may not survive restart.

Changing layouts does not change a conversation's model, workspace, permissions or grants. It does not start inference, load or unload models, cancel work or stop services. Hidden technical panes remain mounted and keep their state. Inline questions, approvals, errors, edited files, model choice and Stop remain available in the conversation. Minimal continues to manage the local server in the menu bar and uses its existing model/API controls.

The built-in TensorFold engine can load a selected model on first send. Initial loading can take longer than later answers. A matching built-in model in its actual starting state displays **Loading model** while the run waits; once ready, the normal thinking/generation display resumes. Other API/CLI providers and remote execution do not borrow that status from the local engine, and model throughput/timing measurements are unchanged.

Use the existing shortcuts in Full: Command/Control+Shift+N opens a new chat, Command/Control+N opens a task, Command/Control+K opens the command palette and Command/Control+, opens Settings. From Chat layout, the sidebar/pane shortcuts first return to the complete workspace; Command/Control+J still opens the terminal for the current task. These are presentation/navigation actions and retain the normal permission checks.

Work activity shows recorded active conversations, rooms, routines and durable jobs, with duplicate linked records combined. Open the item to see questions or approvals, or use **Open full activity** for more detail. The absence of active work does not mean a model or API server has been unloaded.
