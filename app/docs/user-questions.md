# Questions during a run

Local API-model chats can ask one concise clarification with up to three choices and a typed-answer field.
The run waits until you explicitly send an answer or skip the question; no choice is selected automatically.
An answer supplies task information and never changes tool permissions, approval mode or permission to send messages.
Read-only conversations remain read-only after answering.

Questions and answers belong to the exact chat, run and timeline item that created them, and use the same encrypted transcript storage as other conversation content.
The model resumes only after the answer or skip has been saved.
An answer cannot resolve another run's question, and a question can resolve only once.
Incognito conversation persistence follows the app's existing incognito policy.

Stop, archive, application shutdown and local-data erasure interrupt pending questions.
A restarted app marks unfinished questions interrupted; it never replays an old answer or restores an executable resolver.
The run's existing wall-clock limit also applies while it waits for an answer.
Interrupted questions remain visible as history and require a new turn to continue.

This interaction is available to the native local API-model loop, including isolated VM bots as a non-host interaction.
Cloud planning routes, third-party agent CLIs and remote-browser answer controls do not gain a question translator or expanded execution permissions.
The model chooses when clarification is needed; the feature does not guarantee every model asks the right question.
