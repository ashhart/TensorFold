export interface RunRecovery {
  title: string
  guidance: string
  action: string
  destination: 'models' | 'connections' | 'security' | null
}

// Classifications only choose user-invoked navigation; they never retry a tool,
// raise permissions, or infer that an interrupted action left no side effects.
export function runRecovery(text: string): RunRecovery {
  if (/(?:\bHTTP\s*(?:401|403)\b|invalid[_ -]api[_ -]key|authentication failed|not authenticated|unauthorized|sign in required)/i.test(text)) return {
    title: 'Model sign-in needs attention', guidance: 'Check this model connection and its account, then continue the conversation.', action: 'Open connections', destination: 'connections',
  }
  if (/(?:\bHTTP\s*429\b|rate.limit|quota.exceeded|insufficient.quota)/i.test(text)) return {
    title: 'Provider limit reached', guidance: 'Check the provider limit or choose an available model before continuing.', action: 'Choose a model', destination: 'models',
  }
  if (/(?:out.of.memory|insufficient.memory|memory allocation failed|failed to allocate|\bOOM\b)/i.test(text)) return {
    title: 'The model needs more memory', guidance: 'Choose a smaller model or close other memory-heavy apps, then try a new request.', action: 'Open models', destination: 'models',
  }
  if (/(?:context[_ -](?:length|window)|maximum context|too many tokens|prompt is too long)/i.test(text)) return {
    title: 'This conversation is too large', guidance: 'Compact this conversation in AI settings or start a new chat with the context you need.', action: 'Open models', destination: 'models',
  }
  if (/(?:ECONNREFUSED|ENOTFOUND|EHOSTUNREACH|fetch failed|failed to fetch|connection refused|network error|endpoint.*unavailable)/i.test(text)) return {
    title: 'The model endpoint is unavailable', guidance: 'Check that the local server is running or check the provider connection.', action: 'Check connection', destination: 'connections',
  }
  if (/(?:timed?\s*out|timeout|did not respond)/i.test(text)) return {
    title: 'The model took too long', guidance: 'Review the activity for completed actions before continuing or choosing another model.', action: 'Open models', destination: 'models',
  }
  if (/(?:denied by|approval.*denied|blocked by.*(?:policy|security)|permission denied)/i.test(text)) return {
    title: 'An action was blocked', guidance: 'Review the denied action and the current permissions before sending a new instruction.', action: 'Review permissions', destination: 'security',
  }
  return { title: 'The run could not finish', guidance: 'Review the details and completed activity before continuing; earlier actions may already have run.', action: '', destination: null }
}
