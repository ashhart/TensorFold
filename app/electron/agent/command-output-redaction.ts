import { SECRET_MASK } from './security'
import { redactSecretValues } from './secret-detection'

export interface RedactedCommandStreams {
  stdout: string
  stderr: string
  redactionCount: number
}

export function redactCommandStreams(stdout: string, stderr: string): RedactedCommandStreams {
  const cleanStdout = redactSecretValues(stdout, SECRET_MASK)
  const cleanStderr = redactSecretValues(stderr, SECRET_MASK)
  return {
    stdout: cleanStdout.text,
    stderr: cleanStderr.text,
    redactionCount: cleanStdout.hits.length + cleanStderr.hits.length,
  }
}

export function commandOutputWithRedactionNotice(output: string, redactionCount: number): string {
  if (redactionCount < 1) return output
  const noun = redactionCount === 1 ? 'value' : 'values'
  return `${output}\n\n[TensorFold redacted ${redactionCount} sensitive ${noun} from command output.]`
}
