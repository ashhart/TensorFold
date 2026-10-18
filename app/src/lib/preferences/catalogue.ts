import { normalizeLanguage } from '../../../shared/languages'
import { packedText } from './language-packs'
export const english = {
  'common.newChat': 'New chat', 'common.newTask': 'New task', 'common.search': 'Search messages', 'common.settings': 'Settings', 'common.save': 'Save changes', 'common.cancel': 'Cancel', 'common.stop': 'Stop', 'common.close': 'Close', 'common.copy': 'Copy', 'common.unavailable': 'Text unavailable',
  'approval.allow': 'Allow', 'approval.deny': 'Deny', 'approval.waiting': 'Waiting for your approval', 'recovery.retry': 'Try again', 'recovery.stopped': 'Stopped', 'model.choose': 'Choose a model', 'model.loading': 'Loading model…', 'model.unavailable': 'This model is unavailable.',
  'preferences.title': 'Language and reading comfort', 'preferences.description': 'Choose the language, reading direction and spacing for this profile.',
  'preferences.language': 'Interface language', 'preferences.english': 'English', 'preferences.spanish': 'Español',
  'preferences.coverage': 'Spanish covers preferences, App settings, setup, model capability checks, approvals and run recovery; other screens use English. Conversation content is never translated.',
  'preferences.direction': 'Layout direction', 'preferences.auto': 'Automatic for interface language', 'preferences.ltr': 'Left to right', 'preferences.rtl': 'Right to left',
  'preferences.directionHelp': 'Messages and text inputs follow their own writing direction. Code stays left to right.',
  'preferences.chatScale': 'Conversation text size', 'preferences.codeScale': 'Code text size', 'preferences.scale': '{percent}%',
  'preferences.density': 'Spacing', 'preferences.comfortable': 'Comfortable', 'preferences.compact': 'Compact',
  'preferences.preview': 'Reading preview', 'preferences.previewText': 'Your messages can mix languages, citations and file paths without changing their text.',
  'preferences.sampleCode': 'const answer = 42;', 'preferences.mixedSample': 'English / العربية / 日本語 / Русский',
  'preferences.saved': 'Reading preferences saved for this profile.', 'preferences.failed': 'Reading preferences could not be saved. Your previous choices are still active.', 'preferences.saving': 'Saving…', 'preferences.reset': 'Reset to defaults',
  'preferences.savedWhere': 'Saved with this profile through TensorFold settings.', 'preferences.unsaved': 'Save these choices to apply them to the interface.',
  'app.modeTitle': 'How you use TensorFold', 'app.modeDescription': 'Switch at any time; your models and conversations stay available.', 'app.desktopMode': 'Desktop mode', 'app.minimal': 'Minimal', 'app.minimalDescription': 'A menu-bar server with model downloads and an authenticated local API.', 'app.full': 'Full', 'app.fullDescription': 'The complete Chat, Code and Assistant workspace with local model controls.',
  'app.blocked': 'Save or discard your settings draft before switching modes or reopening setup.',
  'app.layoutTitle': 'Full workspace layout', 'app.layoutDescription': 'Choose how much navigation you see in Full. Models, permissions and running work stay the same; Minimal remains the tray/server manager.', 'app.chatLayout': 'Chat layout', 'app.chatLayoutDescription': 'Conversations first, with Models, Bots, Settings and work activity in reach.', 'app.completeLayout': 'Complete workspace', 'app.completeLayoutDescription': 'The project sidebar, saved work panes and all developer views.',
  'app.layoutSaved': 'Saved on this device separately from your model settings. Return with Complete workspace; your sidebar and pane positions are kept.', 'app.welcomeTitle': 'Welcome guide', 'app.welcomeDescription': 'Revisit model setup and the pointers for your chosen mode.', 'app.openWelcome': 'Open welcome guide',
} as const
export type UiTextKey = keyof typeof english
export const spanish: Record<UiTextKey, string> = {
  'common.newChat': 'Nuevo chat', 'common.newTask': 'Nueva tarea', 'common.search': 'Buscar mensajes', 'common.settings': 'Ajustes', 'common.save': 'Guardar cambios', 'common.cancel': 'Cancelar', 'common.stop': 'Detener', 'common.close': 'Cerrar', 'common.copy': 'Copiar', 'common.unavailable': 'Texto no disponible',
  'approval.allow': 'Permitir', 'approval.deny': 'Rechazar', 'approval.waiting': 'Esperando tu aprobación', 'recovery.retry': 'Intentar de nuevo', 'recovery.stopped': 'Detenido', 'model.choose': 'Elegir un modelo', 'model.loading': 'Cargando modelo…', 'model.unavailable': 'Este modelo no está disponible.',
  'preferences.title': 'Idioma y comodidad de lectura', 'preferences.description': 'Elige el idioma, la dirección de lectura y el espaciado para este perfil.',
  'preferences.language': 'Idioma de la interfaz', 'preferences.english': 'English', 'preferences.spanish': 'Español',
  'preferences.coverage': 'El español cubre las preferencias, los ajustes, la configuración inicial, las comprobaciones de modelos, las aprobaciones y la recuperación de tareas; las demás pantallas usan inglés. Las conversaciones nunca se traducen.',
  'preferences.direction': 'Dirección de la interfaz', 'preferences.auto': 'Automática según el idioma', 'preferences.ltr': 'De izquierda a derecha', 'preferences.rtl': 'De derecha a izquierda',
  'preferences.directionHelp': 'Los mensajes y campos de texto siguen su propia dirección de escritura. El código se muestra de izquierda a derecha.',
  'preferences.chatScale': 'Tamaño del texto de conversación', 'preferences.codeScale': 'Tamaño del código', 'preferences.scale': '{percent}%',
  'preferences.density': 'Espaciado', 'preferences.comfortable': 'Cómodo', 'preferences.compact': 'Compacto',
  'preferences.preview': 'Vista previa de lectura', 'preferences.previewText': 'Tus mensajes pueden mezclar idiomas, citas y rutas de archivos sin cambiar el texto.',
  'preferences.sampleCode': 'const answer = 42;', 'preferences.mixedSample': 'English / العربية / 日本語 / Русский',
  'preferences.saved': 'Preferencias de lectura guardadas para este perfil.', 'preferences.failed': 'No se pudieron guardar las preferencias de lectura. Las opciones anteriores siguen activas.', 'preferences.saving': 'Guardando…', 'preferences.reset': 'Restablecer valores',
  'preferences.savedWhere': 'Se guardan en los ajustes de TensorFold de este perfil.', 'preferences.unsaved': 'Guarda estas opciones para aplicarlas a la interfaz.',
  'app.modeTitle': 'Cómo usas TensorFold', 'app.modeDescription': 'Puedes cambiar cuando quieras; tus modelos y conversaciones seguirán disponibles.', 'app.desktopMode': 'Modo de escritorio', 'app.minimal': 'Mínimo', 'app.minimalDescription': 'Un servidor en la barra de menús con descargas de modelos y una API local autenticada.', 'app.full': 'Completo', 'app.fullDescription': 'El espacio completo de Chat, Código y Asistente con controles de modelos locales.',
  'app.blocked': 'Guarda o descarta el borrador de ajustes antes de cambiar de modo o volver a abrir la guía.',
  'app.layoutTitle': 'Diseño del espacio completo', 'app.layoutDescription': 'Elige cuánta navegación quieres ver en el modo Completo. Los modelos, permisos y tareas en curso siguen igual; el modo Mínimo mantiene el servidor en la bandeja.', 'app.chatLayout': 'Diseño de chat', 'app.chatLayoutDescription': 'Las conversaciones primero, con Modelos, Bots, Ajustes y la actividad del trabajo a mano.', 'app.completeLayout': 'Espacio completo', 'app.completeLayoutDescription': 'La barra de proyectos, los paneles guardados y todas las vistas de desarrollo.',
  'app.layoutSaved': 'Se guarda en este dispositivo por separado de los ajustes del modelo. Vuelve con Espacio completo; se conservan la barra lateral y las posiciones de los paneles.', 'app.welcomeTitle': 'Guía de bienvenida', 'app.welcomeDescription': 'Revisa la configuración del modelo y las indicaciones del modo elegido.', 'app.openWelcome': 'Abrir guía de bienvenida',
}
export type UiTextParams = Record<string, string | number>
export function translate(locale: unknown, key: UiTextKey | string, params: UiTextParams = {}): string {
  const fallback = english[key as UiTextKey]
  // Unknown external keys receive readable English, never a raw internal key.
  const template = (fallback ? packedText(locale, fallback) : undefined) ?? (normalizeLanguage(locale).startsWith('es') ? spanish[key as UiTextKey] : undefined) ?? fallback ?? english['common.unavailable']
  return template.replace(/\{([a-zA-Z][a-zA-Z0-9_]*)\}/g, (_match, name) => Object.hasOwn(params, name) ? `\u2068${String(params[name]).replace(/[\u0000-\u001f\u202a-\u202e\u2066-\u2069]/g, ' ').slice(0, 200)}\u2069` : '')
}
