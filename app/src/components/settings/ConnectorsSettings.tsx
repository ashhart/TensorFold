import { SettingsPane, type SettingsSectionContext } from './shared'
import { EmailConnector } from './connectors/EmailConnector'
import { SlackConnector } from './connectors/SlackConnector'
import { DiscordConnector } from './connectors/DiscordConnector'
import { MattermostConnector } from './connectors/MattermostConnector'
import { MatrixConnector } from './connectors/MatrixConnector'
import { GoogleChatConnector } from './connectors/GoogleChatConnector'
import { DocumentsConnector } from './connectors/DocumentsConnector'
import { BrowserConnector } from './connectors/BrowserConnector'
import { BotControlConnector } from './connectors/BotControlConnector'
import { WhatsAppBridgeConnector } from './connectors/WhatsAppBridgeConnector'

export function ConnectorsSettings({ ctx }: { ctx: SettingsSectionContext }) {
  return (
    <SettingsPane>
      <EmailConnector ctx={ctx} />
      <SlackConnector ctx={ctx} />
      <DiscordConnector ctx={ctx} />
      <MattermostConnector ctx={ctx} />
      <MatrixConnector ctx={ctx} />
      <GoogleChatConnector ctx={ctx} />
      <DocumentsConnector ctx={ctx} />
      <BrowserConnector ctx={ctx} />
      <BotControlConnector ctx={ctx} />
      <WhatsAppBridgeConnector ctx={ctx} />
    </SettingsPane>
  )
}
