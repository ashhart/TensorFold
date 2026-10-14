package expo.modules.tensorfoldpinnednetwork

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

class TensorFoldPinnedNetworkModule : Module() {
  override fun definition() = ModuleDefinition {
    Name("TensorFoldPinnedNetwork")
    AsyncFunction("request") { address: String, method: String, headers: Map<String, String>, body: String?, pin: String ->
      PinnedNetwork.request(address, method, headers, body, pin)
    }
  }
}
