package expo.modules.tensorfoldpinnednetwork

import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.net.URI
import java.net.URLDecoder

/** Uses only a disposable explicit QA host; never owner credentials or a personal profile. */
class PrivateBridgeTest {
  @Test fun pairedControlsAndRevocationUseActualAndroidTls() {
    val link = InstrumentationRegistry.getArguments().getString("pairingLink") ?: ""
    assumeTrue("An opt-in disposable host pairing link is required.", link.isNotEmpty())
    val uri = URI(link)
    val fields = uri.rawFragment.split("&").associate { part -> val entry = part.split("=", limit = 2); entry[0] to URLDecoder.decode(entry[1], "UTF-8") }
    val pin = fields.getValue("pin")
    val origin = "https://${uri.host}:${uri.port}"
    fun request(route: String, method: String = "GET", token: String? = null, body: JSONObject? = null) = PinnedNetwork.request(origin + route, method, if (token == null) emptyMap() else mapOf("Authorization" to "Bearer $token"), body?.toString(), pin)
    val claim = JSONObject().put("credential", fields.getValue("pair")).put("clientInfo", "Disposable Android native QA")
    assertThrows(Exception::class.java) { PinnedNetwork.request(origin + "/v1/device-status", "GET", emptyMap(), null, "0".repeat(64)) }
    val paired = request("/v1/auth/pair", "POST", body = claim)
    assertEquals(200, paired["status"])
    val token = JSONObject(paired["body"] as String).getString("token")
    assertEquals(400, request("/v1/auth/pair", "POST", body = claim)["status"])
    assertEquals(200, request("/v1/device-status", token = token)["status"])
    assertEquals(200, request("/v1/snapshot", token = token)["status"])
    assertEquals(404, request("/v1/files", token = token)["status"])
    val message = JSONObject().put("userText", "Native QA fixture instruction")
    assertEquals(202, request("/v1/threads/native-qa-worker/start", "POST", token, message)["status"])
    assertEquals(202, request("/v1/threads/native-qa-worker/guide", "POST", token, message)["status"])
    assertEquals(202, request("/v1/threads/native-qa-worker/cancel", "POST", token, JSONObject())["status"])
    // This fixture revokes its own sessions after stop, proving actual subsequent 401.
    assertEquals(401, request("/v1/device-status", token = token)["status"])
  }
}
