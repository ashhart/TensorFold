package expo.modules.tensorfoldpinnednetwork

import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.tls.HandshakeCertificates
import okhttp3.tls.HeldCertificate
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

class PinnedNetworkTest {
  private fun pin(certificate: HeldCertificate) = MessageDigest.getInstance("SHA-256").digest(certificate.certificate.encoded).joinToString("") { "%02x".format(it.toInt() and 255) }
  private fun withServer(certificate: HeldCertificate, run: (MockWebServer) -> Unit) {
    val server = MockWebServer()
    val tls = HandshakeCertificates.Builder().heldCertificate(certificate).build()
    server.useHttps(tls.sslSocketFactory(), false)
    server.start()
    try { run(server) } finally { server.shutdown() }
  }
  private fun request(server: MockWebServer, pin: String) = PinnedNetwork.request(server.url("/v1/device-status").toString(), "GET", emptyMap(), null, pin)

  @Test fun matchingPinAndHostReturnsActualResponse() {
    val certificate = HeldCertificate.Builder().commonName("fixture").addSubjectAlternativeName("localhost").build()
    withServer(certificate) { server ->
      server.enqueue(MockResponse().setBody("{\"paired\":true}"))
      assertEquals("{\"paired\":true}", request(server, pin(certificate))["body"])
      assertEquals("/v1/device-status", server.takeRequest(1, TimeUnit.SECONDS)!!.path)
    }
  }
  @Test fun changedPinIsRejectedBeforeAnyHttpRequest() {
    val certificate = HeldCertificate.Builder().addSubjectAlternativeName("localhost").build()
    withServer(certificate) { server ->
      assertThrows(Exception::class.java) { request(server, "0".repeat(64)) }
      assertEquals(0, server.requestCount)
    }
  }
  @Test fun matchingPinDoesNotBypassHostnameCheck() {
    val certificate = HeldCertificate.Builder().addSubjectAlternativeName("192.168.1.2").build()
    withServer(certificate) { server ->
      assertThrows(Exception::class.java) { request(server, pin(certificate)) }
      assertEquals(0, server.requestCount)
    }
  }
  @Test fun matchingPinDoesNotBypassExpiryCheck() {
    val certificate = HeldCertificate.Builder().addSubjectAlternativeName("localhost").validityInterval(1_577_836_800_000L, 1_609_459_200_000L).build()
    withServer(certificate) { server ->
      assertThrows(Exception::class.java) { request(server, pin(certificate)) }
      assertEquals(0, server.requestCount)
    }
  }
  @Test fun pinnedHostRedirectsAreRejected() {
    val certificate = HeldCertificate.Builder().addSubjectAlternativeName("localhost").build()
    withServer(certificate) { server ->
      server.enqueue(MockResponse().setResponseCode(302).setHeader("Location", "https://example.invalid"))
      assertThrows(IllegalArgumentException::class.java) { request(server, pin(certificate)) }
      assertEquals(1, server.requestCount)
    }
  }
}
