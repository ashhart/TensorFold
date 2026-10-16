package expo.modules.tensorfoldpinnednetwork

import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.MediaType.Companion.toMediaType
import java.net.URI
import java.security.MessageDigest
import java.security.cert.CertificateException
import java.security.cert.X509Certificate
import java.util.concurrent.TimeUnit
import javax.net.ssl.SSLContext
import javax.net.ssl.X509TrustManager

/** Shared Android implementation exercised through native calls and real JVM TLS tests. */
object PinnedNetwork {
  fun request(address: String, method: String, headers: Map<String, String>, body: String?, pin: String): Map<String, Any> {
      val url = URI(address)
      require(url.scheme == "https" && url.host != null && url.userInfo == null && pin.matches(Regex("[a-fA-F0-9]{64}")) && method in listOf("GET", "POST")) { "Invalid private host request." }
      require((body?.toByteArray(Charsets.UTF_8)?.size ?: 0) <= 65536) { "The request is too large." }
      val expected = pin.lowercase()
      val trust = object : X509TrustManager {
        override fun getAcceptedIssuers(): Array<X509Certificate> = emptyArray()
        override fun checkClientTrusted(chain: Array<X509Certificate>, authType: String) { throw CertificateException("Client certificates are not supported.") }
        override fun checkServerTrusted(chain: Array<X509Certificate>, authType: String) {
          val certificate = chain.firstOrNull() ?: throw CertificateException("The host supplied no certificate.")
          certificate.checkValidity()
          val digest = MessageDigest.getInstance("SHA-256").digest(certificate.encoded).joinToString("") { "%02x".format(it.toInt() and 255) }
          if (digest != expected) throw CertificateException("The host certificate changed; verify it on your Mac and pair again.")
        }
      }
      val context = SSLContext.getInstance("TLS")
      context.init(null, arrayOf(trust), null)
      val client = OkHttpClient.Builder().sslSocketFactory(context.socketFactory, trust)
        // Keep the normal hostname/IP verifier; never install a trust-all verifier.
        .followRedirects(false).followSslRedirects(false).callTimeout(15, TimeUnit.SECONDS).build()
      val request = Request.Builder().url(address)
      headers.forEach { (name, value) -> request.header(name, value) }
      if (method == "POST") request.post((body ?: "{}").toRequestBody("application/json".toMediaType())) else request.get()
      return client.newCall(request.build()).execute().use { response ->
        require(response.code !in 300..399) { "Redirects are not allowed for a paired private host." }
        val bytes = response.body?.byteStream()?.use { input ->
          val output = java.io.ByteArrayOutputStream(); val buffer = ByteArray(8192)
          while (true) {
            val count = input.read(buffer); if (count < 0) break
            require(output.size() + count <= 8 * 1024 * 1024) { "The host response is too large." }
            output.write(buffer, 0, count)
          }
          output.toByteArray()
        } ?: ByteArray(0)
        mapOf("status" to response.code, "headers" to response.headers.names().associateWith { response.header(it) ?: "" }, "body" to bytes.toString(Charsets.UTF_8))
      }

  }
}
