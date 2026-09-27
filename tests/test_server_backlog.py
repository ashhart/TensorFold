"""A burst of clients connecting at once waits in the listen queue instead of being dropped."""

import socket

from tensorfold.server.http import Server


def test_a_burst_of_connections_waits_in_the_backlog():
    # nothing accepts yet: every connection must still complete its handshake in the listen queue
    server = Server(("127.0.0.1", 0), lambda *args: None)
    clients = []
    try:
        for _ in range(64):
            client = socket.create_connection(server.server_address, timeout=1.0)
            clients.append(client)
        assert len(clients) == 64
    finally:
        for client in clients:
            client.close()
        server.server_close()
