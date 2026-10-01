"""One mocked native chat/stream/tool flow through the real shared HTTP server."""
from contextlib import contextmanager
import http.client
import json
from pathlib import Path
import threading
from unittest.mock import patch, MagicMock

import pytest
from tensorfold.cuda.server import Server, make_handler
from tensorfold.cuda.reply_text import StreamDecoder
from tensorfold.families.deepseek_v4.cuda.app import DeepSeekApp, NativeTokenizer, DeepSeekTemplate
from tensorfold.server.cancellation import RequestCancelled

SPECIALS = ('<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>', '<｜User｜>',
            '<｜Assistant｜>', '<think>', '</think>', '｜DSML｜')
TOOLS = [{'type': 'function', 'function': {'name': 'weather', 'parameters': {
    'type': 'object', 'properties': {'city': {'type': 'string'}}}}}]
CALL = ('<｜DSML｜tool_calls>\n<｜DSML｜invoke name="weather">\n'
        '<｜DSML｜parameter name="city" string="true">北京</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n</｜DSML｜tool_calls>')


class Session:
    vocab_size, eos = 263, 257

    def __init__(self):
        self.encoded = []

    def encode(self, text, *, rendered=False):
        self.encoded.append((text, rendered))
        out = []
        while text:
            found = next((i for i, s in enumerate(SPECIALS) if rendered and text.startswith(s)), None)
            if found is not None:
                out.append(256 + found)
                text = text[len(SPECIALS[found]):]
            else:
                out.extend(text[0].encode())
                text = text[1:]
        return out

    def token_text(self, token):
        return bytes([token]) if token < 256 else SPECIALS[token - 256].encode()


class Engine:
    eos, context_window, concurrent = (257,), 8192, False

    def __init__(self):
        self.session = Session()
        self.text = 'Hello 北京'
        self.calls = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, stop_eos=True):
        call = {'prompt': prompt, 'generated': 0}
        self.calls.append(call)
        for token in (self.session.encode(self.text, rendered=True) + [257])[:max_tokens]:
            call['generated'] += 1
            if on_tokens([token]) or (stop_eos and token == 257):
                break
        return {'generated': call['generated'], 'cached': 0, 'drafts': False}


@contextmanager
def http_app(app):
    server = Server(('127.0.0.1', 0), make_handler(app))
    worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
    worker.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)


def request(port, path, body=None):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    try:
        connection.request('GET' if body is None else 'POST', path,
                           body=None if body is None else json.dumps(body),
                           headers={'Content-Type': 'application/json'})
        response = connection.getresponse()
        return response.status, response.read().decode()
    finally:
        connection.close()


def test_native_tokenizer_golden_render_and_utf8_stream():
    session = Session()
    tok = NativeTokenizer(session)
    fixtures = Path(__file__).parent / 'fixtures/deepseek_v4'
    case = json.loads((fixtures / 'test_input_1.json').read_text())
    rendered = DeepSeekTemplate().render(case['messages'], tools=case['tools'], enable_thinking=True)
    # The fixture is history ending in an assistant message; serving adds the
    # next generation prefix, using the existing DeepSeekTokenizer convention.
    assert rendered == (fixtures / 'test_output_1.txt').read_text() + '<｜Assistant｜><think>'
    ids = tok.encode(rendered, add_special_tokens=False).ids
    assert session.encoded[-1] == (rendered, True)
    assert ids[0] == 256 and tok.decode(ids, skip_special_tokens=False) == rendered
    decoder = StreamDecoder(tok)
    for token in tok.encode(' 北京🚀', add_special_tokens=False).ids:
        shown = decoder.add([token])
        assert ' 北京🚀'.startswith(shown) and '\ufffd' not in shown
    assert decoder.final() == ' 北京🚀'
    assert tok.token_to_id('</think>') == 261 and tok.token_to_id('<not-a-token>') is None


def test_shared_http_chat_stream_tools_followup_and_capacity(tmp_path):
    engine = Engine()
    app = DeepSeekApp(engine, tmp_path, 'deepseek-v4-flash', sampling={'temperature': 0})
    body = {'messages': [{'role': 'user', 'content': 'Hi'}], 'max_tokens': 512}
    with http_app(app) as port:
        assert json.loads(request(port, '/v1/models')[1])['data'][0]['id'] == 'deepseek-v4-flash'
        assert json.loads(request(port, '/health')[1])['context_length'] == 8192
        code, raw = request(port, '/v1/chat/completions', body)
        assert code == 200 and json.loads(raw)['choices'][0]['message']['content'] == 'Hello 北京'
        assert engine.session.encoded[0][0].endswith('<｜Assistant｜></think>')
        engine.text = 'Consider</think>\n\nHello 北京'
        code, raw = request(port, '/v1/chat/completions', {**body, 'reasoning_effort': 'medium', 'stream': True})
        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
        deltas = [c['choices'][0]['delta'] for c in chunks]
        assert code == 200 and raw.endswith('data: [DONE]\n\n')
        assert ''.join(d.get('reasoning_content', '') for d in deltas) == 'Consider'
        assert ''.join(d.get('content', '') for d in deltas) == 'Hello 北京'
        engine.text = CALL
        code, raw = request(port, '/v1/chat/completions', {**body, 'tools': TOOLS,
                             'stream': True, 'parallel_tool_calls': False, 'tool_choice': 'required'})
        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
        deltas = [c['choices'][0]['delta'] for c in chunks]
        assert code == 200 and not any('DSML' in d.get('content', '') for d in deltas)
        call = next(d['tool_calls'][0] for d in deltas if d.get('tool_calls'))
        assert call['function']['name'] == 'weather'
        assert json.loads(call['function']['arguments']) == {'city': '北京'}
        assert chunks[-1]['choices'][0]['finish_reason'] == 'tool_calls'
        engine.text = 'Sunny'
        messages = [*body['messages'], {'role': 'assistant', 'content': None, 'tool_calls': [call]},
                    {'role': 'tool', 'tool_call_id': call['id'], 'content': 'Sunny in Beijing'}]
        code, raw = request(port, '/v1/chat/completions', {**body, 'messages': messages, 'tools': TOOLS})
        assert code == 200 and json.loads(raw)['choices'][0]['message']['content'] == 'Sunny'
        assert any('<tool_result>Sunny in Beijing</tool_result>' in text for text, _ in engine.session.encoded)
        before = len(engine.calls)
        code, raw = request(port, '/v1/chat/completions', {**body, 'max_tokens': 8192, 'stream': True})
        assert code == 400 and 'exceed' in raw and len(engine.calls) == before


def test_stop_and_disconnect_reuse_shared_lifecycle(tmp_path):
    engine = Engine()
    app = DeepSeekApp(engine, tmp_path, 'deepseek-v4-flash', sampling={'temperature': 0})
    engine.text = 'Hello STOP hidden'
    body = {'messages': [{'role': 'user', 'content': 'Hi'}], 'max_tokens': 64, 'stop': 'STOP'}
    result = app.run(body, True, lambda _: True)
    assert result['content'] == 'Hello ' and result['finish'] == 'stop'
    with pytest.raises(RequestCancelled):
        app.run({**body, 'stop': None}, True, lambda _: False)
    assert engine.calls[-1]['generated'] == 1


def test_http_shutdown_and_bind_failure_release_native_engine(tmp_path):
    from tensorfold.cuda import http
    engine = Engine()
    engine.close = MagicMock()
    app = DeepSeekApp(engine, tmp_path, 'deepseek-v4-flash')
    server = MagicMock()
    server.serve_forever.side_effect = KeyboardInterrupt
    with patch('signal.signal'), patch.object(http, 'Server', return_value=server):
        http.serve(app, '127.0.0.1', 0)
    engine.close.assert_called_once()
    server.server_close.assert_called_once()
    engine.close.reset_mock()
    with patch('signal.signal'), patch.object(http, 'Server', side_effect=OSError('occupied')):
        with pytest.raises(OSError, match='occupied'):
            http.serve(app, '127.0.0.1', 0)
    engine.close.assert_called_once()
