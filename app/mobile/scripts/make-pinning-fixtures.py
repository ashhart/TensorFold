"""Generate public-only certificate fixtures; private signing keys remain in memory."""
import base64
import datetime
import hashlib
import ipaddress
import json
from pathlib import Path
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID

def certificate(expired=False):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'TensorFold fixture')])
    builder = x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key()).serial_number(x509.random_serial_number())
    builder = builder.not_valid_before(datetime.datetime(2010, 1, 1, tzinfo=datetime.timezone.utc))
    builder = builder.not_valid_after(datetime.datetime(2019 if expired else 2035, 1, 1, tzinfo=datetime.timezone.utc))
    builder = builder.add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]), critical=False)
    builder = builder.add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
    builder = builder.add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
    data = builder.sign(key, hashes.SHA256()).public_bytes(serialization.Encoding.DER)
    return {'certificate': base64.b64encode(data).decode(), 'pin': hashlib.sha256(data).hexdigest()}

valid = certificate()
expired = certificate(True)
cases = [
    {'name': 'Exact pin and matching IP', 'host': '127.0.0.1', 'expected': True, **valid},
    {'name': 'Wrong certificate pin', 'host': '127.0.0.1', 'expected': False, **valid, 'pin': '00' * 32},
    {'name': 'Matching pin but wrong IP', 'host': '127.0.0.2', 'expected': False, **valid},
    {'name': 'Matching pin but expired certificate', 'host': '127.0.0.1', 'expected': False, **expired},
]
destination = Path(__file__).resolve().parent.parent / 'src' / 'fixtures' / 'pinning.json'
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(cases, indent=2) + '\n')
