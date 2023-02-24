import boto3
import logging
from OpenSSL import crypto

def configure_logging(name: str = None, verbose: bool = False):
    """Set up logging."""
    logging_level = logging.DEBUG if verbose else logging.INFO
    root_logger = logging.getLogger(__name__ if name is None else name)
    logging.basicConfig(level=logging_level)
    root_logger.setLevel(logging_level)

    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    formatter = logging.Formatter("%(levelname)-8s %(message)s")
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)
    return root_logger

def cert_gen(key_file: str = "private.key", cert_file: str ="selfsigned.crt"):
    """Create self-signed certificate"""
    # create a key pair
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 2048)
    # create a self-signed cert
    cert = crypto.X509()
    cert.get_subject().C = "US"
    cert.get_subject().ST = "state"
    cert.get_subject().L = "locality"
    cert.get_subject().O = "org"
    cert.get_subject().OU = "orgunit"
    cert.get_subject().CN = "mydomain.com"
    cert.set_serial_number(1)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365*24*60*60) # seconds
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha1')

    with open(key_file, "wt") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k).decode("utf-8"))

    with open(cert_file, "wt") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert).decode("utf-8"))

    logging.info(f"Created key: {key_file}")
    logging.info(f"Created certificate: {cert_file}")

def cert_acm_upload(key_file: str = "private.key", cert_file: str = "selfsigned.crt", region: str = "us-east-1"):
    """Upload certificate to ACM"""
    acm = boto3.client('acm', region_name=region)

    try: 
        with open(key_file, 'rb') as key_file, open(cert_file, 'rb') as cert_file:
            response = acm.import_certificate(
                Certificate=cert_file.read(),
                PrivateKey=key_file.read()
            )
        HTTPStatusCode = response["ResponseMetadata"]["HTTPStatusCode"]
        if HTTPStatusCode == 200:
            arn = response['CertificateArn']
            logging.info(f"\nUploaded certificate to ACM. \nCertificateArn: {arn}\n")
            return arn
        else:
            logging.error(f"\nIssue uploading certificate. Status code {HTTPStatusCode}\n")

    except Exception as e:
        logging.error(e)