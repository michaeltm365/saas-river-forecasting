"""Release downloads preserve local work and reject corrupt payloads."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('download_data', Path(__file__).resolve().parents[1]/'download_data.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)

class CanonicalDownloadTests(unittest.TestCase):
    def run_download(self, root, payload=b'canonical', local='data/model.pt', force=False):
        manifest = {'files':[{'path':'canonical/model.pt','local_path':local,'bytes':9,'sha256':hashlib.sha256(b'canonical').hexdigest()}]}
        def fetch(url, dest, force):
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(payload)
        with patch.object(download,'REPO_ROOT',root), patch.object(download.urllib.request,'urlopen',return_value=io.BytesIO(json.dumps(manifest).encode())), patch.object(download,'_download',side_effect=fetch):
            return download.download_canonical(force)

    def test_download_and_conflict_preservation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.assertEqual(self.run_download(root),0)
            (root/'data/model.pt').write_bytes(b'local work')
            self.assertEqual(self.run_download(root),1)
            self.assertEqual((root/'data/model.pt').read_bytes(),b'local work')
            self.assertEqual(self.run_download(root,force=True),0)

    def test_corrupt_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                self.run_download(Path(d),payload=b'corrupt')

    def test_path_outside_checkout_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'Unsafe manifest path'):
                self.run_download(Path(d),local='../escape')

    def test_previous_hobo_snapshots_never_replace_current_results(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            local='results/paper/baselines/lr_splits.json'
            p=root/local
            p.parent.mkdir(parents=True)
            p.write_bytes(b'calendar targets')
            self.assertEqual(self.run_download(root,local=local,force=True),0)
            self.assertEqual(p.read_bytes(),b'calendar targets')
