"""Exercise fail-closed validation of the portable evidence bundle."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import verify


class EvidenceTests(unittest.TestCase):
    """Keep actual retained evidence as the valid control for each mutation."""

    def setUp(self) -> None:
        """Copy only small offline evidence, without runtime or model files."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = Path(__file__).resolve().parent
        for name in (
            'provenance.json', 'qualification.json', 'evidence-forward.json',
            'evidence-reverse.json', 'requests-forward.jsonl', 'requests-reverse.jsonl',
        ):
            shutil.copyfile(source / name, self.root / name)

    def change_request(self, field: str, value: object, *, arm: str = "candidate",
                       warmup: bool = False) -> None:
        """Mutate a row and renew the container hash to test semantic validation."""
        path = self.root / 'requests-forward.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        row = next(r['request'] for r in rows
                   if r['arm'] == arm and (r['phase'] == 'warmup') == warmup)
        row[field] = value
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        receipt_path = self.root / 'evidence-forward.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['requests']['sha256'] = verify.sha(path)
        receipt_path.write_text(json.dumps(receipt))

    def test_valid_evidence(self) -> None:
        """The control reconstructs all requests and both pairing populations."""
        result = verify.verify(self.root)
        self.assertEqual(result['requests'], 624)
        for order in result['orders'].values():
            self.assertEqual(order['paired_reset_requests'], 104)
            self.assertEqual(order['paired_python_measured_requests'], 102)
            self.assertEqual(order['paired_python_warmup_requests'], 2)
            self.assertEqual(order['paired_python_all_requests'], 104)

    def test_output_mismatch(self) -> None:
        """An otherwise valid fingerprint mismatch fails pairing."""
        self.change_request('output_sha256', '0' * 64)
        with self.assertRaisesRegex(ValueError, 'paired output'):
            verify.verify(self.root)

    def test_python_warmup_output_mismatch(self) -> None:
        """A Python warmup mismatch fails parity without becoming a timed cell."""
        self.change_request('output_sha256', '0' * 64, arm='python', warmup=True)
        with self.assertRaisesRegex(ValueError, 'paired output'):
            verify.verify(self.root)

    def test_body_hash_mismatch(self) -> None:
        """A rehashed envelope cannot hide a contradictory literal body digest."""
        self.change_request('request_body_sha256', '0' * 64)
        with self.assertRaisesRegex(ValueError, 'body hash'):
            verify.verify(self.root)

    def test_cache_mismatch(self) -> None:
        """Native full-N cache evidence cannot silently become a cold result."""
        self.change_request('cached_tokens', 0)
        with self.assertRaisesRegex(ValueError, 'cache state'):
            verify.verify(self.root)

    def test_incomplete_response(self) -> None:
        """Partial SSE output is not accepted as a timing measurement."""
        self.change_request('complete', False)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            verify.verify(self.root)

    def test_missing_record(self) -> None:
        """Request count is enforced independently of a correct file digest."""
        path = self.root / 'requests-forward.jsonl'
        path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
        receipt_path = self.root / 'evidence-forward.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['requests']['sha256'] = verify.sha(path)
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'count drift'):
            verify.verify(self.root)

    def test_wrong_target_repository(self) -> None:
        """A revision alone cannot disguise an incorrect model repository owner."""
        path = self.root / 'provenance.json'
        receipt = json.loads(path.read_text())
        receipt['models'][0]['repo'] = 'TensorFold/Qwen3.8-27B-MLX-4bit'
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'model identity drift'):
            verify.verify(self.root)

    def test_flag_replacement(self) -> None:
        """Five entries alone cannot substitute a different runtime flag."""
        path = self.root / 'provenance.json'
        receipt = json.loads(path.read_text())
        flags = receipt['flag_values']['candidate']
        flags['UNKNOWN_FLAG'] = flags.pop('TF_PR608_EXACT_CACHE_LOGITS')
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'flags drift'):
            verify.verify(self.root)


if __name__ == '__main__':
    unittest.main()
