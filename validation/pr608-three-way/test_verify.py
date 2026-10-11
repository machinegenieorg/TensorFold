"""Exercise fail-closed checks against the retained portable three-way evidence."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import verify


class EvidenceTests(unittest.TestCase):
    """Copy only sanitized evidence, with no model or network dependencies."""

    def setUp(self) -> None:
        """Use the bundled valid evidence as each mutation's control."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = Path(__file__).resolve().parent
        for name in ('provenance.json', 'qualification.json', 'evidence-forward.json',
                     'evidence-reverse.json', 'requests-forward.jsonl', 'requests-reverse.jsonl'):
            shutil.copyfile(source / name, self.root / name)

    def change(self, field: str, value: object, *, cold: bool = False) -> None:
        """Renew the file digest so tests reach the semantic gate."""
        path = self.root / 'requests-forward.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        selected = next(record['request'] for record in rows
                        if record['arm'] == 'candidate'
                        and (record['phase'] == 'uncached_first_use') == cold)
        selected[field] = value
        path.write_text(''.join(json.dumps(record) + '\n' for record in rows))
        receipt_path = self.root / 'evidence-forward.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['requests']['sha256'] = verify.digest(path.read_bytes())
        receipt_path.write_text(json.dumps(receipt))

    def test_valid_control(self) -> None:
        """The complete control has all 104 requests per arm in both orders."""
        result = verify.analyse(self.root)
        self.assertTrue(result['complete'])
        self.assertEqual(set(result['orders']), {'forward', 'reverse'})
        self.assertTrue(all(order['requests_per_arm'] == 104 for order in result['orders'].values()))

    def test_first_use_output_mismatch(self) -> None:
        """Uncached first-use output participates in strict three-way parity."""
        self.change('output_sha256', '0' * 64, cold=True)
        with self.assertRaisesRegex(ValueError, 'parity drift'):
            verify.analyse(self.root)

    def test_changed_decode_rate(self) -> None:
        """The decode check uses actual output count and first/last clocks."""
        self.change('decode_tps', 1.0)
        with self.assertRaisesRegex(ValueError, 'decode_tps arithmetic'):
            verify.analyse(self.root)

    def test_cold_cache_mismatch(self) -> None:
        """An uncached first use must carry observed cache count zero."""
        self.change('cached_tokens', 14, cold=True)
        with self.assertRaisesRegex(ValueError, 'cache count'):
            verify.analyse(self.root)

    def test_literal_body_hash(self) -> None:
        """A rehashed envelope cannot hide an invalid literal-body digest."""
        self.change('request_body_sha256', '0' * 64)
        with self.assertRaisesRegex(ValueError, 'body SHA'):
            verify.analyse(self.root)

    def test_incomplete_response(self) -> None:
        """Partial output cannot become a timing or throughput measurement."""
        self.change('complete', False)
        with self.assertRaisesRegex(ValueError, 'incomplete request'):
            verify.analyse(self.root)

    def test_missing_row(self) -> None:
        """The panel requires its declared row count despite a valid digest."""
        path = self.root / 'requests-forward.jsonl'
        path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
        receipt_path = self.root / 'evidence-forward.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['requests']['sha256'] = verify.digest(path.read_bytes())
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, '312-request'):
            verify.analyse(self.root)

    def test_wrong_model(self) -> None:
        """Model repository owner and revision are both part of provenance."""
        path = self.root / 'provenance.json'
        receipt = json.loads(path.read_text())
        receipt['models'][0]['repo'] = 'TensorFold/Qwen3.8-27B-MLX-4bit'
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'model identity'):
            verify.analyse(self.root)

    def test_wrong_flags(self) -> None:
        """A mislabeled final or stock runtime policy fails closed."""
        path = self.root / 'provenance.json'
        receipt = json.loads(path.read_text())
        receipt['flags']['candidate']['TF_PR608_EXACT_CACHE_SKIP_RESET'] = '0'
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'runtime flag'):
            verify.analyse(self.root)


if __name__ == '__main__':
    unittest.main()
