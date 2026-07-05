# Loop Detector Fixture Corpus

Derived from the AEON issue #14 repeated-thinking evaluation artifacts.

## Source

- **Original bundle**: `/tmp/aeon_issue14_raw_and_sanitized_artifacts_20260704.zip` on the GB10 operator host.
- **Sample size**: 1,841 unique `think`-arm responses.
- **Hard loop rate**: 4.0% (74/1841, Wilson 95% CI 3.2%–5.0%).
- **Mild-or-hard rate**: 22.8% (420/1841, Wilson 95% CI 21.0%–24.8%).

## Fixture Categories

| Category | Count | Expected Severity | Description |
|---|---|---|---|
| `hard_positives` | 8 | `abort_candidate` | Synthetic reasoning streams reproducing hard loop signals |
| `mild_positives` | 5 | `suspect` or `observe` | Borderline cases that should remain telemetry-only |
| `clean_negatives` | 5 | none | Clean controls from sources with 0% hard loops |

## Privacy

All fixture text is **synthetic**. No raw chain-of-thought (CoT) reasoning text from
the AEON evaluation is committed. Fixtures preserve only:

- Repetition patterns (line/token/suffix cycle counts calibrated to original metrics)
- Derived content-free metrics (unique_token_ratio, ngram5_repeat_ratio, etc.)
- Source attribution and original rule descriptions for traceability

## Regeneration

To regenerate fixtures from the original artifact bundle:

```bash
# 1. Extract sanitized CSVs from the bundle
mkdir -p /tmp/aeon_issue14
cd /tmp/aeon_issue14
unzip /tmp/aeon_issue14_raw_and_sanitized_artifacts_20260704.zip 'sanitized/*' 'README.md'

# 2. Verify raw archive checksum (from bundle README)
sha256sum raw/aeon_guard_loop_eval_20260703_165905_with_code_readme.tar.zst
# Expected: 636f76495444ac8666efb90fa6144f1ccd5b2d8a0e914864e3380939ba58cac9

# 3. Run the fixture generator (selects representative samples and derives synthetic patterns)
python3 scripts/generate_loop_fixtures.py \
  --input /tmp/aeon_issue14/sanitized \
  --output crates/llm-guard-proxy-core/tests/fixtures/loop_detector
```

The generator selects diverse hard-loop rule combinations from
`aeon_hard_thinking_loop_record_locations_20260704.csv`, borderline cases from
`aeon_thinking_loop_record_locations_20260704.csv`, and clean controls from
`aeon_thinking_loop_rates_by_source_20260704.csv`.

## Test Coverage

The integration test file `loop_detector_fixtures.rs` loads each fixture and feeds
its synthetic fragments to `ChannelizedLoopDetector`, asserting that:

1. **Hard positives** produce `LoopSeverity::AbortCandidate` signals
2. **Mild positives** produce at most `LoopSeverity::Suspect` (not abort)
3. **Clean negatives** produce no loop signals at all
