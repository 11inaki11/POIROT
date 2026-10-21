# Changelog

## 0.2.0

### Changed
- **The default protocol mode is now `"C1"`.** Each agent votes right after its own
  self-assessment (Phase 1), without peer consultation. To keep the previous
  behaviour (full protocol with peer consultation), pass `mode="C3"`.

### Added
- `mode` parameter in `run_poirot()`, `run_poirot_from_agents()` and `POIROTPipeline`:
  - `"C0"` — vote directly from the session log (no self-assessment phase);
  - `"C1"` — vote after the agent's own self-assessment (default);
  - `"C2"` — vote after reading the self-assessments of all agents;
  - `"C3"` — full protocol with peer consultation.
  Aliases: `"vote_only"`, `"self_assessment"`, `"shared_reports"`, `"full"`.
- `max_parallel_votes` parameter: agents vote concurrently in modes C0–C2.
- The result dictionary includes `"mode"`.
- `poirot.PROTOCOL_MODES` and `poirot.DEFAULT_MODE`.
- Phase 1 returns the context each report was produced in (`"contexts"`).

### Notes
- In modes C0–C2 the vote is requested from a plain chat model with no tools bound,
  so agents cannot re-run their original tools while voting.
- `poirot.__version__` now reports the package version correctly.

## 0.1.0

- First release.
