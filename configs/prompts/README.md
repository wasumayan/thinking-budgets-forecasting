# Prompt rendering rules (implemented in `src/tbf/prompts.py`)

Placeholders:

- `{metadata}` — SPEC §2.1 item 1, always ending with "Frequency: {freq}. Prediction target period: {h_start} to {h_end}." In `context_mode=none` this is replaced by `"{title} ({units}). Frequency: {freq}. Prediction target period: {h_start} to {h_end}."` only.
- `{calendar}` — comma-separated "YYYY-MM-DD Holiday Name" list or `None`. In `none` mode the whole `## Calendar` section is omitted.
- `{events}` — up to 10 lines `- YYYY-MM-DD [Category] text`, newest first, or `None`. In `none` mode the whole section is omitted. In `shuffled` mode the events come from the mapped window in `shuffle_map.parquet`.
- `{reports_section}` — either empty string, or `## Reports (dated)\n- YYYY-MM-DD [Source] text\n...\n\n` (≤2 items). Omitted in `none` mode; shuffled with events in `shuffled` mode.
- `{history}` — 96 lines `YYYY-MM-DD: value`, oldest first, values formatted with the series `precision`.
- `{prior_forecast}` — 12 lines `YYYY-MM-DD: value` (Chronos-2 median unless `prior` says otherwise), same precision.
- `{h_start}`, `{h_end}` — ISO dates of the first and last target timestamp.
- `{freq_desc}` — one of `daily`, `business days`, `weekly, week ending Saturday`, `weekly`.

The rendered user prompt goes in the `user` turn; `system.txt` in the `system` turn. No few-shot examples. No trailing whitespace changes. `tests/test_prompts.py` snapshots the rendered prompt for fixture window `web__Influenza__2025-09-05` in all three context modes and both setups.
