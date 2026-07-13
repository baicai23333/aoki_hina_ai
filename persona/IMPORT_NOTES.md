# Persona v1 import notes

Imported on 2026-07-13 from the user-provided `人格v1.txt` and `persona_evidence_cards.jsonl`.

## Runtime mapping

- Role and disclosure rules → `identity.md`
- Conversational voice and length → `tone.md`
- Response flow and emotional support → `interaction_rules.md`
- Interests and optional music metaphors → `topic_anchors.md`
- Identity, privacy, fact and urgent-safety rules → `boundaries.md`
- The 18 original evidence-card JSON objects → `style_evidence_cards.jsonl`
- Source identities and verification decisions → `source_registry.jsonl`
- Official public facts, split into individual claims → `fact_claims.jsonl`

## Normalization decisions

The source persona said both that this is a non-official bot and that, when asked, it should identify itself as Aoki Hina. Those rules conflict. Runtime behavior follows the explicit non-impersonation boundary: it identifies itself as “青木阳菜 bot，一个非官方粉丝创作 AI 角色”, never as the real person.

The imported `PEC-*` cards reference `SRC-01` through `SRC-50`. The evidence-card artifact itself did not contain URLs, page numbers, program timestamps, or screenshots, so no reference was trusted merely because it had a high confidence label.

The original 50-block corpus was later recovered and compared with accessible primary pages. As of 2026-07-14, the registry contains 52 records: 20 verified, 29 unverified, and 3 rejected. Ten imported style cards have enough verified support to be active; eight remain quarantined. The three former compound fact cards were removed and replaced by 17 granular claims backed by two official, fact-eligible pages.

`SRC-40` and `SRC-46` were rejected because the claimed quotations could not be found anywhere in the three-page LisAni interview. `SRC-08` was rejected because Wikipedia had been misclassified as official material. See `SOURCE_AUDIT.md` for the complete snapshot.

Runtime behavior now enforces these distinctions:

- Style guidance can never support a public fact.
- Unverified, stale, or rejected sources never enter model prompts.
- Supported public fact answers are rendered directly from verified claims.
- Missing or unknown source data fails at startup instead of silently loading an empty store.
