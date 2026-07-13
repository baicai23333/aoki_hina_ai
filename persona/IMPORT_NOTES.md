# Persona v1 import notes

Imported on 2026-07-13 from the user-provided `人格v1.txt` and `persona_evidence_cards.jsonl`.

## Runtime mapping

- Role and disclosure rules → `identity.md`
- Conversational voice and length → `tone.md`
- Response flow and emotional support → `interaction_rules.md`
- Interests and optional music metaphors → `topic_anchors.md`
- Identity, privacy, fact and urgent-safety rules → `boundaries.md`
- The 18 original evidence-card JSON objects → `style_evidence_cards.jsonl`

## Normalization decisions

The source persona said both that this is a non-official bot and that, when asked, it should identify itself as Aoki Hina. Those rules conflict. Runtime behavior follows the explicit non-impersonation boundary: it identifies itself as “青木阳菜 bot，一个非官方粉丝创作 AI 角色”, never as the real person.

The imported `PEC-*` cards reference `SRC-01` through `SRC-50`, but the supplied artifact does not include URLs, publication details, page numbers, timestamps or the source texts. They are therefore loaded as `AOKI_HINA_PUBLIC_STYLE`, usable for response patterns only. They always have `can_support_fact=false` at runtime. Verified public facts remain in `evidence_cards.jsonl` with direct official URLs.
