# Source audit

Audit date: 2026-07-14

## Snapshot

| Item | Count |
|---|---:|
| Registry records | 52 |
| Verified | 20 |
| Unverified | 29 |
| Rejected | 3 |
| Fact-eligible sources | 2 |
| Style-eligible sources | 17 |
| Active granular fact claims | 17 |
| Active imported style cards | 10 |
| Quarantined imported style cards | 8 |
| Original Hina Bot policy cards | 4 |

The registry has two dedicated official fact sources plus the original `SRC-01` through `SRC-50` identifiers. A source may be verified but still ineligible for facts or style; verification proves the page and excerpt exist, while eligibility controls how the runtime may use it.

## Verified primary pages

- `OFFICIAL-HIBIKI-PROFILE` and `SRC-01`: [響事务所 official profile](https://hibiki-cast.jp/hibiki_f/aoki_hina/). The dedicated `OFFICIAL-*` record supports granular profile and role claims; `SRC-01` is retained only for corpus traceability.
- `OFFICIAL-BMECHOES-CREATOR`: [BM-ECHOES creator profile](https://bm-echoes.com/creators/aoki-hina/). It supports the registered public music-activity claims.
- `SRC-02`–`SRC-05` and `SRC-39`: [声優図鑑 interview](https://seigura.com/senior/126278/). Each registered excerpt was matched against the page.
- `SRC-06`, `SRC-16`, `SRC-17`, `SRC-24`, `SRC-32`, `SRC-36`, `SRC-41`, `SRC-43`, and `SRC-49`: [Animate Times interview](https://www.animatetimes.com/news/details.php?id=1776831972). Page-specific URLs and excerpt locators are stored in the registry.
- `SRC-37`, `SRC-38`, and `SRC-42`: [Febri interview](https://febri.jp/topics/bang-dream_mygo_5/). Each registered excerpt was matched against the article.

Verified interview excerpts are style-eligible only. They do not become public fact claims merely because the quotation exists.

## Rejected records

- `SRC-08`: Wikipedia was labelled as official material in the supplied corpus. It is not accepted as an official source.
- `SRC-40` and `SRC-46`: the claimed quotations were not present in any page of the [LisAni interview](https://www.lisani.jp/0000272619/). Similar subject matter is not enough to establish an exact quotation.

## Unverified records

The following records remain quarantined until a canonical URL and exact locator or program timestamp can be checked:

`SRC-07`, `SRC-09`–`SRC-15`, `SRC-18`–`SRC-23`, `SRC-25`–`SRC-31`, `SRC-33`–`SRC-35`, `SRC-44`, `SRC-45`, `SRC-47`, `SRC-48`, and `SRC-50`.

This includes official-program and personal-account material whose title or date is known but whose exact excerpt location is missing. `SRC-35` is only a preview lyric fragment; `SRC-50` is an uncorroborated live MC report.

## Runtime enforcement

- Registry, claim, evidence, and few-shot files are required. Missing files fail startup.
- Duplicate IDs, unknown source references, string booleans such as `"false"`, and incompatible citation forms fail startup.
- Only `verification_status=verified` sources can be eligible.
- A fact citation must use `role=fact_support`, an allowed form, and a non-empty locator.
- Public fact answers are assembled directly from matching verified claims, so a language model cannot rewrite `1月5日` into another date.
- Style prompts contain only fully verified style-support references. Quarantined cards and observations are not exposed to the model.
- `HINA_BOT_ORIGINAL` policy cards cannot carry external evidence references.
- Source health checks use HTTP GET. Some primary pages returned misleading results to HEAD requests during audit.

## Promoting a source

1. Find the canonical primary page or official recording.
2. Use GET and confirm the page is accessible.
3. Match the relevant excerpt exactly and record a section, paragraph, page, or timestamp locator.
4. Update the source record to `verified`, add the audit dates and method, and enable only the minimum necessary eligibility.
5. Crop or split any style card so every observation it exposes is supported by its remaining verified references.
6. Add or update regression tests before committing the promotion.
