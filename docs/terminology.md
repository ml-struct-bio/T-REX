# Manuscript terminology and code

T-REX means **Target-adaptive Rescue–Explore–eXploit**. This glossary connects
the manuscript's Abstract, Introduction, Methods and Supplementary Methods to
the implementation. Serialized identifiers and existing Python interfaces are
retained so configurations, archives and downstream scripts remain compatible.

| Manuscript term | Code or archive representation | Meaning |
| --- | --- | --- |
| Campaign | `CampaignInput`, one archive root | Generation, redesign and evaluation for one target under a compute budget |
| Action space | Action registry and permitted configuration bounds | The full prespecified domain, not the currently available jobs |
| Action family | `backend_family`, `enabled_families` | Registered tool/search variant; four Complexa variants are four families of one generator |
| Candidate job | `ActionCandidate`, `candidate_id`, `candidate_builder.py` | An option constructed by fixed code for validation and selection |
| Hypothesis and follow-up test | `HypothesisCard`, `hypothesis_id`, `planner.py` | An evidence-linked proposal; a card is not a job or a confirmed start |
| Campaign state | `EvidenceSummary.state_label`, `evidence_reducer.classify_state` | Classification of accumulated evidence, separate from R/E/X priority |
| Rescue / Explore / eXploit | `rescue` / `explore` / `exploit` mode values | Priorities for follow-up computation; not fixed tools or campaign states |
| Allocation proportions | Supervisor mode mixture | Proposed proportions of starts, not fractions of GPU time |
| Selected job | `LaunchDecision` | Selection or queue-admission intent; confirm execution using dispatch records |
| Confirmed job start | `DispatchRecord` with `status=started` | Actual worker execution; distinct from a proposed or queued job |
| Qualified design | `strict`, `strict_count`, `is_strict_success` | Common in silico qualification; generator-native acceptance is insufficient |
| Structurally distinct hit (SU) | `run_su_count`, trusted binder-chain clustering | Qualified structural cluster at a declared threshold; not experimental binding or epitope identity |
| Recorded result | `ResultRecord` | A design outcome or failure; one job can yield multiple records |
| Standardized AF2 evaluation | `structure_refilter`, canonical score-conversion role | Evaluation of an existing design; not automatically a new design descendant |
| Route | Route keys and lineage | Jobs grouped by family/configuration and relevant upstream context |
| Campaign archive | `Archive`, JSONL streams and artifact references | Persistent results, decisions and provenance; broader than one evidence summary |

The Planner proposes hypotheses and follow-up tests. Fixed code constructs and
validates candidate jobs. The Supervisor can rank valid candidates and propose
allocation proportions. Deterministic selection and execution enforce resource
constraints and separately admit required standardized evaluations.

The campaign `critic` setting controls a deterministic advisory guard. It does
not enable the standalone optional LLM Critic. Cross-campaign memory is optional
and was not part of the primary manuscript campaigns.

## Accounting and joins

- `worker_wall_gpu_h_total` is the live campaign denominator; completed-job and
  route-attributed costs have different purposes. Route costs are not additive
  estimates of campaign cost.
- Online SU novelty and final post-hoc representatives have different attribution
  rules. A latest evidence summary is not a recomputed paper endpoint.
- `parent_ids` can reference candidate jobs or input results. Do not join it as a
  result-only foreign key. Prefer the supported decision trace and schema export.
- `strict` in archived keys means computational qualification. Missing or pending
  measurements are not qualified designs and are not observed metric failures.

See [archive schema](archive-schema.md), [result interpretation](result-interpretation.md)
and [architecture](architecture.md) for the implementation and analysis contracts.
