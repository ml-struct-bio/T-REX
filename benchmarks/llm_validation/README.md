# LLM validation benchmarks

These source-checkout tools validate Planner/Supervisor structured outputs and
measure Supervisor ranking repeatability on synthetic campaign evidence. They
correspond to the types of checks described in the Supplementary Methods.
They do not generate molecules, measure binding or establish campaign throughput.

Run from the repository root with T-REX installed and an existing LLM endpoint:

```bash
python -m benchmarks.llm_validation.planner_validation --help
python -m benchmarks.llm_validation.supervisor_validation --help
python -m benchmarks.llm_validation.supervisor_repeatability --help
```

| Module | Inputs | Output |
| --- | --- | --- |
| `planner_validation` | Five fixed evidence cases, endpoint/model and repeats | JSON parse/schema validity, confidence, latency and token summaries |
| `supervisor_validation` | Four evidence cases with fixed hypotheses/candidate jobs | JSON validity, priority proportions and ranking checks |
| `supervisor_repeatability` | Fixed five-candidate case, repeats and temperatures | Pairwise ranking agreement and allocation-proportion differences |

Use `--base-url`, `--model` and `--out` explicitly for each command. Validation
scripts default to one repeat; repeatability defaults to five with a temperature
cycle of 0.5–0.9. To match the manuscript's repetition counts, use five repeats
for each validation script, ten repeats with `--temperatures 0` for repeatability,
or ten repeats with `--temperatures 0 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45`
for sampling sensitivity. Matching those counts alone does not reproduce the
reported results: preserve the model/serving configuration and input/prompt
identity as well. New model responses may differ.

The Planner's optional stress case is separate from the five standard cases.
Reports contain synthetic input references and LLM outputs. Review validation
failures and any pruned/omitted candidates before interpreting ranking agreement.
The automated package tests check imports, fixture construction and `--help`;
they do not contact an LLM server or rerun the reported LLM experiment.
