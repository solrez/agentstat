# AgentStat

> A statistical rigor toolkit for LLM/agent evaluation.
> Bootstrap CIs, variance decomposition, power analysis, and significance testing —
> with a focus on **multi-turn agent systems**, where variance is under-explored.

> **Status: under construction.** This README will be rewritten to lead with the
> headline result (a real benchmark ranking that reorders under resampling) once the
> experiments land. For now it documents setup and layout.

## Positioning (honest framing)

Statistical treatment of LLM evals exists in the literature (benchmark variance studies,
bootstrap tutorials, power-analysis recommendations). This project's pitch is:

> These methods are known but rarely applied in practice. Here's a usable tool that makes
> them one function call, plus a reproduction showing rankings are unstable on a benchmark
> people actually use — extended to the multi-turn *agent* case that existing single-turn
> work doesn't cover.

The genuinely under-explored slice is **variance decomposition for agent systems** —
partitioning variance across seed / tool-call nondeterminism / multi-turn trajectory.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev --extra plot   # create venv + install
cp .env.example .env               # add OpenRouter / DeepInfra keys
uv run pytest                      # run the test suite
```

## Layout

```
src/agentstat/
├── core/        # bootstrap, variance, power, significance
├── ranking/     # resample → re-rank → stability metrics
├── harness/     # providers (OpenAI-compatible), disk cache, runner
└── data/        # EvalResult — the format everything consumes
experiments/     # 01_ranking_instability, 02_agent_variance
tests/           # stats core validated against synthetic ground truth
```

## License

MIT
