# Security

## Reporting

Report a vulnerability privately through GitHub:
[Report a vulnerability](https://github.com/gopaljigaur/decide/security/advisories/new).
Please do not open a public issue for one.

Expect an acknowledgement within a week. This is maintained by one person, so
that is a realistic figure, not an aspirational one.

## What is in scope

`pydecide` sends your state and questions to whichever backends you configure,
including hosted APIs when their keys are in the environment. The parts worth
reporting:

- A state reaching a backend the caller did not configure, or reaching a hosted
  backend while `DECIDE_LOCAL_ONLY` is set.
- `decide serve` answering a request that should have failed its
  `Authorization` check.
- An API key appearing in a log line, an exception message, or `meta.raw`.
- Anything in the wire parsing that a malicious backend response can exploit.

## Not in scope

A model giving a wrong or overconfident answer. That is a measurement problem,
not a vulnerability; `docs/gating.md` and the benchmark cover it.
