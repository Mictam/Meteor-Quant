# Security policy

Meteor Quant is a local research and paper-trading application. It intentionally does not support private exchange credentials or live order placement.

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose local files, execute untrusted strategy code unexpectedly, bypass validation, or alter execution accounting. Report it privately to the repository owner with:

- affected version or commit;
- reproduction steps;
- impact assessment;
- suggested mitigation, when available.

## Trust boundaries

- Files in `user_strategies/` execute as local Python code and must be treated as trusted.
- The dashboard should be bound to `127.0.0.1` unless it is placed behind an authenticated reverse proxy.
- Market data and model checkpoints are untrusted inputs and must remain schema-validated.
- No exchange API secret should be placed in `.env`; private trading is outside scope.
