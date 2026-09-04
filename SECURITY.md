# Security Policy

KVWeave is experimental research software. It has no production-hardening or
stable-API guarantee, and no formal supported release series exists yet. The
current `main` branch is the only version considered for security fixes.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability or include sensitive
details in a public discussion.

Report suspected vulnerabilities privately to
`security-kvweave@gabani.org`. Once GitHub private vulnerability reporting is
enabled, reporters may alternatively use the repository's **Security → Report
a vulnerability** flow.

## Security boundaries

- Model tests can download code/data artifacts from the documented external
  model and package sources; review those sources and revisions before use.
- Never load untrusted PyTorch pickle/checkpoint artifacts. Generated `.pt`
  benchmark sidecars are local research artifacts, not exchange formats.
- Do not attach credentials, private prompts, model weights, environment files,
  or raw profiler traces to public issues.

Security fixes will be handled according to impact and maintainer availability.
The project does not currently promise a response or disclosure timeline.
