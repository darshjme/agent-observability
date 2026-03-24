# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please report security issues by emailing **darshjme@gmail.com** with:

- A description of the vulnerability.
- Steps to reproduce the issue.
- Potential impact assessment.
- Any suggested mitigations.

You will receive a response within **48 hours**.  
We aim to release a patch within **7 days** of a confirmed vulnerability.

## Security Design Notes

- **Prompt hashing:** `AgentLogger` never stores raw prompt text — only a truncated SHA-256 hash.
- **No network I/O:** The library makes zero outbound network calls. All exports are local (file/stdout).
- **Zero dependencies:** The stdlib-only design eliminates supply-chain risk from third-party packages.
