# Security

Do not report credentials, private target data, or unpublished generated
structures in a public issue. Report security-sensitive problems privately to
the repository maintainers through GitHub.

T-ReX launches external scientific tools as subprocesses. Treat backend paths,
target files, model files, and configuration files as trusted inputs. Run the
controller under a dedicated account and scheduler allocation; do not expose a
local model endpoint to untrusted networks.
