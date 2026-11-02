# Local CPU serving and container setup

The primary demonstrated deployment is a local Python environment. Training and CUDA
benchmarks are separate offline commands. Ordinary inference can run on CPU without
the training dataset and without a GPU. Model weights are selected by release directory;
the service never downloads a missing model or substitutes a fixture.

## Hash-pinned CPU environment

From the repository root on Windows with `uv` installed:

```powershell
uv venv .venv-cpu --python 3.12
uv pip install --python .venv-cpu/Scripts/python.exe --index-strategy unsafe-best-match -r requirements-cpu.txt
uv pip install --python .venv-cpu/Scripts/python.exe --no-deps -e .
.venv-cpu/Scripts/python.exe -m inspection_dl.api --bundle releases/pcb-ind-ssdlite-v1 --device cpu
```

On Linux, use `.venv-cpu/bin/python` instead of `.venv-cpu/Scripts/python.exe`.
The CPU lock selects `torch==2.7.1+cpu` and `torchvision==0.22.1+cpu`; all transitive
requirements have hashes. The `unsafe-best-match` flag is needed for uv to search both
PyPI and the official PyTorch CPU index for the exact pinned package versions. The
hash checks remain enabled. Without it, uv can stop at a stale package on the PyTorch
index and report that the locked version is unavailable. Pip searches both indexes
without that uv-specific flag.

The source package install uses `--no-deps` after the lock install. It cannot silently
replace the CPU wheels with another runtime. To validate without data or trained weights:

```powershell
.venv-cpu/Scripts/python.exe -m pytest tests/test_api.py tests/test_bundle.py -q
.venv-cpu/Scripts/python.exe -m inspection_dl.api --fixture
```

The fixture command starts a simulated backend and is solely a service/contract check.
Use [the API contract](API.md) for request formats, error semantics, limits, and timing
definitions. The command-line server binds only `127.0.0.1:8000`. Use Ctrl+C to stop it.

## Container recipe

The Dockerfile installs the same hash-pinned CPU dependencies, installs the source
without dependency resolution, and runs as UID 10001. It does not contain datasets,
weights, credentials, or experiment outputs. A readiness healthcheck reports an
unhealthy container when model initialization fails.

```powershell
docker build -t visual-inspection-dl:local .
docker run --rm --name inspection-dl -p 127.0.0.1:8000:8000 --mount "type=bind,source=$((Resolve-Path releases/pcb-ind-ssdlite-v1).Path),target=/models/release,readonly" -e INSPECTION_DL_BUNDLE=/models/release visual-inspection-dl:local
```

For Linux shells, provide the absolute host release path in the bind mount instead.
The container user must be able to read all three release files. The model mount is
read-only and the service accepts no filesystem path in an HTTP request.

The ASGI process binds `0.0.0.0` **inside the container** so Docker port forwarding
works. Publish the host port to `127.0.0.1` as shown to keep it local. The container
uses one worker with at most eight concurrent connections and one active inference.
For a fixture-only container smoke, omit the model mount and set
`-e INSPECTION_DL_BACKEND=fixture`; its responses explicitly report simulation.

Without a valid mounted model, `/health` remains HTTP 200 and `/ready` returns HTTP
503 with `ready: false`. No fixture is selected automatically. Check these endpoints:

```powershell
curl.exe --fail-with-body http://127.0.0.1:8000/health
curl.exe --fail-with-body http://127.0.0.1:8000/ready
```

## Validation record and limits

An initial check found Docker Desktop's Linux daemon unavailable. It became available
later in the session, and the CPU image then built successfully. Actual container HTTP
checks passed for explicit fixture inference and missing-bundle readiness, with host
ports restricted to loopback. The running user was UID 10001 and the runtime reported
PyTorch 2.7.1+cpu, torchvision 0.22.1+cpu, and no CUDA. Both temporary containers were
removed. [Container evidence](../reports/validation/container.json) records the tested
image ID, commands, and results. Its fixture outputs make no trained-model quality
claim.

After training, the released `pcb-ind-ssdlite-v1` bundle was demonstrated over actual
HTTP in both the Windows CPU environment and a rebuilt Linux CPU container. The fixed
attributed 300×300 training fixture produced two proposals at the released threshold
0.2; this demonstrates functioning inference and makes no holdout-quality claim.
Both responses reported `backend: trained`, `simulated: false`, and the expected
artifact digest. A request threshold of 1.0 produced a successful empty result, and
corrupt JPEG bytes returned HTTP 400. The container used a read-only release mount,
UID 10001, and host publication restricted to loopback. Its installed API, schema,
bundle-loader, and model source hashes were verified against the working tree.
The temporary server and container were stopped after capture. Actual responses and
commands are in [native CPU evidence](../reports/validation/real-api.json) and
[container evidence](../reports/validation/real-container.json). The latter separately
records its offline evaluation-module snapshot; that module does not serve requests.

The separately installed Windows CPU environment was verified with Python 3.12.15,
PyTorch 2.7.1+cpu, torchvision 0.22.1+cpu, and `cuda_available: false`. Dependency
consistency passed for all 44 installed packages. All 99 practical tests passed in
24.70 seconds, including real-architecture CPU save/load prediction parity, one
real detector gradient step, API contracts, artifact corruption/provenance, geometry,
metrics, and dataset fixtures. One third-party anyio deprecation warning was emitted.
These checks provide environment validation without claiming full training in CI.
Actual HTTP requests against temporary loopback servers in this CPU-only environment
also passed for explicit fixture inference and missing-bundle readiness. Both servers
were stopped after the check. Local diagnostic evidence is in
[the CPU environment report](../reports/validation/cpu-environment.json); the fixture
results are explicitly simulated. The report preserves the earlier 99-test run;
later source changes and added tests are validated separately.

These local commands provide no authentication, TLS, remote hosting, autoscaling,
or production SLA. External hosting and integration into the existing Visual
Inspection Platform remain separate work.
