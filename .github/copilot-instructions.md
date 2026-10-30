# Copilot instructions for pyrotun

pyrotun is a private, long-running asyncio service that complements an OpenHAB home-automation installation (house heating, power prices, EV/solar, water heater, Polar exercise data, etc.). Code is deployed by systemd (`pyrotun.service`, and `pyelvtun.service` for the separate Elvatun entry point).

## Build, test, lint

```sh
pip install -e ".[tests]"          # version comes from setuptools-scm -> pyrotun/version.py (generated)
pytest                             # all tests (tests/ is small; pytest-asyncio is installed)
pytest tests/test_waterheater.py::test_waterheaterobject   # single test
pre-commit run --all-files         # ruff-check --fix + ruff-format + file hygiene hooks
ruff check . && ruff format .      # same as the hooks
mypy pyrotun                       # mypy.ini only sets ignore_missing_imports
```

Ruff rules are configured in `pyproject.toml` (pylint, bugbear, isort, simplify, etc.; magic values / too-many-args are ignored).

Run the service locally: `python pyrotun/pyrotunservice.py` (loads `.env` via python-dotenv; needs `OPENHAB_URL`, `PUSHOVER_APIKEY`, `PUSHOVER_USER` and other secrets for the individual connections).

## Architecture

- `pyrotun/pyrotunservice.py` is the main entry point. `main()` creates one `PyrotunPersistence`, calls `ainit(requested="all")`, runs `at_startup(pers)` (long-lived tasks / async generators), then `setup_crontabs(pers)` which registers `@aiocron.crontab(...)` coroutines using the `EVERY_*` cron-string constants (aiocron is pinned `<2` because six-field cron with seconds is used). Disabled jobs are left commented out rather than deleted.
- `pyrotun/persist.py` — `PyrotunPersistence` is the shared context object (`pers`) passed to nearly every module's `main(pers)` / `update_openhab(pers)`. It owns the shared `aiohttp` session and one lazily-initialised connection object per backend (`pers.openhab`, `pers.influxdb`, `pers.tibber`, `pers.solis`, `pers.myuplink`, ...). `ainit(requested=[...])` selects which connections to create; `readonly=True` makes connections (e.g. `OpenHABConnection.set_item`) log "would have set" instead of sending values — use it when testing against a live system.
- `pyrotun/connections/` — thin clients for each external system (OpenHAB REST, InfluxDB, Tibber, Solis, Smappee, myUplink, Homely, Home Assistant, Polar, Yr, UniFi Protect, SectorAlarm, ...).
- Top-level `pyrotun/poll*.py` modules poll a backend and push values to OpenHAB items; domain logic lives in modules such as `waterheater.py`, `powercontroller.py`, `poweranalysis.py`, `powermodels.py`, `floors.py`, `elvatunheating.py`, `houseshadow.py`, `vent_calculations.py`. Model parameters/mappings live in YAML next to the code (`floors.yml`, `hasslink.yml`, `*openhab.yml`).
- Polar/exercise pipeline: `polar_dump.py`, `update_polar_nightly.py`, `exercise_uploader.py`, `exercise_analyzer.py` (+ `strava/`); the repo-root `plot_polar_*.py` / `polar_nightly_*.py` scripts are standalone analysis scripts.
- `habapp/` holds HABApp config and rules (runs inside a separate HABApp process, not part of the `pyrotun` package).
- Errors: `global_exception_handler` in `pyrotunservice.py` forwards unhandled asyncio exceptions to Pushover (deduplicated, cleared hourly).

## Conventions

- Get loggers with `pyrotun.getLogger(__name__)` (not `logging.getLogger`); it configures a `log.txt` file handler plus stdout and shortens names.
- Everything is async (aiohttp, aiocron); new periodic work = an `async def` module function taking `pers`, registered in `setup_crontabs`/`at_startup`. Stagger jobs with a small `await asyncio.sleep(...)` at the start as existing jobs do.
- Always write to OpenHAB through `pers.openhab.set_item(...)` so `readonly` is honoured.
- Tests are plain pytest files in `tests/` that exercise pure logic (pandas/numpy/networkx models) without hitting live services.
- Do not add `Co-authored-by` trailers to commit messages.
