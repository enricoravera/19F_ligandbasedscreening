# Combined titration workflow

Standalone workflow files for fitting:

- CPMG / relaxation titration (`fit_cpmg_titration`)
- single-decay DOSY titration (`fit_dosy_single_decay`)
- inversion-recovery T1 titration (`fit_t1_inversion_recovery_titration`)

Run with:

```bash
python -m combined_titration_workflow /path/to/input.json
```

Input JSON can include one or more top-level blocks: `cpmg`, `dosy`, `t1`.
Each block maps directly to the function arguments in `fitting.py`.
