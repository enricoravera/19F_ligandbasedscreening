# Combined titration workflow

Standalone workflow files for fitting:

- CPMG / relaxation titration (`fit_cpmg_titration`)
- single-decay DOSY titration (`fit_dosy_single_decay`)
- inversion-recovery R1 titration (`fit_r1_inversion_recovery_titration`)

Run with:

```bash
python -m combined_titration_workflow /path/to/input.json
```

Input JSON can include one or more top-level blocks: `cpmg`, `dosy`, `r1`.
Each block maps directly to the function arguments in `fitting.py`.

## Guess models used by the workflow

### CPMG / relaxation
- **Bound-state R2 guess**: FASTCSAR-style estimate
  - dipolar contribution: approximately `R2_DD,b ≈ protein molecular weight (kDa)`
  - CSA contribution: computed from `delta_sigma`, `eta`, `B0`, and `tau_c`
  - combined guess: `R2_b_guess = R2_DD,b + R2_CSA,b`
- **Free-state R2 guess**: taken from the zero-protein point when available, otherwise from the minimum observed R2

### DOSY
- **Diffusion guesses**: Stokes–Einstein estimate
  - `D = k_B T / (6 π η r)`
  - used for both free and bound diffusion coefficients
- The user can optionally fix `D_free` and/or `D_bound` to these guessed values

### Inversion recovery
- **R1 guess strategy**: single-exponential inversion-recovery approximation
  - each titration point is first fit to the inversion-recovery model `M(t) = M0 (1 - 2 exp(-t / T1)) + C`
  - the apparent `R1` value is then obtained as `R1 = 1 / T1`
  - the titration model then uses the apparent `R1` values as inputs
- **Free/bound guess values**: derived from the apparent `R1` values, with the zero-protein point used when available
- The user can optionally fix `R1_free` and/or `R1_bound` to the guessed values
