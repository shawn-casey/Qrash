# Data provenance

Source: [Kenneth R. French Data Library, 12 industry portfolios, daily](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

The loader selects the first **Average Value Weighted Returns -- Daily** table, uses BusEq and Enrgy, converts percentages to decimal simple returns, and treats -99.99 and -999 as missing values. Ten-day returns are products of daily gross returns minus one. Missing observations invalidate an entire affected window rather than joining nonadjacent days.

The analysis window is January 1, 2000 through December 31, 2025. The first actual observation is January 3, 2000. Only windows entirely inside this interval are used. Portfolio weights are fixed at the start of each holding period; the example does not assume daily rebalancing.

`fitted_models.json` contains derived model parameters and provenance, including the exact downloaded archive hash, processing time, source header, table, units, date range, and observation counts. Raw source returns are downloaded into ignored `data/raw/`; they are not bundled or relicensed by this project. The included derived parameters make the demo and benchmark reproducible without a network request.

The French library revises historical returns. Re-downloading a later archive can change fitted parameters. Preserve the saved parameter file for exact benchmark reproduction; `prepare-data` intentionally replaces it when recalibrating. Compare the archive hash when reproducing a calibration.

Both models are fitted on the full historical window. This is a descriptive estimator experiment, not a forecasting backtest. Overlapping ten-day observations are dependent; the code does not treat 6,530 windows as independent evidence for forecast accuracy.

The Student model fits each marginal by likelihood with degrees of freedom constrained to [2.1, 100], then estimates Gaussian-copula correlation from transformed observations. It has heavier marginal tails than the Gaussian control. Its Gaussian copula has no asymptotic tail dependence, and fitting does not establish that it captures future crashes.
