@echo off
REM Auto-accumulate the real HN+GDELT dataset. Each run adds new, deduplicated
REM events to data/real_events_raw.pkl (the collector merges + dedupes), growing
REM the corpus toward enough cross-platform transfer cases for the transfer eval.
REM
REM Schedule it (daily) with Windows Task Scheduler, e.g.:
REM   schtasks /Create /TN "SimCity_accumulate" /TR "\"%~f0\"" /SC DAILY /ST 09:00
REM
REM Then, once transfer cases reach ~150+, run:
REM   python -m data.build_real_dataset --pages 2 --gdelt-topics 45   (final pull)
REM   set SIMCITY_DATA=data/real_events.pkl& set SIMCITY_HAWKES_W=10& set SIMCITY_TGN_W=0.1& set SIMCITY_VIRALITY_W=0.1& python train.py
REM   python narrative_transfer_eval.py --preds results/simcity_test_preds.npz

cd /d "%~dp0.."
echo [%date% %time%] accumulate start >> results\accumulate.log
python -m data.build_real_dataset --pages 2 --gdelt-topics 45 --gdelt-timespan 1w --out data\real_events.pkl >> results\accumulate.log 2>&1
echo [%date% %time%] accumulate done (exit %ERRORLEVEL%) >> results\accumulate.log
