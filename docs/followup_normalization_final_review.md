# Independent final normalization and calibration review

Both independent audits pass: all **84 fits / 168 budget artifacts**, and all **56 aggregate cells / 140 paired contrasts / 10 interactions**. CPU replay, raw CSV reconstruction, chronology, frozen hashes, checkpoint selection, metrics and paired streams pass. No numerical implementation or frozen configuration was changed.

## Final-budget MLP versus S4 contrasts

Values are fair CRPS at 6,000-update validation-selected checkpoints, averaged over three paired optimization seeds. Delta = S4 minus MLP: negative favors S4. Intervals are descriptive 95% paired seed/circular-block bootstrap intervals for this already-examined test period, not new-test, independent-dataset or multiplicity-adjusted evidence. The two block sizes are sensitivity analyses, not independent confirmations.

| Pipeline / source | MLP | S4 | Delta | Block-4 interval | Block-8 interval |
|---|---:|---:|---:|---|---|
| global / white | 0.196394 | 0.195861 | -0.000533 | [-0.062282, +0.058069] | [-0.059843, +0.057239] |
| global / gp | 0.216480 | 0.174803 | -0.041677 | [-0.063252, -0.018566] | [-0.065183, -0.016954] |
| global / gp_zero_level | 0.274663 | 0.220776 | -0.053887 | [-0.159284, +0.026302] | [-0.155813, +0.028825] |
| global / gp_white_level | 0.218537 | 0.182593 | -0.035944 | [-0.073951, +0.006902] | [-0.077128, +0.007421] |
| center / white | 0.102324 | 0.099883 | -0.002441 | [-0.005803, +0.001374] | [-0.005839, +0.001628] |
| center / gp | 0.098211 | 0.098071 | -0.000139 | [-0.004664, +0.003774] | [-0.004534, +0.003808] |
| scale / white | 0.131515 | 0.123064 | -0.008451 | [-0.021407, +0.004774] | [-0.021778, +0.003655] |
| scale / gp | 0.117861 | 0.132750 | +0.014888 | [+0.004472, +0.025156] | [+0.004367, +0.025807] |
| history / white | 0.098702 | 0.106788 | +0.008086 | [+0.003152, +0.013312] | [+0.003081, +0.012763] |
| history / gp | 0.098470 | 0.100514 | +0.002044 | [-0.002222, +0.005820] | [-0.001941, +0.005386] |
| history / gp_zero_level | 0.109475 | 0.117519 | +0.008043 | [-0.003536, +0.018247] | [-0.002919, +0.018034] |
| history / gp_white_level | 0.098403 | 0.102988 | +0.004585 | [+0.000393, +0.009016] | [+0.000144, +0.008896] |
| history_rawloss / white | 0.100459 | 0.106378 | +0.005920 | [+0.001115, +0.010908] | [+0.001154, +0.010570] |
| history_rawloss / gp | 0.099193 | 0.101321 | +0.002128 | [-0.002804, +0.006180] | [-0.002931, +0.006083] |

The large global/GP S4 advantage does not survive every change of pipeline. Center/GP is nearly tied; scale/GP favors MLP in all three seeds, as does history/white. Thus there is no normalization-independent head winner. History/GP has a small mean MLP advantage but the within-period intervals span zero. GP level controls also interact with normalization; they are not uniform improvements. No final-budget validation minimum occurs exactly at 6,000 updates (0/84), which is a boundary diagnostic rather than proof of convergence.

## Training-cutpoint calibration strata

Training-history standard-deviation quartiles are 0.18082675 and 0.27478840. The test period contains **102 low, 15 middle and only 1 high** history. This strongly unbalanced distribution is itself important. The high bin is one observed history, not a reliable population stratum. All bins were retained; no test-dependent rebucketing was performed.

For the low bin, which contains 86.4% of test histories:

| Source / head | Pipeline | CRPS | Coverage90 | Width90 | Forecast-mean MSE |
|---|---|---:|---:|---:|---:|
| white / mlp | center | 0.10169 | 0.96242 | 0.85158 | 0.03362 |
| white / mlp | history_rawloss | 0.09991 | 0.73162 | 0.39297 | 0.03299 |
| white / s4 | center | 0.09999 | 0.94050 | 0.76669 | 0.03406 |
| white / s4 | history_rawloss | 0.10680 | 0.67225 | 0.34866 | 0.03623 |
| gp / mlp | center | 0.09842 | 0.94621 | 0.74513 | 0.03121 |
| gp / mlp | history_rawloss | 0.10023 | 0.72644 | 0.38925 | 0.03226 |
| gp / s4 | center | 0.09698 | 0.92538 | 0.66986 | 0.03107 |
| gp / s4 | history_rawloss | 0.10248 | 0.69635 | 0.36462 | 0.03278 |

The large coverage drop from centering to history+raw-unit loss accompanies much narrower intervals in quiet histories. Forecast-mean MSE changes little by comparison; white/MLP even improves slightly. This locates a descriptive width/coverage difference rather than proving a unique causal mechanism. Center-only is not automatically the best pipeline: its wider forecasts can trade coverage against sharpness, and ridge remains a relevant direct baseline.

Nominal 90% intervals are estimated from only 64 ensemble members. Even samples from the true predictive distribution need not attain 90% empirical coverage under finite-sample quantiles. Small deviations from 0.90 are therefore not proof of model miscalibration. Compare coverage, width, CRPS, finite-ensemble reference checks and context counts together. Large paired differences remain descriptive evidence on this test period, not independent validation.

All 168 records reproduce their full metric means after weighting bins by their numbers of contexts. Saved chronological starts, targets and histories match independent raw CSV reconstruction. All source/head cells, both budgets, every seed, ridge forecasts and the paired history_rawloss−center contrasts remain in the JSON/CSV outputs.

Consumed normalization summary SHA256: `843f5507f4203a0902da518de0a303b696c13db6191b80957713c7434520a18b`.
