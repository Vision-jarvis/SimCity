# Cross-platform next-event prediction (timing / transfer)

Which platform fires next, from the conditional intensity lambda_i(t_k). Random top-1 acc = 0.333; random AUC = 0.5.

## All test events
| Model | Top-1 acc ^ | AUC ^ | Log-loss v |
|---|---|---|---|
| SimCity (neural Hawkes) | 0.820 | 0.927 | 0.714 |
| Static MHP | 0.397 | 0.921 | 0.999 |
| Persistence | 0.982 | 0.981 | 0.154 |
| Marginal prior | 0.397 | 0.500 | 21.740 |

## Transition events only (platform changes -> real transfer; persistence = 0 by construction)
_22 transition events of 1202 test events._

| Model | Top-1 acc ^ | AUC ^ | Log-loss v |
|---|---|---|---|
| SimCity (neural Hawkes) | 0.409 | 0.388 | 1.402 |
| Static MHP | 0.500 | 0.248 | 1.186 |
| Marginal prior | 0.500 | 0.500 | 18.022 |

