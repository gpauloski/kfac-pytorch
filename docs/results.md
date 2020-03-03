| Optimizer  | Model    | GPUs | LR    | LR-Decay   | Epochs | Val-Acc (Train) | t/epoch  | time    |
|------------|----------|------|-------|------------|--------|-----------------|----------|---------|
| SGD        | ResNet32 | 1    | 0.1   | 100, 150   | 200    | 92.89% (99.83%) | 00:27.50 | 1:45:22 |
| SGD        | ResNet32 | 4    | 0.4   | 100, 150   | 200    | 92.74% (99.67%) | 00:08.50 | 0:33:22 |
| KFAC (10)  | ResNet32 | 1    | 0.1   | 100, 150   | 200    |                 | 01:20.00 |         |
| KFAC (10)  | ResNet32 | 4    | 0.4   | 100, 150   | 200    | 92.85% (99.56%) | 00:16:00 | 0:57:46 |
| KFAC* (10) | ResNet32 | 4    | 0.4   | 30, 75, 90 | 100    | 92.52% (99.49%) | 00:17:00 | 0:30:42 |
| KFAC* (10) | ResNet32 | 4    | 0.4   | 35, 80, 95 | 105    | 92.43% (99.58%) | 00:17:00 | 0:32:32 |


Notes:
- The value after KFAC is the iterations between inverse update ops. E.g. KFAC (10) means the KFAC optimizer with the inverse update ops run every 10 iterations.
- Expected val-acc from original paper: ResNet20 91.75%, ResNet32 92.49%, ResNet56 93.03%.
