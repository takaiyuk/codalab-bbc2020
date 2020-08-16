## Metric
Submitted results are evaluated by the accuracy

*acc = (number of correctly classified test trajectories) / (total number of test trajectories)*


## Phases

1. test-dev phase: scores are shown with randomness (1/50 of your submission labels are randomly modified for avoiding cheat and overfit) on the leaderboard, which is NOT the final score.
2. test-challenge phase: scores are shown with no randomness. This score is the final one.

---

テストデータの予測対象プレイは382、Public LBでは1/50がランダムに値が変更される。  
このことからテストデータのうち7~8個がランダムに値が変更されるので、Public/Privateでスコアが変動しうる。  
どの程度スコアが変動しうるかというと、例えば提出データが実際には100個正解だとすると

- true: 100/382 -> 0.261
- possible worst -> 0.243
- possible best -> 0.280

**以上のようになるのでPublic/Privateでスコアは約0.02ポイント上下しうる（正解が1個増えるとスコアが0.0026上昇する）**
