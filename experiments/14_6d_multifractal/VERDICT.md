# Experiment 3.3: 6D Multi-fractal Measurement

## Numbers

Sliding window alpha (6D, from Experiment 02 extended data):

| mid_epsilon | alpha   | R2     | predicted slope |
|-------------|---------|--------|-----------------|
| 9.79e-06    | 0.1367  | 0.9991 | -0.0228         |
| 3.06e-05    | 0.1439  | 0.9988 | -0.0240         |
| 9.79e-05    | 0.1529  | 0.9987 | -0.0255         |
| 3.06e-04    | 0.1566  | 0.9990 | -0.0261         |
| 9.79e-04    | 0.1567  | 0.9991 | -0.0261         |
| 3.06e-03    | 0.1428  | 0.9928 | -0.0238         |
| 9.79e-03    | 0.1346  | 0.9962 | -0.0224         |
| 3.06e-02    | 0.1522  | 0.9761 | -0.0254         |

Alpha range: 0.135 to 0.157 (spread 0.022). Not multi-fractal.
Classifier slope: -0.054.
No local alpha predicts the classifier slope. Closest gap: 0.028.

## Verdict

No mechanistic explanation found. The boundary is approximately mono-fractal
(alpha varies by only 0.022 across 4 decades of epsilon). The GOY single-alpha
prediction (-0.024) does not match the measured classifier slope (-0.054) at
any epsilon scale.

The 6D result stands as an honest negative finding: the GOY bound does not
predict classifier accuracy in the 6D phase space.

**D3 status: PASS (honest negative result framing).**
