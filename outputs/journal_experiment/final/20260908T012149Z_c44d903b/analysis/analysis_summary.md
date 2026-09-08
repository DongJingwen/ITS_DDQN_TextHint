# Frozen Experiment

Experiment ID:
20260908T012149Z_c44d903b

## 1. Data Integrity

PASS: 10 DQN and 10 DDQN training runs, 500,000 training interactions; 2,300 held-out episodes and 115,000 held-out interactions across all five policies. Each learned model has 100 held-out episodes; each baseline has one 100-episode cohort. All 20 checkpoint hashes and mappings, seed separation, required fields, action/mastery bounds, and reward reconciliations passed. No raw-data integrity anomalies were detected. No raw files changed. Distributional cautions (sparse low mastery and retained outliers) are described below, not treated as corrupt records.

## 2. RQ1 — Policy Performance

| Policy | Mean Step Reward | Correct Rate | Attempts | Response Time (s) | Simulated Mastery Gain |
| --- | --- | --- | --- | --- | --- |
| Random | 0.799445 | 0.720000 | 1.457400 | 35.137004 | 0.420601 |
| Fixed | 0.810967 | 0.718200 | 1.450000 | 35.127682 | 0.426201 |
| Rule-based | 0.840058 | 0.737000 | 1.417600 | 34.220957 | 0.445259 |
| DQN | 0.832007 | 0.743320 | 1.410620 | 33.366906 | 0.437026 |
| DDQN | 0.812757 | 0.728300 | 1.437020 | 34.174705 | 0.424343 |

Final performance uses held-out evaluation only. DQN/DDQN statistics first average the 100 episodes within each run, then average the 10 runs. Baselines are descriptive single cohorts and are not assigned n=10 training replicates. Mean step reward across-run SD/95% CI: DQN 0.055011 / [0.792655, 0.871359]; DDQN 0.073871 / [0.759914, 0.865601]. Mean episode return is 41.600358 for DQN and 40.637875 for DDQN; these totals are not labeled average step rewards.

The ten matched training-seed pairs support a paired comparison conditional on the common scenario set. Shapiro-Wilk p-values range from 0.184835 to 0.508486, with no extreme 3-IQR paired-difference flag. All five primary tests use paired t-tests, df=9. Normality diagnostics have low power at n=10; moderate skew and ordinary outlier flags are retained and the supplementary signed-rank tests provide a robustness check. All differences below are DDQN minus DQN.

| Metric | t(9) | Raw p | Holm p | Mean difference | 95% CI | Cohen’s dz |
| --- | --- | --- | --- | --- | --- | --- |
| mean_step_reward | -0.6373 | 0.539823 | 1.000000 | -0.019250 | [-0.087582, 0.049083] | -0.2015 |
| correct_rate | -0.8445 | 0.420257 | 1.000000 | -0.015020 | [-0.055253, 0.025213] | -0.2671 |
| average_attempts | 0.7486 | 0.473224 | 1.000000 | 0.026400 | [-0.053380, 0.106180] | 0.2367 |
| average_response_time | 0.8807 | 0.401402 | 1.000000 | 0.807799 | [-1.267180, 2.882778] | 0.2785 |
| mastery_gain | -0.7542 | 0.470008 | 1.000000 | -0.012683 | [-0.050727, 0.025360] | -0.2385 |

None of the five primary differences is significant after Holm correction; their intervals include zero. This is not evidence of equivalence. The DDQN mean step reward is descriptively 0.013312 higher than Random (+1.665%), 0.001790 higher than Fixed (+0.221%), and 0.027301 lower than Rule-based (−3.250%). No independent-run inferential tests against baselines were performed. Baseline contrast intervals in Table 2 quantify DDQN run variability conditional on the observed baseline cohort; they exclude baseline/scenario sampling uncertainty.

## 3. RQ2 — Efficiency and Assistance

| Policy | Attempts | Time (s) | Strong Guidance % | Partial Guidance % | Minimal Scaffold % |
| --- | --- | --- | --- | --- | --- |
| Random | 1.4574 | 35.1370 | 33.860 | 33.200 | 32.940 |
| Fixed | 1.4500 | 35.1277 | 0.000 | 100.000 | 0.000 |
| Rule-based | 1.4176 | 34.2210 | 23.980 | 67.760 | 8.260 |
| DQN | 1.4106 | 33.3669 | 56.054 | 27.590 | 16.356 |
| DDQN | 1.4370 | 34.1747 | 45.538 | 34.826 | 19.636 |

Compared with DQN, DDQN uses 10.516 percentage points less Strong Guidance but has 0.026400 more attempts, 0.807799 s longer response time, and 0.015020 lower correctness on average. Its predefined hint-usage cost is less negative (−0.075301 vs −0.079899), and unnecessary-strong-hint component is less negative (−0.010762 vs −0.016544). These gains are outweighed by lower correctness reward (0.592450 vs 0.614980), lower improvement bonus (0.207030 vs 0.210144), lower time component (0.143042 vs 0.144388), and more negative attempts component (−0.043702 vs −0.041062). Table 5 reconciles the six component means with mean step reward for every policy.

Factual conclusion: DDQN shows an assistance-cost/performance tradeoff relative to DQN, not a demonstrated uniformly favorable efficiency/performance balance. Relative to Rule-based it uses more Strong Guidance (45.538% vs 23.980%) and has lower reward/correctness/mastery gain and more attempts, despite a slightly shorter response time. Lower assistance intensity alone is not a pedagogical benefit.

## 4. RQ3 — Adaptive Hint Selection

### Mastery

- Low: n=35 (0.070% of interactions); Strong Guidance 0.000%, Partial Guidance 0.000%, Minimal Scaffold 100.000%.
- Medium: n=27266 (54.532% of interactions); Strong Guidance 55.256%, Partial Guidance 20.058%, Minimal Scaffold 24.686%.
- High: n=22699 (45.398% of interactions); Strong Guidance 33.935%, Partial Guidance 52.619%, Minimal Scaffold 13.446%.

Pooled Pearson χ²(4)=5947.881495, nominal p<1e-300, Cramer's V=0.243883; minimum expected cell=6.8726. These are policy-state association descriptors with an independent-row reference p-value, not independent-learner or cluster-adjusted significance. Among ten runs, High versus Medium mastery has less Strong Guidance in 8/10 and more Partial Guidance in 9/10. Supplementary whole-run sign-flip contrasts have Holm-adjusted p=0.191406 and 0.070313, respectively, across the 18 state/action contrasts.

Low mastery is severely underrepresented: 35/50,000 interactions (0.070%), 34 from seed 46 and one from seed 49; eight runs never enter this group. Its 100% Minimal Scaffold selection cannot support a general claim about low-mastery adaptation. Mastery_before min/Q1/median/Q3/max = 0.323000 / 0.569416 / 0.677496 / 0.813149 / 1.000000. Medium and High groups contain 27,266 and 22,699 interactions. Mastery reaches its ceiling before 2,129 actions (4.258%).

### Difficulty

- Low: n=17000 (34.000% of interactions); Strong Guidance 58.159%, Partial Guidance 29.588%, Minimal Scaffold 12.253%.
- Medium: n=17000 (34.000% of interactions); Strong Guidance 42.118%, Partial Guidance 38.165%, Minimal Scaffold 19.718%.
- High: n=16000 (32.000% of interactions); Strong Guidance 35.763%, Partial Guidance 36.844%, Minimal Scaffold 27.394%.

Pooled Pearson χ²(4)=2143.463881, nominal p<1e-300, Cramer's V=0.146406; minimum expected cell=3141.76. Difficulty uses actual logged categories. High versus Low difficulty receives less Strong Guidance in 9/10 runs and more Minimal Scaffold in 8/10 runs; the supplementary whole-run sign-flip Holm p-values are 0.099609 and 0.175781. No supporting contrast survives the 18-test Holm correction. The observed pattern does not support the claim that harder questions consistently receive stronger hints. Difficulty distribution is 34% Low, 34% Medium and 32% High under the fixed 50-step ordered curriculum.

Both patterns show state association, but state visitation is policy-dependent and the pooled results mix trained policies. Per-run distributions are necessary to interpret that mixture. Supporting recent-accuracy and previous-attempt analyses remain descriptive and are not main contributions.

## 5. Evidence Assessment

RQ1: NOT SUPPORTED
Reason: DDQN does not demonstrate stronger held-out performance across alternatives. Its mean reward is lower than DQN and Rule-based; all five DDQN/DQN Holm-adjusted p-values equal 1.000000. Small descriptive advantages over Random/Fixed are not independent-run superiority evidence.

RQ2: MIXED
Reason: Lower assistance costs relative to DQN coexist with lower reward/correctness/mastery gain and higher attempts/time. Baseline contrasts vary, so no uniformly favorable balance is established.

RQ3: PARTIALLY SUPPORTED
Reason: Pooled mastery/difficulty distributions differ and several directions recur in 8–9 of 10 runs. However, low mastery has only 35 interactions from two runs, action allocation varies markedly across runs, and the whole-run supporting contrasts do not survive their Holm correction. Association does not establish pedagogical appropriateness or causation.

## 6. Important Limitations Visible in the Data

- Severe mastery-group imbalance and policy-dependent visitation limit Low-mastery conclusions; the pooled Low pattern is almost entirely seed 46.
- Between-seed variation is substantial: DQN run mean step rewards range 0.738759–0.882231; DDQN 0.637456–0.873936. DDQN seed 46 is a Tukey outlier in run-level performance diagnostics; matched seed 51 is flagged for several paired metrics. All ten seeds are retained, with no technical exclusion.
- Action concentration varies by seed. DQN seed 47 selects Strong Guidance for 97.720% of interactions (near-collapse by the descriptive ≥95% criterion). No DDQN run crosses 95% overall, but dominant-action shares and state-conditional patterns vary; absence of total collapse is not proof of effective adaptation.
- DDQN/DQN differences are weak relative to uncertainty; non-significance does not prove equivalence or identical policies.
- Mastery ceiling effects truncate possible gain: DDQN mastery_before equals 1 in 4.258% of interactions. This is simulated mastery, not measured human learning.
- Baselines have one fixed evaluation cohort; no comparable training-run replication or repeated Random-policy randomization is available. All uncertainty is conditional on the fixed common scenario schedule unless explicitly stated otherwise.
- Correctness, time, mastery and assistance-cost relationships arise from the predefined simulator/reward. They cannot identify causal educational effects of hint strength. Shared scenarios and nested interactions invalidate a participant-level interpretation of pooled contingency p-values.

Training-only descriptive check: Early=episodes 1–50, Late=451–500. Mean episode reward changes DQN 39.781410→41.088928 and DDQN 40.432589→41.471313. Mean loss decreases DQN 8.002881→1.946412 and DDQN 6.627791→1.724162. Mean within-run episode-return SD in those windows is DQN 9.354809→8.247613 and DDQN 9.130476→8.596035. The unsmoothed trajectories remain noisy; decreasing loss and modest reward changes do not establish stable convergence or held-out superiority. No non-finite loss/return was found; no obvious aggregate numerical divergence is visible. Figure 1 uses pointwise across-seed CIs without smoothing.

## 7. Five Most Important Numerical Findings

1. DDQN held-out mean step reward is 0.812757 (SD 0.073871; 95% CI [0.759914, 0.865601]), versus DQN 0.832007 (SD 0.055011; CI [0.792655, 0.871359]); paired difference −0.019250, t(9)=−0.637264, raw p=0.539823, Holm p=1.000000, dz=−0.201521.
2. Descriptive baseline mean step rewards are Random 0.799445, Fixed 0.810967 and Rule-based 0.840058. DDQN exceeds the first two by 0.013312 and 0.001790 but is 0.027301 below Rule-based; no independent-run baseline test was performed.
3. DDQN uses Strong Guidance for 45.538% of actions versus DQN 56.054%, while attempts/time are 1.437020/34.174705 s versus 1.410620/33.366906 s and correctness is 0.728300 versus 0.743320.
4. DDQN Strong Guidance falls from 55.256% in Medium mastery to 33.935% in High mastery; Partial Guidance rises from 20.058% to 52.619%. Pooled Cramer's V=0.243883, but Low mastery has only 35 observations (0.070%) from two runs.
5. DDQN Strong Guidance proportions across Low/Medium/High difficulty are 58.159%/42.118%/35.763%; Minimal Scaffold proportions are 12.253%/19.718%/27.394%. Pooled Cramer's V=0.146406; High versus Low Strong Guidance decreases in 9/10 runs, with supporting Holm p=0.099609.

## 8. Claims That Are NOT Supported

- DDQN outperforms all comparison policies, or has demonstrated statistically superior performance to DQN.
- Non-significant DDQN/DQN tests prove equivalence.
- Fewer strong hints reduce learner dependence, prevent over-reliance, or improve long-term learning behavior.
- These simulations demonstrate real-student learning improvement, superiority in real classrooms, or generalization beyond this simulator and quadratic-equation domain.
- State-action associations show causal pedagogical effects, universally appropriate adaptation, or strong evidence for low-mastery learners.
- Harder questions consistently receive stronger assistance from DDQN.
- Entropy-based exploration is superior; it was not part of this primary experiment.
- DDQN itself is a novel algorithmic contribution of these results.
