"""Supplementary diagnostics and factual report, derived only from frozen results."""
from analyze_frozen import *

def f(x,n=6):return format(float(x),'.'+str(n)+'f')

def mdtable(rows,columns):
    return '| '+' | '.join(label for key,label in columns)+' |\n| '+' | '.join('---' for c in columns)+' |\n'+'\n'.join('| '+' | '.join(str(r[key]) for key,label in columns)+' |' for r in rows)

runs=read(OUT/'run_level_policy_metrics.csv')
table=read(OUT/'Table_1_policy_performance.csv')
tests=read(OUT/'Table_2_statistical_comparisons.csv')[:5]
diagnostics=read(OUT/'paired_assumption_diagnostics.csv')
associations=read(OUT/'RQ3_categorical_association_tests.csv')
mastery=read(OUT/'Table_3_hint_by_mastery.csv')
difficulty=read(OUT/'Table_4_hint_by_difficulty.csv')
consistency=read(OUT/'RQ3_run_level_consistency.csv')
components=read(OUT/'Table_5_reward_components.csv')
distribution=read(OUT/'mastery_distribution.csv')[0]
windows=read(OUT/'training_early_late_by_run.csv')
outliers=[];sensitivity=[];collapse=[];ceiling=[];window_stats=[]
for p in ['DQN','DDQN']:
    rr=[r for r in runs if r['Policy']==p]
    for metric in PRIMARY:
        x=vals(rr,metric);q1,q3=np.percentile(x,[25,75]);iqr=q3-q1
        for r,value in zip(rr,x):
            if value<q1-1.5*iqr or value>q3+1.5*iqr:
                outliers.append(dict(Policy=p,Run_ID=r['Run_ID'],Metric=metric,Value=value,
                                     Lower_Fence=q1-1.5*iqr,Upper_Fence=q3+1.5*iqr,Action='retained; no technical defect found'))
    for r in rr:
        props=[float(r[key]) for key in ['strong_percent','partial_percent','minimal_percent']]
        collapse.append(dict(Policy=p,Run_ID=r['Run_ID'],Largest_Action_Percent=max(props),Dominant_Action=HINT[int(np.argmax(props))],
                             Near_Collapse_95_Percent=max(props)>=95,Zero_Use_Actions=';'.join(HINT[i] for i,v in enumerate(props) if v==0) or 'none'))
    for window in ['Early','Late']:
        rrw=[r for r in windows if r['Policy']==p and r['Window']==window]
        for metric in ['total_reward','mean_step_reward','average_loss','average_epsilon','total_reward_within_window_SD','average_loss_within_window_SD']:
            window_stats.append(dict(Policy=p,Window=window,Metric=metric,**desc(vals(rrw,metric))))
for metric in PRIMARY:
    diff=np.array([float(next(r[metric] for r in runs if r['Run_ID']=='ddqn_seed_'+str(s)))-float(next(r[metric] for r in runs if r['Run_ID']=='dqn_seed_'+str(s))) for s in range(42,52)])
    nonzero=diff[diff!=0];ranks=stats.rankdata(abs(nonzero));wp=ranks[nonzero>0].sum();wm=ranks[nonzero<0].sum()
    signs=np.array(list(itertools.product([-1,1],repeat=len(nonzero))))
    p=np.mean(abs(signs@ranks)>=abs(wp-wm)-1e-12)
    sensitivity.append(dict(Metric=metric,N=10,Test='Wilcoxon signed-rank exact sign enumeration; supplementary robustness check',
                            Statistic=min(wp,wm),Raw_P=p,Holm_P=None,Rank_Biserial=(wp-wm)/(wp+wm),
                            Interpretation='does not replace the prespecified primary paired t-test'))
for r,p in zip(sensitivity,holm([r['Raw_P'] for r in sensitivity])):r['Holm_P']=p
for rid in EXPECTED:
    rows=read(BASE/rid/'evaluation_episodes.csv');steps=read(BASE/rid/'evaluation_steps.csv')
    ceiling.append(dict(Run_ID=rid,Policy=rows[0]['policy'],Episodes=len(rows),Final_Mastery_At_1=sum(float(r['final_mastery'])==1 for r in rows),
                        Interactions=len(steps),Pre_Action_Mastery_At_1=sum(float(r['mastery_before'])==1 for r in steps)))
write('run_level_outlier_diagnostics.csv',outliers)
write('paired_wilcoxon_sensitivity.csv',sensitivity)
write('action_concentration_by_run.csv',collapse)
write('ceiling_mastery_by_run.csv',ceiling)
write('training_window_summary.csv',window_stats)

methods='''# Analysis methods and scope

Only frozen experiment 20260908T012149Z_c44d903b is an empirical source. The analysis imports no training/evaluation code and performs no environment/model execution. Raw CSVs, manifests and checkpoints are read only. SHA-256 inventories before/after analysis establish that frozen bytes were preserved. No observations or seeds were removed, imputed, tuned or replaced. Literal NA stays missing; unmeasured entropy is not reconstructed. Units: attempts are counts, response time is seconds, mastery is the simulator's bounded scalar, correctness is a fraction, hint proportions are percentages.

## Replication and intervals

DQN/DDQN: 10 independently seeded training runs per algorithm, each evaluated on the same fixed 100 held-out scenarios. Episode metrics are averaged within each trained run first, then run means are averaged equally. Every episode is 50 steps, so mean episode return divided by 50 equals mean step reward. Reported SD, median, extrema and 95% t intervals use the 10 run aggregates (df=9). These intervals quantify training-run variability conditional on the fixed evaluation scenario set; they do not integrate sampling variability over a new scenario population. All metrics' descriptive statistics are in policy_metric_descriptive_statistics.csv.

Random, Fixed and Rule-based each have one 100-scenario evaluation cohort, zero training replicates. Their descriptive results have no independent-training-run SD/CI or error bars. Repeating their same results 10 times would be pseudo-replication. DDQN-minus-baseline differences have no inferential p-values. Their intervals are t intervals across DDQN runs with the observed baseline cohort held fixed; they exclude uncertainty from baseline randomization and new scenarios. Relative difference uses 100*(DDQN-baseline)/abs(baseline), with no implication that a positive attempts/time change is favorable.

## Paired primary comparison

Pairing uses the 10 prespecified matched training-seed identities, not post-hoc matching. Both algorithms shared initialization/seed design and a common held-out cohort. Equal numeric seeds do not ensure identical stochastic paths after policies diverge; paired tests do not require identical trajectories. The statistical unit is the trained-seed pair, never a transition or one of 100 evaluations of the same model. All tests are two-sided; difference sign is DDQN minus DQN. Positive reward/correctness/mastery differences favor DDQN; negative attempts/time differences favor DDQN.

The selection rule applied uniformly before testing: paired t-test if Shapiro-Wilk p>=0.05 and there are no extreme (3 IQR) paired-difference flags; otherwise Wilcoxon signed-rank. Inspect skewness and Q-Q data alongside this low-power normality diagnostic. All five metrics met the rule; approximate normality is plausible, not proven with n=10. Ordinary 1.5 IQR flags were retained and reported. Primary effect size is Cohen's dz=mean paired difference / SD of paired differences; mean-difference intervals use t(9). Holm correction is applied once across the five primary p-values. CIs are pointwise 95%, not simultaneous or multiplicity-adjusted. A supplementary exact signed-rank sign-enumeration analysis of all five metrics checks sensitivity to nonnormality and retained outliers, with a separately labeled Holm family. It does not replace or select among the primary findings.

## Training

Per-episode means, sample SDs and t confidence intervals across ten seeds are saved for return, mean step reward, loss and epsilon, with valid N. Loss is NA at episode 1 because replay has not reached batch size. Early window is episodes 1-50; late window is 451-500, fixed before observing outcomes. Each run contributes one mean per window. Loss means exclude only explicitly logged NA; no loss is reconstructed. Training evidence describes optimization behavior, not final policy performance. Figure 1 is entirely unsmoothed; bands are pointwise across-run 95% t intervals, not simultaneous bands.

## State-action association

Only DDQN held-out interactions enter mastery/difficulty analyses. Mastery uses the requested <0.4, [0.4,0.7), >=0.7 boundaries; actual logged low/medium/high difficulty is unchanged. Pooled proportions weight interactions. Separate per-run proportions show consistency and distinguish run mix from within-policy patterns. Empty run/group cells retain count 0 and percentage NA; no artificial groups are filled. There are only 35 low-mastery interactions from two trained runs, precluding a stable low-mastery generalization.

Both 3x3 pooled tables have expected counts >=5, so the Pearson chi-square statistic and Cramer's V are calculated without continuity correction. Nominal p-values are only independent-row reference values, not valid independent-learner or cluster-adjusted significance claims. Runs share evaluation scenarios, and interactions repeat within episodes/runs. Therefore nominal tiny pooled p-values must not be the sole basis for RQ3 evidence. Underflowing p-values are reported as <1e-300 with stable log10 survival probabilities (df=4), never literally zero.

Supplementary exact sign-flip tests operate on per-run differences in hint proportions between state groups. Each whole trained run contributes one contrast; all 2^n sign combinations are enumerated, conditional on the shared fixed scenarios. These tests assume symmetry of run-level contrasts under the null. Holm correction covers all 18 reported state-group/action contrasts jointly. Low-mastery contrasts have only two contributing runs and are not strong evidence. These supplementary tests and sign counts assess consistency without treating transitions as independent participants; they do not estimate a population of new evaluation scenarios. Proportions may reflect correlated state variables, time in curriculum, and policy-induced visitation; neither association analysis establishes a causal pedagogical effect.

Supporting recent_accuracy and previous_attempts tables use exact logged values, not reconstructed or optimized bins. Policy action concentration >=95% is a descriptive near-collapse flag; no affected seed is discarded.

## Figures and reproducibility

Seven figure panels are saved as 300-dpi PNG and vector PDF. Each has a matching *_data.csv containing exact plotted means, intervals, and run points where applicable. Policy bars start at zero. Learned-policy bars include ten run points and across-run CIs; baselines have no equivalent error bars. Mastery/difficulty plots show interaction-weighted pooled percentages with group counts. No entropy plots or main-paper supporting-state plots are generated. Analysis bootstrap seed, if the nonparametric primary branch is needed, is 20260908; no bootstrap-generated experimental records are saved or treated as observations.
'''
(OUT/'analysis_methods.md').write_text(methods)

compact=[]
for r in table:
    compact.append(dict(Policy=r['Policy'],Reward=f(r['Mean_Step_Reward']),Correct=f(r['Correct_Rate']),Attempts=f(r['Average_Attempts']),Time=f(r['Average_Response_Time']),Mastery=f(r['Mean_Mastery_Gain'])))
testcompact=[]
for r in tests:
    testcompact.append(dict(Metric=r['Metric'],t=f(r['Statistic'],4),p=f(r['Raw_P'],6),Holm=f(r['Adjusted_P'],6),dz=f(r['Effect_Size'],4),Difference=f(r['Difference']),CI='['+f(r['CI95_Lower'])+', '+f(r['CI95_Upper'])+']'))
assist=[]
for r in table:
    assist.append(dict(Policy=r['Policy'],Attempts=f(r['Average_Attempts'],4),Time=f(r['Average_Response_Time'],4),Strong=f(r['Strong_Guidance_Percent'],3),Partial=f(r['Partial_Guidance_Percent'],3),Minimal=f(r['Minimal_Scaffold_Percent'],3)))
def grouptext(rows):
    return '\n'.join('- '+r['Group']+': n='+r['Interactions']+' ('+f(r['Percent_All_Interactions'],3)+'% of interactions); Strong Guidance '+f(r['Strong_Guidance_Percent'],3)+'%, Partial Guidance '+f(r['Partial_Guidance_Percent'],3)+'%, Minimal Scaffold '+f(r['Minimal_Scaffold_Percent'],3)+'%.' for r in rows)
summary='''# Frozen Experiment

Experiment ID:
20260908T012149Z_c44d903b

## 1. Data Integrity

PASS: 10 DQN and 10 DDQN training runs, 500,000 training interactions; 2,300 held-out episodes and 115,000 held-out interactions across all five policies. Each learned model has 100 held-out episodes; each baseline has one 100-episode cohort. All 20 checkpoint hashes and mappings, seed separation, required fields, action/mastery bounds, and reward reconciliations passed. No raw-data integrity anomalies were detected. No raw files changed. Distributional cautions (sparse low mastery and retained outliers) are described below, not treated as corrupt records.

## 2. RQ1 — Policy Performance

'''+mdtable(compact,[('Policy','Policy'),('Reward','Mean Step Reward'),('Correct','Correct Rate'),('Attempts','Attempts'),('Time','Response Time (s)'),('Mastery','Simulated Mastery Gain')])+'''

Final performance uses held-out evaluation only. DQN/DDQN statistics first average the 100 episodes within each run, then average the 10 runs. Baselines are descriptive single cohorts and are not assigned n=10 training replicates. Mean step reward across-run SD/95% CI: DQN 0.055011 / [0.792655, 0.871359]; DDQN 0.073871 / [0.759914, 0.865601]. Mean episode return is 41.600358 for DQN and 40.637875 for DDQN; these totals are not labeled average step rewards.

The ten matched training-seed pairs support a paired comparison conditional on the common scenario set. Shapiro-Wilk p-values range from 0.184835 to 0.508486, with no extreme 3-IQR paired-difference flag. All five primary tests use paired t-tests, df=9. Normality diagnostics have low power at n=10; moderate skew and ordinary outlier flags are retained and the supplementary signed-rank tests provide a robustness check. All differences below are DDQN minus DQN.

'''+mdtable(testcompact,[('Metric','Metric'),('t','t(9)'),('p','Raw p'),('Holm','Holm p'),('Difference','Mean difference'),('CI','95% CI'),('dz',"Cohen’s dz")])+'''

None of the five primary differences is significant after Holm correction; their intervals include zero. This is not evidence of equivalence. The DDQN mean step reward is descriptively 0.013312 higher than Random (+1.665%), 0.001790 higher than Fixed (+0.221%), and 0.027301 lower than Rule-based (−3.250%). No independent-run inferential tests against baselines were performed. Baseline contrast intervals in Table 2 quantify DDQN run variability conditional on the observed baseline cohort; they exclude baseline/scenario sampling uncertainty.

## 3. RQ2 — Efficiency and Assistance

'''+mdtable(assist,[('Policy','Policy'),('Attempts','Attempts'),('Time','Time (s)'),('Strong','Strong Guidance %'),('Partial','Partial Guidance %'),('Minimal','Minimal Scaffold %')])+'''

Compared with DQN, DDQN uses 10.516 percentage points less Strong Guidance but has 0.026400 more attempts, 0.807799 s longer response time, and 0.015020 lower correctness on average. Its predefined hint-usage cost is less negative (−0.075301 vs −0.079899), and unnecessary-strong-hint component is less negative (−0.010762 vs −0.016544). These gains are outweighed by lower correctness reward (0.592450 vs 0.614980), lower improvement bonus (0.207030 vs 0.210144), lower time component (0.143042 vs 0.144388), and more negative attempts component (−0.043702 vs −0.041062). Table 5 reconciles the six component means with mean step reward for every policy.

Factual conclusion: DDQN shows an assistance-cost/performance tradeoff relative to DQN, not a demonstrated uniformly favorable efficiency/performance balance. Relative to Rule-based it uses more Strong Guidance (45.538% vs 23.980%) and has lower reward/correctness/mastery gain and more attempts, despite a slightly shorter response time. Lower assistance intensity alone is not a pedagogical benefit.

## 4. RQ3 — Adaptive Hint Selection

### Mastery

'''+grouptext(mastery)+'''

Pooled Pearson χ²(4)=5947.881495, nominal p<1e-300, Cramer's V=0.243883; minimum expected cell=6.8726. These are policy-state association descriptors with an independent-row reference p-value, not independent-learner or cluster-adjusted significance. Among ten runs, High versus Medium mastery has less Strong Guidance in 8/10 and more Partial Guidance in 9/10. Supplementary whole-run sign-flip contrasts have Holm-adjusted p=0.191406 and 0.070313, respectively, across the 18 state/action contrasts.

Low mastery is severely underrepresented: 35/50,000 interactions (0.070%), 34 from seed 46 and one from seed 49; eight runs never enter this group. Its 100% Minimal Scaffold selection cannot support a general claim about low-mastery adaptation. Mastery_before min/Q1/median/Q3/max = 0.323000 / 0.569416 / 0.677496 / 0.813149 / 1.000000. Medium and High groups contain 27,266 and 22,699 interactions. Mastery reaches its ceiling before 2,129 actions (4.258%).

### Difficulty

'''+grouptext(difficulty)+'''

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
'''
(OUT/'analysis_summary.md').write_text(summary)
metadata=dict(experiment_id=ID,source_directory=str(BASE),analysis_scope='frozen data only',
              python=sys.version,numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,
              statistical_alpha=.05,primary_holm_family=PRIMARY,ci_level=.95,early_episodes=[1,50],late_episodes=[451,500],
              outliers_removed=0,rq_classifications={'RQ1':'NOT SUPPORTED','RQ2':'MIXED','RQ3':'PARTIALLY SUPPORTED'})
(OUT/'analysis_metadata.json').write_text(json.dumps(metadata,indent=2))
hashes=json.loads((OUT/'raw_file_hashes_before.json').read_text())
assert all(digest(BASE/name)==value for name,value in hashes.items())
print('OUTLIERS',outliers)
print('SENSITIVITY',sensitivity)
print('REPORT COMPLETE; frozen source hashes unchanged')
