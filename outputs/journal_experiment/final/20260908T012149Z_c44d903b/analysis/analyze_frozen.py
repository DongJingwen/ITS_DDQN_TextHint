"""Read only the explicitly frozen experiment; write derived analysis only."""
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
os.environ['MPLCONFIGDIR'] = '/tmp/final_journal_analysis_mpl'
import numpy as np
from scipy import stats
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ID = '20260908T012149Z_c44d903b'
assert BASE.name == ID
POL = ['random', 'fixed', 'rule_based', 'dqn', 'ddqn']
LABEL = dict(zip(POL, ['Random', 'Fixed', 'Rule-based', 'DQN', 'DDQN']))
HINT = ['Strong Guidance', 'Partial Guidance', 'Minimal Scaffold']
METRICS = ['mean_step_reward', 'total_reward', 'correct_rate', 'average_attempts',
           'average_response_time', 'mastery_gain', 'strong_percent', 'partial_percent', 'minimal_percent']
PRIMARY = ['mean_step_reward', 'correct_rate', 'average_attempts', 'average_response_time', 'mastery_gain']
COMP = ['reward_'+x for x in ['correctness', 'improvement', 'response_time', 'attempts',
                             'hint_usage', 'unnecessary_strong_hint']]
EXPECTED = POL[:3] + [p+'_seed_'+str(s) for p in POL[3:] for s in range(42,52)]

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    with p.open(newline='') as f:
        return list(csv.DictReader(f))

def write(name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: ('NA' if row.get(k) is None or isinstance(row.get(k), (float,np.floating))
                            and not np.isfinite(row[k]) else row.get(k)) for k in fields})

def vals(rows, key):
    return np.array([float(r[key]) for r in rows if r[key] != 'NA'])

def desc(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean() if n else np.nan
    sd = x.std(ddof=1) if n > 1 else np.nan
    half = stats.t.ppf(.975, n-1)*sd/np.sqrt(n) if n>1 else np.nan
    return dict(N=n, Mean=mean, SD=sd, Median=np.median(x) if n else np.nan,
                Minimum=min(x) if n else np.nan, Maximum=max(x) if n else np.nan,
                CI95_Lower=mean-half, CI95_Upper=mean+half)

def near(a,b):
    return np.allclose(a,b,rtol=1e-8,atol=1e-8)

def holm(ps):
    ps=np.asarray(ps); order=np.argsort(ps)
    adjusted=np.empty(len(ps)); adjusted[order]=np.minimum(1,np.maximum.accumulate(ps[order]*(len(ps)-np.arange(len(ps)))))
    return adjusted

def integrity():
    issues=[]; data={}; hashes={}; checks=[]
    actual={p.name for p in BASE.iterdir() if p.is_dir() and p.name!='analysis'}
    if actual!=set(EXPECTED): issues.append('Unexpected/missing run directories: '+str(actual ^ set(EXPECTED)))
    for rid in EXPECTED:
        run=BASE/rid
        try:
            m=json.loads((run/'manifest.json').read_text())
            e=read(run/'evaluation_episodes.csv'); s=read(run/'evaluation_steps.csv')
            learned=m['policy'] in POL[3:]
            t=read(run/'training.csv') if learned else []
            data[rid]=dict(m=m,e=e,s=s,t=t)
            assert m['status']=='complete' and m['experiment_id']==ID and m['run_id']==rid
            assert m['evaluation_seed_set']==list(range(10000,10100))
            assert m['configuration']['training_seeds']==list(range(42,52))
            assert not set(m['configuration']['training_seeds']) & set(m['evaluation_seed_set'])
            assert len(e)==100 and len(s)==5000 and (len(t)==500 if learned else len(t)==0)
            for p in run.iterdir():
                if p.is_file(): hashes[str(p.relative_to(BASE))]=digest(p)
            manifest_required='experiment_id run_id policy training_seed evaluation_seed_set checkpoint_path checkpoint_hash number_of_training_episodes steps_per_episode network_architecture learning_rate gamma replay_capacity batch_size target_update_interval exploration_mode epsilon_start epsilon_end environment_configuration simulator_parameters reward_parameters status'.split()
            assert set(manifest_required)<=set(m)
            required={
              'e':'run_id policy training_seed evaluation_seed episode total_reward mean_step_reward correct_rate average_attempts average_response_time initial_mastery final_mastery mastery_gain action_0_count action_1_count action_2_count steps',
              's':'experiment_id run_id policy training_seed evaluation_seed checkpoint_id episode step simulated_student_id knowledge_point question_difficulty mastery_before recent_accuracy previous_response_time previous_attempts action hint_level correct response_time attempts reward mastery_after mastery_gain done q_value_action_0 q_value_action_1 q_value_action_2',
              't':'run_id policy training_seed episode total_reward mean_step_reward average_loss average_epsilon average_entropy average_attempts average_response_time action_0_count action_1_count action_2_count final_mastery steps'}
            for kind,rows in [('e',e),('s',s),('t',t)]:
                if not rows: continue
                assert set(required[kind].split())<=set(rows[0])
                for row in rows:
                    assert row['run_id']==rid and row['policy']==m['policy']
                    assert row['training_seed']==str(m['training_seed'])
                    for k,v in row.items():
                        assert v is not None and v!='', (kind,k)
                        if v=='NA':
                            assert (not learned and k in ['training_seed','checkpoint_id','q_value_action_0','q_value_action_1','q_value_action_2']) or (kind=='t' and k=='average_entropy') or (kind=='t' and k=='average_loss' and row['episode']=='1'), (kind,k)
            if learned:
                cp=Path(m['checkpoint_path'])
                assert cp.resolve()==(run/'checkpoint.pt').resolve()
                assert digest(cp)==m['checkpoint_hash']==m['checkpoint_sha256']
                assert m['number_of_training_episodes']==500
                assert m['epsilon_start']==.8 and m['epsilon_end']==.05 and m['exploration_mode']=='linear_decay'
                assert set(r['checkpoint_id'] for r in s)=={m['checkpoint_hash']}
                assert [int(r['episode']) for r in t]==list(range(1,501))
                assert all(int(r['steps'])==50 for r in t)
                assert near(vals(t,'mean_step_reward'),vals(t,'total_reward')/50)
                assert near(vals(t,'average_epsilon'),np.linspace(.8,.05,500))
                assert all(r['average_entropy']=='NA' for r in t)
                assert np.all((vals(t,'final_mastery')>=0)&(vals(t,'final_mastery')<=1))
            assert set(r['experiment_id'] for r in s)=={ID}
            assert set(int(r['action']) for r in s)<={0,1,2}
            if rid=='fixed': assert set(int(r['action']) for r in s)=={1}
            for k in ['mastery_before','mastery_after','previous_mastery','mastery']:
                assert np.all((vals(s,k)>=0)&(vals(s,k)<=1))
            assert near(vals(s,'reward'),sum(vals(s,k) for k in COMP))
            assert near(vals(s,'mastery_gain'),vals(s,'mastery_after')-vals(s,'mastery_before'))
            assert near(vals(s,'mastery_before'),vals(s,'previous_mastery'))
            assert near(vals(s,'mastery_after'),vals(s,'mastery'))
            if learned:
                q=np.column_stack([vals(s,'q_value_action_'+str(a)) for a in range(3)])
                assert np.isfinite(q).all() and np.array_equal(q.argmax(axis=1),vals(s,'action'))
            for i,row in enumerate(e,1):
                group=s[(i-1)*50:i*50]
                assert int(row['episode'])==i and int(row['evaluation_seed'])==9999+i
                assert int(row['steps'])==50
                assert [int(r['step']) for r in group]==list(range(1,51))
                assert all(int(r['evaluation_seed'])==9999+i and int(r['episode'])==i for r in group)
                assert [r['done'] for r in group]==['False']*49+['True']
                assert near(float(row['total_reward']),vals(group,'reward').sum())
                assert near(float(row['mean_step_reward']),vals(group,'reward').mean())
                for field in ['initial_mastery','final_mastery']: assert 0<=float(row[field])<=1
                assert near(float(row['initial_mastery']),float(group[0]['mastery_before']))
                assert near(float(row['final_mastery']),float(group[-1]['mastery_after']))
                assert near(float(row['mastery_gain']),float(row['final_mastery'])-float(row['initial_mastery']))
                assert near(float(row['correct_rate']),np.mean([r['correct']=='True' for r in group]))
                assert near(float(row['average_attempts']),vals(group,'attempts').mean())
                assert near(float(row['average_response_time']),vals(group,'response_time').mean())
                for a in range(3): assert int(row['action_'+str(a)+'_count'])==sum(int(r['action'])==a for r in group)
                assert near(vals(group,'previous_response_time'),[0]+[float(r['response_time']) for r in group[:-1]])
                assert near(vals(group,'previous_attempts'),[0]+[int(r['attempts']) for r in group[:-1]])
            checks.append(rid+': passed')
        except Exception as exc:
            issues.append(rid+': '+repr(exc))
    counts=dict(dqn_runs=sum(d['m']['policy']=='dqn' for d in data.values()),
                ddqn_runs=sum(d['m']['policy']=='ddqn' for d in data.values()),
                training_interactions=sum(sum(int(r['steps']) for r in d['t']) for d in data.values()),
                evaluation_episodes=sum(len(d['e']) for d in data.values()),
                evaluation_interactions=sum(len(d['s']) for d in data.values()))
    if counts!=dict(dqn_runs=10,ddqn_runs=10,training_interactions=500000,evaluation_episodes=2300,evaluation_interactions=115000): issues.append(str(counts))
    for name in ['validation_report.json','execution_validation.py']:
        if (BASE/name).exists(): hashes[name]=digest(BASE/name)
    lines=['# Frozen data integrity report','', 'Experiment: '+ID, '', 'Outcome: '+('PASS' if not issues else 'FAIL'),'',str(counts),'',
           'Only the 23 explicitly named run directories in this frozen experiment were read. No experiment code was imported or executed. No GUI, pilot, historical or real-user data were included.',
           'Checked expected runs, unique seed identities, episode/interaction counts, checkpoint content hashes and row mappings, seed separation, mandatory fields/allowed NA, actions, greedy argmax, mastery bounds, episode ordering, reward-component sums and episode summaries.',
           'Tolerance: absolute 1e-8 plus relative 1e-8. Expected NA: baseline training seeds/checkpoints/Q-values; unmeasured training entropy; first-episode loss before replay warm-up.','',
           'Anomalies: '+('; '.join(issues) if issues else 'None detected.'),'',
           'Independent runs mean distinct training seeds; statistical independence is a design assumption, not a property proved by file counts.', '',*checks]
    (OUT/'data_integrity_report.md').write_text('\n'.join(lines)+'\n')
    hash_path=OUT/'raw_file_hashes_before.json'
    if hash_path.exists():
        assert json.loads(hash_path.read_text())==hashes, 'Raw data changed since first audit'
    else:
        hash_path.write_text(json.dumps(hashes,indent=2))
    if issues: raise RuntimeError('Integrity anomalies; see report. Analysis stopped.')
    print('INTEGRITY PASS',counts,flush=True)
    return data,hashes,counts

def calculate(data):
    runs=[]; long=[]; table=[]
    for rid,d in data.items():
        row=dict(Policy=LABEL[d['m']['policy']],Run_ID=rid,Training_Seed=d['m']['training_seed'],
                 Independent_Training_Run=d['m']['policy'] in POL[3:],Evaluation_Episodes=len(d['e']),Evaluation_Interactions=len(d['s']))
        for key in METRICS[:6]: row[key]=vals(d['e'],key).mean()
        for a,key in enumerate(METRICS[6:]): row[key]=100*np.mean(vals(d['s'],'action')==a)
        for key in COMP: row[key]=vals(d['s'],key).mean()
        runs.append(row)
    write('run_level_policy_metrics.csv',runs)
    for p in POL:
        rr=[r for r in runs if r['Policy']==LABEL[p]]
        for metric in METRICS+COMP:
            long.append(dict(Policy=LABEL[p],Metric=metric,Statistical_Unit='training run' if p in POL[3:] else 'single fixed evaluation cohort; descriptive',**desc([r[metric] for r in rr])))
        reward=desc([r['mean_step_reward'] for r in rr])
        row=dict(Policy=LABEL[p],Independent_Runs=len(rr) if p in POL[3:] else 0,
                 Evaluation_Cohorts=len(rr),Evaluation_Episodes=sum(r['Evaluation_Episodes'] for r in rr),
                 Evaluation_Interactions=sum(r['Evaluation_Interactions'] for r in rr),
                 Mean_Step_Reward=reward['Mean'],SD_Step_Reward=reward['SD'],
                 CI95_Lower_Step_Reward=reward['CI95_Lower'],CI95_Upper_Step_Reward=reward['CI95_Upper'])
        for key,target in zip(METRICS[1:],['Mean_Episode_Return','Correct_Rate','Average_Attempts','Average_Response_Time','Mean_Mastery_Gain','Strong_Guidance_Percent','Partial_Guidance_Percent','Minimal_Scaffold_Percent']): row[target]=np.mean([r[key] for r in rr])
        table.append(row)
    write('policy_metric_descriptive_statistics.csv',long); write('Table_1_policy_performance.csv',table)
    rq2=[]; comp=[]
    for row,p in zip(table,POL):
        components={k:np.mean([r[k] for r in runs if r['Policy']==LABEL[p]]) for k in COMP}
        rq2.append({**row,**components})
        comp.append(dict(Policy=LABEL[p],**components,Mean_Step_Reward=row['Mean_Step_Reward'],Component_Sum=sum(components.values())))
    write('Table_RQ2_assistance_efficiency.csv',rq2);write('Table_5_reward_components.csv',comp)
    return runs,long,table,comp

def comparisons(runs):
    rows=[]; diagnostics=[]; difference_rows=[]
    rng=np.random.RandomState(20260908)
    for metric in PRIMARY:
        a=np.array([next(r[metric] for r in runs if r['Run_ID']=='ddqn_seed_'+str(s)) for s in range(42,52)])
        b=np.array([next(r[metric] for r in runs if r['Run_ID']=='dqn_seed_'+str(s)) for s in range(42,52)])
        diff=a-b
        sw=stats.shapiro(diff); q1,q3=np.percentile(diff,[25,75]);iqr=q3-q1
        extreme=(diff<q1-3*iqr)|(diff>q3+3*iqr)
        tukey=(diff<q1-1.5*iqr)|(diff>q3+1.5*iqr)
        use_t=sw.pvalue>=.05 and not extreme.any()
        diagnostics.append(dict(Metric=metric,Shapiro_W=sw.statistic,Shapiro_P=sw.pvalue,
                                Skewness=stats.skew(diff,bias=False),Tukey_Flag_Seeds=';'.join(str(42+i) for i in np.where(tukey)[0]) or 'none',
                                Extreme_Flag_Seeds=';'.join(str(42+i) for i in np.where(extreme)[0]) or 'none',
                                Selected_Test='paired t-test' if use_t else 'Wilcoxon signed-rank',Outliers_Removed=0))
        normal_quantiles=stats.norm.ppf((np.arange(1,11)-.375)/(10+.25))
        for i in range(10): difference_rows.append(dict(Metric=metric,Training_Seed=42+i,Difference=diff[i],Sorted_Difference=np.sort(diff)[i],Normal_Quantile=normal_quantiles[i]))
        if use_t:
            test=stats.ttest_rel(a,b);summary=desc(diff)
            statistic=test.statistic;p=test.pvalue;effect=diff.mean()/diff.std(ddof=1)
            estimate=diff.mean();lo=summary['CI95_Lower'];hi=summary['CI95_Upper'];etype='Cohen dz';estimator='mean paired difference';ci='paired t interval for mean difference'
        else:
            nonzero=diff[diff!=0];ranks=stats.rankdata(abs(nonzero));wp=ranks[nonzero>0].sum();wm=ranks[nonzero<0].sum()
            statistic=min(wp,wm);observed=abs(wp-wm)
            signs=np.array(list(itertools.product([-1,1],repeat=len(nonzero))))
            p=np.mean(abs(signs@ranks)>=observed-1e-12)
            effect=(wp-wm)/(wp+wm) if wp+wm else 0
            samples=diff[rng.randint(0,10,size=(50000,10))]
            lo,hi=np.percentile(np.median(samples,axis=1),[2.5,97.5]);estimate=np.median(diff)
            etype='matched rank-biserial';estimator='median paired difference';ci='50000 paired-run percentile bootstrap median interval; not test inversion'
        rows.append(dict(Metric=metric,Comparison='DDQN vs DQN',Statistical_Unit='matched training-seed pair; fixed 100-scenario evaluation cohort',N=10,
                         Test='paired t-test' if use_t else 'Wilcoxon signed-rank, exhaustive sign enumeration',Statistic=statistic,Raw_P=p,Adjusted_P=None,
                         Difference=estimate,Difference_Estimator=estimator,Mean_Difference=diff.mean(),Median_Difference=np.median(diff),
                         CI95_Lower=lo,CI95_Upper=hi,CI_Method=ci,Effect_Size_Type=etype,Effect_Size=effect,
                         Interpretation='positive difference means higher DDQN value; lower attempts/time is favorable'))
    for row,p in zip(rows,holm([r['Raw_P'] for r in rows])): row['Adjusted_P']=p
    for base in ['Random','Fixed','Rule-based']:
        for metric in PRIMARY:
            x=np.array([r[metric] for r in runs if r['Policy']=='DDQN'])
            baseline=next(r[metric] for r in runs if r['Policy']==base)
            d=desc(x-baseline)
            rows.append(dict(Metric=metric,Comparison='DDQN vs '+base,Statistical_Unit='DDQN training runs; baseline cohort fixed',N=10,
                             Test='descriptive only; no independent-run baseline test',Statistic=None,Raw_P=None,Adjusted_P=None,
                             Difference=d['Mean'],Difference_Estimator='mean difference',CI95_Lower=d['CI95_Lower'],CI95_Upper=d['CI95_Upper'],
                             CI_Method='t interval across DDQN runs conditional on these scenarios and this baseline realization; excludes baseline/scenario uncertainty',
                             Relative_Difference_Percent=100*d['Mean']/abs(baseline) if baseline else None,Effect_Size_Type=None,Effect_Size=None,
                             Interpretation='descriptive contrast, not population-level baseline superiority inference'))
    write('Table_2_statistical_comparisons.csv',rows);write('paired_assumption_diagnostics.csv',diagnostics);write('paired_differences_and_qq_data.csv',difference_rows)
    return rows,diagnostics

def training(data):
    rows=[]; windows=[]
    for p in ['dqn','ddqn']:
        logs=[data[p+'_seed_'+str(s)]['t'] for s in range(42,52)]
        for episode in range(500):
            row=dict(Policy=LABEL[p],Episode=episode+1,Independent_Runs=10)
            for key in ['total_reward','mean_step_reward','average_loss','average_epsilon']:
                x=[float(log[episode][key]) for log in logs if log[episode][key]!='NA']
                for stat,value in desc(x).items(): row[key+'_'+stat]=value
            rows.append(row)
        for seed,log in zip(range(42,52),logs):
            for label,start,end in [('Early',0,50),('Late',450,500)]:
                window=log[start:end]
                row=dict(Policy=LABEL[p],Training_Seed=seed,Window=label,Start_Episode=start+1,End_Episode=end)
                for key in ['total_reward','mean_step_reward','average_loss','average_epsilon']:
                    x=vals(window,key);row[key]=x.mean();row[key+'_within_window_SD']=x.std(ddof=1)
                windows.append(row)
    write('training_summary.csv',rows);write('training_early_late_by_run.csv',windows)
    return rows,windows

def categorical(data):
    allrows=[r for s in range(42,52) for r in data['ddqn_seed_'+str(s)]['s']]
    associations=[];consistency=[]
    def group(r,dim):
        if dim=='mastery':
            m=float(r['mastery_before']);return 'Low' if m<.4 else 'Medium' if m<.7 else 'High'
        return r['question_difficulty'].capitalize()
    def aggregate(rows,dim,denominator,rid=None):
        result=[]
        for g in ['Low','Medium','High']:
            subset=[r for r in rows if group(r,dim)==g];n=len(subset)
            row=dict(Group=g,Interactions=n,Percent_All_Interactions=100*n/denominator,Run_ID=rid)
            for a,h in enumerate(HINT):
                c=sum(int(r['action'])==a for r in subset)
                row[h.replace(' ','_')+'_Count']=c;row[h.replace(' ','_')+'_Percent']=100*c/n if n else None
            row.update(Mean_Mastery_Before=vals(subset,'mastery_before').mean() if n else None,
                       Correct_Rate=np.mean([r['correct']=='True' for r in subset]) if n else None,
                       Average_Attempts=vals(subset,'attempts').mean() if n else None,
                       Average_Response_Time=vals(subset,'response_time').mean() if n else None,
                       Mean_Step_Reward=vals(subset,'reward').mean() if n else None)
            result.append(row)
        return result
    tables={}
    for dim,num in [('mastery',3),('difficulty',4)]:
        pooled=aggregate(allrows,dim,len(allrows));byrun=[]
        for s in range(42,52):
            rid='ddqn_seed_'+str(s);byrun.extend(aggregate(data[rid]['s'],dim,5000,rid))
        write('Table_'+str(num)+'_hint_by_'+dim+'.csv',pooled);write('hint_by_'+dim+'_by_run.csv',byrun)
        observed=np.array([[row[h.replace(' ','_')+'_Count'] for h in HINT] for row in pooled])
        nonempty=observed.sum(1)>0;active=observed[nonempty][:,observed.sum(0)>0]
        chi,p,df,expected=stats.chi2_contingency(active,correction=False)
        # Only compute conventional chi-square when all expected counts meet 5.
        assert expected.min()>=5, 'Sparse contingency requires separate alternative, no invalid test emitted'
        v=np.sqrt(chi/(active.sum()*min(active.shape[0]-1,active.shape[1]-1)))
        logp=(-chi/2+math.log1p(chi/2)) if df==4 else stats.chi2.logsf(chi,df)
        associations.append(dict(State=dim,Test='pooled Pearson chi-square; independence reference only',Statistic=chi,DF=df,P_Value='<1e-300' if p==0 else p,
                                 Log10_Nominal_P=logp/math.log(10),
                                 P_Value_Note='below floating-point range; report p < 1e-300' if p==0 else 'nominal, not cluster-adjusted',
                                 Cramers_V=v,Minimum_Expected_Cell=expected.min(),Interactions=len(allrows),Independent_Trained_Runs=10,
                                 Interpretation='nested observations violate independent-row inference; use effect size and by-run patterns'))
        write(dim+'_contingency_observed.csv',[dict(Group=g,**dict(zip(HINT,row))) for g,row in zip(['Low','Medium','High'],observed)])
        write(dim+'_contingency_expected.csv',[dict(Group=g,**dict(zip([h for h,ok in zip(HINT,observed.sum(0)>0) if ok],row))) for g,row in zip(np.array(['Low','Medium','High'])[nonempty],expected)])
        # Supplement: sign-flip tests of run-level within-policy proportion contrasts;
        # each trained run is one vector, no row-level independent participant claim.
        for g1,g2 in itertools.combinations(['Low','Medium','High'],2):
            for h in HINT:
                key=h.replace(' ','_')+'_Percent';diff=[]
                for s in range(42,52):
                    a=next(r for r in byrun if r['Run_ID']=='ddqn_seed_'+str(s) and r['Group']==g1)
                    b=next(r for r in byrun if r['Run_ID']=='ddqn_seed_'+str(s) and r['Group']==g2)
                    if a[key] is not None and b[key] is not None:diff.append(b[key]-a[key])
                if not diff:continue
                diff=np.array(diff);signs=np.array(list(itertools.product([-1,1],repeat=len(diff))))
                pflip=np.mean(abs(signs@diff/len(diff))>=abs(diff.mean())-1e-12)
                consistency.append(dict(State=dim,Contrast=g2+' minus '+g1,Hint=h,N_Runs=len(diff),
                                        Mean_Difference_Percentage_Points=diff.mean(),Positive_Runs=int(sum(diff>0)),Negative_Runs=int(sum(diff<0)),Zero_Runs=int(sum(diff==0)),
                                        Exact_Sign_Flip_P=pflip,Holm_P=None,
                                        Interpretation='supporting across-run association conditional on fixed scenarios; assumes symmetric run-level contrasts'))
        tables[dim]=(pooled,byrun)
    for r,p in zip(consistency,holm([r['Exact_Sign_Flip_P'] for r in consistency])):r['Holm_P']=p
    write('RQ3_categorical_association_tests.csv',associations);write('RQ3_run_level_consistency.csv',consistency)
    m=vals(allrows,'mastery_before');qs=np.percentile(m,[0,25,50,75,100])
    distribution=dict(zip(['Minimum','Q1','Median','Q3','Maximum'],qs))
    distribution.update(Low_Count=int(sum(m<.4)),Medium_Count=int(sum((m>=.4)&(m<.7))),High_Count=int(sum(m>=.7)),At_Ceiling_Count=int(sum(m==1)))
    write('mastery_distribution.csv',[distribution])
    supporting=[]
    for key in ['recent_accuracy','previous_attempts']:
        values=sorted(set(r[key] for r in allrows),key=float)
        for value in values:
            subset=[r for r in allrows if r[key]==value]
            for a,h in enumerate(HINT):
                count=sum(int(r['action'])==a for r in subset)
                supporting.append(dict(State_Variable=key,Logged_Value=value,Interactions=len(subset),Hint=h,Hint_Count=count,Hint_Percent=100*count/len(subset)))
    write('supporting_state_action_analysis.csv',supporting)
    return tables,associations,consistency,distribution

def figures(table,runs,train,tables):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':12,'axes.titlesize':13,
                         'legend.fontsize':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#4477AA','#EE6677','#228833']; policies=[LABEL[p] for p in POL]
    def save(fig,name,rows):
        fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight')
        fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
        write(name+'_data.csv',rows);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.6));plotdata=[]
    for p,c in zip(['DQN','DDQN'],colors):
        rows=[r for r in train if r['Policy']==p];x=np.array([r['Episode'] for r in rows]);y=np.array([r['total_reward_Mean'] for r in rows])
        lo=np.array([r['total_reward_CI95_Lower'] for r in rows]);hi=np.array([r['total_reward_CI95_Upper'] for r in rows])
        ax.plot(x,y,color=c,lw=1,label=p);ax.fill_between(x,lo,hi,color=c,alpha=.17)
        plotdata.extend(dict(Policy=p,Episode=int(a),Mean_Total_Reward=b,CI95_Lower=d,CI95_Upper=e) for a,b,d,e in zip(x,y,lo,hi))
    ax.set(xlabel='Training episode',ylabel='Mean episode total reward',title='Training reward across 10 seeds per algorithm');ax.legend();fig.tight_layout()
    save(fig,'Figure_1_training_reward',plotdata)
    def policy_plot(metric,ylabel,name):
        fig,ax=plt.subplots(figsize=(7.5,4.6));rows=[]
        for i,p in enumerate(policies):
            x=np.array([r[metric] for r in runs if r['Policy']==p]);d=desc(x)
            ax.bar(i,d['Mean'],color=colors[0] if p!='DDQN' else colors[1],alpha=.8,width=.65)
            if len(x)>1:
                ax.errorbar(i,d['Mean'],yerr=d['Mean']-d['CI95_Lower'],color='black',capsize=4,fmt='none')
                offsets=np.linspace(-.18,.18,len(x));ax.scatter(i+offsets,x,color='black',s=14,zorder=3)
            rows.append(dict(Policy=p,Record_Type='summary',Value=d['Mean'],CI95_Lower=d['CI95_Lower'],CI95_Upper=d['CI95_Upper'],Training_Seed=None,X=i))
            if len(x)>1:
                rr=[r for r in runs if r['Policy']==p]
                rows.extend(dict(Policy=p,Record_Type='run_point',Value=r[metric],CI95_Lower=None,CI95_Upper=None,Training_Seed=r['Training_Seed'],X=i+offset) for r,offset in zip(rr,offsets))
        ax.set_xticks(range(5),policies);ax.set_ylabel(ylabel);ax.set_ylim(bottom=0)
        fig.text(.5,.01,'Dots: trained runs; error bars: 95% CI across runs. Baselines: descriptive cohort means.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.055,1,1]);save(fig,name,rows)
    policy_plot('mean_step_reward','Mean step reward','Figure_2_policy_performance')
    policy_plot('average_attempts','Average attempts per interaction','Figure_5a_average_attempts')
    policy_plot('average_response_time','Average response time (seconds)','Figure_5b_response_time')
    for dim,num in [('mastery',3),('difficulty',4)]:
        rows=tables[dim][0];fig,ax=plt.subplots(figsize=(8,4.8));x=np.arange(3);width=.23
        for a,h in enumerate(HINT):
            y=[r[h.replace(' ','_')+'_Percent'] for r in rows]
            ax.bar(x+(a-1)*width,[v if v is not None else np.nan for v in y],width,label=h,color=colors[a])
        for i,r in enumerate(rows):ax.text(i,102,'n = '+format(r['Interactions'],','),ha='center',fontsize=10)
        ax.set_xticks(x,[r['Group']+'\n'+('Mastery' if dim=='mastery' else 'Difficulty') for r in rows]);ax.set(ylabel='Percentage of hint selections',ylim=(0,112))
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.20),ncol=3,frameon=False)
        fig.tight_layout();save(fig,'Figure_'+str(num)+'_hint_by_'+dim,rows)
    fig,ax=plt.subplots(figsize=(8,4.6));bottom=np.zeros(5)
    for a,h in enumerate(HINT):
        y=np.array([r[['Strong_Guidance_Percent','Partial_Guidance_Percent','Minimal_Scaffold_Percent'][a]] for r in table])
        ax.bar(range(5),y,bottom=bottom,color=colors[a],label=h,width=.65);bottom+=y
    ax.set_xticks(range(5),policies);ax.set(ylabel='Percentage of hint selections',ylim=(0,100))
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.18),ncol=3,frameon=False);fig.tight_layout()
    save(fig,'Figure_6_hint_distribution_by_policy',[{k:r[k] for k in ['Policy','Strong_Guidance_Percent','Partial_Guidance_Percent','Minimal_Scaffold_Percent']} for r in table])

if __name__=='__main__':
    data,hashes,counts=integrity()
    runs,long,table,components=calculate(data)
    tests,diagnostics=comparisons(runs)
    train,windows=training(data)
    tables,associations,consistency,distribution=categorical(data)
    figures(table,runs,train,tables)
    assert all(digest(BASE/name)==value for name,value in hashes.items()), 'Frozen source changed'
    (OUT/'raw_file_hashes_after.json').write_text(json.dumps(hashes,indent=2))
    print('TABLE1',json.dumps(table,default=float),flush=True)
    print('PRIMARY',json.dumps(tests[:5],default=float),flush=True)
    print('ASSOCIATION',json.dumps(associations,default=float),flush=True)
    print('MASTERY',json.dumps(distribution,default=float),flush=True)
    print('DIAGNOSTICS',json.dumps(diagnostics,default=float),flush=True)
    print('CALCULATIONS AND FIGURES COMPLETE; raw hashes unchanged',flush=True)
