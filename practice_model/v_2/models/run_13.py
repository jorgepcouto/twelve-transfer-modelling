"""NB 13 — Enhanced LassoCV ablation (lightweight script)."""
import pandas as pd
import numpy as np
import glob, os, json
from sklearn.linear_model import LassoCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

parent = glob.glob('/Users/jorgepadilla/Documents/Documents*Jorge*MacBook*')[0]
BASE = os.path.join(parent, 'thesis_data')
V3 = f'{BASE}/processed_data/model_dataset/v_3'

QUALITIES = [
    'Active defence','Aerial threat','Box threat','Chance prevention',
    'Composure','Defensive heading','Dribbling','Effectiveness',
    'Finishing','Hold-up play','Intelligent defence','Involvement',
    'Passing quality','Poaching','Pressing','Progression',
    'Providing teammates','Run quality','Territorial dominance','Winning duels'
]
TQ = ['DEFENCE','DEFENSIVE_TRANSITION','ATTACKING_TRANSITION',
      'ATTACK','PENETRATION','CHANCE_CREATION','OUTCOME']
POSITIONS = ['Midfielder','Central Defender','Full Back','Winger','Striker']

def valid_q(df, qualities, prefix='pre_'):
    return [q for q in qualities if f'{prefix}{q}' in df.columns and df[f'{prefix}{q}'].isnull().mean() <= 0.50]

def style_dist(df, suffix=''):
    sq = sum((df[f'from_q_proj_{q}{suffix}'] - df[f'to_q_{q}'])**2
             for q in TQ if f'from_q_proj_{q}{suffix}' in df.columns and f'to_q_{q}' in df.columns)
    return np.sqrt(sq)

results = []

for ds_name, fname in [('ALL','enhanced_model_df.parquet'), ('Transfers Only','enhanced_model_df_transfers_only.parquet')]:
    df_full = pd.read_parquet(f'{V3}/{fname}')
    df_full['style_distance'] = style_dist(df_full)
    df_full['style_distance_t2'] = style_dist(df_full, '_t2')
    print(f'\n{"="*50}\n{ds_name}: {len(df_full):,} rows\n{"="*50}')

    for pos in POSITIONS:
        dp = df_full[df_full['position'] == pos].copy()
        train, test = dp[dp['to_season'] <= 2023], dp[dp['to_season'] == 2024]
        vq = valid_q(dp, QUALITIES, 'pre_')
        vq_post = valid_q(dp, QUALITIES, 'post_')
        targets = [f'post_{q}' for q in vq_post]

        base = [f'from_q_proj_{q}' for q in TQ] + [f'to_q_{q}' for q in TQ] + [f'pre_{q}' for q in vq] + ['style_distance','pre_minutes']
        configs = {
            'Baseline': base,
            '+ Age': base + ['player_season_age','age_squared'],
            '+ Tenure': base + ['years_at_club'],
            '+ T2 Player': base + [f'pre_{q}_t2' for q in vq if f'pre_{q}_t2' in dp.columns] + ['pre_minutes_t2','has_t2_history'],
            '+ T2 Team': base + [f'from_q_proj_{q}_t2' for q in TQ] + ['style_distance_t2'],
            'Full v3': base + ['player_season_age','age_squared','years_at_club']
                       + [f'pre_{q}_t2' for q in vq if f'pre_{q}_t2' in dp.columns] + ['pre_minutes_t2','has_t2_history']
                       + [f'from_q_proj_{q}_t2' for q in TQ] + ['style_distance_t2'],
        }

        print(f'\n  {pos}: {len(train)} train, {len(test)} test, {len(targets)} targets')

        for cfg_name, feat_cols in configs.items():
            feat_cols = [c for c in feat_cols if c in dp.columns]
            use = feat_cols + targets
            tr_c = train[use].dropna()
            te_c = test[use].dropna()
            if len(tr_c) < 50 or len(te_c) < 10:
                continue

            Xtr, ytr = tr_c[feat_cols].values, tr_c[targets].values
            Xte, yte = te_c[feat_cols].values, te_c[targets].values

            # n_jobs=1 to avoid memory issues
            m = MultiOutputRegressor(LassoCV(cv=5, max_iter=10000, n_jobs=1))
            m.fit(Xtr, ytr)

            r2tr = r2_score(ytr, m.predict(Xtr), multioutput='uniform_average')
            r2te = r2_score(yte, m.predict(Xte), multioutput='uniform_average')

            results.append({'dataset':ds_name,'position':pos,'config':cfg_name,
                           'n_features':len(feat_cols),'r2_train':r2tr,'r2_test':r2te})
            print(f'    {cfg_name:15s}  feats={len(feat_cols):3d}  R2 train={r2tr:.3f}  test={r2te:.3f}')

# Save results
res = pd.DataFrame(results)
res.to_csv(f'{V3}/ablation_results.csv', index=False)

# Summary
print('\n\n' + '='*60)
print('SUMMARY: Mean R2 (test) by config')
print('='*60)
config_order = ['Baseline','+ Age','+ Tenure','+ T2 Player','+ T2 Team','Full v3']
pivot = res.pivot_table(values='r2_test', index='config', columns='dataset', aggfunc='mean')
pivot = pivot.reindex(config_order)
print(pivot.round(4).to_string())

print('\nDelta vs Baseline:')
delta = pivot.subtract(pivot.loc['Baseline'])
print(delta.round(4).to_string())

print('\nBest config per position:')
for ds in ['ALL','Transfers Only']:
    print(f'\n  {ds}:')
    for pos in POSITIONS:
        pr = res[(res['dataset']==ds) & (res['position']==pos)]
        if len(pr) > 0:
            best = pr.loc[pr['r2_test'].idxmax()]
            bl = pr[pr['config']=='Baseline']['r2_test'].values[0]
            print(f'    {pos:20s}  {best["config"]:15s}  R2={best["r2_test"]:.3f}  D={best["r2_test"]-bl:+.4f}')

print('\nDone. Results saved to ablation_results.csv')
