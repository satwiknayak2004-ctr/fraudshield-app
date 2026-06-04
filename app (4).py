import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='FraudShield', page_icon='🛡️', layout='wide')

@st.cache_resource
def load_models():
    xgb      = pickle.load(open('/content/fraud_model_xgb.pkl',  'rb'))
    lgbm     = pickle.load(open('/content/fraud_model_lgbm.pkl', 'rb'))
    rf       = pickle.load(open('/content/fraud_model_rf.pkl',   'rb'))
    cfg      = pickle.load(open('/content/ensemble_config.pkl',  'rb'))
    explainer = shap.TreeExplainer(xgb)
    return xgb, lgbm, rf, cfg, explainer

xgb, lgbm, rf, cfg, explainer = load_models()

# ── build_features — leakage features removed ────────────────────────────────
def build_features(amount, txn_type, hour,
                   old_bal_orig, new_bal_orig,
                   old_bal_dest, new_bal_dest):

    amount_to_balance_ratio = amount / (old_bal_orig + 1)
    is_zero_balance_orig    = int(old_bal_orig == 0)
    is_zero_balance_dest    = int(old_bal_dest == 0)
    is_night_transaction    = int((hour >= 23) or (hour <= 5))
    is_large_transaction    = int(amount > 1000000)
    balance_drain_pct       = amount / (old_bal_orig + 1e-6)
    is_full_drain           = int((new_bal_orig == 0) and (old_bal_orig > 0))

    return {
        'step'                   : hour,
        'amount'                 : amount,
        'oldbalanceOrg'          : old_bal_orig,
        'newbalanceOrig'         : new_bal_orig,
        'oldbalanceDest'         : old_bal_dest,
        'newbalanceDest'         : new_bal_dest,
        'amount_to_balance_ratio': amount_to_balance_ratio,
        'is_zero_balance_orig'   : is_zero_balance_orig,
        'is_zero_balance_dest'   : is_zero_balance_dest,
        'hour'                   : hour,
        'is_night_transaction'   : is_night_transaction,
        'is_large_transaction'   : is_large_transaction,
        'balance_drain_pct'      : balance_drain_pct,
        'is_full_drain'          : is_full_drain,
        # Graph features — defaulted to 0 (unknown at single-txn inference)
        'orig_out_degree'        : 0,
        'dest_in_degree'         : 0,
        'orig_unique_receivers'  : 0,
        'dest_unique_senders'    : 0,
        'dest_fan_ratio'         : 0,
        'dest_only_receives'     : 0,
        'pair_tx_count'          : 0,
        'orig_fraud_exposure'    : 0,
        # Velocity features — defaulted to 0 (unknown at single-txn inference)
        'amount_zscore'          : 0,
        'tx_count_total_feat'    : 0,
        'amt_mean_sender'        : 0,
        'step_gap'               : 0,
        'dest_is_new'            : 1,
        # Type one-hot
        'type_TRANSFER'          : int(txn_type == 'TRANSFER'),
        'type_CASH_OUT'          : int(txn_type == 'CASH_OUT'),
    }

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Outfit:wght@400;600;700&display=swap');
* { font-family: 'Outfit', sans-serif !important; }
.stApp { background: #04080f; }
h1 { font-family: 'JetBrains Mono',monospace !important; font-size:2.8rem !important;
     background: linear-gradient(135deg,#00d4ff,#7c3aed,#ff6b35);
     -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.stButton > button { background: linear-gradient(135deg,#00d4ff,#7c3aed) !important;
    color:#000 !important; font-family:'JetBrains Mono',monospace !important;
    font-weight:700 !important; border:none !important; border-radius:10px !important;
    padding:14px !important; letter-spacing:2px !important; }
</style>
""", unsafe_allow_html=True)

st.title('🛡️ FraudShield')
st.markdown('**Ensemble Model: XGBoost + LightGBM + Random Forest**')

for k, v in [('history', []), ('total_checked', 0), ('total_fraud', 0), ('total_legit', 0)]:
    if k not in st.session_state:
        st.session_state[k] = v

tab1, tab2, tab3 = st.tabs(['🔍 Analyze Transaction', '📊 Dashboard', '📈 Model Info'])

# ── TAB 1 : Analyze ──────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2, gap='large')

    with col1:
        st.markdown('#### Transaction Info')
        amount   = st.number_input('Amount (₹)', min_value=0.01, max_value=10000000.0, value=50000.0, step=1000.0)
        txn_type = st.selectbox('Transaction Type', ['TRANSFER', 'CASH_OUT'])
        hour     = st.slider('Hour of Day', 0, 23, 14)

    with col2:
        st.markdown('#### Balance Details')
        old_bal_orig = st.number_input('Sender Balance Before (₹)',   min_value=0.0, value=100000.0, step=1000.0)
        new_bal_orig = st.number_input('Sender Balance After (₹)',    min_value=0.0, value=50000.0,  step=1000.0)
        old_bal_dest = st.number_input('Receiver Balance Before (₹)', min_value=0.0, value=10000.0,  step=1000.0)
        new_bal_dest = st.number_input('Receiver Balance After (₹)',  min_value=0.0, value=60000.0,  step=1000.0)

    if st.button('🔍 ANALYZE TRANSACTION', use_container_width=True):

        if amount > old_bal_orig:
            st.error(f'❌ Invalid: Sender has ₹{old_bal_orig:,.2f} but tries to send ₹{amount:,.2f}')
            st.stop()

        with st.spinner('Running ensemble analysis...'):
            time.sleep(0.8)

        feats = build_features(amount, txn_type, hour,
                               old_bal_orig, new_bal_orig,
                               old_bal_dest, new_bal_dest)

        row = pd.DataFrame([feats])

        row_xgb  = row.reindex(columns=xgb.feature_names_in_,  fill_value=0)
        row_lgbm = row.reindex(columns=lgbm.feature_name_,     fill_value=0)
        row_rf   = row.reindex(columns=rf.feature_names_in_,   fill_value=0)

        p_xgb  = xgb.predict_proba(row_xgb)[0][1]
        p_lgbm = lgbm.predict_proba(row_lgbm)[0][1]
        p_rf   = rf.predict_proba(row_rf)[0][1]

        prob = (cfg['w_xgb']  * p_xgb  +
                cfg['w_lgbm'] * p_lgbm +
                cfg['w_rf']   * p_rf)
        pct  = prob * 100

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric('XGBoost',      f'{p_xgb*100:.1f}%')
        mc2.metric('LightGBM',     f'{p_lgbm*100:.1f}%')
        mc3.metric('RandomForest', f'{p_rf*100:.1f}%')
        mc4.metric('🎯 Ensemble',  f'{pct:.1f}%')

        if prob >= cfg['threshold']:
            st.error(f'🚨 FRAUD DETECTED — {pct:.1f}% probability')
            verdict = 'FRAUD'
        elif prob >= cfg['threshold'] * 0.5:
            st.warning(f'⚠️ SUSPICIOUS — {pct:.1f}% probability')
            verdict = 'SUSPICIOUS'
        else:
            st.success(f'✅ LEGITIMATE — {pct:.1f}% probability')
            verdict = 'LEGITIMATE'

        st.progress(float(min(prob, 1.0)))

        # ── SHAP plain-English explanation ────────────────────────────────────
        st.markdown('#### 🧠 Why did the model decide this?')
        try:
            shap_vals   = explainer.shap_values(row_xgb)
            shap_series = pd.Series(shap_vals[0], index=row_xgb.columns)

            top_risk = shap_series.nlargest(3)
            top_safe = shap_series.nsmallest(3)

            PLAIN = {
                'balance_drain_pct'      : 'Sender is draining a large % of their balance',
                'is_full_drain'          : 'Sender emptied their entire account',
                'dest_is_new'            : 'Money is going to a new/unknown receiver',
                'amount_zscore'          : 'Amount is unusually large for this sender',
                'orig_fraud_exposure'    : 'Sender has transacted with known fraudsters before',
                'dest_fan_ratio'         : 'Receiver gets money from many different senders',
                'dest_in_degree'         : 'Receiver receives from many accounts (mule pattern)',
                'is_large_transaction'   : 'Transaction amount is very large',
                'is_night_transaction'   : 'Transaction happened late at night',
                'tx_same_hour'           : 'Multiple transactions in the same hour',
                'velocity_x_new_dest'    : 'Rapid transactions to a new receiver',
                'new_account_high_amount': 'Large amount sent to a newly seen account',
                'drain_x_new_dest'       : 'Balance drained to a new receiver',
            }

            def plain(feat):
                return PLAIN.get(feat, feat.replace('_', ' ').capitalize())

            if prob >= cfg['threshold']:
                st.markdown('**🚨 Top reasons this looks like FRAUD:**')
                for feat, val in top_risk.items():
                    if val > 0:
                        st.error(f'⬆️ {plain(feat)}')
                st.markdown('**✅ Factors that reduce suspicion:**')
                for feat, val in top_safe.items():
                    if val < 0:
                        st.success(f'⬇️ {plain(feat)}')
            else:
                st.markdown('**✅ Top reasons this looks LEGITIMATE:**')
                for feat, val in top_safe.items():
                    if val < 0:
                        st.success(f'⬇️ {plain(feat)}')
                st.markdown('**⚠️ Minor risk signals:**')
                for feat, val in top_risk.items():
                    if val > 0:
                        st.warning(f'⬆️ {plain(feat)}')

        except Exception as e:
            st.info(f'Explanation unavailable: {e}')

        # ── Session state update ──────────────────────────────────────────────
        st.session_state.total_checked += 1
        if verdict == 'FRAUD':
            st.session_state.total_fraud += 1
        elif verdict == 'LEGITIMATE':
            st.session_state.total_legit += 1
        st.session_state.history.append({
            'Amount' : f'₹{amount:,.0f}',
            'Type'   : txn_type,
            'Prob'   : f'{pct:.1f}%',
            'Verdict': verdict
        })

# ── TAB 2 : Dashboard ─────────────────────────────────────────────────────────
with tab2:
    c1, c2, c3 = st.columns(3)
    c1.metric('Total Analyzed', st.session_state.total_checked)
    c2.metric('Fraud Flagged',  st.session_state.total_fraud)
    c3.metric('Approved',       st.session_state.total_legit)

    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

# ── TAB 3 : Model Info ────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    ### Model Architecture
    | Component | Detail |
    |---|---|
    | Model 1 | XGBoost — 500 trees, eval_metric=aucpr, scale_pos_weight |
    | Model 2 | LightGBM — 500 trees, is_unbalance=True, early stopping |
    | Model 3 | Random Forest — 50 trees, SMOTE inside pipeline (no leakage) |
    | Ensemble | Weighted average by individual PR-AUC performance |
    | Explainability | Real SHAP TreeExplainer on XGBoost (per-prediction) |
    | Graph Features | in/out degree, fan ratio, unique counterparties, fraud exposure ratio |
    | Velocity Features | amount_zscore, balance_drain_pct, is_full_drain, dest_is_new |
    | Split Strategy | Temporal split by step column — train on past, test on future |

    ### Research Paper Connections
    | Our Design Choice | Paper | Conference |
    |---|---|---|
    | Temporal train/test split | Real-world deployment standard | — |
    | orig_fraud_exposure feature | CARE-GNN — Dou et al. 2020 | CIKM |
    | balance_inconsistency feature | FRAUDRE — Zhang et al. 2021 | ICDM |
    | SMOTE inside pipeline only | FRAUDRE — Zhang et al. 2021 | ICDM |
    | Threshold-moving (F1 optimal) | PC-GNN — Liu et al. 2021 | WWW |
    | GMean metric reported | PC-GNN — Liu et al. 2021 | WWW |
    | eval_metric = aucpr | BWGNN — Tang et al. 2022 | ICML |
    | amount_zscore (local deviation) | BWGNN — Tang et al. 2022 | ICML |
    | Weighted ensemble by PR-AUC | DGA-GNN — Duan et al. 2024 | AAAI |

    ### Why This Is Better Than Standard Approaches
    - **No data leakage**: all graph and velocity stats computed on training data only, then mapped to test
    - **Temporal realism**: split by time step, not random — matches how banks actually deploy models
    - **Paper-grounded features**: orig_fraud_exposure (CARE-GNN) and balance_inconsistency (FRAUDRE) go beyond standard tutorials
    - **Correct metric**: PR-AUC and GMean, not accuracy — dataset is 0.13% fraud so accuracy is meaningless
    - **Three models + ensemble**: stronger than any single model, weights adapt to each model's PR-AUC

    ### Dataset
    PaySim synthetic mobile money transactions — 6.3M rows, 0.13% fraud rate.
    Acknowledged limitation: synthetic data may underrepresent real-world fraud diversity.
    """)
