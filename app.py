import sys
import os
import time

# Add the project root directory to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import plotly.graph_objects as go
import plotly.express as px

from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA

# Local Imports
from src.config import REPOS, GROUND_TRUTH_RULES
from src.repo_manager import download_repo
from src.java_parser import parse_java_project
from src.llm_engine import generate_embeddings_for_classes, generate_service_name

# Import all 8 Optimizers
from src.optimizer import CSA, GA, PSO, WOA, GWO, SSA, FA, DE
from src.metrics import combined_fitness, calculate_ground_truth_similarity

# ==========================================
# 0. HELPER FUNCTION FOR PARALLELIZATION
# ==========================================
def run_single_experiment(AlgClass, run_id, n_pop, dim, n_iter, G, Emb, w1, w2, repo_name):
    """
    Worker function that runs a single optimization routine.
    IMPORTANT: All inputs here must be standard Python types (int, list, array), 
    not Streamlit session_state objects.
    """
    name = AlgClass.__name__
    
    # Initialize the specific algorithm
    optimizer = AlgClass(name, n_pop, dim, n_iter)
    
    # Fitness Wrapper
    def fit_func(x): 
        return combined_fitness(x, G, Emb, w1, w2)[0]
    
    # Execute Optimization
    start_t = time.time()
    best_score, best_pos = optimizer.optimize(fit_func)
    end_t = time.time()
    
    # Calculate Final Detailed Metrics
    _, mq, sem, clusters = combined_fitness(best_pos, G, Emb, w1, w2)
    
    # Calculate Ground Truth Accuracy (ARI)
    # We pass list(G.nodes()) implicitly via G inside calculate_ground_truth_similarity 
    # but the helper expects the list of class names, which are the nodes of G.
    gt_acc = calculate_ground_truth_similarity(list(G.nodes()), clusters, repo_name, GROUND_TRUTH_RULES)
    
    return {
        "Algorithm": name,
        "Run": run_id,
        "Fitness": best_score,
        "MQ": mq,
        "Cohesion": sem,
        "GT_Accuracy": gt_acc,
        "Time": end_t - start_t,
        "History": optimizer.loss_history,
        "Clusters": clusters if name == "CSA" else None 
    }

# ==========================================
# 1. STATE & CONFIG
# ==========================================
st.set_page_config(page_title="RAG-CSA Research Framework", layout="wide")

keys = [
    'G', 'classes', 'embeddings', 'node_map', 'results', 
    'convergence', 'weights', 'raw_fitness', 'best_clusters', 
    'repo_name', 'docs'
]
for k in keys:
    if k not in st.session_state: st.session_state[k] = None

st.title("🧪 Synergizing LLMs & Circle Search for Software Architecture")

tabs = st.tabs(["1. Data & RAG", "2. Optuna Tuning", "3. Experiments", "4. Statistics & Plots", "5. LLM Naming"])

# ==========================================
# TAB 1: DATA INGESTION
# ==========================================
with tabs[0]:
    st.header("Step 1: Dataset & Embeddings")
    
    col1, col2 = st.columns(2)
    with col1:
        repo_select = st.selectbox("Select Target System", list(REPOS.keys()))
        
        if st.button("Download & Parse Code"):
            with st.status(f"Processing {repo_select}...", expanded=True) as status:
                st.write("Cloning repository...")
                path = download_repo(repo_select)
                if path:
                    st.write("Parsing Abstract Syntax Trees (AST)...")
                    G, docs = parse_java_project(path)
                    st.session_state.G = G
                    st.session_state.docs = docs
                    st.session_state.classes = list(G.nodes())
                    st.session_state.node_map = {i: n for i, n in enumerate(st.session_state.classes)}
                    st.session_state.repo_name = repo_select
                    status.update(label="Ingestion Complete!", state="complete", expanded=False)
                    st.success(f"Loaded {len(G.nodes)} Classes from {repo_select}")
                else:
                    status.update(label="Download Failed", state="error")

    with col2:
        if st.session_state.G:
            st.write("### Semantic Analysis")
            st.info(f"Ready to generate embeddings for {len(st.session_state.classes)} classes.")
            
            if st.button("Generate RAG Embeddings"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_ui(current, total, msg):
                    progress_bar.progress(current / total)
                    status_text.text(f"{msg} ({current}/{total})")

                rel_docs = {k: v for k, v in st.session_state.docs.items() if k in st.session_state.classes}
                keys_list, embs = generate_embeddings_for_classes(rel_docs, progress_callback=update_ui)
                
                status_text.text("Aligning vectors...")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()
                
                ordered_embs = []
                for cls in st.session_state.classes:
                    if cls in keys_list:
                        idx = keys_list.index(cls)
                        ordered_embs.append(embs[idx])
                    else:
                        ordered_embs.append(np.zeros(1024))
                
                st.session_state.embeddings = np.array(ordered_embs)
                st.success("Embeddings Generated!")
                
                with st.expander("View Vector Space (PCA)", expanded=True):
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(st.session_state.embeddings)
                    df_pca = pd.DataFrame(coords, columns=["x", "y"])
                    st.scatter_chart(df_pca, x="x", y="y")

# ==========================================
# TAB 2: OPTUNA TUNING
# ==========================================
with tabs[1]:
    st.header("Hyper-parameter Tuning")
    
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        n_trials = st.slider("Number of Trials", min_value=5, max_value=100, value=15, step=5)
    with col_opt2:
        st.write("Search Space: w1 [0.0 - 1.0]")
    
    if st.button("Start Tuning"):
        if st.session_state.embeddings is None:
            st.error("Load data first.")
        else:
            opt_bar = st.progress(0)
            opt_status = st.empty()
            
            class StreamlitCallback:
                def __call__(self, study, trial):
                    progress = len(study.trials) / n_trials
                    opt_bar.progress(min(progress, 1.0))
                    opt_status.text(f"Trial {len(study.trials)}/{n_trials} | Best Value: {study.best_value:.4f}")

            def objective(trial):
                w1 = trial.suggest_float("w1", 0.0, 1.0)
                w2 = 1.0 - w1
                # FIXED: Extract len into variable
                dim_tune = len(st.session_state.classes)
                opt = CSA("Tune", 10, dim_tune, 15) 
                def wrap(x): return combined_fitness(x, st.session_state.G, st.session_state.embeddings, w1, w2)[0]
                score, _ = opt.optimize(wrap)
                return score

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, callbacks=[StreamlitCallback()])
            
            best_w1 = study.best_params['w1']
            st.session_state.weights = (best_w1, 1.0 - best_w1)
            
            opt_status.success(f"Tuning Finished! Optimal Weights: w1={best_w1:.2f}, w2={1.0-best_w1:.2f}")

            st.divider()
            st.subheader("Optimization History")
            fig = optuna.visualization.plot_optimization_history(study)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Detailed Trials Data")
            df_trials = study.trials_dataframe()
            cols_to_show = [c for c in ['number', 'value', 'params_w1', 'duration', 'state'] if c in df_trials.columns]
            df_display = df_trials[cols_to_show].rename(columns={'params_w1': 'w1 Value', 'value': 'Fitness'})
            st.dataframe(df_display.style.highlight_max(axis=0, subset=['Fitness']), use_container_width=True)

# ==========================================
# TAB 3: EXPERIMENTS (PARALLELIZED)
# ==========================================
with tabs[2]:
    st.header("Comparative Experiments")
    st.markdown("Run the **full benchmark** using parallel processing.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        n_runs = st.slider("Runs per Algorithm", 1, 30, 5)
    with c2:
        n_pop = st.slider("Population Size", 10, 200, 20)
    with c3:
        n_iter = st.slider("Iterations", 10, 500, 50)
    
    if st.button("🚀 Run Benchmark"):
        if st.session_state.embeddings is None:
            st.error("No Data Loaded.")
        else:
            # 1. EXTRACT SESSION STATE VARIABLES TO LOCALS BEFORE PARALLELIZATION
            # This fixes the AttributeError/KeyError in joblib threads
            w1, w2 = st.session_state.weights if st.session_state.weights else (0.6, 0.4)
            G = st.session_state.G
            Emb = st.session_state.embeddings
            repo = st.session_state.repo_name
            dim_size = len(st.session_state.classes) # <--- CRITICAL FIX
            
            alg_classes = [CSA, GA, PSO, WOA, GWO, SSA, FA, DE] 
            
            main_bar = st.progress(0)
            status_txt = st.empty()
            
            results_data = []
            raw_fitness_data = {} 
            convergence_data = {}
            total_algs = len(alg_classes)
            
            for idx, Alg in enumerate(alg_classes):
                name = Alg.__name__
                status_txt.markdown(f"**Processing {name}...** ({idx+1}/{total_algs})")
                
                # Use extracted local variable 'dim_size' instead of st.session_state
                batch_results = Parallel(n_jobs=-1)(
                    delayed(run_single_experiment)(
                        Alg, r+1, n_pop, dim_size, n_iter, G, Emb, w1, w2, repo
                    ) 
                    for r in range(n_runs)
                )
                
                fitness_scores = []
                histories = []
                
                for res in batch_results:
                    clean_res = {k:v for k,v in res.items() if k not in ['History', 'Clusters']}
                    results_data.append(clean_res)
                    fitness_scores.append(res['Fitness'])
                    histories.append(res['History'])
                    
                    if name == "CSA" and res['Clusters'] is not None:
                        current_max = max(raw_fitness_data.get("CSA", [-1])) if "CSA" in raw_fitness_data else -1
                        if res['Fitness'] > current_max:
                            st.session_state.best_clusters = res['Clusters']

                raw_fitness_data[name] = fitness_scores
                max_len = max(len(h) for h in histories)
                padded = [h + [h[-1]]*(max_len-len(h)) for h in histories]
                convergence_data[name] = np.mean(np.array(padded), axis=0).tolist()
                
                main_bar.progress((idx + 1) / total_algs)

            st.session_state.results = pd.DataFrame(results_data)
            st.session_state.raw_fitness = raw_fitness_data
            st.session_state.convergence = convergence_data
            status_txt.success("Benchmark Complete! Go to Tab 4.")

# ==========================================
# TAB 4: STATISTICS & PLOTS
# ==========================================
with tabs[3]:
    st.header("Manuscript Artifacts & Deep Analysis")
    
    if st.session_state.results is not None:
        df = st.session_state.results
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results CSV", data=csv, file_name=f"results_{st.session_state.repo_name}.csv", mime="text/csv")
        
        st.subheader("1. Performance Summary")
        summary = df.groupby("Algorithm").agg({
            "Fitness": ["mean", "std"], "MQ": ["mean", "std"], 
            "Cohesion": ["mean", "std"], "GT_Accuracy": ["mean", "std"], "Time": ["mean"]
        })
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        st.dataframe(summary.style.highlight_max(axis=0, subset=['Fitness_mean', 'MQ_mean', 'GT_Accuracy_mean']), use_container_width=True)

        st.subheader("2. Wilcoxon Test (vs CSA)")
        if "CSA" in st.session_state.raw_fitness:
            csa_scores = st.session_state.raw_fitness["CSA"]
            rows = []
            for alg, scores in st.session_state.raw_fitness.items():
                if alg == "CSA": continue
                try:
                    s, p = wilcoxon(csa_scores, scores)
                    rows.append({"Comparison": f"CSA vs {alg}", "p-value": f"{p:.5e}", "Result": "✅ Significant" if p < 0.05 else "❌ Not Sig."})
                except:
                    rows.append({"Comparison": f"CSA vs {alg}", "p-value": "-", "Result": "Error"})
            st.table(pd.DataFrame(rows))

        st.subheader("3. Visualizations")
        df_norm = df.groupby("Algorithm").mean(numeric_only=True).reset_index()
        cols_norm = ['Fitness', 'MQ', 'Cohesion', 'GT_Accuracy']
        for c in cols_norm:
            mn, mx = df_norm[c].min(), df_norm[c].max()
            if mx > mn: df_norm[c] = (df_norm[c] - mn) / (mx - mn)
        
        fig_radar = go.Figure()
        for i, row in df_norm.iterrows():
            fig_radar.add_trace(go.Scatterpolar(r=[row[c] for c in cols_norm], theta=cols_norm, fill='toself', name=row['Algorithm']))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), height=400)
        st.plotly_chart(fig_radar, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**B. Fitness Distribution**")
            st.plotly_chart(px.box(df, x="Algorithm", y="Fitness", color="Algorithm"), use_container_width=True)
        with c2:
            st.markdown("**C. Convergence**")
            fig_conv = plt.figure(figsize=(6, 4))
            for name, hist in st.session_state.convergence.items():
                plt.plot(hist, label=name)
            plt.legend(fontsize='small'); plt.grid(True, alpha=0.3)
            st.pyplot(fig_conv)
    else:
        st.warning("No results yet.")

# ==========================================
# TAB 5: LLM NAMING
# ==========================================
with tabs[4]:
    st.header("Automated Naming")
    if st.session_state.best_clusters is not None:
        if st.button("Generate Names"):
            clusters_map = {}
            for i, c_id in enumerate(st.session_state.best_clusters):
                if c_id not in clusters_map: clusters_map[c_id] = []
                clusters_map[c_id].append(st.session_state.classes[i])
            
            naming_bar = st.progress(0)
            unique_ids = [k for k, v in clusters_map.items() if len(v) >= 2]
            cols = st.columns(3)
            
            for idx, c_id in enumerate(unique_ids):
                members = clusters_map[c_id]
                s_name = generate_service_name(members)
                with cols[idx % 3]:
                    st.info(f"**{s_name}**")
                    with st.expander("Classes"): st.text("\n".join(members))
                naming_bar.progress((idx + 1) / len(unique_ids))
    else:
        st.warning("Run experiments first.")