#!/usr/bin/env python
# coding: utf-8

# ## Mitochondria specific overexpression of Bola3 Proteomic Analysis from TDA282

# #### *** Samples - NODOX(3), mBola3_NODOX(3), mBola3_DOX(3)

# ### Core Biological Questions:

# ### 1. Does Bola3 overepresssion in the mitochondria rescue or supress ferroptosis

# In[1]:


# Install all the missing packages
get_ipython().system('pip install statsmodels matplotlib plotly')

# Now import all the required libraries
import os
import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats.multitest import multipletests  # This should now work after installation

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt  # This should now work after installing matplotlib
import plotly.graph_objects as go  # This should now work after installing plotly


# #### 1. Load data and convert #NUM! to missing values

# In[2]:


DATA_PATH = "Input data/mBola3Oe_v5.csv"

OUTDIR = "mBola3_analysis_outputs"
os.makedirs(OUTDIR, exist_ok=True)

gene_col = "PG.Genes"
desc_col = "PG.ProteinDescriptions"

control_cols = ["NODOX-1", "NODOX-2", "NODOX-3"]
test_cols    = ["mBola3_Dox-1", "mBola3_Dox-2"]

sample_cols = control_cols + test_cols


# In[3]:


df = pd.read_csv(
    DATA_PATH,
    na_values=["#NUM!", "NUM!", "NaN", "nan", "", "NA"]
)

print(df.columns.tolist())


# In[4]:


# Make sure intensity columns are numeric
all_possible_sample_cols = [
    c for c in df.columns
    if c not in [gene_col, desc_col]
]

for c in all_possible_sample_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print("Detected sample columns:")
print(all_possible_sample_cols)


# #### 2. Basic missingness QC

# In[5]:


missing_summary = pd.DataFrame({
    "sample": all_possible_sample_cols,
    "missing_count": [df[c].isna().sum() for c in all_possible_sample_cols],
    "detected_count": [df[c].notna().sum() for c in all_possible_sample_cols],
})

missing_summary["missing_percent"] = (
    missing_summary["missing_count"] / len(df) * 100
)

missing_summary = missing_summary.sort_values("missing_percent", ascending=False)

missing_summary


# In[6]:


missing_summary.to_csv(
    os.path.join(OUTDIR, "sample_missingness_summary.csv"),
    index=False
)


# In[7]:


plt.figure(figsize=(7, 4))
plt.bar(missing_summary["sample"], missing_summary["missing_percent"])
plt.ylabel("Missing values (%)")
plt.xlabel("Sample")
plt.title("Missingness per sample")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "missingness_per_sample.png"), dpi=300)
plt.show()


# #### 3. Check BOLA3 overexpression

# In[8]:


bola3 = df[df[gene_col].astype(str).str.contains(r"\bBOLA3\b", case=False, regex=True, na=False)]

bola3[[gene_col, desc_col] + all_possible_sample_cols]


# #### 4. Log2 transform

# In[9]:


df_log2 = df.copy()

for c in all_possible_sample_cols:
    df_log2[c] = np.log2(df_log2[c])

df_log2[[gene_col, desc_col] + sample_cols].head()

for c in all_possible_sample_cols:
    n_inf = np.isinf(df_log2[c]).sum()
    print(c, "infinite values:", n_inf)


# #### 5. Median normalization
# 
# This subtracts the sample median from each sample. It preserves fold-change structure while correcting sample-level intensity shifts.

# In[10]:


df_norm = df_log2.copy()

sample_medians = df_norm[all_possible_sample_cols].median(axis=0, skipna=True)

sample_medians

for c in all_possible_sample_cols:
    df_norm[c] = df_norm[c] - sample_medians[c]

df_norm[[gene_col, desc_col] + sample_cols].head()

df_norm.to_csv(
    os.path.join(OUTDIR, "log2_median_normalized_data.csv"),
    index=False
)


# #### 6. Intensity distribution before and after normalization

# In[11]:


plt.figure(figsize=(8, 5))

for c in sample_cols:
    values = df_log2[c].dropna()
    plt.hist(values, bins=80, alpha=0.35, label=c)

plt.xlabel("log2 intensity")
plt.ylabel("Protein count")
plt.title("Log2 intensity distributions before normalization")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "log2_distributions_before_normalization.png"), dpi=300)
plt.show()


# In[12]:


plt.figure(figsize=(8, 5))

for c in sample_cols:
    values = df_norm[c].dropna()
    plt.hist(values, bins=80, alpha=0.35, label=c)

plt.xlabel("Median-normalized log2 intensity")
plt.ylabel("Protein count")
plt.title("Log2 intensity distributions after median normalization")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "log2_distributions_after_normalization.png"), dpi=300)
plt.show()


# #### 7. Sample-sample correlation heatmap

# In[13]:


corr = df_norm[sample_cols].corr(method="pearson", min_periods=1000)

corr


# In[14]:


plt.figure(figsize=(6, 5))
plt.imshow(corr.values)
plt.colorbar(label="Pearson correlation")

plt.xticks(range(len(sample_cols)), sample_cols, rotation=45, ha="right")
plt.yticks(range(len(sample_cols)), sample_cols)

for i in range(len(sample_cols)):
    for j in range(len(sample_cols)):
        plt.text(j, i, f"{corr.iloc[i, j]:.3f}", ha="center", va="center")

plt.title("Sample-sample correlation")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "sample_correlation_heatmap.png"), dpi=300)
plt.show()


# #### 8. PCA

# In[15]:


pca_input = df_norm[[gene_col, desc_col] + sample_cols].dropna(subset=sample_cols).copy()

X = pca_input[sample_cols].T

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
pca_coords = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "sample": sample_cols,
    "group": ["NoDox"] * len(control_cols) + ["Dox"] * len(test_cols),
    "PC1": pca_coords[:, 0],
    "PC2": pca_coords[:, 1]
})

pca_df


# In[16]:


plt.figure(figsize=(6, 5))

for group in pca_df["group"].unique():
    sub = pca_df[pca_df["group"] == group]
    plt.scatter(sub["PC1"], sub["PC2"], s=100, label=group)

    for _, row in sub.iterrows():
        plt.text(row["PC1"], row["PC2"], row["sample"], fontsize=9)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
plt.title("PCA using complete-case proteins")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "PCA_complete_case_proteins.png"), dpi=300)
plt.show()


# #### 9. Detection filtering

# In[17]:


analysis_df = df_norm[[gene_col, desc_col] + sample_cols].copy()

analysis_df["n_detected_control"] = analysis_df[control_cols].notna().sum(axis=1)
analysis_df["n_detected_dox"] = analysis_df[test_cols].notna().sum(axis=1)

quant_df = analysis_df[
    (analysis_df["n_detected_control"] >= 2) &
    (analysis_df["n_detected_dox"] == 2)
].copy()

print("Proteins used for quantitative analysis:", quant_df.shape[0])


# In[18]:


analysis_df.to_csv(
    os.path.join(OUTDIR, "complete_case_proteins_3control_2dox.csv"),
    index=False
)


# #### 10. Differential abundance analysis

# In[19]:


results = quant_df.copy()

results["mean_control"] = results[control_cols].mean(axis=1, skipna=True)
results["mean_dox"] = results[test_cols].mean(axis=1, skipna=True)

results["log2FC"] = results["mean_dox"] - results["mean_control"]

results["sd_control"] = results[control_cols].std(axis=1, skipna=True)
results["sd_dox"] = results[test_cols].std(axis=1, skipna=True)

results["valid_control"] = results[control_cols].notna().sum(axis=1)
results["valid_dox"] = results[test_cols].notna().sum(axis=1)

pvals = []

for _, row in results.iterrows():
    control_vals = row[control_cols].astype(float).dropna().values
    dox_vals = row[test_cols].astype(float).dropna().values

    if len(control_vals) >= 2 and len(dox_vals) >= 2:
        _, pval = stats.ttest_ind(
            dox_vals,
            control_vals,
            equal_var=False
        )
    else:
        pval = np.nan

    pvals.append(pval)

results["pval"] = pvals

valid_p = results["pval"].notna()

results["FDR"] = np.nan
results.loc[valid_p, "FDR"] = multipletests(
    results.loc[valid_p, "pval"],
    method="fdr_bh"
)[1]

epsilon = 1e-300
results["neg_log10_pval"] = -np.log10(results["pval"] + epsilon)
results["neg_log10_FDR"] = -np.log10(results["FDR"] + epsilon)

results = results.sort_values("log2FC", ascending=False)

results.to_csv(
    os.path.join(OUTDIR, "mBola3_Dox_vs_NoDox_differential_results.csv"),
    index=False
)

results.head(20)


# #### 11. Define significant proteins

# In[20]:


log2fc_threshold = 0.50
p_threshold = 0.05
fdr_threshold = 0.10


# In[21]:


results["regulation_pval"] = "Not significant"

results.loc[
    (results["log2FC"] >= log2fc_threshold) &
    (results["pval"] < p_threshold),
    "regulation_pval"
] = "Upregulated in Dox"

results.loc[
    (results["log2FC"] <= -log2fc_threshold) &
    (results["pval"] < p_threshold),
    "regulation_pval"
] = "Downregulated in Dox"

results["regulation_FDR"] = "Not significant"

results.loc[
    (results["log2FC"] >= log2fc_threshold) &
    (results["FDR"] < fdr_threshold),
    "regulation_FDR"
] = "Upregulated in Dox"

results.loc[
    (results["log2FC"] <= -log2fc_threshold) &
    (results["FDR"] < fdr_threshold),
    "regulation_FDR"
] = "Downregulated in Dox"

results["regulation_pval"].value_counts()


# In[22]:


results["regulation_FDR"].value_counts()


# In[23]:


up_pval = results[results["regulation_pval"] == "Upregulated in Dox"].copy()
down_pval = results[results["regulation_pval"] == "Downregulated in Dox"].copy()

up_fdr = results[results["regulation_FDR"] == "Upregulated in Dox"].copy()
down_fdr = results[results["regulation_FDR"] == "Downregulated in Dox"].copy()

up_pval.to_csv(os.path.join(OUTDIR, "upregulated_pval_0p05.csv"), index=False)
down_pval.to_csv(os.path.join(OUTDIR, "downregulated_pval_0p05.csv"), index=False)

up_fdr.to_csv(os.path.join(OUTDIR, "upregulated_FDR_0p10.csv"), index=False)
down_fdr.to_csv(os.path.join(OUTDIR, "downregulated_FDR_0p10.csv"), index=False)


# #### 12. Condition-specific proteins
# These are proteins detected in both Dox samples but missing in both controls, or vice versa.
# 
# Do not convert their missing values to zero. Keep them as a separate biological category.

# In[24]:


condition_specific_df = analysis_df.copy()

dox_specific = condition_specific_df[
    (condition_specific_df["n_detected_control"] == 0) &
    (condition_specific_df["n_detected_dox"] == 2)
].copy()

control_specific = condition_specific_df[
    (condition_specific_df["n_detected_control"] >= 2) &
    (condition_specific_df["n_detected_dox"] == 0)
].copy()

print("Dox-specific proteins:", dox_specific.shape[0])
print("Control-specific proteins:", control_specific.shape[0])

dox_specific.to_csv(
    os.path.join(OUTDIR, "dox_specific_detected_2of2_missing_controls.csv"),
    index=False
)

control_specific.to_csv(
    os.path.join(OUTDIR, "control_specific_detected_controls_missing_dox.csv"),
    index=False
)


# In[25]:


dox_specific.to_csv(
    os.path.join(OUTDIR, "dox_specific_detected_2of2_missing_controls.csv"),
    index=False
)

control_specific.to_csv(
    os.path.join(OUTDIR, "control_specific_detected_2of2_missing_dox.csv"),
    index=False
)


# #### 13. Volcano Plot

# In[26]:


import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# Volcano plot for mBola3-Dox vs NoDox
# Same layout/style as your previous volcano plot
# ============================================================

# Output folder
OUTDIR = "mBola3_analysis_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------------------------------------------
# Input dataframe
# ------------------------------------------------------------
# This assumes your differential analysis dataframe is called `results`
# and contains: log2FC, pval, FDR

res_dox_df = results.copy()

# Standardized column names for this plot
fc_col = "log2fc_DOX_vs_NODOX"
p_col = "pval_DOX_vs_NODOX"
fdr_col = "FDR_DOX_vs_NODOX"

gene_col = "PG.Genes"
desc_col = "PG.ProteinDescriptions"

# Create volcano-specific columns from current results table
res_dox_df[fc_col] = res_dox_df["log2FC"]
res_dox_df[p_col] = res_dox_df["pval"]
res_dox_df[fdr_col] = res_dox_df["FDR"]

# ------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------
p_threshold = 0.05
log2fc_threshold = 0.50

# ------------------------------------------------------------
# Appearance settings
# ------------------------------------------------------------
point_size_not_sig = 8
point_size_sig = 11
label_font_size = 18
axis_font_size = 18
title_font_size = 22
legend_font_size = 15

# Edit this list as needed
label_genes = [
    "BOLA3",
    "FDXR",
    "HK1",
    "SHMT1",
    "FASN",
    "NDUFAF2"
]

label_genes = sorted(set([g.upper() for g in label_genes]))


# In[27]:


# ============================================================
# Helper functions
# ============================================================

def split_gene_symbols(x):
    if pd.isna(x):
        return []

    parts = re.split(r"[;,|]+", str(x))

    genes = []
    for part in parts:
        gene = part.strip().upper()
        if gene not in ["", "NAN", "NONE"]:
            genes.append(gene)

    return sorted(set(genes))


def row_contains_label_gene(row):
    row_genes = split_gene_symbols(row[gene_col])
    matched = sorted(set(row_genes).intersection(label_genes))
    return "; ".join(matched)


def make_customdata(df_in, columns):
    if df_in.shape[0] == 0:
        return np.empty((0, len(columns)))

    arrays = []

    for col in columns:
        if col in df_in.columns:
            arrays.append(df_in[col].astype(str).to_numpy())
        else:
            arrays.append(np.array([""] * df_in.shape[0]))

    return np.stack(arrays, axis=-1)


# In[28]:


# ============================================================
# Prepare dataframe
# ============================================================

volcano_df = res_dox_df.copy()

volcano_df[fc_col] = pd.to_numeric(volcano_df[fc_col], errors="coerce")
volcano_df[p_col] = pd.to_numeric(volcano_df[p_col], errors="coerce")

if fdr_col in volcano_df.columns:
    volcano_df[fdr_col] = pd.to_numeric(volcano_df[fdr_col], errors="coerce")
else:
    volcano_df[fdr_col] = np.nan

volcano_df = volcano_df.dropna(subset=[fc_col, p_col]).copy()

volcano_df[p_col] = volcano_df[p_col].clip(lower=1e-300)
volcano_df["neg_log10_pval"] = -np.log10(volcano_df[p_col])


# In[29]:


# ============================================================
# Define volcano categories
# ============================================================

volcano_df["Volcano_Category"] = "Not significant"

volcano_df.loc[
    (volcano_df[p_col] < p_threshold) &
    (volcano_df[fc_col] > log2fc_threshold),
    "Volcano_Category"
] = "Upregulated in Dox"

volcano_df.loc[
    (volcano_df[p_col] < p_threshold) &
    (volcano_df[fc_col] < -log2fc_threshold),
    "Volcano_Category"
] = "Downregulated in Dox"

# Add selected text labels
volcano_df["Selected_Label"] = volcano_df.apply(row_contains_label_gene, axis=1)


# In[30]:


# ============================================================
# Split data
# ============================================================

not_sig = volcano_df[volcano_df["Volcano_Category"] == "Not significant"].copy()
up = volcano_df[volcano_df["Volcano_Category"] == "Upregulated in Dox"].copy()
down = volcano_df[volcano_df["Volcano_Category"] == "Downregulated in Dox"].copy()
selected = volcano_df[volcano_df["Selected_Label"] != ""].copy()

up_count = up.shape[0]
down_count = down.shape[0]
not_sig_count = not_sig.shape[0]

print("Volcano plot summary:")
print("Upregulated in Dox:", up_count)
print("Downregulated in Dox:", down_count)
print("Not significant:", not_sig_count)
print("Selected labeled proteins found:", selected.shape[0])

print("\nSelected proteins found in dataset:")
if selected.shape[0] > 0:
    display(
        selected[
            [gene_col, "Selected_Label", "Volcano_Category", fc_col, p_col, fdr_col]
        ].sort_values("Selected_Label")
    )


# In[31]:


# ============================================================
# Build volcano plot
# ============================================================

fig = go.Figure()

# Not significant
fig.add_trace(
    go.Scattergl(
        x=not_sig[fc_col],
        y=not_sig["neg_log10_pval"],
        mode="markers",
        name=f"Not significant ({not_sig_count})",
        marker=dict(
            color="lightgray",
            size=point_size_not_sig,
            opacity=0.50
        ),
        text=not_sig[gene_col].astype(str),
        customdata=make_customdata(not_sig, [desc_col, p_col, fdr_col]),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Description: %{customdata[0]}<br>"
            "log2FC: %{x:.3f}<br>"
            "-log10(p): %{y:.3f}<br>"
            "raw p: %{customdata[1]}<br>"
            "FDR: %{customdata[2]}<br>"
            "<extra></extra>"
        )
    )
)

# Upregulated in Dox
fig.add_trace(
    go.Scattergl(
        x=up[fc_col],
        y=up["neg_log10_pval"],
        mode="markers",
        name=f"Upregulated in Dox ({up_count})",
        marker=dict(
            color="green",
            size=point_size_sig,
            opacity=0.85
        ),
        text=up[gene_col].astype(str),
        customdata=make_customdata(up, [desc_col, p_col, fdr_col]),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Description: %{customdata[0]}<br>"
            "log2FC: %{x:.3f}<br>"
            "-log10(p): %{y:.3f}<br>"
            "raw p: %{customdata[1]}<br>"
            "FDR: %{customdata[2]}<br>"
            "<extra></extra>"
        )
    )
)

# Downregulated in Dox
fig.add_trace(
    go.Scattergl(
        x=down[fc_col],
        y=down["neg_log10_pval"],
        mode="markers",
        name=f"Downregulated in Dox ({down_count})",
        marker=dict(
            color="red",
            size=point_size_sig,
            opacity=0.85
        ),
        text=down[gene_col].astype(str),
        customdata=make_customdata(down, [desc_col, p_col, fdr_col]),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Description: %{customdata[0]}<br>"
            "log2FC: %{x:.3f}<br>"
            "-log10(p): %{y:.3f}<br>"
            "raw p: %{customdata[1]}<br>"
            "FDR: %{customdata[2]}<br>"
            "<extra></extra>"
        )
    )
)


# In[32]:


# ============================================================
# Add offset annotations for selected proteins
# Labels below dots; BOLA3 slightly larger
# ============================================================

for _, row in selected.iterrows():
    label_text = str(row["Selected_Label"]).strip()

    if label_text == "":
        continue

    current_font_size = (
        label_font_size + 7
        if "BOLA3" in label_text.upper()
        else label_font_size
    )

    fig.add_annotation(
        x=row[fc_col],
        y=row["neg_log10_pval"],
        text=label_text,
        showarrow=False,
        yshift=-18,
        xshift=0,
        font=dict(
            size=current_font_size,
            color="black"
        ),
        xanchor="center",
        yanchor="top",
        align="center"
    )


# In[33]:


# ============================================================
# Threshold lines
# ============================================================

fig.add_hline(
    y=-np.log10(p_threshold),
    line_dash="dash",
    line_color="black",
    line_width=2,
    annotation_text=f"p = {p_threshold}",
    annotation_position="top left",
    annotation_font_size=14
)

fig.add_vline(
    x=log2fc_threshold,
    line_dash="dot",
    line_color="black",
    line_width=2,
    annotation_text="+0.5",
    annotation_position="top right",
    annotation_font_size=14
)

fig.add_vline(
    x=-log2fc_threshold,
    line_dash="dot",
    line_color="black",
    line_width=2,
    annotation_text="-0.5",
    annotation_position="top left",
    annotation_font_size=14
)


# In[34]:


# ============================================================
# Layout
# ============================================================

fig.update_layout(
    title=dict(
        text=(
            f"Volcano Plot: mBola3-Dox vs NoDox<br>"
            f"Raw p < {p_threshold}, |log2FC| > {log2fc_threshold}; "
            f"{up_count} up, {down_count} down"
        ),
        font=dict(size=title_font_size)
    ),
    xaxis_title="log2 fold change, Dox vs NoDox",
    yaxis_title="-log10(raw p-value)",
    template="simple_white",
    width=1200,
    height=900,
    legend_title_text="Protein category",
    legend=dict(
        font=dict(size=legend_font_size)
    )
)

fig.update_xaxes(
    title_font=dict(size=axis_font_size),
    tickfont=dict(size=15),
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor="black"
)

fig.update_yaxes(
    title_font=dict(size=axis_font_size),
    tickfont=dict(size=15)
)

fig.show()


# In[35]:


# ============================================================
# Save outputs
# ============================================================

volcano_html = os.path.join(
    OUTDIR,
    "mBola3_Dox_vs_NoDox_volcano_plot_larger_dots_labels_log2fc_boundaries.html"
)

selected_csv = os.path.join(
    OUTDIR,
    "mBola3_Dox_vs_NoDox_selected_labeled_proteins_on_volcano.csv"
)

full_volcano_csv = os.path.join(
    OUTDIR,
    "mBola3_Dox_vs_NoDox_full_volcano_table.csv"
)

fig.write_html(volcano_html)
selected.to_csv(selected_csv, index=False)
volcano_df.to_csv(full_volcano_csv, index=False)

print("\nSaved:")
print(volcano_html)
print(selected_csv)
print(full_volcano_csv)


# #### 15. Heatmap of top changed proteins

# In[36]:


top_n = 25

top_up = results.sort_values("log2FC", ascending=False).head(top_n)
top_down = results.sort_values("log2FC", ascending=True).head(top_n)

top_heatmap_df = pd.concat([top_up, top_down], axis=0).drop_duplicates(subset=[gene_col])

heatmap_matrix = top_heatmap_df[sample_cols].copy()
heatmap_matrix.index = top_heatmap_df[gene_col].astype(str)

# Row z-score
heatmap_z = heatmap_matrix.sub(heatmap_matrix.mean(axis=1), axis=0)
heatmap_z = heatmap_z.div(heatmap_matrix.std(axis=1), axis=0)


# In[37]:


plt.figure(figsize=(7, 12))

plt.imshow(heatmap_z.values, aspect="auto")
plt.colorbar(label="Row z-score")

plt.xticks(range(len(sample_cols)), sample_cols, rotation=45, ha="right")
plt.yticks(range(len(heatmap_z.index)), heatmap_z.index, fontsize=7)

plt.title("Top up/down proteins: mBola3-Dox vs NoDox")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "top_up_down_heatmap.png"), dpi=300)
plt.show()


# In[38]:


top_heatmap_df.to_csv(
    os.path.join(OUTDIR, "top_up_down_proteins_for_heatmap.csv"),
    index=False
)


# #### 18. Export ranked gene list for pathway analysis
# 
# For GO / KEGG / Reactome later, save ranked genes by log2FC.

# In[39]:


ranked = results[[gene_col, desc_col, "log2FC", "pval", "FDR"]].copy()

# For gene sets, use first gene symbol if multiple symbols are separated by ;
ranked["GeneSymbol"] = ranked[gene_col].astype(str).str.split(";").str[0]

ranked = ranked.sort_values("log2FC", ascending=False)

ranked.to_csv(
    os.path.join(OUTDIR, "ranked_gene_list_for_pathway_analysis.csv"),
    index=False
)

ranked.head()


# In[40]:


up_genes = ranked[
    (ranked["log2FC"] >= log2fc_threshold) &
    (ranked["pval"] < p_threshold)
].copy()

down_genes = ranked[
    (ranked["log2FC"] <= -log2fc_threshold) &
    (ranked["pval"] < p_threshold)
].copy()

up_genes["GeneSymbol"].dropna().drop_duplicates().to_csv(
    os.path.join(OUTDIR, "upregulated_gene_symbols.txt"),
    index=False,
    header=False
)

down_genes["GeneSymbol"].dropna().drop_duplicates().to_csv(
    os.path.join(OUTDIR, "downregulated_gene_symbols.txt"),
    index=False,
    header=False
)


# #### 19. Optional: run GO / Reactome enrichment with g:Profiler

# In[41]:


get_ipython().system('pip install gprofiler-official')


# In[42]:


from gprofiler import GProfiler

gp = GProfiler(return_dataframe=True)

up_gene_list = up_genes["GeneSymbol"].dropna().drop_duplicates().tolist()
down_gene_list = down_genes["GeneSymbol"].dropna().drop_duplicates().tolist()

up_enrich = gp.profile(
    organism="hsapiens",
    query=up_gene_list,
    sources=["GO:BP", "GO:CC", "GO:MF", "REAC", "KEGG"]
)

down_enrich = gp.profile(
    organism="hsapiens",
    query=down_gene_list,
    sources=["GO:BP", "GO:CC", "GO:MF", "REAC", "KEGG"]
)


# In[43]:


up_enrich.to_csv(
    os.path.join(OUTDIR, "GO_Reactome_KEGG_upregulated_enrichment.csv"),
    index=False
)

down_enrich.to_csv(
    os.path.join(OUTDIR, "GO_Reactome_KEGG_downregulated_enrichment.csv"),
    index=False
)

up_enrich.head(20)


# In[44]:


down_enrich.head(20)


# #### Ranked gene set analysis with Mitocarta

# In[71]:


import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import plotly.graph_objects as go

# ============================================================
# Ranked gene-set analysis for mBola3 overexpression
# Dox vs NoDox
# Same structure/style as previous knockout / WT-vs-Dox analysis
# ============================================================

OUTDIR = "mBola3_analysis_outputs"
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# 1. Load all mBola3-Dox vs NoDox differential results
# ============================================================

# Preferred saved result file from the mBola3 overexpression analysis.
all_results_file = os.path.join(
    OUTDIR,
    "mBola3_Dox_vs_NoDox_differential_results.csv"
)

# If results dataframe already exists in memory, use it.
# Otherwise read from the saved CSV.
if "results" in globals():
    print("Using existing `results` dataframe from memory.")
    res_dox_df = results.copy()
elif os.path.exists(all_results_file):
    print("Reading result file:")
    print(all_results_file)
    res_dox_df = pd.read_csv(all_results_file)
else:
    raise FileNotFoundError(
        "Could not find `results` in memory or the saved differential result file. "
        f"Expected file: {all_results_file}"
    )

gene_col = "PG.Genes"
desc_col = "PG.ProteinDescriptions"

# Original columns from the mBola3 differential analysis
raw_fc_col = "log2FC"
raw_p_col = "pval"
raw_fdr_col = "FDR"

# Standardized names to preserve your previous code style
fc_col = "log2fc_DOX_vs_NODOX"
p_col = "pval_DOX_vs_NODOX"
fdr_col = "FDR_DOX_vs_NODOX"

# Create standardized columns
if fc_col not in res_dox_df.columns:
    res_dox_df[fc_col] = res_dox_df[raw_fc_col]

if p_col not in res_dox_df.columns:
    res_dox_df[p_col] = res_dox_df[raw_p_col]

if fdr_col not in res_dox_df.columns:
    if raw_fdr_col in res_dox_df.columns:
        res_dox_df[fdr_col] = res_dox_df[raw_fdr_col]
    else:
        res_dox_df[fdr_col] = np.nan

print("All-protein mBola3-Dox result table:")
print(res_dox_df.shape)

print("\nColumns used:")
print("Gene column:", gene_col)
print("Description column:", desc_col)
print("Fold-change column:", fc_col)
print("p-value column:", p_col)
print("FDR column:", fdr_col)


# In[72]:


# ============================================================
# 2. Helper functions
# ============================================================

def split_gene_symbols(x):
    if pd.isna(x):
        return []

    # Includes ; , | and /, matching your earlier code
    parts = re.split(r"[;,|/]+", str(x))

    genes = []
    for part in parts:
        gene = part.strip().upper()
        if gene not in ["", "NAN", "NONE", "NULL"]:
            genes.append(gene)

    return sorted(set(genes))


def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    fdr = np.full_like(pvals, np.nan, dtype=float)

    valid = ~np.isnan(pvals)

    if valid.sum() == 0:
        return fdr

    p = pvals[valid]
    order = np.argsort(p)
    ranked_p = p[order]

    m = len(p)
    bh = ranked_p * m / np.arange(1, m + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.minimum(bh, 1.0)

    temp = np.empty_like(bh)
    temp[order] = bh

    fdr[valid] = temp

    return fdr


def safe_display(x):
    try:
        display(x)
    except NameError:
        print(x)


# In[73]:


# ============================================================
# 3. Create gene-level ranked table
# ============================================================

rank_df = res_dox_df.copy()

rank_df[fc_col] = pd.to_numeric(rank_df[fc_col], errors="coerce")
rank_df[p_col] = pd.to_numeric(rank_df[p_col], errors="coerce")

if fdr_col in rank_df.columns:
    rank_df[fdr_col] = pd.to_numeric(rank_df[fdr_col], errors="coerce")
else:
    rank_df[fdr_col] = np.nan

rank_df = rank_df.dropna(subset=[gene_col, fc_col, p_col]).copy()

# Avoid infinite -log10 values
rank_df[p_col] = rank_df[p_col].clip(lower=1e-300)

rank_df["Gene"] = rank_df[gene_col].apply(split_gene_symbols)
rank_df = rank_df.explode("Gene").copy()
rank_df = rank_df.dropna(subset=["Gene"])
rank_df = rank_df[rank_df["Gene"].astype(str).str.len() > 0].copy()

# Signed score:
# positive = increased in mBola3-Dox
# negative = decreased in mBola3-Dox
rank_df["signed_score"] = np.sign(rank_df[fc_col]) * (-np.log10(rank_df[p_col]))
rank_df["abs_signed_score"] = rank_df["signed_score"].abs()

# If multiple protein groups map to one gene, keep strongest differential evidence
rank_gene_df = (
    rank_df.sort_values(
        ["Gene", "abs_signed_score", p_col],
        ascending=[True, False, True]
    )
    .drop_duplicates(subset=["Gene"], keep="first")
    .reset_index(drop=True)
)

rank_gene_df = rank_gene_df[
    [
        "Gene",
        gene_col,
        desc_col,
        fc_col,
        p_col,
        fdr_col,
        "signed_score",
        "abs_signed_score"
    ]
].copy()

rank_gene_output = os.path.join(
    OUTDIR,
    "mBola3OE_gene_level_ranked_scores_all_detected_proteins.csv"
)

rank_gene_df.to_csv(rank_gene_output, index=False)

print("Detected gene-level table:")
print(rank_gene_df.shape)

print("\nSigned score summary:")
print(rank_gene_df["signed_score"].describe())

print("\nSaved:")
print(rank_gene_output)


# In[74]:


# ============================================================
# 4. Manual hypothesis-driven gene sets
# ============================================================

manual_gene_sets = {
    "Ferroptosis suppressive defense": {
        "genes": [
            "GPX4", "GGT1", "SLC7A11", "SLC3A2", "GCLC", "GCLM",
            "GSS", "GSR", "AIFM2", "FSP1", "GCH1", "DHODH",
            "NFE2L2", "KEAP1", "TXNRD1", "PRDX1", "PRDX6"
        ],
        "source": "Manual hypothesis set",
        "category": "Ferroptosis",
        "hierarchy": "Ferroptosis protection / antioxidant defense"
    },

    "Ferroptosis drivers / lipid peroxidation": {
        "genes": [
            "ACSL4", "LPCAT3", "ALOX5", "ALOX12", "ALOX15",
            "ALOX15B", "NCOA4", "TFRC", "HMOX1", "SAT1",
            "CHAC1", "NOX1", "NOX4"
        ],
        "source": "Manual hypothesis set",
        "category": "Ferroptosis",
        "hierarchy": "Ferroptosis drivers / lipid peroxidation"
    },

    "Iron regulation / Fe-S biology core": {
        "genes": [
            "BOLA3", "FTH1", "FTL", "TFRC", "IREB2", "ACO1",
            "NFS1", "ISCU", "FXN", "GLRX5", "ABCB7",
            "SLC25A37", "SLC25A28", "HMOX1", "HMOX2", "HSCB",
            "ISCA1", "ISCA2", "IBA57", "NFU1", "FDX1", "FDX2",
            "FDXR", "NUBPL", "LYRM4"
        ],
        "source": "Manual hypothesis set",
        "category": "Iron / Fe-S biology",
        "hierarchy": "Iron regulation and Fe-S biogenesis"
    },

    "BOLA3-related mitochondrial lipoate / 4Fe-4S biology": {
        "genes": [
            "BOLA3", "GLRX5", "NFU1", "ISCA1", "ISCA2", "IBA57",
            "NFS1", "ISCU", "FXN", "HSCB", "HSPA9", "LYRM4",
            "LIAS", "LIPT1", "LIPT2", "DLAT", "DLD", "PDHA1",
            "PDHB", "OGDH", "DLST", "SDHA", "SDHB", "ACO2"
        ],
        "source": "Manual hypothesis set",
        "category": "BOLA3 / mitochondrial Fe-S biology",
        "hierarchy": "BOLA3-associated Fe-S and lipoate-dependent metabolism"
    }
}


# In[75]:


# ============================================================
# 5. Load uploaded MitoCarta / mitochondrial pathway gene sets
# ============================================================

def find_mito_gsea_file():
    candidate_files = []

    candidate_files.extend(glob.glob("*GSEA*.csv"))
    candidate_files.extend(glob.glob("*Mito*.csv"))
    candidate_files.extend(glob.glob("/mnt/data/*GSEA*.csv"))
    candidate_files.extend(glob.glob("/mnt/data/*Mito*.csv"))

    required_cols = {"MitoPathway", "MitoPathway Hierarchy", "Genes"}

    for file in candidate_files:
        try:
            temp = pd.read_csv(file, nrows=5)
            if required_cols.issubset(set(temp.columns)):
                return file
        except Exception:
            pass

    return None


mito_file = find_mito_gsea_file()

if mito_file is None:
    print("No MitoCarta/MitoPathway GSEA file found. Proceeding with manual gene sets only.")
    mito_df = pd.DataFrame(columns=["MitoPathway", "MitoPathway Hierarchy", "Genes"])
else:
    print("Using MitoCarta/MitoPathway file:")
    print(mito_file)
    mito_df = pd.read_csv(mito_file)

safe_display(mito_df.head())


def split_gene_list_from_mito_file(x):
    if pd.isna(x):
        return []

    # For uploaded MitoCarta-style gene lists, whitespace can separate genes too
    parts = re.split(r"[,;|\s]+", str(x))

    genes = []
    for part in parts:
        gene = part.strip().upper()
        if gene not in ["", "NAN", "NONE", "NULL"]:
            genes.append(gene)

    return sorted(set(genes))


# Build all MitoCarta pathway sets
mito_gene_sets = {}

for _, row in mito_df.iterrows():
    pathway = str(row["MitoPathway"]).strip()
    hierarchy = str(row["MitoPathway Hierarchy"]).strip()
    genes = split_gene_list_from_mito_file(row["Genes"])

    if len(genes) == 0:
        continue

    set_name = f"{hierarchy} | {pathway}"

    mito_gene_sets[set_name] = {
        "genes": genes,
        "source": "MitoCarta / uploaded mitochondrial pathway file",
        "category": "Mitochondrial pathway",
        "hierarchy": hierarchy
    }


# In[76]:


# ============================================================
# 6. Focused uploaded MitoCarta modules
# ============================================================

uploaded_set_patterns = {
    "Electron carriers": [
        r"electron carrier"
    ],

    "Selenoproteins / ROS / glutathione metabolism": [
        r"selenoprotein",
        r"ros",
        r"reactive oxygen",
        r"glutathione",
        r"oxidative stress"
    ],

    "Fe-S containing proteins": [
        r"fe-s",
        r"iron-sulfur",
        r"iron sulfur"
    ],

    "NAD biosynthesis and metabolism": [
        r"\bnad\b",
        r"nicotinamide",
        r"nicotinate"
    ],

    "Iron homeostasis": [
        r"iron homeostasis",
        r"iron regulation",
        r"iron metabolism"
    ],

    "CoQ metabolism": [
        r"\bcoq\b",
        r"coenzyme q",
        r"ubiquinone"
    ],

    "Kynurenine metabolism": [
        r"kynurenine",
        r"tryptophan catabolism",
        r"tryptophan metabolism"
    ],

    "Lipid metabolism": [
        r"lipid metabolism",
        r"fatty acid",
        r"beta oxidation",
        r"β-oxidation",
        r"sphingolipid",
        r"phospholipid",
        r"cholesterol",
        r"sterol"
    ],

    "Lipoate metabolism": [
        r"lipoate",
        r"lipoic acid",
        r"lipoyl"
    ],

    "TCA cycle": [
        r"tca",
        r"tricarboxylic",
        r"citric acid cycle"
    ],

    "OXPHOS / respiratory chain": [
        r"oxphos",
        r"oxidative phosphorylation",
        r"respiratory chain",
        r"complex i",
        r"complex ii",
        r"complex iii",
        r"complex iv",
        r"complex v"
    ]
}


focused_uploaded_sets = {}

if mito_df.shape[0] > 0:

    mito_df["Search_Text"] = (
        mito_df["MitoPathway"].astype(str) +
        " " +
        mito_df["MitoPathway Hierarchy"].astype(str)
    ).str.lower()

    matched_rows = []

    for module_name, patterns in uploaded_set_patterns.items():

        module_genes = []

        for _, row in mito_df.iterrows():
            search_text = row["Search_Text"]

            matched = any(re.search(pattern, search_text) for pattern in patterns)

            if matched:
                genes = split_gene_list_from_mito_file(row["Genes"])
                module_genes.extend(genes)

                matched_rows.append({
                    "Focused_module": module_name,
                    "MitoPathway": row["MitoPathway"],
                    "MitoPathway Hierarchy": row["MitoPathway Hierarchy"],
                    "n_genes_in_row": len(genes)
                })

        module_genes = sorted(set(module_genes))

        if len(module_genes) > 0:
            focused_uploaded_sets[module_name] = {
                "genes": module_genes,
                "source": "MitoCarta / uploaded mitochondrial pathway file",
                "category": "Focused mitochondrial module",
                "hierarchy": module_name
            }

    matched_rows_df = pd.DataFrame(matched_rows)

    matched_rows_output = os.path.join(
        OUTDIR,
        "mBola3OE_focused_uploaded_MitoCarta_modules_matched_rows.csv"
    )

    matched_rows_df.to_csv(matched_rows_output, index=False)

    print("Focused uploaded modules found:")
    for k, v in focused_uploaded_sets.items():
        print(k, len(v["genes"]))

    print("\nSaved:")
    print(matched_rows_output)


# Combined sets
focused_gene_sets = {}
focused_gene_sets.update(manual_gene_sets)
focused_gene_sets.update(focused_uploaded_sets)

all_gene_sets = {}
all_gene_sets.update(manual_gene_sets)
all_gene_sets.update(mito_gene_sets)

print("\nManual gene sets:", len(manual_gene_sets))
print("Focused uploaded gene sets:", len(focused_uploaded_sets))
print("All uploaded mitochondrial gene sets:", len(mito_gene_sets))
print("All combined gene sets:", len(all_gene_sets))


# In[77]:


# ============================================================
# 7. Score gene sets using all detected proteins
# ============================================================

detected_genes = set(rank_gene_df["Gene"])

score_lookup = rank_gene_df.set_index("Gene")["signed_score"].to_dict()
fc_lookup = rank_gene_df.set_index("Gene")[fc_col].to_dict()

all_scores = rank_gene_df["signed_score"].dropna().values


def score_gene_set(set_name, set_info):
    genes = sorted(set([g.upper() for g in set_info["genes"]]))

    detected_set_genes = sorted(set(genes).intersection(detected_genes))
    missing_genes = sorted(set(genes) - detected_genes)

    in_scores = rank_gene_df[
        rank_gene_df["Gene"].isin(detected_set_genes)
    ]["signed_score"].dropna().values

    out_scores = rank_gene_df[
        ~rank_gene_df["Gene"].isin(detected_set_genes)
    ]["signed_score"].dropna().values

    in_fc = rank_gene_df[
        rank_gene_df["Gene"].isin(detected_set_genes)
    ][fc_col].dropna().values

    if len(in_scores) >= 2 and len(out_scores) >= 2:
        mw_p = mannwhitneyu(
            in_scores,
            out_scores,
            alternative="two-sided"
        ).pvalue
    else:
        mw_p = np.nan

    detected_gene_score_df = rank_gene_df[
        rank_gene_df["Gene"].isin(detected_set_genes)
    ].copy()

    up_genes = detected_gene_score_df[
        detected_gene_score_df[fc_col] > 0
    ]["Gene"].tolist()

    down_genes = detected_gene_score_df[
        detected_gene_score_df[fc_col] < 0
    ]["Gene"].tolist()

    return {
        "Gene_Set": set_name,
        "Source": set_info["source"],
        "Category": set_info["category"],
        "Hierarchy": set_info["hierarchy"],

        "n_genes_defined": len(genes),
        "n_genes_detected": len(detected_set_genes),

        "mean_signed_score": np.mean(in_scores) if len(in_scores) > 0 else np.nan,
        "median_signed_score": np.median(in_scores) if len(in_scores) > 0 else np.nan,

        "mean_log2FC": np.mean(in_fc) if len(in_fc) > 0 else np.nan,
        "median_log2FC": np.median(in_fc) if len(in_fc) > 0 else np.nan,

        "n_up_in_Dox": len(up_genes),
        "n_down_in_Dox": len(down_genes),

        "fraction_up_in_Dox": len(up_genes) / len(detected_set_genes) if len(detected_set_genes) > 0 else np.nan,
        "fraction_down_in_Dox": len(down_genes) / len(detected_set_genes) if len(detected_set_genes) > 0 else np.nan,

        "mannwhitney_p": mw_p,

        "detected_genes": "; ".join(detected_set_genes),
        "up_genes_in_Dox": "; ".join(sorted(up_genes)),
        "down_genes_in_Dox": "; ".join(sorted(down_genes)),
        "missing_genes": "; ".join(missing_genes)
    }


def run_gene_set_scoring(gene_sets, output_prefix):
    records = []

    for set_name, set_info in gene_sets.items():
        records.append(score_gene_set(set_name, set_info))

    score_df = pd.DataFrame(records)

    score_df["mannwhitney_FDR"] = benjamini_hochberg(
        score_df["mannwhitney_p"].values
    )

    score_df = score_df.sort_values(
        ["mannwhitney_p", "n_genes_detected"],
        ascending=[True, False]
    ).copy()

    output_file = os.path.join(
        OUTDIR,
        f"{output_prefix}.csv"
    )

    score_df.to_csv(output_file, index=False)

    print("Saved:")
    print(output_file)

    return score_df


focused_score_df = run_gene_set_scoring(
    focused_gene_sets,
    "mBola3OE_ranked_scores_FOCUSED_modules"
)

all_mito_score_df = run_gene_set_scoring(
    all_gene_sets,
    "mBola3OE_ranked_scores_ALL_MitoCarta_and_manual_modules"
)

print("\nFocused ranked module scores:")
safe_display(focused_score_df)


# In[78]:


# ============================================================
# 8. Plot focused module scores
# Increased in mBola3-Dox = green; decreased in mBola3-Dox = red
# ============================================================

plot_module_df = focused_score_df.copy()

# Keep modules with at least 2 detected genes
plot_module_df = plot_module_df[
    plot_module_df["n_genes_detected"] >= 2
].copy()

plot_module_df = plot_module_df.sort_values(
    "mean_signed_score",
    ascending=True
).copy()

# Define bar colors
plot_module_df["Bar_Color"] = np.where(
    plot_module_df["mean_signed_score"] >= 0,
    "green",
    "red"
)

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=plot_module_df["mean_signed_score"],
        y=plot_module_df["Gene_Set"],
        orientation="h",
        marker=dict(
            color=plot_module_df["Bar_Color"]
        ),
        customdata=np.stack(
            [
                plot_module_df["n_genes_detected"].astype(str),
                plot_module_df["mean_log2FC"].astype(str),
                plot_module_df["median_log2FC"].astype(str),
                plot_module_df["mannwhitney_p"].astype(str),
                plot_module_df["mannwhitney_FDR"].astype(str),
                plot_module_df["up_genes_in_Dox"].astype(str),
                plot_module_df["down_genes_in_Dox"].astype(str)
            ],
            axis=-1
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Mean signed score: %{x:.3f}<br>"
            "Detected genes: %{customdata[0]}<br>"
            "Mean log2FC: %{customdata[1]}<br>"
            "Median log2FC: %{customdata[2]}<br>"
            "Mann-Whitney p: %{customdata[3]}<br>"
            "Mann-Whitney FDR: %{customdata[4]}<br>"
            "Up in mBola3-Dox: %{customdata[5]}<br>"
            "Down in mBola3-Dox: %{customdata[6]}<br>"
            "<extra></extra>"
        )
    )
)

fig.add_vline(
    x=0,
    line_dash="dash",
    line_color="black",
    line_width=1
)

fig.update_layout(
    title=(
        "mBola3-Dox overexpression focused ranked gene-set scores<br>"
        "Green = increased in mBola3-Dox; red = decreased in mBola3-Dox"
    ),
    xaxis_title="Mean signed pathway score",
    yaxis_title="Pathway / module",
    template="simple_white",
    width=1100,
    height=750,
    font=dict(size=14)
)

fig.show()

focused_plot_file = os.path.join(
    OUTDIR,
    "mBola3OE_focused_ranked_module_scores.html"
)

fig.write_html(focused_plot_file)

print("Saved:")
print(focused_plot_file)


# In[79]:


# ============================================================
# 9. Plot top shifted mitochondrial/manual pathways
# ============================================================

plot_all_df = all_mito_score_df.copy()

plot_all_df = plot_all_df[
    plot_all_df["n_genes_detected"] >= 3
].copy()

plot_all_df["abs_mean_signed_score"] = plot_all_df["mean_signed_score"].abs()

plot_all_df = plot_all_df.sort_values(
    "abs_mean_signed_score",
    ascending=False
).head(25).copy()

plot_all_df = plot_all_df.sort_values(
    "mean_signed_score",
    ascending=True
).copy()

plot_all_df["Bar_Color"] = np.where(
    plot_all_df["mean_signed_score"] >= 0,
    "green",
    "red"
)

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=plot_all_df["mean_signed_score"],
        y=plot_all_df["Gene_Set"],
        orientation="h",
        marker=dict(
            color=plot_all_df["Bar_Color"]
        ),
        customdata=np.stack(
            [
                plot_all_df["Category"].astype(str),
                plot_all_df["n_genes_detected"].astype(str),
                plot_all_df["mean_log2FC"].astype(str),
                plot_all_df["median_log2FC"].astype(str),
                plot_all_df["mannwhitney_p"].astype(str),
                plot_all_df["mannwhitney_FDR"].astype(str),
                plot_all_df["detected_genes"].astype(str)
            ],
            axis=-1
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Mean signed score: %{x:.3f}<br>"
            "Category: %{customdata[0]}<br>"
            "Detected genes: %{customdata[1]}<br>"
            "Mean log2FC: %{customdata[2]}<br>"
            "Median log2FC: %{customdata[3]}<br>"
            "Mann-Whitney p: %{customdata[4]}<br>"
            "Mann-Whitney FDR: %{customdata[5]}<br>"
            "Detected genes: %{customdata[6]}<br>"
            "<extra></extra>"
        )
    )
)

fig.add_vline(
    x=0,
    line_dash="dash",
    line_color="black",
    line_width=1
)

fig.update_layout(
    title=(
        "mBola3-Dox overexpression top shifted mitochondrial/manual pathways<br>"
        "Ranked using all detected proteins"
    ),
    xaxis_title="Mean signed score",
    yaxis_title="Pathway/module",
    template="simple_white",
    width=1200,
    height=1000,
    font=dict(size=13)
)

fig.show()

top_shifted_plot_file = os.path.join(
    OUTDIR,
    "mBola3OE_top_shifted_MitoCarta_manual_pathways.html"
)

fig.write_html(top_shifted_plot_file)

print("Saved:")
print(top_shifted_plot_file)


# In[80]:


# ============================================================
# 10. Save top increased and decreased pathway tables
# ============================================================

top_increased_pathways = (
    all_mito_score_df[
        all_mito_score_df["n_genes_detected"] >= 3
    ]
    .sort_values("mean_signed_score", ascending=False)
    .head(50)
    .copy()
)

top_decreased_pathways = (
    all_mito_score_df[
        all_mito_score_df["n_genes_detected"] >= 3
    ]
    .sort_values("mean_signed_score", ascending=True)
    .head(50)
    .copy()
)

top_increased_file = os.path.join(
    OUTDIR,
    "mBola3OE_top_50_increased_MitoCarta_manual_pathways.csv"
)

top_decreased_file = os.path.join(
    OUTDIR,
    "mBola3OE_top_50_decreased_MitoCarta_manual_pathways.csv"
)

top_increased_pathways.to_csv(top_increased_file, index=False)
top_decreased_pathways.to_csv(top_decreased_file, index=False)

print("Saved:")
print(top_increased_file)
print(top_decreased_file)

print("\nTop increased pathways in mBola3-Dox:")
safe_display(
    top_increased_pathways[
        [
            "Gene_Set",
            "Category",
            "n_genes_detected",
            "mean_signed_score",
            "median_signed_score",
            "mean_log2FC",
            "median_log2FC",
            "n_up_in_Dox",
            "n_down_in_Dox",
            "mannwhitney_p",
            "mannwhitney_FDR"
        ]
    ]
)

print("\nTop decreased pathways in mBola3-Dox:")
safe_display(
    top_decreased_pathways[
        [
            "Gene_Set",
            "Category",
            "n_genes_detected",
            "mean_signed_score",
            "median_signed_score",
            "mean_log2FC",
            "median_log2FC",
            "n_up_in_Dox",
            "n_down_in_Dox",
            "mannwhitney_p",
            "mannwhitney_FDR"
        ]
    ]
)


# In[ ]:




