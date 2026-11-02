# 🛒 SmartCart Customer Segmentation

Unsupervised machine learning project that groups **2,237 customers into 4 behavioral segments** using PCA, K-Means and Hierarchical (Agglomerative) clustering. Part of my ML learning series.

📄 [**View the project carousel (PDF)**](assets/SmartCart_Segmentation_Carousel.pdf)

## 🎯 Problem

One marketing campaign for everyone wastes money. With no labels available, can the data reveal its own natural customer groups that a marketing team can act on?

## 📦 Dataset

`smartcart\_customers.csv`: 2,240 customers, 22 features covering demographics, income, product spending, purchase channels, campaign response and complaints.

## 🔧 Approach

1. **Cleaning:** filled 24 missing `Income` values with the median.
2. **Feature engineering:** `Age`, `Customer\_Tenure\_Days`, `Total\_Spending`, `Total\_Children`, `Education` (5 → 3 levels), `Living\_With` (Partner / Alone). Dropped 12 redundant columns (27 → 15).
3. **Outliers:** inspected with pair-plots and filtered unrealistic ages (≥ 90), leaving 2,237 customers.
4. **EDA:** correlation heatmap (e.g. catalog purchases ↔ total spending: 0.78, income ↔ total spending: 0.66).
5. **Preprocessing:** one-hot encoding (18 features) and `StandardScaler`.
6. **Dimensionality reduction:** PCA to 3 components (\~44% of variance) for clustering and 2D/3D visualization.
7. **Choosing K:** elbow method (`KneeLocator` → k = 4) and silhouette score (≈ 0.36 at k = 4).
8. **Clustering:** K-Means and Agglomerative (Ward linkage), k = 4.
9. **Profiling:** per-cluster feature means to interpret each segment.

## 📊 Results

|Segment (my label)|Customers|Avg income|Avg spend|Children|Campaign response|
|-|-|-|-|-|-|
|🟢 Affluent solos|300|72.1K|1,258|0.4|**37.7%**|
|🔵 Affluent couples|583|70.4K|1,098|0.5|13.4%|
|🔴 Budget families|860|40.1K|258|1.3|9.3%|
|🟡 Budget solos|494|39.9K|235|1.2|12.8%|

**Takeaways**

* *Affluent solos* respond to campaigns roughly 3–4× more than other groups, so they are a priority for targeted offers.
* *Affluent couples* spend heavily and favor catalog and store channels, a fit for loyalty perks.
* *Budget groups* spend 4–5× less, buy more on deals and visit the website more often, so coupons and web promotions suit them.

These are patterns in the data, not proof of cause. An A/B test would validate them.

## ⚠️ Limitations and next steps

* Tighter outlier handling (a few very high incomes remain).
* 3 PCA components keep only \~44% of the variance; try more components.
* Compare with Gaussian Mixture Models and DBSCAN.
* Validate segments with an A/B campaign test.

## 🛠️ Tech stack

Python · pandas · NumPy · scikit-learn · kneed · matplotlib · seaborn · Jupyter

## ▶️ How to run

```bash
git clone https://github.com/RohanRathod81/smartcart-project.git
cd smartcart-project
pip install -r requirements.txt
jupyter notebook smartcart.ipynb
```

## 📬 Connect

Rohan Rathod · [GitHub](https://github.com/RohanRathod81) · [LinkedIn](https://www.linkedin.com/in/rohanrathod03/)

