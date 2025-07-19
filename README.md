# football-player-classification
Classify football players in the Big 8 European leagues into positions and roles based on in-game data, using supervised and unsupervised learning, all in one pipeline.
## Files
- `unsupervised_role_pipeline.py` — Main pipeline script  
- `fbref_big8_raw_2023-2024.csv` — Raw player stats scraped from FBRef: output of modified scraping module from other project (https://github.com/dannllabore411/fbref-data-analysis)   
---
## Other Files
Intermediate files and visualization outputs have been added - to list
---
## Pipeline Overview
1. **Data Loading & Cleaning**  
   Loads FBRef event data, filters out low-minute players, and handles missing values.  
   *data wrangling and preprocessing*
2. **Supervised Position Group Prediction (`PosGrp`)**  
   Trains an XGBoost classifier to assign each player to one of 7 position groups (GK, CB, FB, DM, AM, WF, CF) based only on raw stats.  
   *supervised classification, model training, label encoding*
3. **Feature Engineering & Standardization**  
   Applies z-score normalization per position group to enable fair comparisons across stats. Creates positional metrics to derive average position (verticality & width) from event data.  
   *feature engineering, standardization, data engineering*
4. **Unsupervised Role Clustering**  
   Uses KMeans to cluster players into 3 roles within each position group (e.g., CB-1, CB-2, CB-3).  
   *unsupervised learning, K-means clustering*
5. **Dimensionality Reduction & Visualization**  
   Projects standardized features into 2D using UMAP for intuitive visualization of player-role similarity.  
   *UMAP dimension reduction, data visualization with Seaborn*
6. **Example Player Extraction**  
   For each cluster, finds the players closest to the centroid — archetypal examples of that role.  
   *data interpretation*
7. **Export Results & Trained Models**  
   Saves player classifications, visual plots, cluster examples, and the trained PosGrp model for reuse.  
   *data export, model export*
---
## How to Run
Place `fbref_big8_raw_2023-2024.csv` in your working directory and run:
```bash
python unsupervised_role_pipeline.py
