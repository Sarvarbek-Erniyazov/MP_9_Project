import pandas as pd
import numpy as np
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.logger import logging_instance

class FeatureSelector:
    def __init__(self, model_dir="models", plots_dir="plots"):
        self.model_dir = model_dir
        self.plots_dir = plots_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def check_vif(self, X):
        """Multicollinearity-ni tekshirish (VIF)"""
        logging_instance.info("VIF tahlili boshlandi (Multicollinearity tekshiruvi).")
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        return vif_data.sort_values(by="VIF", ascending=False)

    def analyze_importance(self, X, y):
        try:
            logging_instance.info("Advanced Feature Selection jarayoni...")

            # 1. Mutual Information
            mi_scores = mutual_info_regression(X, y, random_state=42)
            mi_results = pd.Series(mi_scores, name="MI_Score", index=X.columns).sort_values(ascending=False)

            # 2. VIF Tahlili
            vif_results = self.check_vif(X)
            
            # 3. Grafik: MI Scores
            plt.figure(figsize=(10, 6))
            sns.barplot(x=mi_results.values, y=mi_results.index, palette="viridis")
            plt.title("Xususiyatlarning Target-ga ta'siri (Mutual Information)")
            plt.savefig(f"{self.plots_dir}/mi_importance.png")
            plt.close()

            # 4. Grafik: Correlation Heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(X.corr(method='spearman'), annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Xususiyatlararo korrelyatsiya (Spearman)")
            plt.savefig(f"{self.plots_dir}/correlation_heatmap.png")
            plt.close()

            # Tanlash mantiqi: MI > 0.01 va juda yuqori VIF-ga ega bo'lmaganlar
            # (VIF > 10 bo'lsa, bu ustun boshqasini deyarli 100% takrorlaydi degani)
            selected_features = mi_results[mi_results > 0.01].index.tolist()
            
            # Saqlash
            with open(os.path.join(self.model_dir, "selected_features.json"), "w") as f:
                json.dump({"selected_features": selected_features, "vif_report": vif_results.to_dict()}, f, indent=4)

            logging_instance.info(f"Top 5 Feature: \n{mi_results.head(5)}")
            return selected_features, mi_results

        except Exception as e:
            logging_instance.error(f"Feature Selectionda xato: {str(e)}")
            raise e