import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os
from collections import Counter

# 👇 AJUSTA ESTA RUTA AL CSV QUE TE CREÓ LARAVEL
CSV_PATH = r"C:\nginx\html\smartbet-api\storage\app\exports\nba_games.csv"

def main():
    if not os.path.exists(CSV_PATH):
        print(f"No se encontró el archivo CSV en: {CSV_PATH}")
        return

    print(f"Cargando datos desde: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # Filtrar filas con datos indispensables
    df = df.dropna(subset=["closing_total_line", "over_odds", "over_hit"])

    if df.empty:
        print("No hay datos suficientes para entrenar el modelo.")
        return

    # =========
    # FEATURES
    # =========
    df["implied_over_prob"] = 1.0 / df["over_odds"]
    X = df[["closing_total_line", "implied_over_prob"]]
    y = df["over_hit"].astype(int)

    # =========================
    # REVISAR DISTRIBUCIÓN Y
    # DECIDIR SI HACEMOS SPLIT
    # =========================
    counts = Counter(y)
    print(f"Distribución de clases over_hit: {counts}")

    # ¿Hay al menos 2 clases y cada una con 2 o más ejemplos?
    can_split = (
        len(counts) == 2 and
        min(counts.values()) >= 2 and
        len(df) >= 4
    )

    model = LogisticRegression()

    if can_split:
        print("Hay datos suficientes, usando train_test_split con stratify.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        model.fit(X_train, y_train)

        # ====== Métricas ======
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auc = None

        print("\n--- RESULTADOS DEL MODELO ---")
        print(f"Accuracy: {acc:.3f}")
        if auc is not None:
            print(f"ROC AUC: {auc:.3f}")
        else:
            print("ROC AUC: no se pudo calcular (muy pocos datos).")

        # Demo de predicción
        if len(X_test) > 0:
            sample = X_test.iloc[[0]]
            prob_over = model.predict_proba(sample)[0, 1]
            print("\nEjemplo de predicción con un partido del dataset:")
            print(sample)
            print(f"Probabilidad estimada de Over: {prob_over:.3f}")

    else:
        print("Muy pocos datos o solo una clase; entrenando con TODO el dataset sin split.")
        model.fit(X, y)

    # ======================
    # GUARDAR EL MODELO
    # ======================
    MODEL_PATH = "nba_over_model.joblib"
    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo guardado en: {os.path.abspath(MODEL_PATH)}")

if __name__ == "__main__":
    main()
