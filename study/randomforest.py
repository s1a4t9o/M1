import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

FEATURES = ["x__1", "x__2", "x__3", "x__4", "x__5", "x__6", "x__7", "x__8", "x__9", "x__10", "x__11", "x__12", "x__13", "x__14", "x__15", "x__16", "x__17", "x__18", "x__19", "x__20", "x__21", "x__22", "x__23", "x__24", "x__25", "x__26", "x__27", "x__28", "x__29", "x__30", "x__31", "x__32"]
LABEL    = "y"

# 将来、CSVに「src（元ファイル名）」「time」「id」など識別用の列があるならここに列名を足してください
IDENT_COLS_CANDIDATES = ["src", "filename", "time", "id", "row_id"]

def main():
    run_id = f"PID={os.getpid()} @ {time.strftime('%H:%M:%S')}"
    print(f"[START] {run_id}")

    # ===== 学習データ =====
    train = pd.read_csv("study/1020_train_ver2.csv", usecols=FEATURES + [LABEL], dtype="float64")
    X_train = train.drop(columns=[LABEL])
    y_train = train[LABEL].astype(int)

    # ===== 検証データ（識別用メタ列があれば一緒に読む） =====
    # まず全列読み -> あれば IDENT_COLS を拾う（無ければ使わない）
    test_full = pd.read_csv("study/1020_valid_ver2.csv")
    ident_cols = [c for c in IDENT_COLS_CANDIDATES if c in test_full.columns]
    # 特徴＋ラベルだけで学習/評価、ident は後でくっつける
    test = test_full[FEATURES + [LABEL]].copy()
    X_test = test.drop(columns=[LABEL])
    y_test = test[LABEL].astype(int)

    # ===== 学習 =====
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ===== 予測・評価 =====
    y_pred = model.predict(X_test)
    n_classes = len(np.unique(y_test))
    avg = "binary" if n_classes == 2 else "macro"

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=avg, zero_division=0)
    recall    = recall_score(y_test, y_pred, average=avg, zero_division=0)
    f1        = f1_score(y_test, y_pred, average=avg, zero_division=0)
    conf_mat  = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision ({avg}): {precision:.4f}")
    print(f"Recall    ({avg}): {recall:.4f}")
    print(f"F1-score  ({avg}): {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_mat)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # ===== 予測確率と詳細明細を作成 =====
    # classes_ の順序と predict_proba の列は対応
    classes = model.classes_
    proba = model.predict_proba(X_test)  # shape: (n_samples, n_classes)
    proba_df = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes], index=X_test.index)

    # 予測確信度（最大確率）と真のクラス確率
    pred_conf = proba_df.max(axis=1)
    class_to_col = {c: i for i, c in enumerate(classes)}
    true_conf = pd.Series(
        [proba[i, class_to_col[y]] for i, y in enumerate(y_test.to_numpy())],
        index=X_test.index,
        name="true_conf"
    )

    # 明細データフレーム
    detail = pd.DataFrame({
        "row_id": X_test.index,  # CSV内の行番号代わり
        "y_true": y_test.to_numpy(),
        "y_pred": y_pred,
        "pred_conf": pred_conf,
        "true_conf": true_conf,
        "correct": (y_pred == y_test.to_numpy())
    }, index=X_test.index)

    # もし識別用メタ列（src, time, id 等）があれば結合
    if ident_cols:
        detail = pd.concat([test_full[ident_cols], detail], axis=1)

    # 各クラス確率も連結（必要に応じて後で見返せる）
    detail = pd.concat([detail, proba_df], axis=1)

    # 誤分類だけ
    errors = detail[~detail["correct"]].copy()

    # ===== 出力（CSV保存 & 端末表示） =====
    os.makedirs("study/models", exist_ok=True)
    ts = time.strftime('%Y%m%d-%H%M%S')
    eval_csv = f"study/models/eval_detail_{ts}.csv"
    err_csv  = f"study/models/errors_{ts}.csv"

    detail.to_csv(eval_csv, index=False)
    errors.to_csv(err_csv, index=False)

    print(f"\n🧾 Saved eval detail: {eval_csv}")
    print(f"❌ Saved misclassifications: {err_csv}  (count={len(errors)}/{len(detail)})")

    # 端末表示を少しリッチに
    if len(errors) > 0:
        # 確信度が高いのに外した上位10件
        cols_show = ["row_id", "y_true", "y_pred", "pred_conf", "true_conf"] + [c for c in ident_cols]
        top_hard = errors.sort_values("pred_conf", ascending=False).head(10)[cols_show]
        print("\nTop-10 high-confidence wrong predictions:")
        print(top_hard.to_string(index=False))

        # クラス別の誤分類数（真のクラス基準）
        by_true = errors.groupby("y_true").size().rename("errors").sort_values(ascending=False)
        print("\nErrors by true class:")
        print(by_true)

        # 予測先（どこに間違えたか）の分布
        by_pred = errors.groupby("y_pred").size().rename("predicted_as").sort_values(ascending=False)
        print("\nErrors grouped by predicted class:")
        print(by_pred)
    else:
        print("\nNo misclassifications 🎉")

    # ===== モデル保存 =====
    os.makedirs("models", exist_ok=True)
    model_path = "study/models/pose_model_test.sav"
    joblib.dump(model, model_path)
    print(f"✅ Saved: {model_path}  ({run_id})")

    print(f"[DONE]  {run_id}")

if __name__ == "__main__":
    main()
    sys.exit(0)
