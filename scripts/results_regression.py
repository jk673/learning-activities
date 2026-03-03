"""Kriging 모델 학습 → 평가 → 임의 입력 예측 테스트."""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (어디서든 실행 가능)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.model_kriging import load_yacht_data, build_kriging, evaluate


if __name__ == "__main__":
    X, y = load_yacht_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("Fitting Kriging model...")
    model = build_kriging(X_train_scaled, y_train)
    print(f"Optimized kernel: {model.kernel_}")

    y_pred, y_std, rmse, r2 = evaluate(model, X_test, y_test, scaler)

    # --- 핵심 메트릭 ---
    mae = np.mean(np.abs(y_test - y_pred))
    max_err = np.max(np.abs(y_test - y_pred))
    # MAPE: 실측값이 매우 작은 점은 제외 (0.05 미만은 상대오차가 왜곡됨)
    mask = y_test > 0.05
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

    print(f"\n{'='*50}")
    print(f"  Train: {len(y_train)} samples, Test: {len(y_test)} samples")
    print(f"{'='*50}")
    print(f"  R²:         {r2:.4f}")
    print(f"  RMSE:       {rmse:.4f}")
    print(f"  MAE:        {mae:.4f}")
    print(f"  MAPE:       {mape:.2f}%")
    print(f"  Max Error:  {max_err:.4f}")
    print(f"{'='*50}")

    # --- Ground Truth vs Prediction 비교 (전체 테스트셋) ---
    sort_idx = np.argsort(y_test)
    print(f"\n{'Actual':>10s}  {'Predicted':>10s}  {'Error':>10s}  {'Rel%':>8s}  {'±Std':>8s}")
    print(f"{'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")
    for i in sort_idx:
        err = y_pred[i] - y_test[i]
        rel = abs(err / y_test[i]) * 100 if y_test[i] > 0.05 else float('nan')
        print(f"{y_test[i]:10.4f}  {y_pred[i]:10.4f}  {err:+10.4f}  {rel:7.1f}%  {y_std[i]:8.4f}")
