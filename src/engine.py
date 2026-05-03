import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from src.utils import set_seed
from src.model import build_model_bin
from src.dataset import ScrewDataset, train_transform, val_transform
from typing import Dict, List, Tuple


def measure_inference_speed_with_gpu(model, device, input_size=(1, 3, 224, 224), num_runs=100):
    """
    """
    model.eval()

    # 1. 더미 데이터 생성
    dummy_input = torch.randn(input_size).to(device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input) # 워밍업 연산

    # 2. GPU가 워밍업 연산을 끝낼 때까지 CPU 대기
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()

    # 3. 실제 측정 구간
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    # 4. GPU 연산이 끝날 때까지 타이머 멈추지않고 대기
    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()

    # 5. 수치 계산
    total_time = end_time - start_time
    latency_ms = (total_time / num_runs) * 1000  # 1장당 걸린 시간 (밀리초)
    fps = num_runs / total_time                  # 1초당 처리 장수

    print(f"[GPU 기준] 측정 결과: {fps:.1f} FPS (1장당 {latency_ms:.2f} ms)")
    return fps, latency_ms


def measure_inference_speed_with_cpu(model, input_size=(1, 3, 224, 224), num_runs=100):
    """
    """
    cpu_device = torch.device('cpu')

    original_device = next(model.parameters()).device
    model.to(cpu_device)
    model.eval()

    dummy_input = torch.randn(input_size).to(cpu_device)
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    end_time = time.time()

    total_time = end_time - start_time
    latency_ms = (total_time / num_runs) * 1000
    fps = num_runs / total_time

    model.to(original_device)

    print(f"[CPU 기준] 측정 결과: {fps:.1f} FPS (1장당 {latency_ms:.2f} ms)")
    return fps, latency_ms


def run_grid_search_with_kfold_cv(
    model_name,
    all_paths,
    all_labels_np,
    bad_weight_list,
    n_splits,
    num_epochs,
    batch_size,
    num_workers,
    device,
    run_dirs,
    early_stop_patience,
    champion_select
) -> Tuple:
    """
    Grid Search + Stratified K-Fold CV
    
    Returns:
        best_weight: 최적 가중치
        weight_f2: {weight: [fold1_f2, fold2_f2, ...]}
        db: 전체 학습 히스토리
    """
    if champion_select not in ("robust", "mean"):
        raise ValueError("champion_select must be 'robust' or 'mean'")

    bad_weight_list = list(bad_weight_list)
    fold_keys = [f"fold{i}" for i in range(1, n_splits + 1)]
    
    # 데이터 저장 구조
    db: Dict[float, Dict[str, dict]] = {
        w: {
            fk: {
                "train_loss": [], 
                "val_loss": [], 
                "f2_score": [], 
                "best_epoch": -1, 
                "best_f2": -1.0,
                "best_val_loss": -1.0
            } for fk in fold_keys
        } for w in bad_weight_list
    }
    weight_f2: Dict[float, List[float]] = {w: [] for w in bad_weight_list}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    k_fold_splits = list(skf.split(all_paths, all_labels_np))

    weights_dir = run_dirs.get("weights", "./outputs/")
    os.makedirs(weights_dir, exist_ok=True)

    # ===== Grid Search Loop =====
    for w in bad_weight_list:
        set_seed(42)
        print(f"\n{'='*60}")
        print(f"[Weight {w:.1f}]")
        print(f"{'='*60}")

        class_weight = torch.tensor([1.0, float(w)], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)

        # ===== K-Fold Loop =====
        for fold_idx, (train_idx, val_idx) in enumerate(k_fold_splits):
            set_seed(42 + fold_idx)
            fk = fold_keys[fold_idx]
            print(f"\n  [{fk}] Training...")

            # 데이터 준비
            train_path_fold = [all_paths[i] for i in train_idx]
            val_path_fold   = [all_paths[i] for i in val_idx]

            train_dataset = ScrewDataset(train_path_fold, all_labels_np[train_idx], 
                                        transform=train_transform)
            val_dataset   = ScrewDataset(val_path_fold, all_labels_np[val_idx], 
                                        transform=val_transform)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                     shuffle=True, num_workers=num_workers, 
                                     pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                   shuffle=False, num_workers=num_workers, 
                                   pin_memory=True)
            
            # 모델 초기화
            model = build_model_bin(model_name).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            
            # Early Stopping 변수
            best_f2 = 0.0  # ← 추가!
            best_epoch = -1
            best_val_loss = float('inf')
            no_improve_f2 = 0

            # ===== Epoch Loop =====
            for epoch in range(num_epochs):
                # --- Train ---
                model.train()
                running_train_loss = 0.0
                
                for imgs, lbls in train_loader:
                    imgs = imgs.to(device)
                    lbls = (lbls > 0).long().to(device)
                    
                    optimizer.zero_grad()
                    loss = criterion(model(imgs), lbls)
                    loss.backward()
                    optimizer.step()
                    
                    running_train_loss += loss.item()

                avg_train_loss = running_train_loss / max(len(train_loader), 1)

                # --- Validation ---
                model.eval()
                running_val_loss = 0.0
                y_true_ep, y_pred_ep = [], []
                
                with torch.no_grad():  # ← 수정!
                    for imgs, lbls in val_loader:
                        imgs = imgs.to(device)
                        lbls = (lbls > 0).long().to(device)
                        
                        logits = model(imgs)
                        running_val_loss += criterion(logits, lbls).item()
                        
                        y_true_ep.extend(lbls.cpu().numpy())
                        y_pred_ep.extend(logits.argmax(dim=1).cpu().numpy())

                avg_val_loss = running_val_loss / max(len(val_loader), 1)

                # F2-score 계산
                f2_ep = fbeta_score(y_true_ep, y_pred_ep, beta=2, pos_label=1, 
                                   zero_division=0)
                
                # 히스토리 저장 (매 epoch)
                db[w][fk]["train_loss"].append(avg_train_loss)
                db[w][fk]["val_loss"].append(avg_val_loss)
                db[w][fk]["f2_score"].append(f2_ep)
                
                # Early Stopping 체크
                if f2_ep > best_f2:
                    best_f2 = f2_ep
                    best_epoch = epoch
                    best_val_loss = avg_val_loss
                    no_improve_f2 = 0
                    
                    # 모델 저장 (선택사항)
                    # save_path = f"{weights_dir}/w{w}_fold{fold_idx+1}_best.pth"
                    # torch.save(model.state_dict(), save_path)
                else:
                    no_improve_f2 += 1

                # 출력
                if epoch == 0 or (epoch + 1) % 10 == 0:
                    print(f"    Ep {epoch+1:02d} | "
                          f"TrL {avg_train_loss:.4f} | "
                          f"VaL {avg_val_loss:.4f} | "
                          f"F2 {f2_ep:.4f} | "
                          f"Best {best_f2:.4f}")

                # Early Stop
                if early_stop_patience > 0 and no_improve_f2 >= early_stop_patience:
                    print(f"    -> Early stop at epoch {epoch + 1} "
                          f"(no improve for {early_stop_patience} epochs)")
                    break
            
            # Fold 종료 - Best 값 저장
            db[w][fk]["best_epoch"] = best_epoch
            db[w][fk]["best_f2"] = best_f2
            db[w][fk]["best_val_loss"] = best_val_loss
            weight_f2[w].append(best_f2)  # ← 중요!
            
            print(f"  [{fk}] Finished | Best F2: {best_f2:.4f} at epoch {best_epoch+1}")

        # Weight 종료 - 평균 계산
        mean_f2 = np.mean(weight_f2[w])
        std_f2 = np.std(weight_f2[w])
        print(f"\n[Weight {w:.1f}] Summary:")
        print(f"  Fold F2s: {[f'{f:.4f}' for f in weight_f2[w]]}")
        print(f"  Mean F2:  {mean_f2:.4f} ± {std_f2:.4f}")

    # ===== Champion Selection =====
    print(f"\n{'='*60}")
    print("Grid Search Results:")
    print(f"{'='*60}")
    
    results_table = []
    for w in bad_weight_list:
        mean_f2 = np.mean(weight_f2[w])
        std_f2 = np.std(weight_f2[w])
        
        if champion_select == "robust":
            score = mean_f2 - std_f2  # Robust score
        else:  # "mean"
            score = mean_f2
        
        results_table.append({
            'weight': w,
            'mean_f2': mean_f2,
            'std_f2': std_f2,
            'score': score
        })
        
        print(f"Weight {w:.1f}: Mean F2 = {mean_f2:.4f} ± {std_f2:.4f} "
              f"({'Robust' if champion_select == 'robust' else 'Score'}: {score:.4f})")
    
    # 최적값 선정
    best_result = max(results_table, key=lambda x: x['score'])
    best_weight = best_result['weight']
    
    print(f"\n{'='*60}")
    print(f"🏆 Champion: Weight {best_weight:.1f}")
    print(f"   Mean F2: {best_result['mean_f2']:.4f} ± {best_result['std_f2']:.4f}")
    print(f"{'='*60}\n")

    return best_weight, weight_f2, db


def train_with_best_weight(model_name, weight):
    # 1) 전체 데이터 로드
    TRAIN_DIR = "./data/new_k-fold_data/train"
    all_paths = []
    all_labels = []
    
    class_names = ["good", "type1", "type2", "type3", "type4", "type5"]
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(TRAIN_DIR, class_name)
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_paths.append(os.path.join(class_dir, fname))
                all_labels.append(class_idx)

    print(f"전체 데이터: {len(all_paths)}장")  # 425

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, 
        all_labels, 
        test_size=0.2,  # 85장
        stratify=all_labels,  # 클래스 비율 유지
        random_state=42
    )

    train_dataset = ScrewDataset(train_paths, np.array(train_labels), train_transform)
    val_dataset = ScrewDataset(val_paths, np.array(val_labels), val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model_bin(model_name).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, weight]).to(device)  # 1단계 최적값
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 6) 학습 (Loss 기록)
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f2': []
    }

    for epoch in range(40):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = (labels > 0).long().to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels_bin = (labels > 0).long().to(device)
                
                outputs = model(imgs)
                loss = criterion(outputs, labels_bin)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_bin.cpu().numpy())
        
        val_loss /= len(val_loader)
        from sklearn.metrics import fbeta_score
        val_f2 = fbeta_score(all_labels, all_preds, beta=2)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f2'].append(val_f2)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/40 | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val F2: {val_f2:.4f}")
    
    TEST_DIR         = "./data/new_k-fold_data/test"
    class_names_test = ["good", "type1", "type2", "type3", "type4", "type5"]
    valid_exts       = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    test_paths, test_labels = [], []
    for class_idx, class_name in enumerate(class_names_test):
        class_dir = os.path.join(TEST_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(valid_exts):
                test_paths.append(os.path.join(class_dir, fname))
                test_labels.append(class_idx)

    if not test_paths:
        raise ValueError(f"{TEST_DIR} 에서 이미지를 찾지 못함")

    test_ds     = ScrewDataset(test_paths, np.array(test_labels, dtype=np.int64), transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                             num_workers=0, pin_memory=True)

    y_true_test, y_pred_test = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            lbls = (lbls > 0).long()
            preds = model(imgs).argmax(dim=1).cpu()
            y_true_test.extend(lbls.numpy())
            y_pred_test.extend(preds.numpy())
    
    return y_true_test, y_pred_test, model, history