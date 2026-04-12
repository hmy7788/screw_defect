import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import copy
from sklearn.metrics import classification_report
from src.model import build_model # model.py에서 가져옴
import time
from src.dataset import ScrewDataset

# 모델 학습 및 평가
def train_and_evaluate_model(model_name, dataloaders, dataset_sizes, class_names, device, run_dirs, num_epochs=40, use_amp=True, early_stop_patience=7):
    """
    use_amp: GPU일 때 Mixed Precision 사용 (속도·메모리 개선)
    early_stop_patience: Val loss가 이 수의 epoch 동안 개선 없으면 해당 가중치 학습 조기 종료 (0이면 비활성화)
    """
    use_amp = use_amp and device.type == 'cuda'
    if use_amp:
        if hasattr(torch.amp, 'GradScaler'):
            scaler = torch.amp.GradScaler('cuda')
        else:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    print(f"\n{'='*60}")
    print(f"[ 시작 ] 모델 학습 및 평가: {model_name.upper()}" + (" [AMP ON]" if use_amp else ""))
    print(f"{'='*60}")

    bad_weights = np.arange(1.0, 10.5, 0.5).tolist()
    
    # Validation 기록용 리스트
    val_bad_recalls, val_bad_f1s, val_f2_scores = [], [], []

    best_f2 = -1.0 
    best_model_weights = None
    best_train_loss_history, best_val_loss_history = [], []
    optimal_weight = 1.0

    # 1단계: 가중치 탐색 스윕
    for current_weight in bad_weights:
        print(f"\n{'='*60}")
        print(f"현재 탐색 중인 bad weight : {current_weight}")

        current_train_loss_history = []
        current_val_loss_history = []

        # 매 가중치마다 새 모델과 옵티마이저 생성 (초기화)
        model = build_model(model_name).to(device)
        weights_tensor = torch.tensor([current_weight, 1.0], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            # Train Phase (AMP 사용 시 autocast)
            model.train()
            running_loss = 0.0
            for inputs, labels in dataloaders['train']:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                if use_amp:
                    autocast_ctx = torch.amp.autocast('cuda') if hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast()
                    with autocast_ctx:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            
            scheduler.step()
            train_epoch_loss = running_loss / dataset_sizes['train']
            current_train_loss_history.append(train_epoch_loss)

            # Validation Phase
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for inputs, labels in dataloaders['val']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    if use_amp:
                        autocast_ctx = torch.amp.autocast('cuda') if hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast()
                        with autocast_ctx:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)
            
            val_epoch_loss = val_running_loss / dataset_sizes['val']
            current_val_loss_history.append(val_epoch_loss)

            # Early stopping
            if early_stop_patience > 0:
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"Early stop at Epoch {epoch+1}/{num_epochs} (val loss {epochs_no_improve} epochs no improve)")
                    break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:02d}/{num_epochs} | Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")
            
        # val 평가
        model.eval()
        val_y_true, val_y_pred = [], []
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_y_true.extend(labels.cpu().numpy())
                val_y_pred.extend(preds.cpu().numpy())

        report_dict = classification_report(val_y_true, val_y_pred, target_names=class_names, output_dict=True, zero_division=0)
        curr_bad_precision = report_dict[class_names[0]]['precision']
        curr_bad_recall = report_dict[class_names[0]]['recall']
        curr_bad_f1 = report_dict[class_names[0]]['f1-score']
        
        if (4 * curr_bad_precision + curr_bad_recall) == 0:
            curr_f2 = 0.0
        else:
            curr_f2 = 5 * curr_bad_precision * curr_bad_recall / (4 * curr_bad_precision + curr_bad_recall)

        val_bad_recalls.append(curr_bad_recall)
        val_bad_f1s.append(curr_bad_f1)
        val_f2_scores.append(curr_f2)

        print(f"[Validation] 가중치 {current_weight} | F2: {curr_f2:.4f} | Recall: {curr_bad_recall:.4f}")

        if curr_f2 > best_f2:
            best_f2 = curr_f2
            optimal_weight = current_weight
            best_model_weights = copy.deepcopy(model.state_dict())
            best_train_loss_history = current_train_loss_history.copy()
            best_val_loss_history = current_val_loss_history.copy()
    
    print(f"\n{'='*60}")
    print(f"최적 Class Weight: {optimal_weight} (최고 Val F2: {best_f2:.4f})")
    print(f"{'='*60}")

    save_path = os.path.join(run_dirs['weights'], f'{model_name}_best_weight_{optimal_weight}.pth')
    torch.save(best_model_weights, save_path)
    print(f"best 가중치 파일 저장 완료: {save_path}")

    # 2단계: 최적 모델로 test 셋 최종 평가
    model.load_state_dict(best_model_weights)
    model.eval()
    test_y_true, test_y_pred = [], []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_y_true.extend(labels.cpu().numpy())
            test_y_pred.extend(preds.cpu().numpy())

    print(f"\n[ {model_name.upper()} 최종 TEST SET 성능 지표 (Weight: {optimal_weight}) ]")
    print(classification_report(test_y_true, test_y_pred, target_names=class_names, digits=4))

    return model, val_bad_recalls, val_bad_f1s, val_f2_scores, test_y_true, test_y_pred, best_train_loss_history, best_val_loss_history




# main6.ipynb
# ==================================================================
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



from sklearn.model_selection import StratifiedKFold
from src.utils import set_seed
from torch.utils.data import WeightedRandomSampler, DataLoader
from src.model import build_model_bin
from sklearn.metrics import classification_report
from src.dataset import ScrewDataset
from src.dataset import train_transform, val_transform


def f2_score(precision, recall):
    denom = 4 * precision + recall
    return 5 * precision * recall / denom if denom > 0 else 0.0


def run_kfold_experiment(model_name, all_paths, all_labels_np,
                         bad_weight_list, n_splits, num_epochs,
                         batch_size, num_workers, device,
                         run_dirs, early_stop_patience):
    db = {
        w: {
            f'fold{fold}': {
                'train_loss': [],
                'val_loss': [],
                'f2_score': [],
                'y_true': [],
                'best_epoch': -1,
                'best_f2': -1
            } for fold in range(1, 6)
        } for w in bad_weight_list
    }

    weight_f2 = {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    bad_weight_list = bad_weight_list

    weights_dir = run_dirs.get('weights', './outputs/')
    os.makedirs(weights_dir, exist_ok=True)
    # model_save_dir = os.path.join(weights_dir, 'weights')
    # os.makedirs(model_save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"모델: {model_name.upper()} | {n_splits}-Fold CV | bad_weight sweep: {bad_weight_list}")
    print(f"{'='*60}")

    k_fold_splits = list(skf.split(all_paths, all_labels_np))

    global_best_robust = -1.0
    champion_paths = []
    champion_weight = None
    # w_map = {}

    for w in bad_weight_list:
        set_seed(42)
        print(f'[Weight {w}/{bad_weight_list[-1]}]')
        current_w_paths = []

        class_weight = torch.tensor([1.0, float(w)], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)

        for fold_idx, (train_idx, val_idx) in enumerate(k_fold_splits):
            set_seed(42)
            print(f'[Fold {fold_idx+1}/{n_splits}]')

            train_labels_fold = all_labels_np[train_idx]
            train_labels_bin = (train_labels_fold > 0).astype(int)
            class_counts = np.bincount(train_labels_bin)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[train_labels_bin]

            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

            train_path_fold = [all_paths[i] for i in train_idx]
            val_path_fold = [all_paths[i] for i in val_idx]

            train_dataset = ScrewDataset(train_path_fold, all_labels_np[train_idx], transform=train_transform)
            val_dataset = ScrewDataset(val_path_fold, all_labels_np[val_idx], transform=val_transform)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

            model = build_model_bin(model_name).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

            best_f2 = -1
            best_f2_epoch = -1
            best_f2_val_loss = float('inf')
            min_val_loss = float('inf')
            best_state = None
            no_improve = 0

            for epoch in range(num_epochs):
                model.train()
                running_train_loss = 0.0

                for imgs, lbls in train_loader:
                    imgs = imgs.to(device)
                    lbls = (lbls > 0).long().to(device)

                    optimizer.zero_grad()
                    outputs = model(imgs)

                    loss = criterion(outputs, lbls)
                    loss.backward()
                    optimizer.step()
                    
                    running_train_loss += loss.item()
                
                avg_train_loss = running_train_loss / len(train_loader)

                model.eval()
                running_val_loss = 0.0
                y_true_ep, y_pred_ep = [], []

                with torch.no_grad():
                    for imgs, lbls in val_loader:
                        imgs = imgs.to(device)
                        lbls = (lbls > 0).long().to(device)
                        
                        outputs = model(imgs)
                        running_val_loss += criterion(outputs, lbls).item()
                
                        preds = outputs.argmax(dim=1)
                        y_true_ep.extend(lbls.cpu().numpy())
                        y_pred_ep.extend(preds.cpu().numpy())
                    
                avg_val_loss = running_val_loss / len(val_loader)
                scheduler.step(avg_val_loss)

                report_ep = classification_report(y_true_ep, y_pred_ep, output_dict=True, zero_division=0)
                precision = report_ep.get('1', {}).get('precision', 0.0)
                recall = report_ep.get('1', {}).get('recall', 0.0)
                f2_ep = f2_score(precision, recall)

                if (epoch+1) % 10 == 0 or epoch == 0:
                    print(f"  Epoch {epoch+1:02d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | F2(bad): {f2_ep:.4f}")

                db[w][f'fold{fold_idx+1}']['train_loss'].append(avg_train_loss)
                db[w][f'fold{fold_idx+1}']['val_loss'].append(avg_val_loss)
                db[w][f'fold{fold_idx+1}']['f2_score'].append(f2_ep)

                # f2 score 더 좋을시 best 모델 갱신 및 db 값 갱신
                if f2_ep > best_f2:
                    best_f2 = f2_ep
                    best_f2_epoch = epoch+1
                    best_f2_val_loss = avg_val_loss
                    best_state = copy.deepcopy(model.state_dict())

                    db[w][f'fold{fold_idx+1}']['best_epoch'] = best_f2_epoch
                    db[w][f'fold{fold_idx+1}']['best_f2'] = best_f2

                elif f2_ep == best_f2 and avg_val_loss < best_f2_val_loss:
                    best_f2_epoch = epoch+1
                    best_f2_val_loss = avg_val_loss
                    best_state = copy.deepcopy(model.state_dict())

                    db[w][f'fold{fold_idx+1}']['best_epoch'] = best_f2_epoch
                    db[w][f'fold{fold_idx+1}']['best_f2'] = best_f2
                
                # early stopping / val loss 기준으로 판단
                if avg_val_loss < min_val_loss:
                    min_val_loss = avg_val_loss # val loss 갱신
                    no_improve = 0
                else:
                    no_improve += 1

                if early_stop_patience > 0 and no_improve >= early_stop_patience:
                    print(f"    -> early stop at epoch {epoch+1}")
                    break

            # epoch 반복문 끝
            
            # 여기서부터 좀 어려움
            if best_state is not None:
                temp_path = os.path.join(weights_dir, f"temp_w{w}_fold{fold_idx+1}.pth")
                torch.save(best_state, temp_path)
                current_w_paths.append(temp_path)
        

        # fold 반복문 끝
        f2_list = [db[w][f'fold{i}']['best_f2'] for i in range(1, 6)]

        mean_f2, std_f2 = np.mean(f2_list), np.std(f2_list)
        robust_score = mean_f2 - std_f2

        weight_f2[w] = f2_list

        print(f"\n Weight {w} 결산 | 평균: {mean_f2:.4f} | Std: {std_f2:.4f} | 보장점수(Robust): {robust_score:.4f}")

        if robust_score > global_best_robust:
            print(f'\n [robust score 갱신]: {global_best_robust:.4f} -> {robust_score}')
            global_best_robust = robust_score
            champion_weight = w

            for p in champion_paths:
                if os.path.exists(p):
                    os.remove(p)
            champion_paths = current_w_paths
        
        else:
            print()
            for p in current_w_paths:
                if os.path.exists(p):
                    os.remove(p)

    # weight 반복문 끝 (for문 아에 끝)
    print(f"\n{'='*60}")
    print(f"{model_name} / 모든 학습 종료! best 단일 모델을 선발 시작")

    champion_f2_scores = [db[champion_weight][f'fold{i}']['best_f2'] for i in range(1, 6)]
    best_fold_idx = np.argmax(champion_f2_scores) # (0~4)
    final_best_path = champion_paths[best_fold_idx]
    best_single_f2 = champion_f2_scores[best_fold_idx]

    for i, p in enumerate(champion_paths):
        if i != best_fold_idx and os.path.exists(p):
            os.remove(p)
        
    ultimate_champion_path = os.path.join(weights_dir, f"{model_name}_CHAMPION_w{champion_weight}_fold{best_fold_idx+1}.pth")
    os.rename(final_best_path, ultimate_champion_path)
            
    print(f"\n[{model_name} 최종] Weight: {champion_weight} / Fold: {best_fold_idx+1} (F2 Score: {best_single_f2:.4f})")
    print(f"최종 모델 파일: {ultimate_champion_path}")
    print(f"{'='*60}")


    # Test data 평가
    best_model = build_model_bin(model_name).to(device)
    best_model.load_state_dict(torch.load(ultimate_champion_path))
    best_model.eval()

    TEST_DIR = './data/k-fold_data/test'
    CLASS_NAMES = ['good', 'type1', 'type2', 'type3', 'type4', 'type5']

    test_paths, test_labels = [], []
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

    for class_idx, class_name in enumerate(CLASS_NAMES): 
        class_dir = os.path.join(TEST_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(valid_exts):
                test_paths.append(os.path.join(class_dir, fname))
                test_labels.append(class_idx) 

    test_labels_np = np.array(test_labels, dtype=np.int64)

    test_ds = ScrewDataset(test_paths, test_labels_np, transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    y_true_test, y_pred_test = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            lbls = (lbls > 0).long().to(device)

            outputs = best_model(imgs)
            preds = outputs.argmax(dim=1)
        
            y_true_test.extend(lbls.cpu().numpy())
            y_pred_test.extend(preds.cpu().numpy())

    best_train_loss = db[champion_weight][f'fold{best_fold_idx+1}']['train_loss']
    best_val_loss = db[champion_weight][f'fold{best_fold_idx+1}']['val_loss']

    return best_model, weight_f2, best_train_loss, best_val_loss, y_true_test, y_pred_test, db