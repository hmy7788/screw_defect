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


def measure_inference_speed(model, device, input_size=(1, 3, 224, 224), num_runs=100):
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

    print(f"측정 결과: {fps:.1f} FPS (1장당 {latency_ms:.2f} ms)")
    return fps, latency_ms