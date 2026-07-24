import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
import torch
sys.path.append('.')
from src.utils import *
from src.visualization import *
from src.engine import *
from src.dataset import *
from src.loc_eval import *
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0  # Windows/Jupyter에서는 0이 가장 안정적
print(f"Device: {device} | num_workers: {NUM_WORKERS}")

DATA_ROOT = './data/new_k-fold_data_gt'
CLASS_NAMES = ['good', 'type1', 'type2', 'type3', 'type4', 'type5']
SEED = 42
BEST_MODEL_PATH = './outputs/best_model.pth'


def load_kfold_gt(data_root=DATA_ROOT):
    """new_k-fold_data_gt를 record 리스트(path/type/label/mask)로 로드 (train/test는 폴더로 이미 고정 분할됨)."""
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

    def collect(split):
        records = []
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(data_root, split, class_name)
            label = 0 if class_name == 'good' else 1
            for fname in sorted(os.listdir(class_dir)):
                if not fname.lower().endswith(valid_exts):
                    continue
                path = os.path.join(class_dir, fname)
                if class_name == 'good':
                    mask = None
                else:
                    stem = fname.split('.')[0]
                    mask = os.path.join(data_root, 'ground_truth', class_name, f'{stem}_mask.png')
                records.append({'path': path, 'type': class_name, 'label': label, 'mask': mask})
        return records

    return collect('train'), collect('test')


def _run_training(model, train_loader, val_loader, bad_weight, epochs=40, early_stop_patience=10, save_path=None, verbose=False):
    """학습 루프 공통 로직. save_path가 주어지면 best val F2 시점의 state_dict를 저장한다."""
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor([1.0, bad_weight]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_f2 = 0.0
    no_improve_f2 = 0

    for ep in range(epochs):
        model.train()
        running_train_loss = 0.0

        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = (lbls > 0).long().to(device)

            optimizer.zero_grad()
            loss = loss_function(model(imgs), lbls)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / max(len(train_loader), 1)

        model.eval()
        running_val_loss = 0.0
        y_true_ep, y_pred_ep = [], []

        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                lbls = (lbls > 0).long().to(device)

                logits = model(imgs)
                running_val_loss += loss_function(logits, lbls).item()

                y_true_ep.extend(lbls.cpu().numpy())
                y_pred_ep.extend(logits.argmax(dim=1).cpu().numpy())

        avg_val_loss = running_val_loss / max(len(val_loader), 1)
        f2_ep = fbeta_score(y_true_ep, y_pred_ep, beta=2, pos_label=1, zero_division=0)

        if f2_ep > best_f2:
            best_f2 = f2_ep
            no_improve_f2 = 0
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
        else:
            no_improve_f2 += 1

        if verbose:
            print(f"    Ep {ep+1:02d} | TrL {avg_train_loss:.4f} | VaL {avg_val_loss:.4f} | "
                  f"F2 {f2_ep:.4f} | Best {best_f2:.4f}")

        if early_stop_patience > 0 and no_improve_f2 >= early_stop_patience:
            if verbose:
                print(f"    -> Early stop at epoch {ep + 1} (no improve for {early_stop_patience} epochs)")
            break

    return best_f2


def train_final_model(train_records, bad_weight, epochs=40, early_stop_patience=10):
    """champion weight로 Train/Val 분할 후 최종 재학습, best checkpoint 저장."""
    train_paths = [r['path'] for r in train_records]
    train_labels = [r['label'] for r in train_records]
    train_types = [r['type'] for r in train_records]

    tr_paths, val_paths, tr_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=0.2,
        stratify=train_types,
        random_state=SEED
    )

    train_loader = DataLoader(ScrewDataset(tr_paths, tr_labels, transform=train_transform),
                               batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(ScrewDataset(val_paths, val_labels, transform=val_transform),
                             batch_size=16, shuffle=False, num_workers=NUM_WORKERS)

    set_seed(SEED)
    model = build_model_bin('resnet18').to(device)
    best_f2 = _run_training(model, train_loader, val_loader, bad_weight=bad_weight,
                             epochs=epochs, early_stop_patience=early_stop_patience,
                             save_path=BEST_MODEL_PATH, verbose=True)
    print(f"최종 재학습 완료 | Best Val F2={best_f2:.4f}")


def test_model(test_records, model_path=BEST_MODEL_PATH):
    model = build_model_bin('resnet18').to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = DataLoader(
        ScrewDataset([r['path'] for r in test_records], [r['label'] for r in test_records], transform=val_transform),
        batch_size=16, shuffle=False, num_workers=NUM_WORKERS
    )

    y_true_test, y_pred_test, y_prob_test = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            logits = model(imgs.to(device))
            sf = torch.softmax(logits, dim=1)[:, 1]

            y_true_test.extend(lbls.cpu().numpy())
            y_pred_test.extend(logits.argmax(dim=1).cpu().numpy())
            y_prob_test.extend(sf.cpu().numpy())

    print(f'precision: {precision_score(y_true_test, y_pred_test):.4f}',
            f'recall: {recall_score(y_true_test, y_pred_test):.4f}',
            f'f2: {fbeta_score(y_true_test, y_pred_test, beta=2, pos_label=1):.4f}',
            f'이미지 auroc: {roc_auc_score(y_true_test, y_prob_test):.4f}')

    loc_metrics = evaluate_localization(model, test_records, device)
    print(f"pixel auroc: {loc_metrics['pixel_auroc']:.4f}",
            f"aupro: {loc_metrics['aupro']:.4f}")

    loc_by_type = evaluate_localization_by_type(model, test_records, device)
    for t, m in loc_by_type.items():
        print(f"  [{t:20s}] n={m['n']:3d} | pixel auroc: {m['pixel_auroc']:.4f} | aupro: {m['aupro']:.4f}")

    show_localization_grid(model, test_records, device)

    show_gradcam_grid_fixed(model, val_transform, base_dir='test', device=device,
                             save_dir='./outputs', model_name='ResNet-18',
                             data_root=DATA_ROOT)


if __name__ == '__main__':
    train_records, test_records = load_kfold_gt()

    best_weight = 2.0  # 논문의 ResNet-18 champion weight 그대로 사용 (그리드서치 생략)

    # train_final_model(train_records, best_weight)
    test_model(test_records)
