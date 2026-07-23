import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
import torch
sys.path.append('.')
from src.utils import *
from src.visualization import *
from src.engine import *
from src.dataset import *
from src.mvtec_data import *
from torch.utils.data import DataLoader
from src.loc_eval import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0  # Windows/Jupyter에서는 0이 가장 안정적
print(f"Device: {device} | num_workers: {NUM_WORKERS}")

DATA_ROOT = r'./data/screw'
TEST_SIZE = 0.4
SEED = 42
BEST_MODEL_PATH = './outputs/best_model.pth'


def prepare_pool():
    """Test를 먼저 완전히 격리하고, 남은 pool(향후 CV 탐색 + 최종 재학습용)을 반환."""
    train_pool_records, test_records = load_mvtec_screw(DATA_ROOT, TEST_SIZE, SEED)
    return train_pool_records, test_records


def make_loader(records, transform, batch_size=32, shuffle=False):
    paths = [r['path'] for r in records]
    labels = [r['label'] for r in records]
    dataset = ScrewDataset(paths, labels, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _run_training(model, train_loader, val_loader, bad_weight, epochs=40, early_stop_patience=8, save_path=None, verbose=False):
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


def grid_search_weight(train_pool_records, weight_range, n_splits=5, epochs=40, early_stop_patience=8):
    """Stratified 5-Fold CV로 bad_weight별 Mean Max F2를 계산해 최적 weight를 선정."""
    types = [r['type'] for r in train_pool_records]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_splits = list(skf.split(train_pool_records, types))

    results = {}
    for w in weight_range:
        w = round(float(w), 2)
        fold_f2s = []

        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            fold_train = [train_pool_records[i] for i in train_idx]
            fold_val = [train_pool_records[i] for i in val_idx]

            train_loader = make_loader(fold_train, train_transform, shuffle=True)
            val_loader = make_loader(fold_val, val_transform, shuffle=False)

            set_seed(SEED)
            model = build_model_bin('resnet18').to(device)
            fold_f2 = _run_training(model, train_loader, val_loader, bad_weight=w,
                                     epochs=epochs, early_stop_patience=early_stop_patience)
            fold_f2s.append(fold_f2)
            print(f"    weight={w:.1f} fold={fold_idx+1}/{n_splits} max_f2={fold_f2:.4f}")

        mean_f2 = float(np.mean(fold_f2s))
        results[w] = {'fold_f2s': fold_f2s, 'mean_f2': mean_f2}
        print(f"  weight={w:.1f} | Mean Max F2={mean_f2:.4f}")

    best_weight = max(results, key=lambda k: results[k]['mean_f2'])
    print(f"\nBest weight: {best_weight} (Mean Max F2={results[best_weight]['mean_f2']:.4f})")
    return best_weight, results


def train_final_model(train_pool_records, bad_weight, epochs=40, early_stop_patience=8):
    """champion weight로 Train/Val 분할 후 최종 재학습, best checkpoint 저장."""
    train_records, val_records = train_test_split(
        train_pool_records,
        test_size=0.2,
        random_state=SEED,
        stratify=[r['type'] for r in train_pool_records]
    )

    train_loader = make_loader(train_records, train_transform, shuffle=True)
    val_loader = make_loader(val_records, val_transform, shuffle=False)

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

    test_loader = make_loader(test_records, val_transform, shuffle=False)

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


if __name__ == '__main__':
    train_pool_records, test_records = prepare_pool()

    best_weight = 2.0  # 논문의 ResNet-18 champion weight 그대로 사용 (그리드서치 생략)

    # train_final_model(train_pool_records, best_weight)
    test_model(test_records)
