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

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0  # Windows/Jupyter에서는 0이 가장 안정적
print(f"Device: {device} | num_workers: {NUM_WORKERS}")


def prepare_data():
    data_root = r'./data/screw'
    test_size = 0.4
    seed = 42

    train_records, test_records = load_mvtec_screw(data_root, test_size, seed)

    train_records, val_records = train_test_split(
        train_records,
        test_size=0.2,
        random_state=seed,
        stratify=[r['type'] for r in train_records]
    )

    train_paths = [r['path'] for r in train_records]
    val_paths = [r['path'] for r in val_records]

    train_labels = [r['label'] for r in train_records]
    val_labels = [r['label'] for r in val_records]

    train_dataset = ScrewDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = ScrewDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_records


def train_model(train_loader, val_loader):
    model = build_model_bin('resnet18').to(device)
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_f2 = 0.0
    best_epoch = -1
    best_val_loss = float('inf')
    no_improve_f2 = 0
    epochs = 40
    early_stop_patience = 8

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
            best_epoch = ep
            best_val_loss = avg_val_loss
            no_improve_f2 = 0

            save_path = f'./outputs/best_model.pth'
            torch.save(model.state_dict(), save_path)
        else:
            no_improve_f2 += 1

        # if ep == 0 or (ep+1)%10 == 0:
        print(f"    Ep {ep+1:02d} | "
            f"TrL {avg_train_loss:.4f} | "
            f"VaL {avg_val_loss:.4f} | "
            f"F2 {f2_ep:.4f} | "
            f"Best {best_f2:.4f}")

        if early_stop_patience > 0 and no_improve_f2 >= early_stop_patience:
            print(f"    -> Early stop at epoch {ep + 1} "
                f"(no improve for {early_stop_patience} epochs)")
            break


def test_model(test_records):
    model = build_model_bin('resnet18').to(device)
    save_path = f'./outputs/best_model.pth'
    model.load_state_dict(torch.load(save_path))

    from sklearn.metrics import roc_auc_score

    y_true_test = []
    y_pred_test = []
    y_prob_test = []
    model.eval()
    test_paths = [r['path'] for r in test_records]
    test_labels = [r['label'] for r in test_records]

    
    test_dataset = ScrewDataset(test_paths, test_labels, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

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





if __name__ == '__main__':
    train_loader, val_loader, test_records = prepare_data()
    # train_model(train_loader, val_loader)
    test_model(test_records)