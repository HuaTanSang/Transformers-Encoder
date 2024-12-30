import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import torch
from shutil import copyfile
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from vocab import Vocab
from dataset import ViOCD_Dataset, collate_fn
from transformer_encoder_model import TransformerEncoderModel_positional, TransformerEncoderModel_learnedpositional
from transformers_from_pytorch import TransformersEncoder_pytorch
device = "cuda" if torch.cuda.is_available() else "cpu"
scorers = {
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score
}

def train(epoch: int, model: nn.Module, dataloader: DataLoader, optim: torch.optim.Optimizer):
    model.train()

    running_loss = .0
    with tqdm(desc='Epoch %d - Training' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, items in enumerate(dataloader):
            # forward pass
            input_ids = items["input_ids"].to(device)
            labels = items["labels"].to(device)
            
            _, loss = model(input_ids, labels)
            
            # backward pass
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()

            # update the training status
            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

def compute_scores(predictions: list, labels: list) -> dict:
    scores = {}
    for scorer_name in scorers:
        scorer = scorers[scorer_name]
        scores[scorer_name] = scorer(labels, predictions, average="micro")

    return scores

def evaluate_metrics(epoch: int, model: nn.Module, dataloader: DataLoader) -> dict:
    model.eval()
    all_labels = []
    all_predictions = []
    scores = {}
    with tqdm(desc='Epoch %d - Evaluating' % epoch, unit='it', total=len(dataloader)) as pbar:
        for items in dataloader:
            input_ids = items["input_ids"].to(device)
            labels = items["labels"].to(device)
            with torch.no_grad():
                logits, _ = model(input_ids, labels)

            predictions = logits.argmax(dim=-1).long()
    
            labels = labels.view(-1).cpu().numpy()
            predictions = predictions.view(-1).cpu().numpy()

            all_labels.extend(labels)
            all_predictions.extend(predictions)

            pbar.update()
        # Calculate metrics
    scores = compute_scores(all_predictions, all_labels)

    return scores

def save_checkpoint(dict_to_save: dict, checkpoint_path: str):
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    torch.save(dict_to_save, os.path.join(f"{checkpoint_path}", "last_model.pth"))

def main(
        d_model: int,
        layer_dim: int,
        head: int,
        d_ff: int,
        dropout: float,
        train_path: str, 
        dev_path: str, 
        test_path: str,
        learning_rate: float,
        checkpoint_path: str = "checkpoints"):

    vocab = Vocab(
        train_path, dev_path, test_path
    )

    train_dataset = ViOCD_Dataset(train_path, vocab)
    dev_dataset = ViOCD_Dataset(dev_path, vocab)
    test_dataset = ViOCD_Dataset(test_path, vocab)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    epoch = 0
    score_name = "f1"
    allowed_patience = 5
    best_score = 0


    print("Choose model")
    print("1. Transformers with positinal Encoding")
    print("2. Transformers with Learned Positinal Embedding")
    print("3. Transformer from Pytorch")
    
    n = int(input())
    if n == 1: 
        model = TransformerEncoderModel_positional(
            d_model, head, layer_dim, d_ff, dropout, vocab
        )

    elif n == 2: 
        model = TransformerEncoderModel_learnedpositional(
            d_model, head, layer_dim, d_ff, dropout, vocab 
        )
    elif n == 3: 
        model = TransformersEncoder_pytorch(
            d_model, head, layer_dim, d_ff, dropout, vocab
            )        
    else:
        raise ValueError("Value is not valid")

    
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    while True:
        train(epoch, model, train_dataloader, optim)
        # val scores
        scores = evaluate_metrics(epoch, model, dev_dataloader)
        print(f"Dev scores: {scores}")
        score = scores[score_name]

        # Prepare for next epoch
        is_the_best_model = False
        if score > best_score:
            best_score = score
            patience = 0
            is_the_best_model = True
        else:
            patience += 1

        exit_train = False

        if patience == allowed_patience:
            exit_train = True

        save_checkpoint({
            "epoch": epoch,
            "best_score": best_score,
            "patience": patience,
            "state_dict": model.state_dict(),
            "optimizer": optim.state_dict()
        }, checkpoint_path)

        if is_the_best_model:
            copyfile(
                os.path.join(checkpoint_path, "last_model.pth"), 
                os.path.join(checkpoint_path, "best_model.pth")
            )

        if exit_train:
            break

        epoch += 1

    scores = evaluate_metrics(epoch, model, test_dataloader)
    print(f"Test scores: {scores}")

if __name__ == "__main__":
    main(
        d_model=512,
        head=8,
        layer_dim=3,
        d_ff=4086,
        dropout=0.5,    
        learning_rate=0.00035,
        train_path=r"D:\UIT\Tài liệu\Năm 3\Deep Learning\UIT-ViOCD\train.json",
        dev_path=r"D:\UIT\Tài liệu\Năm 3\Deep Learning\UIT-ViOCD\dev.json",
        test_path=r"D:\UIT\Tài liệu\Năm 3\Deep Learning\UIT-ViOCD\test.json",
        checkpoint_path=r"D:\UIT\Tài liệu\Năm 3\Deep Learning\UIT-ViOCD"
    )
