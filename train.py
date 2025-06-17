import torch
import torch.nn.functional as F
import argparse
import glob
import os
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from model.dualformer import DUALFormer_Model
from util import set_seed, DataLoader, dataset_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, args, data, optimizer, split):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    train_loss = F.nll_loss(output[split['train']], data.y[split['train']])
    train_loss.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        with torch.no_grad():
            output = model(data.x, data.edge_index)
            val_loss = F.nll_loss(output[split['valid']], data.y[split['valid']])
    return train_loss.item(), val_loss.item()

def test(model, data, split, best_epoch):
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        y_test = data.y[split['test']].detach().cpu().numpy()
        y_pred = output[split['test']].argmax(-1).detach().cpu().numpy()
        test_micro = f1_score(y_test, y_pred, average='micro')
        test_macro = f1_score(y_test, y_pred, average='macro')
        test_acc = accuracy_score(y_test, y_pred)
        return {
            'micro_f1': test_micro,
            'macro_f1': test_macro,
            'acc': test_acc}

def main(args):
    dataset = DataLoader(args.dataset)
    data = dataset[0].to(device)
    data.name = args.dataset
    activation = ({'relu': nn.ReLU, 'prelu': nn.PReLU, 'lrelu': nn.LeakyReLU, 'elu': nn.ELU})[args.activation]
    mi_list, ma_list, acc_list = [], [], []

    for run_id in range(args.runs):
        set_seed(args.seed)
        model = DUALFormer_Model(input_dim=data.num_features,
                    hidden_dim=args.hidden,
                    output_dim=dataset.num_classes,
                    activation=activation,
                    num_gnns=args.num_gnns,
                    num_trans=args.num_sa,
                    num_heads=args.num_heads,
                    dropout_trans=args.dropout_sa,
                    dropout=args.dropout,
                    alpha=args.alpha,
                    use_bn=args.use_bn,
                    lammda=args.lammda,
                    GraphConv=args.graphconv
                    ).to(device)

        optimizer = torch.optim.Adam([
                        {'params': model.params1, 'weight_decay': args.weight_decay_sa, 'lr': args.lr_sa},
                        {'params': model.params2, 'weight_decay': args.weight_decay, 'lr': args.lr}])

        min_val_loss = 1e9
        best_epoch = 0
        bad_counter = 0
        split = dataset_split(data, run_id)

        with tqdm(total=args.epochs, desc='(T)') as pbar:
            for epoch in range(0, args.epochs):
                train_loss, val_loss = train(model, args, data, optimizer, split)
                pbar.set_postfix({'Train Loss': train_loss, 'Val Loss': val_loss})
                pbar.update()
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
                    best_epoch = epoch
                    bad_counter = 0
                else:
                    bad_counter += 1
                if bad_counter == args.patience:
                    break
                files = glob.glob('*.pkl')
                for file in files:
                    epoch_nb = int(file.split('.')[0])
                    if epoch_nb < best_epoch:
                        os.remove(file)

            files = glob.glob('*.pkl')
            for file in files:
                epoch_nb = int(file.split('.')[0])
                if epoch_nb > best_epoch:
                    os.remove(file)

        result = test(model, data, split, best_epoch)
        mic, mac, acc = result["micro_f1"], result["macro_f1"], result["acc"]
        print('{}-th run,'.format(run_id+1), 'micro-f1:{:.2f}%,'.format(mic*100), 'macor-f1:{:.2f}%;'.format(mac*100), 'acc:{:.2f}%;'.format(acc*100))
        mi_list.append(mic)
        ma_list.append(mac)
        acc_list.append(acc)

    mi_mean, mi_std = np.mean(mi_list), np.std(mi_list)
    ma_mean, ma_std = np.mean(ma_list), np.std(ma_list)
    ac_mean, ac_std = np.mean(acc_list), np.std(acc_list)

    print("After", args.runs, "runs on ", args.dataset, "!")
    print("Micro-F1, mean ± std: {:.2f}%±{:.2f}".format(mi_mean*100,mi_std*100))
    print("Macro-F1, mean ± std: {:.2f}%±{:.2f}".format(ma_mean*100,ma_std*100))
    print("ACC, mean ± std: {:.2f}%±{:.2f}".format(ac_mean * 100, ac_std * 100))

    filename = f'results/{args.graphconv}_{args.dataset}.csv'
    print(f"Saving results to the'{filename}'")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f"mi:{mi_mean*100:.2f} ± {mi_std*100:.2f},"
                        + f"ma:{ma_mean*100:.2f} ± {ma_std*100:.2f},"
                        + f"ma:{ac_mean * 100:.2f} ± {ac_std * 100:.2f},"
                        + f"seed:{args.seed},"
                        + f"epochs:{args.epochs},"
                        + f"patience:{args.patience},"
                        + f"lr:{args.lr},"
                        + f"wd:{args.weight_decay},"
                        + f"lr_trans:{args.lr_sa},"
                        + f"wd_trans:{args.weight_decay_sa},"
                        + f"activation:{args.activation},"
                        + f"hidden:{args.hidden},"
                        + f"num_gnns:{args.num_gnns},"
                        + f"num_sa:{args.num_sa},"
                        + f"num_heads:{args.num_heads},"
                        + f"dropout:{args.dropout},"
                        + f"dropout_sa:{args.dropout_sa},"
                        + f"alpha:{args.alpha},"
                        + f"lammda:{args.lammda},"
                        + f"use_bn:{args.use_bn},"
                        + f"runs:{args.runs}\n"
                        )

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--fastmode', type=bool, default=False)
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--dataset', type=str, default='cora') #photo
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--lr_sa', type=float, default=0.001)
parser.add_argument('--weight_decay_sa', type=float, default=5e-4)
parser.add_argument('--activation', type=str, default='relu')#prelu, elu, relu, lrelu
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--num_gnns', type=int, default=2)
parser.add_argument('--num_sa', type=int, default=1)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--dropout_sa', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--lammda', type=float, default=0.1)
parser.add_argument('--graphconv', type=str, default='sgc')#'sgc', 'gcn', 'appnp'
parser.add_argument('--use_bn', type=bool, default=True)
args = parser.parse_args()
print("data:", args.dataset)
main(args)
