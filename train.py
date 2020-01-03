import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
from collections import Counter
from dataloader import *
from model import *
from tqdm import tqdm
import logging
from datetime import datetime
import argparse
from sklearn.metrics import precision_recall_curve, auc


def train(data, model, optim, criterion, max_clip_norm=0):
    input = data[:, :-1].to(device)
    label = data[:,-1].to(device)
    model.train()
    optim.zero_grad()
    logits = model(input)
    loss = criterion(logits, label)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip_norm)
    loss.backward()
    optim.step()
    return loss.item()


def evaluate(model, data_iter, length):
    model.eval()
    y_pred = np.zeros(length)
    y_true = np.zeros(length)
    y_prob = np.zeros(length)
    pointer = 0
    for data in data_iter:
        batch_size = data.size()[0]
        input = data[:, :-1].to(device)
        label = data[:, -1].to(device)
        output = model(input)
        _, predicted = torch.max(output.data, 1)
        probability = torch.softmax(output.data, 1)[:, 1]
        y_true[pointer: pointer + batch_size] = label.cpu().numpy()
        y_pred[pointer: pointer + batch_size] = predicted.cpu().numpy()
        y_prob[pointer: pointer + batch_size] = probability.cpu().numpy()
        pointer += batch_size
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)


if torch.cuda.is_available():
    device = 'cuda'
else:
    raise Exception('No CUDA')
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=int, default=512, help='size of input size')
parser.add_argument('--output', type=int, default=512, help='size of output size')
parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
parser.add_argument('--batch',  type=int, default=16, help='batch size')
parser.add_argument('--dropout',  default='0.4', type=float, help='dropout rate')
parser.add_argument('--alpha',  default='0.15', type=float, help='activation rate')
parser.add_argument('--lr',  default='0.0001', type=float, help='learning rate')
parser.add_argument('--epoch',  default=20, type=int, help='max epoch')
parser.add_argument('--path',  default='./data/', help='path of files')
parser.add_argument('--result',  default='./result', help='path of results')


opt = parser.parse_args()

path = opt.path
epoch = opt.epoch
BATCH_SIZE = opt.batch
n_heads = opt.heads
in_feature = opt.input
out_feature = opt.output
BATCH_SIZE = opt.batch
dropout = opt.dropout
alpha = opt.alpha
lr = opt.lr

num_of_nodes = 3589
train_idx_name = 'train_idx.pkl'
val_idx_name = 'val_idx.pkl'
test_idx_name = 'test_idx.pkl'

s = datetime.now().strftime('%Y%m%d%H%M%S')
result_root = '%s/%s-lr_%s-input_%s-output_%s-dropout_%s'%(opt.result, s, opt.lr, opt.input, opt.output, opt.dropout)
if not os.path.exists(result_root):
    os.mkdir(result_root)

logging.basicConfig(filename='%s/train.log'%result_root, format='%(asctime)s %(message)s', level=logging.INFO)
data = OriginalData(path)
device_ids = list(np.arange(torch.cuda.device_count(), dtype='int'))
device_ids = range(torch.cuda.device_count())
model = GAT(in_feature, out_feature, num_of_nodes, n_heads, dropout=dropout, alpha = alpha).to(device)
model = nn.DataParallel(model, device_ids=device_ids)
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda count: 0.9 ** count])
val_x, val_y = data.datasampler(val_idx_name, train=False)
val_loader = DataLoader(dataset=EHRData(val_x, val_y), batch_size=BATCH_SIZE,
                        collate_fn=collate_fn, num_workers=torch.cuda.device_count(), shuffle=False)

for epoch in range(epoch):
    total_loss = 0
    x_train, y_train = data.datasampler(train_idx_name, train=True)
    ratio = Counter(y_train)
    train_loader = DataLoader(dataset=EHRData(x_train,y_train), batch_size=BATCH_SIZE,
                              collate_fn=collate_fn, num_workers=torch.cuda.device_count(), shuffle=True)
    weight = torch.from_numpy((ratio[True]+ratio[False])/(2 * np.array([ratio[False],ratio[True]]))).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    if epoch % 5 == 0:
        val_auprc = evaluate(model, val_loader, len(val_y))
        logging.info('epoch:%d AUPRC:%f' % (epoch + 1, val_auprc))

    t = tqdm(iter(train_loader), leave=True, total=len(train_loader))
    for idx, batch_data in enumerate(t):
        batch_size = batch_data.size()[0]
        prev_row = batch_data
        loss = train(batch_data, model, optimizer, criterion, 5)
        total_loss += loss * BATCH_SIZE
        if idx % 50 == 0:
            t.set_description('[train epoch:%d] loss: %.8f' % (epoch + 1, total_loss))
            t.refresh()
    if epoch % 5 == 0:
        scheduler.step()
    torch.save(model.state_dict(), "%s/parameter%d" % (result_root, epoch + 1))






