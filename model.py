import math
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# hparams
dim_in = 200
dim_out = 4
n_hid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
n_layers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
n_head = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
batch_size = 64
eval_batch_size = 32
bptt = 35
lr = 5.0 # learning rate

model = GoingMarry(dim_in, dim_out, n_head, n_hid, n_layers, dropout, batch_size, eval_batch_size, bptt).to(device)
# model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)


class GoingMarry(nn.Module):
    def __init__(self, dim_in, dim_out, n_head, n_hid, n_layers, dropout, batch_size, eval_batch_size, bptt, n_epochs, lr, code):
        super(GoingMarry, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.preprocessor = None
        self.n_epochs = n_epochs
        self.lr = lr
        self.code = code
        self.model = TransformerModel(dim_in, dim_out, n_head, n_hid, n_layers, dropout)
    
    def set_data(self, train_data, val_data, test_data):
        self.train_data = batchify(train_data, batch_size)
        self.val_data = batchify(val_data, eval_batch_size)
        self.test_data = batchify(test_data, eval_batch_size)
    
    def save_model(self, model):
        torch.save(model, os.path.join(os.getcwd(), 'save', '-'.join(self.train_start_time, self.code) + '.pt'))

    def train(self):
        self.train_start_time = time.strftime("%Y%m%d-%H%M%S")
        self.total_loss = 0.
        self.best_val_loss = 0.
        self.best_model = self.model
        self.save_model(self.best_model)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        for epoch in range(1, self.n_epochs + 1):
            epoch_start_time = time.time()
            self.train_epoch()  # Train one epoch
            val_loss = self.evaluate()  # Validate
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = self.model
                self.save_model(self.best_model)

            self.scheduler.step()

    def train_epoch(self):
        self.model.train() # Turn on the train mode
        start_time = time.time()
        src_mask = self.model.generate_square_subsequent_mask(self.bptt).to(self.device)
        for batch, i in enumerate(range(0, self.train_data.size(0) - 1, self.bptt)):
            data, targets = get_batch(self.train_data, i)
            optimizer.zero_grad()
            if data.size(0) != self.bptt:
                src_mask = self.model.generate_square_subsequent_mask(data.size(0)).to(self.device)
            output = self.model(data, src_mask)
            loss = criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            # Logs
            log_interval = 200
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                    'lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(train_data) // self.bptt, scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
            
    def evaluate(self):
        self.model.eval() # Turn on the evaluation mode
        total_loss = 0.
        src_mask = self.model.generate_square_subsequent_mask(bptt).to(device)
        with torch.no_grad():
            for i in range(0, self.val_data.size(0) - 1, bptt):
                data, targets = get_batch(self.val_data, i)
                if data.size(0) != bptt:
                    src_mask = self.model.generate_square_subsequent_mask(data.size(0)).to(device)
                output = self.model(data, src_mask)
                total_loss += len(data) * criterion(output, targets).item()
        return total_loss / (len(self.val_data) - 1)
    
    def batchify(self, data, bsz):
        # Divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(self.device)
    
    def get_batch(self, source, i):
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target

    def test(self):
        test_loss = self.evaluate(self.best_model, self.test_data)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)


class TransformerModel(nn.Module):
    def __init__(self, dim_in, dim_out, n_head, n_hid, n_layers, dropout):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(dim_in, dropout)
        encoder_layers = TransformerEncoderLayer(dim_in, n_head, n_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        # self.encoder = nn.Embedding(ntoken, dim_in)
        self.dim_in = dim_in
        self.decoder = nn.Linear(dim_in, dim_out)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.dim_in)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
