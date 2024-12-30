import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    def __init__(self, configs):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # self.decompsition = series_decomp(configs.moving_avg)
        self.channels = configs.enc_in

        self.coarse_linear = nn.Linear(self.seq_len, self.pred_len)
        # self.seasonallinear = nn.Linear(self.seq_len, self.pred_len)
        self.fine_linear = nn.Linear(self.seq_len + self.pred_len, self.pred_len)

        self.coarse_linear.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        
        # kaiming + eye
        # with torch.no_grad():
        #     self.fine_linear.weight[:, self.seq_len:] = torch.eye(self.pred_len)

        # mean + mean
        # self.fine_linear.weight = nn.Parameter(
        #     (1 / self.seq_len + self.pred_len) * torch.ones([self.pred_len, self.seq_len + self.pred_len]))
        
        # zero + eye
        # self.fine_linear.weight = nn.Parameter(torch.cat([
        #     torch.zeros([self.pred_len, self.seq_len]), torch.eye(self.pred_len)
        # ], dim=-1))

        # zero + zero
        self.fine_linear.weight = nn.Parameter(torch.cat([
            torch.zeros([self.pred_len, self.seq_len + self.pred_len])
        ], dim=-1))

        # frozen
        # for param in self.fine_linear.parameters():
        #     param.requires_grad = False
    
    def coarse_forecast(self, x):
        y = self.coarse_linear(x)
        return y

    def fine_forecast(self, xy):
        y = self.fine_linear(xy)
        return y

    def forecast(self, x):
        # revin
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
 
        # coarse forward
        x = x.permute(0, 2, 1)
        y = self.coarse_forecast(x)

        # fine forward
        xy = torch.cat([x, y], dim=-1)
        # y = self.fine_forecast(xy)
        y = self.fine_forecast(xy) + y

        # revin
        y = y.permute(0, 2, 1)
        y = y * \
            (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        y = y + \
            (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return y

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
