import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp, moving_avg



class Model(nn.Module):
    """
    """

    def __init__(self, configs):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        self.channels = configs.enc_in

        self.decomposition = moving_avg(configs.moving_avg, stride=1)

        self.trend_predictor = nn.Linear(self.seq_len, self.pred_len)
        self.seasonal_embed = nn.ModuleList([nn.Embedding(n_embd, self.channels) for n_embd in [13, 32, 8, 24]])

        self.trend_mapper = nn.Linear(self.seq_len + self.pred_len, self.seq_len + self.pred_len)
        self.seasonal_mapper = nn.Linear(self.seq_len + self.pred_len, self.seq_len + self.pred_len)

        # for s_embd in self.seasonal_embed:
        #     s_embd.weight 
        self.trend_predictor.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.trend_mapper.weight = nn.Parameter(
            (1 / (self.seq_len + self.pred_len)) * torch.ones([self.seq_len+self.pred_len, self.seq_len+self.pred_len]))
        self.seasonal_mapper.weight = nn.Parameter(
            (1 / (self.seq_len + self.pred_len)) * torch.ones([self.seq_len+self.pred_len, self.seq_len+self.pred_len]))

        if self.task_name == 'classification':
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x_obs, x_tim, y_tim):
        # y_tim.shape: (batch, pred_len, k)
        y_tim = y_tim[:, -self.pred_len:, :]
        timestamp = torch.cat([x_tim, y_tim], dim=1).int()

        seasonal_init = torch.zeros((timestamp.shape[0], timestamp.shape[1], self.channels), 
            device=y_tim.device)
        for t, embd_layer in enumerate(self.seasonal_embed):
            seasonal_init += embd_layer(timestamp[:, :, t])
        seasonal_init = seasonal_init.permute(0, 2, 1)

        trend_history = self.decomposition(x_obs).permute(0, 2, 1)
        trend_future = self.trend_predictor(trend_history)
        trend_init = torch.cat([trend_history, trend_future], dim=-1)
        
        trend_output = self.trend_mapper(trend_init)
        seasonal_output = self.seasonal_mapper(seasonal_init)

        xy = (trend_output + seasonal_output).permute(0, 2, 1)
        return xy

    def forecast(self, x_obs, x_tim, y_tim):
        # Encoder
        return self.encoder(x_obs, x_tim, y_tim)

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

    def forward(self, x_obs, x_tim, _x_dec, y_tim, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_obs, x_tim, y_tim)
            return dec_out[:, :self.seq_len + self.pred_len, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_obs)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_obs)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_obs)
            return dec_out  # [B, N]
        return None
