import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp, moving_avg

class series_decomp_trend(nn.Module):
    def __init__(self, kernel_size=25, sigma_type='math', seq_len=96):
        super(series_decomp_trend, self).__init__()
        kernel_size = 5
        self.series_decomp = series_decomp(kernel_size)

        self.sigma_type = sigma_type
     
        if self.sigma_type == "math":
            # bug
            self.moving_avg = moving_avg(kernel_size, stride=1)
        elif self.sigma_type == "linear":
            self.sigma_proj = nn.Linear(seq_len, seq_len)
    

    def forward(self, x):
        res_x, trend_mu = self.series_decomp(x)
        if self.sigma_type == 'math':
            res_x = torch.pow(res_x, 2)
            res_x = self.moving_avg(res_x)
            trend_sigma = torch.sqrt(res_x)
        elif self.sigma_type == "linear":
            trend_sigma = self.sigma_proj(res_x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            raise NotImplementedError()
            pass
        # print(torch.tensor([torch.std(x[0, i:i+5, 0]).item() for i in range(2, 7)]))
        # exit()
        return trend_mu, trend_sigma 

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, sigma_type="linear", eps=1e-5):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        self.sigma_type = sigma_type
        self.decompsition = series_decomp_trend(kernel_size=configs.moving_avg, sigma_type=self.sigma_type, seq_len=self.seq_len)
        self.channels = configs.enc_in
        self.eps = eps

        self.Linear_Trend_Mu = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend_Sigma = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)

        self.Linear_Trend_Mu.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.Linear_Trend_Sigma.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        if self.task_name == 'classification':
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x_obs, x_tim, y_tim):
        trend_mu_init, trend_sigma_init = self.decompsition(x_obs)

        seasonal_init = (x_obs - trend_mu_init) / (trend_sigma_init + self.eps)
        seasonal_init = x_obs - trend_mu_init

        trend_mu_init, trend_sigma_init = trend_mu_init.permute(
            0, 2, 1), trend_sigma_init.permute(0, 2, 1)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        
        trend_mu_output = self.Linear_Trend_Mu(trend_mu_init)
        trend_sigma_output = self.Linear_Trend_Sigma(trend_sigma_init)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        
        x = trend_mu_output + seasonal_output * (trend_sigma_output + self.eps)
        # x = trend_mu_output + seasonal_output
        return x.permute(0, 2, 1)

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
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
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
