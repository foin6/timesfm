import numpy as np
import timesfm
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("/home/mist/my_projs/timesfm/pretrain_model/timesfm-2.5-200m-pytorch", torch_compile=True, proxies=None, resume_download=True)

model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)
input_1 = np.linspace(0, 1, 100)
input_2 = np.sin(np.linspace(0, 20, 67))
point_forecast, quantile_forecast = model.forecast_naive(
    horizon=12,
    inputs=[
        input_1,
        input_2,
    ],  # Two dummy inputs
)
print(point_forecast.shape)  # (2, 12)
print(quantile_forecast.shape)  # (2, 12, 10): mean, then 10th to 90th quantiles.