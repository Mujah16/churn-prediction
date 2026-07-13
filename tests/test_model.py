import torch

from models.lstm_churn_model import LSTMChurnModel


def test_lstm_churn_model_forward_shape() -> None:
    model = LSTMChurnModel(input_dim=10, hidden_dim=8, num_layers=1, dropout=0.1)
    batch = torch.randn(4, 1, 10)

    output = model(batch)

    assert output.shape == (4, 1)
