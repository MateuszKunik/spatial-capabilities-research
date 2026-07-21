import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(
            self,
            num_inputs: int = 2,
            num_outputs: int = 2,
            hidden_size: int = 8,
            num_layers: int = 1,
            batch_first: bool = True
    ):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=num_inputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
        )

        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs,
        )

    def forward(self, x):
        output, hidden_state = self.rnn(x)

        predictions = self.fc(output)

        return predictions