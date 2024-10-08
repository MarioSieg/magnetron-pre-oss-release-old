from msml.core import *

class MultilayerPerceptron:
    def __init__(self, ctx: Context, input_size: int, hidden_size: int, output_size: int):
        self.ctx = ctx

        self.weights_input_hidden = Tensor.with_data(ctx, [2, 2], data=
            [10.0, -10.0, 10.0, 10.0]
        )
        self.bias_hidden = Tensor.with_data(ctx, [2, 1], data=
            [-5.0, -15.0]
        )
        self.weights_hidden_output = Tensor.with_data(ctx, [2, 1], data=
            [10.0, 10.0]
        )
        self.bias_output = Tensor.with_data(ctx, [1, 1], data=
            [-10.0]
        )

    def forward(self, inputs):
        hidden_input = (inputs @ self.weights_input_hidden) + self.bias_hidden
        hidden_output = hidden_input.sigmoid()
        output_input = (hidden_output @ self.weights_hidden_output) + self.bias_output
        output_output = output_input.sigmoid()
        return output_output()
