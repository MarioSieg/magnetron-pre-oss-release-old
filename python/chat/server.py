
from dotenv import load_dotenv
from flask import Flask, render_template, request
import wavelet.core as wl
from wavelet.models import *

load_dotenv()

EPOCHS: int = 10000
LEARNING_RATE: float = 0.8

# Inputs
inputs = [
    wl.Tensor.const([0.0, 0.0]),
    wl.Tensor.const([0.0, 1.0]),
    wl.Tensor.const([1.0, 0.0]),
    wl.Tensor.const([1.0, 1.0])
]

# Targets
targets = [
    wl.Tensor.const([0.0]),
    wl.Tensor.const([1.0]),
    wl.Tensor.const([1.0]),
    wl.Tensor.const([0.0])
]

mlp = SequentialModel([
    DenseLayer(2, 4),
    DenseLayer(4, 1)
])

# Train model XOR
print('Training XOR model...')
losses = mlp.train(inputs, targets, EPOCHS, LEARNING_RATE)

print('Launching Flask server...')
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/v1/bot_response')
def get_response():
    try:
        message = request.args.get('message')
        splits = message.split(' ')
        a: float = float(splits[0])
        b: float = float(splits[1])
        result: float = mlp.forward(wl.Tensor.const([a, b]), activation=wl.Operator.HARD_SIGMOID).scalar()
        return f'{a} ^ {b} = {result}'
    except:
        return 'Please enter a valid input'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', use_reloader=False) #  host='0.0.0.0'
