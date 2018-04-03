
const dl = require('deeplearn');
const split = require('./split.js');

class BasicNLSTMCell {
  constructor(numUnits, depth, forgetBias, variables) {
    this._numUnits = numUnits;
    this._depth = depth;
    this._forgetBias = forgetBias;
    this._activation = dl.tanh.bind(dl);

    this._variables = variables
  }

  zeroState() {
    const h = dl.zeros([1, this._numUnits]);
    const cs = [];
    for (let d = 0; d < this._depth; d++) {
      cs.push(dl.zeros([1, this._numUnits]));
    }

    return [h, ...cs];
  }

  async _recurrence(inputs, hiddenState, cellStates, depth) {
    // Parameters of gates are concatenated into one multiply for efficiency
    const c = cellStates[depth];
    const h = hiddenState;

    let gateInputs = dl.matMul(
        dl.concat([inputs, h], 1), await this._variables.get(`kernel_${depth}`)
    )
    gateInputs = dl.add(gateInputs, await this._variables.get(`bias_${depth}`))

    // i = input_gate, j = new_input, f = forget_gate, o = output_gate
    const [i, j, f, o] = split(gateInputs, 4, 1);

    const innerHidden = dl.mulStrict(
        c,
        dl.sigmoid(dl.add(f, this._forgetBias))
    );
    let innerInput;
    if (depth === 0) {
        innerInput = dl.mulStrict(dl.sigmoid(i), j);
    } else {
        innerInput = dl.mulStrict(dl.sigmoid(i), this._activation(j));
    }

    // For the final layer, just do what a normal LSTM cell would do
    let newC, newCs;
    if (depth == this._depth - 1) {
        newC = dl.add(innerHidden, innerInput);
        newCs = [newC];
    // For the other layers, calculate the cell state using a nested LSTM
    } else {
        [newC, newCs] = await this._recurrence(
            innerInput,
            innerHidden,
            cellStates,
            depth + 1
        );
    }

    const newH = dl.mulStrict(this._activation(newC), dl.sigmoid(o));
    newCs = [newH, ...newCs];

    return [newH, newCs];
  }

  async call(inputs, states) {
    const [hiddenState, ...cellStates] = states;

    const [outputs, newState] = await this._recurrence(
      inputs,
      hiddenState,
      cellStates,
      0
    )

    return [outputs, newState]; // [output, new_state]
  }
}

module.exports = BasicNLSTMCell
