
const dl = require('deeplearn');
const split = require('./split.js');

class BasicGRUCell {
  constructor(numUnits, variables) {
    this._numUnits = numUnits;
    this._activation = dl.tanh.bind(dl);

    this._variables = variables;
  }

  zeroState() {
    const h = dl.zeros([1, this._numUnits]);

    return [h];
  }

  async call(inputs, states) {
    const [state] = states;

    const gateKernel = await this._variables.get('gates/kernel');
    const gateBias = await this._variables.get('gates/bias');
    const candidateKernel = await this._variables.get('candidate/kernel');
    const candidateBias = await this._variables.get('candidate/bias');

    let gateInputs = dl.matMul(dl.concat([inputs, state], 1), gateKernel);
    gateInputs = dl.add(gateInputs, gateBias);

    const value = dl.sigmoid(gateInputs);
    const [r, u] = split(value, 2, 1);

    const rState = dl.mulStrict(r, state);

    let candidate = dl.matMul(dl.concat([inputs, rState], 1), candidateKernel);
    candidate = dl.add(candidate, candidateBias);

    const c = this._activation(candidate);
    const newH = dl.add(
      dl.mulStrict(u, state),
      dl.mulStrict(dl.sub(dl.ones([1]), u), c)
    );

    return [newH, [newH]]; // [output, new_state]
  }
}

module.exports = BasicGRUCell
