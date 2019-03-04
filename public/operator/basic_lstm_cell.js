
const dl = require('deeplearn');

class BasicLSTMCell {
  constructor(numUnits, forgetBias, variables) {
    this._numUnits = numUnits;
    this._forgetBias = forgetBias;
    this._variables = variables;
  }

  zeroState() {
    const h = dl.zeros([1, this._numUnits]);
    const c = dl.zeros([1, this._numUnits]);

    return [h, c];
  }

  async call(inputs, states) {
    const [h, c] = states;

    const [newC, newH] = dl.basicLSTMCell(
      this._forgetBias,
      await this._variables.get('kernel'),
      await this._variables.get('bias'),
      inputs,
      c,
      h
    );
    const newState = [newH, newC];

    return [newH, newState]; // [output, new_state]
  }
}

module.exports = BasicLSTMCell
