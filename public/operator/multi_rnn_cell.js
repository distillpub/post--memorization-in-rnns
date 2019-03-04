
class MultiRNNCell {
  constructor(layers) {
    this._layers = layers;
  }

  zeroState() {
    const state = [];
    for (const layer of this._layers) {
      state.push(layer.zeroState());
    }
    return state;
  }

  async call(inputs, states) {

    let newStates = [];
    let nextInputs = inputs;
    for (let i = 0; i < this._layers.length; i++) {
      const layer = this._layers[i];
      const [output, newState] = await layer.call(nextInputs, states[i]);

      newStates.push(newState);
      nextInputs = output;
    }

    return [nextInputs, newStates]; // [output, new_state]
  }
}

module.exports = MultiRNNCell
