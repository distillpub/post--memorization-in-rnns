
const dl = require('deeplearn')

async function dynamicRnn(cell, inputs, sequenceLength) {
  const outputs = [];
  const states = [];

  let nextState = cell.zeroState();
  for (let t = 0; t < sequenceLength; t++) {
    const inputAtTimeT = dl.gather(inputs, dl.tensor([t], [1], 'int32'));
    const [output, state] = await cell.call(inputAtTimeT, nextState);

    outputs.push(output);
    states.push(state);
    nextState = state;
  }

  return [outputs, states];
}

module.exports = dynamicRnn;
