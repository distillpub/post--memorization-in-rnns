
const dl = require('deeplearn');

async function denseLayer(input, variables) {
  return dl.add(
    dl.matMul(input, await variables.get('kernel')
  ), await variables.get('bias'));
}

module.exports = denseLayer;
