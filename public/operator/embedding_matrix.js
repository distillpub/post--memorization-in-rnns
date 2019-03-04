
const dl = require('deeplearn');

function embeddingMatrix(initialization) {
  // Create embedding matrix with zeros for <eos>
  return dl.concat([
    dl.zeros([1, initialization.shape[1]], 'float32'),
    initialization
  ], 0)
}

module.exports = embeddingMatrix
