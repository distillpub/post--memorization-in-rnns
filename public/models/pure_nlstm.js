
const dl = require('deeplearn');
const AbstactModel = require('./abstact_model.js');
const embeddingMatrix = require('../operator/embedding_matrix.js');
const BasicNLSTMCell = require('../operator/basic_nlstm_cell.js');
const MultiRNNCell = require('../operator/multi_rnn_cell.js');
const dynamicRnn = require('../operator/dynamic_rnn.js');
const denseLayer = require('../operator/dense_layer.js');

class PureNLSTM extends AbstactModel {
  constructor(dataDirectory, name='autocomplete_nlstm') {
    super({
      'embedding_size': 600,
      'depth': 2,
      'layers': 1
    }, dataDirectory, name);
  }

  async _model_fn(features, variables, params) {
    const source = features.source;
    const length = features.length;

    // Create embedding matrix with zeros for <eos>
    const sourceEmbedding = embeddingMatrix(
      await variables.get('embedding-source')
    );

    // shape = [time, embedding_size]
    let logits = dl.gather(sourceEmbedding, source);

    // Setup RNN layers
    const rnnLayers = []
    for (let l = 0; l < params.layers; l++) {
      rnnLayers.push(new BasicNLSTMCell(
        params['embedding_size'],
        params['depth'],
        dl.scalar(1),
        variables.scope(`rnn/multi_rnn_cell/cell_${l}/basic_nlstm_cell`)
      ));
    }

    const multiRnnCell = new MultiRNNCell(rnnLayers);
    const [outputs, states] = await dynamicRnn(
      multiRnnCell,
      logits,
      source.shape[0]
    );

    // Get the output for the last layer for the last time step
    logits = outputs[outputs.length - 1];

    // Do a tf.layers.dense
    logits = await denseLayer(
      logits,
      variables.scope('dense')
    );

    // Compute predicted labels
    const predict = dl.argMax(logits, 1)

    // Compute softmax
    const probability = dl.softmax(logits)

    return [logits, probability]
  }
}

module.exports = PureNLSTM;
