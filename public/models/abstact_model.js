
const dl = require('deeplearn');
const VariableScope = require('../operator/variable_scope.js');

class LoaderCache {
  constructor(loaderpath) {
    this._loader = new dl.CheckpointLoader(loaderpath);
    this._cache = new Map();
  }

  initialize() {
    return this._loader.getCheckpointManifest();
  }

  async get(path) {
    if (this._cache.has(path)) {
      return await this._cache.get(path);
    }

    this._cache.set(path, this._loader.getVariable(path));
    return await this._cache.get(path);
  }
}

class AbstactModel {
  constructor(params, dataDirectory, name='unnamed') {
    this._params = params;
    this._loader = new LoaderCache(dataDirectory + `save/${name}`);
    this._variables = new VariableScope(this._loader, 'model');
    this.ready = false;
  }

  async load() {
    await this._loader.initialize();
    const [logitsTensor, probabilityTensor] = await this._model_fn(
      {
        source: dl.tensor([0], [1], 'int32'),
        length: dl.tensor(1, [], 'int32')
      },
      this._variables,
      this._params
    );
    this.ready = true;
  }

  async * predict(dataset) {
    await this._loader.initialize();

    for (let i = 0; i < dataset.source.length; i++) {
      const length = dl.tensor(dataset.length[i], [], 'int32');
      const source = dl.tensor(dataset.source[i], [
        dataset.source[i].length
      ], 'int32');

      const [logitsTensor, probabilityTensor] = await this._model_fn(
        { source, length }, this._variables, this._params
      );
      this.ready = true;

      const [logits, probability] = await Promise.all([
        logitsTensor.data(), probabilityTensor.data()
      ])

      yield { logits, probability };
    }
  }
}

module.exports = AbstactModel
