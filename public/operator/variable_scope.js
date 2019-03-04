
class VariableScope {
  constructor(loader, path) {
    this._loader = loader;
    this._path = path;
  }

  get(name) {
    return this._loader.get(`${this._path}/${name}`)
  }

  scope(path) {
    return new VariableScope(this._loader, this._path + '/' + path)
  }
}

module.exports = VariableScope;
