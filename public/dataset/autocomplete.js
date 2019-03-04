
const d3 = require('../d3.js');

class AutoComplete {
  constructor(assertDirectory) {
    this._loader = d3.json(assertDirectory + 'data/autocomplete.json');

    this._charMap = null;
    this._charMapInverse = null;
    this._wordMap = null;
    this._wordMapInverse = null;
  }

  async getCharMap() {
    if (this._charMap === null) {
      const maps = await this._loader;
      this._charMap = maps['char_map'];
    }

    return this._charMap;
  }

  async getCharMapInverse() {
    if (this._charMapInverse === null) {
      const charMap = await this.getCharMap();
      this._charMapInverse = new Map();
      for (let i = 0; i < charMap.length; i++) {
        this._charMapInverse.set(charMap[i], i);
      }
    }

    return this._charMapInverse;
  }

  async getWordMap() {
    if (this._wordMap === null) {
      const maps = await this._loader;
      this._wordMap = maps['word_map'];
    }

    return this._wordMap;
  }

  async getWordMapInverse() {
    if (this._wordMapInverse === null) {
      const wordMap = await this.getWordMap();
      this._wordMapInverse = new Map();
      for (let i = 0; i < wordMap.length; i++) {
        this._wordMapInverse.set(wordMap[i], i);
      }
    }

    return this._wordMapInverse;
  }

  async encodeSource(texts) {
    const charMapInverse = await this.getCharMapInverse();
    const source = [];
    const length = [];

    for (const text of texts) {
      const chars = text.split('');
      const encoded = new Int32Array(chars.length);
      for (let i = 0; i < chars.length; i++) {
        encoded[i] = charMapInverse.get(chars[i]);
      }
      source.push(encoded);
      length.push(chars.length);
    }

    return { length, source };
  }
}

module.exports = AutoComplete;
