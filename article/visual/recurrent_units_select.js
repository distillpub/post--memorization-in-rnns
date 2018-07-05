
const events = require('events');
const d3 = require('../d3.js');

class RecurrentUnitsSelect extends events.EventEmitter {
  constructor({ container }) {
    super();
    this._container = d3.select(container);

    this._menu = this._container.insert('ul', ':first-child');
    this._menu.selectAll('li')
      .data([
        { text: 'Nested LSTM', id: 'nlstm' },
        { text: 'LSTM', id: 'lstm' },
        { text: 'GRU', id: 'gru' }
      ], (d) => d.id)
      .enter()
      .append('li')
      .text((d) => d.text)
      .on('click', (d) => this.emit('select', d.id))

    this._selectedId = 'nlstm';
  }

  setSelect(id) {
    this._selectedId = id;
  }

  draw() {
    const selection = this._menu
      .selectAll('li')
      .classed('selected', (d) => d.id === this._selectedId);
    this._container.attr('data-show', this._selectedId);
  }
}

module.exports = RecurrentUnitsSelect;
