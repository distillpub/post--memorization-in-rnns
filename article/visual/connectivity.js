
const d3 = require('../d3.js');
const events = require('events');

class Connectivity extends events.EventEmitter {
  constructor({container, assertDirectory, filename, name}) {
    super();

    this._name = name;

    this._data = d3.json(assertDirectory + 'data/' + filename);
    this._isInitialized = false;
    this._selectedCharIndex = 0;

    this._container = d3.select(container);

    this._textArea = this._container
      .append('div')
      .classed('textarea', true)

    this._predictArea = this._container
      .append('div')
      .classed('predictarea', true)

    this._predictArea
      .selectAll('.prediction')
      .data(["", "", ""])
      .enter()
      .append('div')
      .classed('prediction', true)

    this._nameArea = this._container
      .append('div')
      .classed('namearea', true)
    this._nameArea
      .append('span')
      .text(name)
  }

  _initializeTextDraw(data) {
    if (this._isInitialized) return;
    this._isInitialized = true;

    this._textArea
      .selectAll('span')
      .data(data)
      .enter()
      .append('span')
      .text((d) => d.char)
      .on('mouseover', (d, i) => this.emit('mouseover', i));
  }

  setSelected(index) {
    this._selectedCharIndex = index;
  }

  async draw() {
    const data = await this._data;
    this._initializeTextDraw(data);

    const charData = data[this._selectedCharIndex];

    const connectivity = charData.connectivity;
    const connectivityRescaled = [];
    const highestConnectivity = Math.max(...connectivity);
    for (const stength of connectivity) {
      // 2) Use only the middle 1/3 of the color scale
      connectivityRescaled.push(stength / highestConnectivity)
    }

    this._predictArea
      .selectAll('.prediction')
      .data(charData.predict.slice(0, 3))
      .text((d) => d)

    this._textArea
      .selectAll('span')
      .data(connectivityRescaled)
      .style('background-color', interpolateViridisSubset)
      .classed('selected', (d, i) => this._selectedCharIndex === i);
  }
}

function interpolateViridisSubset(ratio) {
  const low = 0.29;
  const high = 2/3;

  return d3.interpolateViridis(low + ratio * (high - low));
}

module.exports = Connectivity;
