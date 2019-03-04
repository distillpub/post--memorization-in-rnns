
const d3 = require('../d3.js')
const events = require('events');

const blockWidth = 12;
const blockSpace = 14;

function arange(length) {
  const output = [];
  for (let i = 0; i < length; i++) {
    output.push(i);
  }
  return output;
}

class VisualRNN extends events.EventEmitter {
  constructor({container, colorize}) {
    super();

    this._colorize = colorize;
    this._colorizeColumn = Infinity;
    this._colorScale = d3.interpolateRgb('#f3f3f3', '#9BC0DB');

    this._container = d3.select(container)
      .classed('ar-visual-rnn', true);

    this._descriptions = this._container
      .append('div')
      .classed('descriptions', true)

    this._descriptions
      .selectAll('figcaption')
      .data([
        'Softmax Layer',
        'Recurrent Layer',
        'Recurrent Layer',
        'Input Layer'
      ])
      .enter()
      .append('figcaption')
      .text((d) => d)

    this._svg = this._container
      .append('svg')

    const arrowDefs = this._svg
      .append('defs')

    arrowDefs
      .append('marker')
      .attr('id', this._arrowId('right-inactive'))
      .attr('markerWidth', 3)
      .attr('markerHeight', 3)
      .attr('refX', 2)
      .attr('refY', 1.5)
      .append('path')
        .attr('d', 'M0,0 L0,3 L3,1.5 Z')
        .attr('fill', '#BBBBBB')

    arrowDefs
      .append('marker')
      .attr('id', this._arrowId('right-active'))
      .attr('markerWidth', 3)
      .attr('markerHeight', 3)
      .attr('refX', 2)
      .attr('refY', 1.5)
      .append('path')
        .attr('d', 'M0,0 L0,3 L3,1.5 Z')
        .attr('fill', '#9BC0DB')

    arrowDefs
      .append('marker')
      .attr('id', this._arrowId('up-inactive'))
      .attr('markerWidth', 3)
      .attr('markerHeight', 3)
      .attr('refX', 1.5)
      .attr('refY', 2)
      .append('path')
        .attr('d', 'M0,3 L3,3 L1.5,0 Z')
        .attr('fill', '#BBBBBB')

    arrowDefs
      .append('marker')
      .attr('id', this._arrowId('up-active'))
      .attr('markerWidth', 3)
      .attr('markerHeight', 3)
      .attr('refX', 1.5)
      .attr('refY', 2)
      .append('path')
        .attr('d', 'M0,3 L3,3 L1.5,0 Z')
        .attr('fill', '#9BC0DB')

    this._softmax = this._svg
      .append('g')
      .attr('transform', 'translate(0, 0)')

    this._recurrentTop = this._svg
      .append('g')
      .attr('transform', 'translate(0, 60)')

    this._recurrentBottom = this._svg
      .append('g')
      .attr('transform', 'translate(0, 120)')

    this._input = this._svg
      .append('g')
      .attr('transform', 'translate(0, 180)')
  }

  _arrowId(direction) {
    return this._container.attr('id') + '-visual-rnn-arrow-' + direction
  }

  setColorizeColumn(column) {
    this._colorizeColumn = column;
  }

  draw() {
    const svgWidth = this._svg.node()
      .getBoundingClientRect().width;
    const blocks = 1 + Math.floor(
      (svgWidth - blockWidth) / (blockWidth + blockSpace)
    );

    const colors = this._colorize(
      Math.min(this._colorizeColumn, blocks - 1), blocks
    );

    this._softmax
      .call(this._drawBoxes({blockWidth, blockSpace, blocks, colors: colors[0]}))

    this._recurrentTop
      .call(this._drawBoxes({blockWidth, blockSpace, blocks, colors: colors[1]}))
      .call(this._drawUpArrow({blockWidth, blockSpace, blocks}))
      .call(this._drawRightArrow({blockWidth, blockSpace, blocks}))

    this._recurrentBottom
      .call(this._drawBoxes({blockWidth, blockSpace, blocks, colors: colors[2]}))
      .call(this._drawUpArrow({blockWidth, blockSpace, blocks}))
      .call(this._drawRightArrow({blockWidth, blockSpace, blocks}))

    this._input
      .call(this._drawBoxes({blockWidth, blockSpace, blocks, colors: colors[3]}))
      .call(this._drawUpArrow({blockWidth, blockSpace, blocks}))
  }

  _drawBoxes({ blockWidth, blockSpace, blocks, colors }) {
    const mouseenter = this.emit.bind(this, 'mouseenter');
    const colorScale = this._colorScale.bind(this);

    return function (element) {
      const selection = element
        .selectAll('rect.unit')
        .data(arange(blocks))
        .attr('fill', (i) => colorScale(colors[i]))

      selection
        .enter()
        .append('rect')
        .classed('unit', true)
        .attr('width', blockWidth)
        .attr('height', 30)
        .attr('rx', 3)
        .attr('x', (i) => i * (blockWidth + blockSpace))
        .attr('y', 15)
        .on('mouseenter', mouseenter)
        .attr('fill', (i) => colorScale(colors[i]))

      selection
        .exit()
        .remove()
    }
  }

  _drawUpArrow({ blockWidth, blockSpace, blocks }) {
    const arrowActiveUrl = `url(#${this._arrowId('up-active')})`;
    const arrowInactiveUrl = `url(#${this._arrowId('up-inactive')})`;
    const activeColumn = this._colorizeColumn;

    return function (element) {
      const selection = element
        .selectAll('path.arrow-up')
        .data(arange(blocks))
        .attr('marker-end', (i) => i <= activeColumn ? arrowActiveUrl : arrowInactiveUrl)
        .classed('active', (i) => i <= activeColumn)
        .classed('inactive', (i) => i > activeColumn);

      selection
        .enter()
        .append('path')
        .classed('arrow-up', true)
        .attr('d', (i) => `M ${i * (blockWidth + blockSpace) + blockWidth/2} 15 v -15`)
        .attr('marker-end', (i) => i <= activeColumn ? arrowActiveUrl : arrowInactiveUrl)
        .classed('active', (i) => i <= activeColumn)
        .classed('inactive', (i) => i > activeColumn);

      selection
        .exit()
        .remove();
    }
  }

  _drawRightArrow({ blockWidth, blockSpace, blocks }) {
    const arrowActiveUrl = `url(#${this._arrowId('right-active')})`;
    const arrowInactiveUrl = `url(#${this._arrowId('right-inactive')})`;
    const activeColumn = this._colorizeColumn;

    return function (element) {
      const selection = element
        .selectAll('path.arrow-right')
        .data(arange(blocks - 1))
        .attr('marker-end', (i) => i < activeColumn ? arrowActiveUrl : arrowInactiveUrl)
        .classed('active', (i) => i < activeColumn)
        .classed('inactive', (i) => i >= activeColumn);

      selection
        .enter()
        .append('path')
        .classed('arrow-right', true)
        .attr('d', (i) => `M ${i * (blockWidth + blockSpace) + blockWidth} 30 h 10`)
        .attr('marker-end', (i) => i < activeColumn ? arrowActiveUrl : arrowInactiveUrl)
        .classed('active', (i) => i < activeColumn)
        .classed('inactive', (i) => i >= activeColumn);

      selection
        .exit()
        .remove();
    }
  }
}

module.exports = VisualRNN
