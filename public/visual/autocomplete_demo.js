
const d3 = require('../d3.js');
const events = require('events');

function smoothFlowPathNarrow({ topX, bottomX, topWidth, bottomWidth }) {
  const path = d3.path();
  path.moveTo(topX, 0);
  path.quadraticCurveTo(topX, 15, bottomX, 30);
  path.lineTo(bottomX + bottomWidth, 30);
  path.quadraticCurveTo(topX + topWidth, 15, topX + topWidth, 0);
  path.closePath();
  return path;
}

function smoothFlowPathSameWidth({ x, borderWidth, midWidth }) {
  const path = d3.path();
  path.moveTo(x, 0);
  path.quadraticCurveTo(x + (borderWidth / 2) - (midWidth / 2), 15, x, 30);
  path.lineTo(x + borderWidth, 30);
  path.quadraticCurveTo(x + (borderWidth / 2) + (midWidth / 2), 15, x + borderWidth, 0);
  path.closePath();
  return path;
}

class AutoCompleteDemo extends events.EventEmitter {
  constructor(dataDirectory, model, dataset) {
    super();
    this._model = model;
    this._text = ' ';
    this._probabilityCacheGood = false;
    this._probabilityCache = null;
    this._probabilityPrecomputed = d3.json(
      dataDirectory + 'data/demo-precompute.json'
    ).then((val) => new Map(val));

    this._dataset = dataset;

    this._inputContainer = d3.select('#ar-demo-input-content');
    this._inputBackground = this._inputContainer
      .append('div')
      .classed('background', true);
    this._inputUnderline = this._inputContainer
      .append('div')
      .classed('underline', true);
    this._inputUnderline
      .append('span')
      .classed('underline-letter', true)
      .text(' ');
    this._inputField = this._inputContainer
      .append('input')
      .property('value', this._text)
      .on('input', () => this.emit('input',
        this._inputField.property('value').trimLeft()
      ));

    this._rnnContainer = d3.select('#ar-demo-rnn-content');
    this._rnnSvg = this._rnnContainer
      .append('svg')
      .attr('height', 75)
      .attr('width', '100%');
    this._rnnSvgCells = this._rnnSvg
      .append('g')
      .classed('cells', true)
      .attr('transform', 'translate(0, 15)');
    this._rnnFlowInSvg = this._rnnSvg
      .append('g')
      .classed('flow-in', true);
    this._rnnFlowOutSvg = this._rnnSvg
      .append('g')
      .attr('transform', 'translate(0, 45)')
      .classed('flow-out', true);
    this._rnnFlowOutPath = this._rnnFlowOutSvg
      .append('path')

    this._rnnFlowArrow = this._rnnSvg
      .append('path')
      .attr('transform', 'translate(0, 30)')
      .attr('marker-start', 'url(#ar-demo-marker-circle)')
      .attr('marker-mid', 'url(#ar-demo-marker-circle)')
      .attr('marker-end', 'url(#ar-demo-marker-arrow)')
      .classed('flow-arrow', true);
    const arrowDefs = this._rnnSvg
      .append('defs');
    arrowDefs
      .append('marker')
      .attr('id', 'ar-demo-marker-circle')
      .attr('markerWidth', 3)
      .attr('markerHeight', 3)
      .attr('refX', 1.5)
      .attr('refY', 1.5)
      .append('circle')
        .attr('r', 1.5)
        .attr('cx', 1.5)
        .attr('cy', 1.5)
        .attr('fill', 'rgb(70, 130, 180)')
    arrowDefs
      .append('marker')
      .attr('id', 'ar-demo-marker-arrow')
      .attr('markerWidth', 3)
      .attr('markerHeight', 3)
      .attr('refX', 2)
      .attr('refY', 1.5)
      .append('path')
        .attr('d', 'M0,0 L0,3 L3,1.5 Z')
        .attr('fill', 'rgb(70, 130, 180)')
    this._loader = this._rnnContainer.append('div')
      .attr('id', 'ar-demo-loader')
      .style('display', 'none')
    this._loader.append('span').text('custom input, loading ...')

    this._outputContainer = d3.select('#ar-demo-output-content');
    this._outputCanvas = this._outputContainer
      .append('canvas')
      .attr('height', 30);
    this._outputFlowOutSvg = this._outputContainer
      .append('svg')
      .attr('height', 30)
      .attr('width', '100%');
    this._outputFlowOutSvg
      .selectAll('path')
      .data([0, 0, 0])
      .enter().append('path')

    this._finalContainer = d3.select('#ar-demo-final-content');
    this._finalSuggestions = this._finalContainer
      .append('div')
      .classed('suggestions', true);
  }

  disableInput() {
    this._inputField.attr('disabled', 'disabled');
  }

  _getMaxLength() {
    // Get width of each letter
    const letterNode = this._inputUnderline
      .select('span.underline-letter')
      .node();
    const letterStyle = window.getComputedStyle(letterNode);
    const letterWidth = (
      parseFloat(letterStyle.width) + parseFloat(letterStyle.marginRight)
    );

    // Get input size
    const inputWidth = this._inputField
      .node()
      .getBoundingClientRect()
      .width;

    // Calculate the maximum number of letters
    return Math.floor(inputWidth / letterWidth);
  }

  setText(originalText) {
    const oldText = this._text;

    // remove invalid charecters
    const validText = originalText.toLowerCase().replace(/[^a-z ]/g, '');

    // For the input to start with a space
    this._text = (' ' + validText).slice(0, this._getMaxLength());
    if (this._text !== oldText) this._probabilityCacheGood = false;
  }

  _showLoader() {
    this._loader.style('display', 'flex');
  }

  _hideLoader() {
    this._loader.style('display', 'none');
  }

  async getProperbility(text) {
    const precomputed = await this._probabilityPrecomputed;
    if (this._probabilityCacheGood) return this._probabilityCache;

    if (precomputed.has(text)) {
      this._probabilityCacheGood = true;
      this._probabilityCache = precomputed.get(text);
      return this._probabilityCache;
    }

    let shouldLoad = !this._model.ready;
    if (shouldLoad) {
      this._showLoader();
    }

    // Compute properbiliies
    const inputDataset = await this._dataset.encodeSource([text]);
    for await (const output of this._model.predict(inputDataset)) {
      this._probabilityCache = output.probability;
      break;
    }
    this._probabilityCacheGood = true;

    if (shouldLoad) {
      this._hideLoader();
    }

    return this._probabilityCache;
  }

  async draw() {
    this._inputField.property('value', this._text);

    // Compute properbiliies
    const probability = await this.getProperbility(this._text);

    // The 3 most likely words
    const sorted = Array.from(probability)
      .map((d, i) => ({ probability: d, index: i }))
      .sort((a, b) => b.probability - a.probability)
    const wordMap = await this._dataset.getWordMap();
    const mostLikelyWords = sorted.slice(0, 3)
      .sort((a, b) => a.index - b.index)
      .map(({probability, index}) => ({
        probability: probability,
        index: index,
        word: wordMap[index]
      }))

    this._drawInputUnderline();

    const cellSizing = this._getCellSizing();
    this._drawRnn(cellSizing);
    this._drawOutput(cellSizing, probability, mostLikelyWords);
    this._drawFinal(probability, mostLikelyWords);
  }

  _getCellSizing() {
    const relativeParentRect = this._inputContainer
      .node()
      .getBoundingClientRect();

    return this._inputUnderline
      .selectAll('span.underline-letter')
      .nodes()
      .map(function (node) {
        const nodeRect = node.getBoundingClientRect();
        return {
          left: nodeRect.left - relativeParentRect.left,
          width: nodeRect.width
        }
      })
  }

  _drawInputUnderline() {
    const maxLength = this._getMaxLength();
    const data = (this._text + ' '.repeat(maxLength - this._text.length))
      .split('')

    const selection = this._inputUnderline
      .selectAll('span.underline-letter')
      .data(data)
      .classed('empty', (d, i) => i >= this._text.length)
      .text((d) => d)
    selection.enter()
      .append('span')
      .classed('underline-letter', true)
      .classed('empty', (d, i) => i >= this._text.length)
      .text((d) => d);
    selection.exit()
      .remove();
  }

  _drawRnn(cellSizing) {
    const lastCellSizing = cellSizing[this._text.length - 1];
    const parentWidth = this._outputContainer
      .node()
      .clientWidth;

    const cellSelection = this._rnnSvgCells
      .selectAll('rect')
      .data(cellSizing)
      .attr('x', (d) => d.left)
      .attr('width', d => d.width);
    cellSelection.enter()
      .append('rect')
      .attr('x', (d) => d.left)
      .attr('width', (d) => d.width)
      .attr('height', 30);
    cellSelection.exit()
      .remove();

    const flowInSelection = this._rnnFlowInSvg
      .selectAll('rect')
      .data(cellSizing.slice(0, this._text.length))
      .attr('x', (d) => d.left)
      .attr('width', d => d.width)
    flowInSelection.enter()
      .append('rect')
      .attr('x', (d) => d.left)
      .attr('width', (d) => d.width)
      .attr('height', 45);
    flowInSelection.exit()
      .remove();

    this._rnnFlowOutPath
      .attr('d', smoothFlowPathNarrow({
        topX: lastCellSizing.left,
        topWidth: lastCellSizing.width,
        bottomX: 0,
        bottomWidth: parentWidth
      }));

    const arrowPath = d3.path();
    arrowPath.moveTo(cellSizing[0].left + cellSizing[0].width / 2, 0);
    for (const cell of cellSizing.slice(1, this._text.length)) {
      arrowPath.lineTo(cell.left + cell.width / 2, 0);
    }

    const flowArrowSelection = this._rnnFlowArrow
      .attr('d', arrowPath)
  }

  _drawOutput(cellSizing, probability, mostLikelyWords) {
    const lastCellSizing = cellSizing[this._text.length - 1];
    const parentWidth = this._outputContainer
      .node()
      .clientWidth;
    const suggestionWidth = (parentWidth - 20) / 3;
    const wordsPerPixed = Math.floor(probability.length / (parentWidth / 4));

    // draw flow
    const flowOutSelection = this._outputFlowOutSvg
      .selectAll('path')
      .data(mostLikelyWords)
      .attr('d', (d, i) => smoothFlowPathNarrow({
        topX: Math.floor(d.index / wordsPerPixed) * 4,
        topWidth: 4,
        bottomX: (suggestionWidth + 10) * i,
        bottomWidth: suggestionWidth
      }));

    // draw canvas
    this._outputCanvas
      .attr('width', parentWidth);
    const ctx = this._outputCanvas.node().getContext('2d');

    // Interpolate data into a color array of width parentWidth
    const highestProperbility = Math.max(...probability);
    for (let i = 0; i < parentWidth; i++) {
      const highestInRange = Math.max(
        ...probability.slice(i * wordsPerPixed, (i + 1) * wordsPerPixed)
      );
      // 1) Rescale such the highest probability is 1
      // 2) Use only the middle 1/3 of the color scale
      const colorRatio = (highestInRange / highestProperbility)/3 + 1/3;
      const colorHex = d3.interpolateViridis(colorRatio);

      ctx.beginPath();
      ctx.fillStyle = colorHex;
      ctx.fillRect(i * 4, 0, 4, 30);
      ctx.stroke();
    }
  }

  _drawFinal(probability, mostLikelyWords) {
    const parentWidth = this._outputContainer
      .node()
      .clientWidth;
    const suggestionWidth = (parentWidth - 20) / 3;
    const wordsPerPixed = Math.floor(probability.length / (parentWidth / 4));

    // draw content
    const contentSelection = this._finalSuggestions
      .selectAll('div.suggestion')
      .data(mostLikelyWords)
      .each(function (d) {
        d3.select(this)
          .select('span.word')
          .text(d.word)
        d3.select(this)
          .select('span.probability')
          .text(`(${(d.probability * 100).toFixed(2)}%)`)
      })
    contentSelection.enter()
      .append('div')
      .classed('suggestion', true)
      .each(function (d) {
        d3.select(this)
          .append('span')
          .classed('word', true)
          .text(d.word)
        d3.select(this)
          .append('span')
          .classed('probability', true)
          .text(`(${(d.probability * 100).toFixed(2)}%)`)
      })
    contentSelection.exit()
      .remove()
  }
}

module.exports = AutoCompleteDemo;
