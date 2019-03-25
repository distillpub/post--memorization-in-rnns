const d3 = require('../d3.js');

const margin = { top: 10, right: 10, bottom: 0, left: 10 }
const xAxisHeight = 40;
const yAxisWidth = 45;
const xLabelHeight = 15;
const yLabelWidth = 15;
const legendItemRectSize = 24;
const legendItemTextMargin = 4;

const lineSetup = {
  'GRU': "#F8766D",
  'LSTM': "#00BA38",
  'Nested LSTM': "#619CFF"
};

const topSuggestionSetup = {
  1: '8 0',
  2: '8 8',
  3: '2 2'
};

class AccuracyGraph {
  constructor({ container, height }) {
    this._container = d3.select(container);
    this._height = height;
    this._innerHeight = height - margin.top - margin.bottom - xAxisHeight - xLabelHeight;
    this._data = null;

    this._svg = this._container.append('svg')
      .attr('height', this._height)
      .classed('ar-line-graph', true)

    this._legend = this._container.append('div')
      .classed('legend', true)
      .style('margin-left', `${margin.left + yAxisWidth}px`)
      .style('margin-right', `${margin.right}px`);

    this._graph = this._svg.append('g')
      .attr('transform',
            'translate(' + (margin.left + yAxisWidth + yLabelWidth) + ',' + margin.top + ')');

    // Create background
    this._background = this._graph.append("rect")
      .attr("class", "background")
      .attr("height", this._innerHeight);

    // define scales
    this._xScale = d3.scaleLinear()
      .domain([-1, 21]);
    this._yScale = d3.scaleLinear()
      .domain([1, 0])
      .range([0, this._innerHeight]);

    // create grid
    this._xGrid = d3.axisBottom(this._xScale)
      .ticks(10)
      .tickSize(-this._height);
    this._xGridElement = this._graph.append("g")
        .attr("class", "grid")
        .attr("transform", "translate(0," + this._innerHeight + ")");

    // create grid
    this._yGrid = d3.axisLeft(this._yScale)
      .ticks(8);
    this._yGridElement = this._graph.append("g")
      .attr("class", "grid");

    // define axis
    this._xAxis = d3.axisBottom(this._xScale)
      .ticks(5);
    this._xAxisElement = this._graph.append('g')
      .attr("class", "axis")
      .attr('transform', 'translate(0,' + this._innerHeight + ')');

    this._yAxis = d3.axisLeft(this._yScale)
      .ticks(4)
      .tickFormat(d3.format(".0%"));
    this._yAxisElement = this._graph.append('g')
      .attr("class", "axis");
    this._yAxisTitle = this._graph.append('g')
      .attr("class", "axis-title");

    this._xLabel = this._svg.append('text')
      .attr('text-anchor', 'middle')
      .attr('y', height - margin.bottom - xLabelHeight)
      .text('characters from word');

    const yLabelTextPath = this._svg.append('path')
      .attr('d', `M${margin.left + yLabelWidth / 2},${margin.top + this._innerHeight} V${margin.top}`)
      .attr('id', 'ar-accuracy-graph-ylabel-vline');
    this._yLabel = this._svg.append('text')
      .append('textPath')
      .attr('startOffset', '50%')
      .attr('href', '#ar-accuracy-graph-ylabel-vline')
      .attr('text-anchor', 'middle')
      .text('accuracy');
    this._yLabel.node()
      .setAttributeNS('http://www.w3.org/1999/xlink', 'xlink:href', `#ar-training-graph-${name}-ylabel`);

    this._lines = this._graph.append("g")
      .classed('lines-container', true);
    this._lines.selectAll('g.lines-type')
      .data(['GRU', 'LSTM', 'Nested LSTM'], (d) => d)
      .enter().append('g')
      .classed('lines-type', true)
      .style('stroke', (d) => lineSetup[d])
      .each(function () {
        d3.select(this).selectAll('g.line')
          .data([1, 2, 3])
          .enter().append('path')
          .classed('line', true)
          .style('stroke-dasharray', (d) => topSuggestionSetup[d]);
      });

    this._lineDrawer = d3.line()
      .curve(d3.curveLinear)
      .x((d, i) => this._xScale(i))
      .y((d, i) => this._yScale(d));

    this._data = require('../data/word-length-results.json');

    this._drawLegend();
  }

  draw() {
    const width = this._container.node().clientWidth;
    const innerWidth = width - (margin.left + margin.right + yAxisWidth + yLabelWidth);

    this._svg
      .attr('width', width);

    // set background
    this._background
      .attr("width", innerWidth);

    // set the ranges
    this._xScale.range([0, innerWidth]);

    // update grid
    this._yGrid.tickSize(-innerWidth);
    const yTicksMajors = this._yScale.ticks(5);
    const xTicksMajors = this._xScale.ticks(5);
    this._yGridElement.call(this._yGrid);
    this._yGridElement
      .selectAll('.tick')
      .classed('minor', (d) => !yTicksMajors.includes(d))
    this._xGridElement.call(this._xGrid);
    this._xGridElement
      .selectAll('.tick')
      .classed('minor', (d) => !yTicksMajors.includes(d))

    // update axis
    this._xAxisElement.call(this._xAxis);
    this._yAxisElement
      .call(this._yAxis);

    // draw lines
    const lineDrawer = this._lineDrawer;
    const data = this._data;
    const linesSelect = this._lines.selectAll('g.lines-type')
      .each(function (name) {
        d3.select(this).selectAll('path.line')
          .data(data[name])
          .attr('d', (d) => lineDrawer(d));
      });

    this._xLabel.attr('x', innerWidth / 2 + margin.left + yLabelWidth + yAxisWidth);
  }

  _drawLegend() {
    const legendSelectPhase1 = this._legend.selectAll('svg.legend-item')
      .data([
        { type: 'GRU', top: 1 },
        { type: 'GRU', top: 2 },
        { type: 'GRU', top: 3 },
        { type: 'LSTM', top: 1 },
        { type: 'LSTM', top: 2 },
        { type: 'LSTM', top: 3 },
        { type: 'Nested LSTM', top: 1 },
        { type: 'Nested LSTM', top: 2 },
        { type: 'Nested LSTM', top: 3 }
      ]);

    const lengedGroupEnter = legendSelectPhase1
      .enter().append('svg')
      .attr('height', legendItemRectSize)
      .classed('legend-item', true);
    lengedGroupEnter.append('rect')
      .classed('legend-background', true)
      .attr('width', legendItemRectSize)
      .attr('height', legendItemRectSize);
    lengedGroupEnter.append('path')
      .classed('legend-color', true)
      .style('stroke', ({ type, top }) => lineSetup[type])
      .style('stroke-dasharray', ({ type, top }) => topSuggestionSetup[top])
      .attr('d', 'M0,12 L24,12');
    lengedGroupEnter.append('text')
      .classed('legend-text', true)
      .attr('x', legendItemRectSize + legendItemTextMargin)
      .attr('y', legendItemRectSize / 2)
      .text(({ type, top }) => `${type} - top ${top}`);
  }
}

module.exports = AccuracyGraph;
