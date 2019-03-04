
const d3 = require('../d3.js');

const margin = {top: 0, right: 10, bottom: 10, left: 60}
const facetWidth = 30;
const legendHeight = 40;
const xAxisHeight = 20;
const xLabelHeight = 20;

class SubGraph {
  constructor({ container, topName, name, lineSetup, outerHeight, drawXAxis, xlimTime, xlimEpochs, ylim }) {
    this.height = outerHeight - margin.top - margin.bottom;
    this.xlimTime = xlimTime;
    this.xlimEpochs = xlimEpochs;

    this.lineSetup = lineSetup;
    this.container = container;
    this.graph = this.container.append('g')
      .attr('transform',
            'translate(' + margin.left + ',' + margin.top + ')');

    this.facet = this.container.append('g')
      .classed('facet', true);
    this.facet.append('rect')
      .classed('facet-background', true)
      .attr('width', facetWidth)
      .attr('height', this.height);
    const facetTextPath = this.facet.append('path')
      .attr('d', `M10,0 V${this.height}`)
      .attr('id', `ar-training-graph-${topName}-${name}-facet-text`);
    const facetText = this.facet.append('text')
      .append('textPath')
      .attr('startOffset', '50%')
      .attr('href', `#ar-training-graph-${topName}-${name}-facet-text`)
      .attr('text-anchor', 'middle')
      .text(name);
    facetText.node()
      .setAttributeNS('http://www.w3.org/1999/xlink', 'xlink:href', `#ar-training-graph-${topName}-${name}-facet-text`);

    // Create background
    this.background = this.graph.append("rect")
      .attr("class", "background")
      .attr("height", this.height);

    // define scales
    this.yScale = d3.scaleLinear()
      .domain(ylim)
      .range([this.height, 0]);

    // create grid
    this.xGridElement = this.graph.append("g")
        .attr("class", "grid")
        .attr("transform", "translate(0," + this.height + ")");

    // create grid
    this.yGrid = d3.axisLeft(this.yScale)
      .ticks(8);
    this.yGridElement = this.graph.append("g")
        .attr("class", "grid");

    // define axis
    this.xAxisElement = this.graph.append('g')
      .attr("class", "axis")
      .classed('hide-axis', !drawXAxis)
      .attr('transform', 'translate(0,' + this.height + ')');

    this.yAxis = d3.axisLeft(this.yScale)
      .ticks(4);
    this.yAxisElement = this.graph.append('g')
      .attr("class", "axis");

    // Define drawer functions and line elements
    this.lineDrawers = [];
    this.lineElements = [];
    const self = this;
    for (let i = 0; i < lineSetup.length; i++) {
      const lineDrawer = d3.line()
          .x((d) => self.xScale(d[this.dataColumn]))
          .y((d) => this.yScale(d.lossSmooth));

      this.lineDrawers.push(lineDrawer);

      const lineElement = this.graph.append('path')
          .attr('class', 'line')
          .attr('stroke', lineSetup[i].color);

      this.lineElements.push(lineElement);
    }
  }

  setData (data) {
    // Update domain of scales
    for (let i = 0; i < this.lineSetup.length; i++) {
      const lineData = data[this.lineSetup[i].name];
      this.lineElements[i].data([lineData]);
    }
  }

  setXAxis(xAxisName) {
    this.xAxisName = xAxisName;

    if (xAxisName === 'time') {
      this.dataColumn = 'sec';
      this.xScale = d3.scaleTime()
        .domain(this.xlimTime);
      this.xGrid = d3.axisBottom(this.xScale)
        .ticks(d3.timeMinute.every(30))
        .tickSize(-this.height);
      this.xAxis = d3.axisBottom(this.xScale)
        .ticks(d3.timeMinute.every(60))
        .tickFormat(function (d) {
          const hour = Math.floor(d.getTime() / (60 * 60 * 1000));
          return `0${hour}:00:00`;
        });
    } else if (xAxisName === 'mini-batches') {
      this.dataColumn = 'step';
      this.xScale = d3.scaleLinear()
        .domain(this.xlimEpochs);
      this.xGrid = d3.axisBottom(this.xScale)
        .ticks(17)
        .tickSize(-this.height);
      this.xAxis = d3.axisBottom(this.xScale)
        .ticks(9);
    }
  }

  xTicksMajors() {
    if (this.xAxisName === 'time') {
      const xTicksMajors = this.xScale.ticks(d3.timeMinute.every(60))
        .map((d) => d.getTime());

      return function (d) {
        return !xTicksMajors.includes(d.getTime());
      }
    } else if (this.xAxisName === 'mini-batches') {
      const xTicksMajors = this.xScale.ticks(4)

      return function (d) {
        return !xTicksMajors.includes(d);
      }
    }
  }

  draw({ outerWidth, outerHeight }) {
    const graphWidth = outerWidth - facetWidth - margin.left - margin.right;

    // set background
    this.background
      .attr("width", graphWidth);

    // set facet
    this.facet
      .attr('transform', `translate(${margin.left + graphWidth}, ${margin.top})`);

    // set the ranges
    this.xScale.range([0, graphWidth]);

    // update grid
    this.yGrid.tickSize(-graphWidth);
    const yTicksMajors = this.yScale.ticks(4);
    this.yGridElement.call(this.yGrid);
    this.yGridElement
      .selectAll('.tick')
      .classed('minor', function (d) {
        return !yTicksMajors.includes(d);
      })
    this.xGridElement.call(this.xGrid);
    this.xGridElement
      .selectAll('.tick')
      .classed('minor', this.xTicksMajors());

    // update axis
    this.xAxisElement.call(this.xAxis);
    this.yAxisElement.call(this.yAxis);

    // update lines
    for (let i = 0; i < this.lineSetup.length; i++) {
      this.lineElements[i].attr('d', this.lineDrawers[i]);
    }
  }
}

class TrainingGraph {
  constructor({ container, dataDirectory, name, filename, height, xlimTime, xlimEpochs, ylim }) {
    this._data = d3.csv(
      dataDirectory + 'data/' + filename,
      (d) => ({
        dataset: d.dataset,
        loss: parseFloat(d.loss),
        lossSmooth: parseFloat(d['loss smooth']),
        model: d.model,
        sec: new Date(parseFloat(d.sec) * 1000),
        time: new Date(parseFloat(d['wall time']) * 1000),
        step: parseInt(d.step)
      })
    );
    this._dataLoaded = false;

    const lineSetup = [
      {name: 'GRU', color: "#F8766D"},
      {name: 'LSTM', color: "#00BA38"},
      {name: 'Nested LSTM', color: "#619CFF"}
    ];

    const graphHeight = (height - legendHeight - xLabelHeight - xAxisHeight) / 2;

    this._container = d3.select(container)
      .attr('height', height)
      .attr('xmlns:xlink', 'http://www.w3.org/1999/xlink');

    this._train = new SubGraph({
      container: this._container.append('g'),
      topName: name,
      name: 'training',
      lineSetup: lineSetup,
      outerHeight: graphHeight,
      drawXAxis: false,
      xlimTime: xlimTime,
      xlimEpochs: xlimEpochs,
      ylim: ylim
    });
    this._valid = new SubGraph({
      container: this._container.append('g')
        .attr('transform', `translate(0, ${graphHeight})`),
      topName: name,
      name: 'validation',
      lineSetup: lineSetup,
      outerHeight: graphHeight,
      drawXAxis: true,
      xlimTime: xlimTime,
      xlimEpochs: xlimEpochs,
      ylim: ylim
    });

    this._labels = this._container
      .append('g')
      .attr('transform', `translate(0, ${margin.top})`);

    const combinedOuterHeight = height - legendHeight;
    const combinedInnerHeight = combinedOuterHeight - margin.bottom - margin.top - xAxisHeight;

    const yLabelTextPath = this._labels.append('path')
      .attr('d', `M${margin.left / 2},${combinedInnerHeight} V0`)
      .attr('id', `ar-training-graph-${name}-ylabel`);
    this._yLabel = this._labels.append('text')
      .append('textPath')
      .attr('startOffset', '50%')
      .attr('href', `#ar-training-graph-${name}-ylabel`)
      .attr('text-anchor', 'middle')
      .text('cross entropy loss');
    this._yLabel.node()
      .setAttributeNS('http://www.w3.org/1999/xlink', 'xlink:href', `#ar-training-graph-${name}-ylabel`);
    this._xLabel = this._labels.append('text')
      .attr('text-anchor', 'middle')
      .attr('y', combinedOuterHeight - 12);

    this._legend = this._container
      .append('g')
      .classed('legned', true)
      .attr('transform', `translate(${margin.left}, ${combinedOuterHeight})`);
    this._legendOfsset = this._legend.append('g')

    let currentOffset = 0;
    for (const {name, color} of lineSetup) {
      this._legendOfsset.append('rect')
        .attr('width', 25)
        .attr('height', 25)
        .attr('x', currentOffset);
      this._legendOfsset.append('line')
        .attr('x1', currentOffset + 2)
        .attr('x2', currentOffset + 25 - 2)
        .attr('y1', 25/2)
        .attr('y2', 25/2)
        .attr('stroke', color);

      const textNode = this._legendOfsset.append('text')
        .attr('x', currentOffset + 30)
        .attr('y', 19)
        .text(name);
      const textWidth = textNode.node().getComputedTextLength();
      currentOffset += 30 + textWidth + 20;
    }
    this._legendWidth = currentOffset - 20;

    // Default config
    this._xAxisName = '';
    this.setXAxis('time');
  }

  setXAxis (xAxisName) {
    this._xAxisName = xAxisName;
    this._train.setXAxis(xAxisName);
    this._valid.setXAxis(xAxisName);
  }

  getGraphWidth () {
    const outerSize = this._container.node().getBoundingClientRect()
    return outerSize.width;
  }

  async draw() {
    if (!this._dataLoaded) {
      const data = await this._data;

      const grouped = d3.nest()
        .key((d) => d.dataset)
        .key((d) => d.model)
        .object(data);

      this._train.setData(grouped.train);
      this._valid.setData(grouped.valid);
      this._dataLoaded = true;
    }

    const outerWidth = this.getGraphWidth();
    const innerWidth = outerWidth - margin.left - margin.right - facetWidth;

    this._train.draw({
      outerWidth: outerWidth
    });
    this._valid.draw({
      outerWidth: outerWidth
    });
    this._xLabel
      .attr('x', innerWidth / 2 + margin.left)
      .text(this._xAxisName);
    this._legendOfsset
      .attr('transform', `translate(${(innerWidth - this._legendWidth) / 2}, 0)`)
  }
}

module.exports = TrainingGraph
