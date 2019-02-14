
const AutoComplete = require('./dataset/autocomplete.js');
const PureGRU = require('./models/pure_gru.js');
const AutoCompleteDemo = require('./visual/autocomplete_demo.js');
const VisualRNN = require('./visual/visual_rnn.js');
const HeroConnectivity = require('./visual/hero_connectivity.js');
const Connectivity = require('./visual/connectivity.js');
const TrainingGraph = require('./visual/training_graph.js');
const RecurrentUnitsSelect = require('./visual/recurrent_units_select.js');
const Walkthrough = require('./visual/walkthrough.js');
const AccuracyGraph = require('./visual/accuracy_graph.js');
const dl = require('deeplearn');

//const dataDirectory = '/uploads/blogpost-recurrent-units-in-rnn/';
//const dataDirectory = '/blog-recurrent-units/';
const dataDirectory = '/';

async function setupHeroDiagram() {
  const lstmConnectivity = new HeroConnectivity({
    container: document.querySelector('#ar-hero-diagram-lstm'),
    dataDirectory: dataDirectory,
    filename: 'connectivity_lstm.json',
    name: 'LSTM'
  });
  lstmConnectivity.on('mouseover', userInput);

  const nlstmConnectivity = new HeroConnectivity({
    container: document.querySelector('#ar-hero-diagram-nlstm'),
    dataDirectory: dataDirectory,
    filename: 'connectivity_nlstm.json',
    name: 'Nested LSTM'
  });
  nlstmConnectivity.on('mouseover', userInput);

  const gruConnectivity = new HeroConnectivity({
    container: document.querySelector('#ar-hero-diagram-gru'),
    dataDirectory: dataDirectory,
    filename: 'connectivity_gru.json',
    name: 'GRU'
  });
  gruConnectivity.on('mouseover', userInput);

  async function drawConnectivity(i) {
    lstmConnectivity.setSelected(i);
    nlstmConnectivity.setSelected(i);
    gruConnectivity.setSelected(i);
    await lstmConnectivity.draw();
    await nlstmConnectivity.draw();
    await gruConnectivity.draw();
  }

  let animationTimer;
  function animation(index, minIndex, maxIndex) {
    drawConnectivity(index).catch((err) => { throw err; });

    const nextLetterIndex = index >= maxIndex ? minIndex : index + 1;
    animationTimer = setTimeout(animation, 200,
                                nextLetterIndex, minIndex, maxIndex);
  }

  function userInput(i) {
    clearTimeout(animationTimer);
    drawConnectivity(i).catch((err) => { throw err; });
  }

  window.heroReset = function () {
    clearTimeout(animationTimer);
    animation(12, 12, 35);
  };

  window.heroSetIndex = function (i) {
    userInput(i);
  };

  window.heroReset();
}

async function setupRecurrentUnitsSelect() {
  const recurrentUnitsSelect = new RecurrentUnitsSelect({
    container: document.querySelector('#ar-recurrent-units')
  });

  recurrentUnitsSelect.on('select', function (id) {
    recurrentUnitsSelect.setSelect(id);
    recurrentUnitsSelect.draw();
  });

  recurrentUnitsSelect.setSelect('lstm');
  recurrentUnitsSelect.draw();
}

async function setupAutocompleteDemo(model) {
  const autocomplete = new AutoComplete(dataDirectory);
  const demo = new AutoCompleteDemo(dataDirectory, model, autocomplete);

  let animationTimer = null;
  let inputText = '';

  async function changeText(text) {
    inputText = text;
    demo.setText(text);
    await demo.draw();
  }

  async function userInput(text) {
    clearTimeout(animationTimer);
    await changeText(text);
  }

  (async function recursive(fullText, index) {
    await changeText(fullText.slice(0, index));
    if (index < fullText.length) {
      animationTimer = setTimeout(recursive, 500, fullText, index + 1);
    }
  })('parts of north af', 0);


  const isSafari = /^((?!chrome|android|crios|fxios).)*safari/i.test(navigator.userAgent);
  const inputNotice = document.querySelector('#ar-demo-input-notice');
  // Safari doesn't support a float texture in WebGL, thus custom input is disabled.
  // Just in case, the backend is also set to be CPU.
  if (isSafari) {
    dl.setBackend('cpu');
    demo.disableInput();
    inputNotice.innerHTML = 'Safari doesn\'t support custom text.';
  } else {
    demo.on('input', userInput);
    inputNotice.innerHTML = 'You can also type in your own text.';
  }

  window.addEventListener('resize', function () {
    demo.setText(inputText);
    demo.draw().catch((err) => { throw err });
  });

  window.arDemoShort = function () {
    userInput('parts of north ').catch((err) => { throw err });
  };

  window.arDemoReset = function () {
    userInput('parts of north af').catch((err) => { throw err });
  };
}

async function setupMemorizationProblemRNN() {
  const memorizationProblemRNN = new VisualRNN({
    container: document.querySelector('#ar-memorization-problem-rnn'),
    colorize: function (column, columnsTotal) {

      const colors = [];
      for (let row = 0; row < 4; row++) {
        const colorsRow = [];

        for (let col = 0; col < columnsTotal; col++) {
          if (row === 0) {
            if (col === column) colorsRow.push(1);
            else colorsRow.push(0);
          } else {
            if (col <= column) {
              colorsRow.push(Math.pow(0.8, row + (column - col)));
            } else colorsRow.push(0);
          }
        }
        colors.push(colorsRow);
      }

      return colors;
    }
  });
  memorizationProblemRNN.draw();
  memorizationProblemRNN.on('mouseenter', function (column) {
    memorizationProblemRNN.setColorizeColumn(column);
    memorizationProblemRNN.draw();
  });

  window.addEventListener('resize', function () {
    memorizationProblemRNN.draw();
  });
}

async function setupConnectivity() {
  const lstmConnectivity = new Connectivity({
    container: document.querySelector('#ar-connectivity-lstm'),
    dataDirectory: dataDirectory,
    filename: 'connectivity_lstm.json',
    name: 'LSTM'
  });
  lstmConnectivity.on('mouseover', drawConnectivity);

  const nlstmConnectivity = new Connectivity({
    container: document.querySelector('#ar-connectivity-nlstm'),
    dataDirectory: dataDirectory,
    filename: 'connectivity_nlstm.json',
    name: 'Nested LSTM'
  });
  nlstmConnectivity.on('mouseover', drawConnectivity);

  const gruConnectivity = new Connectivity({
    container: document.querySelector('#ar-connectivity-gru'),
    dataDirectory: dataDirectory,
    filename: 'connectivity_gru.json',
    name: 'GRU'
  });
  gruConnectivity.on('mouseover', drawConnectivity);

  async function drawConnectivity(i) {
    lstmConnectivity.setSelected(i);
    nlstmConnectivity.setSelected(i);
    gruConnectivity.setSelected(i);
    await lstmConnectivity.draw();
    await nlstmConnectivity.draw();
    await gruConnectivity.draw();
  }

  await drawConnectivity(106);

  window.connectivitySetIndex = function (index) {
    document.querySelector('#ar-connectivity-gru').scrollIntoView({
      behavior: "smooth",
      block: "nearest",
      inline: "start"
    });

    drawConnectivity(index === null ? 106 : index)
      .catch((err) => { throw err; });
  };
}

async function setupWalkthrough() {
  const walkthrough = new Walkthrough({
    container: document.querySelector('#ar-walkthrough')
  });

  walkthrough.select(1);
  walkthrough.draw();

  walkthrough.on('click', function (pageNumber, element) {
    walkthrough.select(pageNumber);
    walkthrough.draw();
    window.connectivitySetIndex(parseInt(element.dataset.connectivityIndex, 10));
  });
}

async function setupAccuracyGraph() {
  const accuracyGraph = new AccuracyGraph({
    container: document.querySelector('#ar-accuracy-graph'),
    height: 360
  });

  accuracyGraph.draw();

  window.addEventListener('resize', function () {
    accuracyGraph.draw();
  });
}

async function setupAutocompleteTrainingGraph() {
  const offset = 10 * 60 * 1000;
  const hour = 60 * 60 * 1000;

  const autocomplete = new TrainingGraph({
    container: document.querySelector('#ar-autocomplete-training'),
    dataDirectory: dataDirectory,
    name: 'autocomplete',
    filename: 'autocomplete-training.csv',
    ylim: [0.5, 10.5],
    xlimTime: [-offset, 7 * hour + offset],
    xlimEpochs: [-200, 15200],
    height: 360
  });

  await autocomplete.draw();

  window.addEventListener('resize', function () {
    autocomplete.draw().catch((err) => { throw err; });
  });

  window.setAutocompleteTrainingGraphXAxis = function (xAxisName) {
    autocomplete.setXAxis(xAxisName);
    autocomplete.draw().catch((err) => { throw err; });
  };
}

document.addEventListener('DOMContentLoaded', async function () {
  try {
    // Render LaTeX elements first, as their size is unknown.
  	var elements = document.querySelectorAll('math-latex');
  	Array.from(elements).forEach(function processElement(element) {
  		window.katex.render(element.getAttribute('latex'), element, {
  			displayMode: element.hasAttribute('display-mode')
  		});
  	});
  } catch (e) {
    console.error(e);
  }

  const model = new PureGRU(dataDirectory);
  await Promise.all([
    setupHeroDiagram(),
    setupMemorizationProblemRNN(),
    setupRecurrentUnitsSelect(),
    setupAutocompleteDemo(model),
    setupConnectivity(),
    setupWalkthrough(),
    setupAccuracyGraph(),
    setupAutocompleteTrainingGraph()
  ]);

  // Do this last, as it takes some time to load
  model.load().catch((err) => { throw err });
});
