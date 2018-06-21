
const AutoComplete = require('./dataset/autocomplete.js');
const PureGRU = require('./models/pure_gru.js');
const AutoCompleteDemo = require('./visual/autocomplete_demo.js');
const VisualRNN = require('./visual/visual_rnn.js');
const HeroConnectivity = require('./visual/hero_connectivity.js');
const Connectivity = require('./visual/connectivity.js');
const TrainingGraph = require('./visual/training_graph.js');
const dl = require('deeplearn');

//const assertDirectory = '/uploads/blogpost-recurrent-units-in-rnn/';
//const assertDirectory = '/blog-recurrent-units/';
const assertDirectory = '/';

async function setupHeroDiagram() {
  const lstmConnectivity = new HeroConnectivity({
    container: document.querySelector('#ar-hero-diagram-lstm'),
    assertDirectory: assertDirectory,
    filename: 'connectivity_lstm.json',
    name: 'LSTM'
  });
  lstmConnectivity.on('mouseover', userInput);

  const nlstmConnectivity = new HeroConnectivity({
    container: document.querySelector('#ar-hero-diagram-nlstm'),
    assertDirectory: assertDirectory,
    filename: 'connectivity_nlstm.json',
    name: 'Nested LSTM'
  });
  nlstmConnectivity.on('mouseover', userInput);

  const gruConnectivity = new HeroConnectivity({
    container: document.querySelector('#ar-hero-diagram-gru'),
    assertDirectory: assertDirectory,
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

async function setupAutocompleteDemo(model) {
  const autocomplete = new AutoComplete(assertDirectory);
  const demo = new AutoCompleteDemo(assertDirectory, model, autocomplete);

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
    assertDirectory: assertDirectory,
    filename: 'connectivity_lstm.json',
    name: 'LSTM'
  });
  lstmConnectivity.on('mouseover', drawConnectivity);

  const nlstmConnectivity = new Connectivity({
    container: document.querySelector('#ar-connectivity-nlstm'),
    assertDirectory: assertDirectory,
    filename: 'connectivity_nlstm.json',
    name: 'Nested LSTM'
  });
  nlstmConnectivity.on('mouseover', drawConnectivity);

  const gruConnectivity = new Connectivity({
    container: document.querySelector('#ar-connectivity-gru'),
    assertDirectory: assertDirectory,
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

  await drawConnectivity(105);

  window.connectivitySetIndex = function (index) {
    drawConnectivity(index === null ? 105 : index)
      .catch((err) => { throw err; });
  };
}

async function setupTrainingGraph() {
  const offset = 10 * 60 * 1000;
  const hour = 60 * 60 * 1000;

  const autocomplete = new TrainingGraph({
    container: document.querySelector('#ar-autocomplete-training'),
    assertDirectory: assertDirectory,
    name: 'autocomplete',
    filename: 'autocomplete-training.csv',
    ylim: [0.5, 10.5],
    xlimTime: [-offset, 2.5 * hour + offset],
    xlimEpochs: [-200, 7300],
    height: 360
  });

  await autocomplete.draw();

  window.addEventListener('resize', function () {
    autocomplete.draw().catch((err) => { throw err; });
  });

  window.setTrainingGraphXAxis = function (xAxisName) {
    autocomplete.setXAxis(xAxisName);
    autocomplete.draw().catch((err) => { throw err; });
  };
}

document.addEventListener('DOMContentLoaded', async function () {
  // Render LaTeX elements first, as their size is unknown.
	var elements = document.querySelectorAll('math-latex');
	Array.from(elements).forEach(function processElement(element) {
		window.katex.render(element.getAttribute('latex'), element, {
			displayMode: element.hasAttribute('display-mode')
		});
	});

  const model = new PureGRU(assertDirectory);
  await Promise.all([
    setupHeroDiagram(),
    setupMemorizationProblemRNN(),
    setupAutocompleteDemo(model),
    setupConnectivity(),
    setupTrainingGraph()
  ]);

  // Do this last, as it takes some time to load
  model.load().catch((err) => { throw err });
});
