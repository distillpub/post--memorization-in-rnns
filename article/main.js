
const AutoComplete = require('./dataset/autocomplete.js');
const PureGRU = require('./models/pure_gru.js');
const ArticleDemo = require('./visual/article_demo.js');
const VisualRNN = require('./visual/visual_rnn.js');
const Connectivity = require('./visual/connectivity.js');
const TrainingGraph = require('./visual/training_graph.js');
const dl = require('deeplearn');

//const assertDirectory = '/uploads/blogpost-recurrent-units-in-rnn/';
//const assertDirectory = '/blog-recurrent-units/';
const assertDirectory = '/';

async function setupArticleDemo(model) {
  const autocomplete = new AutoComplete(assertDirectory);
  const demo = new ArticleDemo(assertDirectory, model, autocomplete);

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

  const isSafari = /^((?!chrome|android|crios|fxios).)*safari/i.test(navigator.userAgent);
  // Safari doesn't support a float texture, thus the backend is set to CPU.
  // This is very slow, so skip the animation.
  if (isSafari) dl.setBackend('cpu');

  (async function recursive(fullText, index) {
    await changeText(fullText.slice(0, index));
    if (index < fullText.length) {
      animationTimer = setTimeout(recursive, 500, fullText, index + 1);
    }
  })('parts of north af', 0);

  demo.on('input', userInput);

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

async function setupRecurentUnitRNN() {
  const recurentUnitRNN = new VisualRNN({
    container: document.querySelector('#ar-recurrent-unit-rnn'),
    caption: '<strong>Recurrent Neural Network:</strong> as used in ' +
             'autocomplete example. Shows how the network in theory knows ' +
             'about every part of the sequence that came before.',
    colorize: function (column, columnsTotal) {

      const colors = [];
      for (let row = 0; row < 4; row++) {
        const colorsRow = [];

        for (let col = 0; col < columnsTotal; col++) {
          if (row === 0) {
            if (col === column) colorsRow.push(1);
            else colorsRow.push(0);
          } else {
            if (col < column) colorsRow.push(0.5);
            else if (col === column) colorsRow.push(1);
            else colorsRow.push(0);
          }
        }
        colors.push(colorsRow);
      }

      return colors;
    }
  });
  recurentUnitRNN.draw();
  recurentUnitRNN.on('mouseenter', function (column) {
    recurentUnitRNN.setColorizeColumn(column);
    recurentUnitRNN.draw();
  });

  window.addEventListener('resize', async function () {
    recurentUnitRNN.draw();
  });
}

async function setupMemorizationProblemRNN() {
  const memorizationProblemRNN = new VisualRNN({
    container: document.querySelector('#ar-memorization-problem-rnn'),
    caption: '<strong>Vanishing Gradient:</strong> where the contribution from the earlier steps becomes insignificant.',
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

  window.addEventListener('resize', async function () {
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
      .catch((err) => {throw err;});
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
    xlim: [-offset, 2.5 * hour + offset],
    height: 360
  });

  let generate = null;
  if (!!document.querySelector('#ar-generate-training')) {
    generate = new TrainingGraph({
      container: document.querySelector('#ar-generate-training'),
      assertDirectory: assertDirectory,
      name: 'autocomplete',
      filename: 'generate-training.csv',
      ylim: [0.5, 5.5],
      xlim: [-offset, 4 * hour + offset],
      height: 360
    });
  }

  await autocomplete.draw();
  if (generate) await generate.draw();

  window.addEventListener('resize', async function () {
    await autocomplete.draw();
    if (generate) await generate.draw();
  });
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
    setupArticleDemo(model),
    setupRecurentUnitRNN(),
    setupMemorizationProblemRNN(),
    setupConnectivity(),
    setupTrainingGraph()
  ]);

  // Do this last, as it takes some time to load
  model.load().catch((err) => { throw err });
});
