
function split(tensor, numSplits, axis) {
  if (tensor.shape[axis] % numSplits !== 0) {
    throw new Error(`number of splits ${numSplits} does not ` +
                    `divide shape ${tensor.shape} on axis ${axis}`);
  }

  const sliceSize = [...tensor.shape]; // copy
  sliceSize[axis] = tensor.shape[axis] / numSplits;

  const sliceStart = [];
  for (let d = 0; d < tensor.shape.length; d++) sliceStart[d] = 0;

  const outputs = [];
  for (let splitIndex = 0; splitIndex < numSplits; splitIndex++) {
    outputs.push(tensor.slice(sliceStart, sliceSize));
    sliceStart[axis] += sliceSize[axis]; // Move along slice axis
  }

  return outputs;
}

module.exports = split;
