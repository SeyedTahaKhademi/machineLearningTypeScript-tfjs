const tf = require("@tensorflow/tfjs");

const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

async function train(xs: number[], ys: number[], epochs: number) {
  const tensorX = tf.tensor(xs);
  const tensorY = tf.tensor(ys);
  await model.fit(tensorX, tensorY, { epochs });
}

function predict(input: number): number {
  const tensorInput = tf.tensor1d([input]);
  const result = model.predict(tensorInput);
  return result.dataSync()[0]; // چون فقط یه عدده، از [0] استفاده می‌کنیم
}

async function fap(xs: number[], ys: number[], input: number) {
  await train(xs, ys, 1000);
  const output = predict(input);
  console.log("Predicted output:", output.toFixed(2));
}

const xs = [1, 2, 3, 4];
const ys = [1, 3, 5, 7];

fap(xs, ys, 5);
