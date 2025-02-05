// See https://aka.ms/new-console-template for more information

using MNIST.IO;


/*
 Loads the MNIST Dataset ref https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
 */
var data = FileReaderMNIST.LoadImagesAndLables(
         @"Assets\train-labels-idx1-ubyte.gz"
        , @"Assets\train-images-idx3-ubyte.gz"
    ).ToList();


var mnistNetwork = new NeuralNetwork(new NetworkDefinition(
       InputNodes: 784,
       HiddenLayerNodeDescription: [784, 392, 196, 98],
       OutputNodes: 10,
       ActivationStrategy: new SigmoidActivationStrategy(5)
    ));

var trainingDataSet = new List<TrainingData>();

foreach (var node in data[0..1000]) {
    trainingDataSet.Add(new TrainingData(Flatten(node.Image), ByteToFlatOutput(node.Label)));
}



var trainingParameters = new TrainingParameters(
        TrainingDataSet: trainingDataSet.ToArray(),
        Epochs: 1,
        TrainingRate: 1
    );


mnistNetwork.Train(trainingParameters);

var output = mnistNetwork.Predict(trainingDataSet[10].input);

Console.WriteLine($"Output : [{String.Join(",", output.Select(o => o.ToString("n")))}]");

double[] ByteToFlatOutput(byte label)
{
    double[] output = new double[10];
    output[label] = 1;
    return output;
}

double[] Flatten(byte[,] image)
{
    var flat = new List<double>();
    foreach (var item in image) { 
        flat.Add(NormalizeByte(item)); 
    }

    return flat.ToArray();
}

double NormalizeByte(byte input) => input / 255;

Console.ReadLine();