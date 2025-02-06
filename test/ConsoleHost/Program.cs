// See https://aka.ms/new-console-template for more information

using iluvadev.ConsoleProgressBar;
using MNIST.IO;
using System.Diagnostics;
using System.Runtime.CompilerServices;


/*
 Loads the MNIST Dataset ref https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
 */
var data = FileReaderMNIST.LoadImagesAndLables(
         @"Assets\train-labels-idx1-ubyte.gz"
        , @"Assets\train-images-idx3-ubyte.gz"
    ).ToList();


var trainingDataSet = new List<TrainingData>();

foreach (var node in data[0..1000]) {
    trainingDataSet.Add(new TrainingData(Flatten(node.Image), ByteToFlatOutput(node.Label)));
}

var trainingParameters = new TrainingParameters(
        TrainingDataSet: trainingDataSet.ToArray(),
        Epochs: 60,
        TrainingRate: 0.128
    );

TrainingData set;
double[] output;

using (var pb = new ProgressBar  { Maximum= trainingDataSet.Count*trainingParameters.Epochs, FixedInBottom=false })
{
    var mnistNetwork = new NeuralNetwork(new NetworkDefinition(
       InputNodes: 784,
       HiddenLayerNodeDescription: [784],
       OutputNodes: 10,
       ActivationStrategy: new SigmoidActivationStrategy(seed: 1337),
       NotificationCallback: (current, total, description) => { pb.PerformStep(description); }
    ));

    var watch = Stopwatch.StartNew();
    mnistNetwork.Train(trainingParameters);
    watch.Stop();
    pb.WriteLine($"Training Completed on: {watch.ElapsedMilliseconds}ms");

    // Pick random element
    var rnd = new Random();

    set = trainingDataSet[rnd.Next(trainingDataSet.Count)];

    output = mnistNetwork.Predict(set.input);
}

Console.WriteLine($"Predicting: [{String.Join(",", set.expectedOutput.Select(o => o.ToString("n")))}]");
Console.WriteLine($"Output    : [{String.Join(",", output.Select(o => o.ToString("n")))}]");
Console.WriteLine("Press any key to exit");
Console.ReadLine();


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