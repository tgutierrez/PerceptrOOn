// See https://aka.ms/new-console-template for more information

using iluvadev.ConsoleProgressBar;
using MNIST.IO;
using PerceptrOOn.Exporters;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text.Json;


/*
 Loads the MNIST Dataset ref https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
 */
var data = FileReaderMNIST.LoadImagesAndLables(
         @"Assets\train-labels-idx1-ubyte.gz"
        , @"Assets\train-images-idx3-ubyte.gz"
    ).ToList();

var labels = new byte[] { 0, 1, 2, 3 };
data = data.Where(z => labels.Any(l => l == z.Label)).ToList();

var trainingDataSet = new List<TrainingData>();

foreach (var node in data) {
    trainingDataSet.Add(new TrainingData(Flatten(node.Image), ByteToFlatOutput(node.Label, labels.Length)));
}

var trainingParameters = new TrainingParameters(
        TrainingDataSet: trainingDataSet.ToArray(),
        Epochs: 3,
        TrainingRate: 0.05
    );

TrainingData set;
double[] output;

using (var pb = new ProgressBar  { Maximum= trainingDataSet.Count*trainingParameters.Epochs, FixedInBottom=false })
{
    var mnistNetwork = new NeuralNetwork(new NetworkDefinition(
       InputNodes: 784,
       HiddenLayerNodeDescription: [64],
       OutputNodes: labels.Length, // Size of the label set will dictate the length
       ActivationStrategy: new SigmoidActivationStrategy(seed: 1337),
       NotificationCallback: (current, total, description) => { pb.PerformStep(description); }
    ));

    mnistNetwork.Train(trainingParameters);

    // Pick random element
    var rnd = new Random();

    set = trainingDataSet[rnd.Next(trainingDataSet.Count)];

    output = mnistNetwork.Predict(set.input);

    File.WriteAllText("weights.json", mnistNetwork.ExportWith<JSONExporter, string>());
}

Console.WriteLine($"Predicting: [{String.Join(",", set.expectedOutput.Select(o => o.ToString("n")))}]");
Console.WriteLine($"Output    : [{String.Join(",", output.Select(o => o.ToString("n")))}]");
Console.WriteLine("Press any key to exit");
Console.ReadLine();




double[] ByteToFlatOutput(byte label, int outputSize)
{
    double[] output = new double[outputSize];
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

double NormalizeByte(byte input) => input / 255d;