// See https://aka.ms/new-console-template for more information

using MNIST.IO;
using PerceptrOOn;
using PerceptrOOn.Exporters;

var data = FileReaderMNIST.LoadImagesAndLables(
         @"Assets/train-labels-idx1-ubyte.gz"
        , @"Assets/train-images-idx3-ubyte.gz"
    ).ToList();

var mnistJSON = File.ReadAllText(@"Assets/mnist.json");


var importer = new JSONImporter();
var layers = importer.Import(mnistJSON);

var xorNetwork = new NeuralNetwork(layers, new SigmoidActivationStrategy());

var random = new Random();

foreach (var index in Enumerable.Range(1, 10))
{
    var testCase = data[random.Next(data.Count)];

    var testImage = testCase.Image.Flatten2DMatrix();
    var testLabel = testCase.Label.ByteToFlatOutput(10);
    var result = xorNetwork.Predict(testImage);
    Console.WriteLine($"-{index}-----------");
    Console.WriteLine($"Predicting  : [{String.Join(",", testLabel.Select(o => o.ToString("n")))}]");
    Console.WriteLine($"Output      : [{String.Join(",", result.Select(o => o.ToString("n")))}]");
}

Console.WriteLine("Press any key to exit...");
Console.ReadLine();

