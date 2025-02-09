// See https://aka.ms/new-console-template for more information

using iluvadev.ConsoleProgressBar;
using MNIST.IO;
using PerceptrOOn;
using PerceptrOOn.Exporters;
using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text.Json;


/*
 Loads the MNIST Dataset ref https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
 */
var data = FileReaderMNIST.LoadImagesAndLables(
         @"Assets/train-labels-idx1-ubyte.gz"
        , @"Assets/train-images-idx3-ubyte.gz"
    ).ToList();

// var labels = new byte[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }; // Uncomment to train full. Warning, 30 epochs, on a Ryzen7700 takes +/- 15m

var labels = new byte[] { 0, 1, 2, 3 }; // Demo training with a subset of 4 labels
data = data.Where(z => labels.Any(l => l == z.Label)).ToList();

var trainingDataSet = new List<TrainingData>();

foreach (var node in data) {
    trainingDataSet.Add(new TrainingData(node.Image.Flatten2DMatrix() , node.Label.ByteToFlatOutput(labels.Length)));
}

var trainingParameters = new TrainingParameters(
        TrainingDataSet: trainingDataSet.ToArray(),
        Epochs: 3,
        TrainingRate: 0.01
    );

var cpuInfo = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString();
var osInfo = System.Runtime.InteropServices.RuntimeInformation.OSDescription.ToString();
var hasVectorSupport = Vector.IsHardwareAccelerated;

TrainingData set;
var watch = new Stopwatch();
var random  = new Random();
using (var pb = new ProgressBar  { Maximum= trainingDataSet.Count*trainingParameters.Epochs, FixedInBottom=false })
{
    pb.Text.Description.Processing.AddNew().SetValue(pb => $"CPU Architecture        : {cpuInfo}");
    pb.Text.Description.Processing.AddNew().SetValue(pb => $"OS Info                 : {osInfo}");
    pb.Text.Description.Processing.AddNew().SetValue(pb => $"Vector Hardware Support : {hasVectorSupport}");
    var mnistNetwork = new NeuralNetwork(new NetworkDefinition(
       InputNodes: 784,
       HiddenLayerNodeDescription: [128],
       OutputNodes: labels.Length, // Size of the label set will dictate the length
       ActivationStrategy: new SigmoidActivationStrategy(seed: 1337),
       NotificationCallback: (current, total, description) => { pb.PerformStep(description); }
    ));
    
    watch.Start(); 
    await mnistNetwork.Train(trainingParameters);
    watch.Stop();
    // Pick random element
    

    foreach (var index in Enumerable.Range(0, 10))
    {
        set = trainingDataSet[new Random().Next(trainingDataSet.Count)];
        var result = mnistNetwork.Predict(set.input);
        Console.WriteLine($"-{index}-----------");
        Console.WriteLine($"Predicting  : [{String.Join(",", set.expectedOutput.Select(o => o.ToString("n")))}]");
        Console.WriteLine($"Output      : [{String.Join(",", result.Select(o => o.ToString("n")))}]");
    }


    File.WriteAllText("weights.json", mnistNetwork.ExportWith<JSONExporter, string>());
}

Console.ReadLine();

