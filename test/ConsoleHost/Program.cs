using MNIST.IO;
using PerceptrOOn;
using PerceptrOOn.Exporters;
using Spectre.Console;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;

TrainingData set;
NeuralNetwork? mnistNetwork = null;
var trainingDataSet = new List<TrainingData>();

var labels = new byte[] { 0, 1, 2, 3 }; // Demo training with a subset of 4 labels

await AnsiConsole.Progress()
    .AutoClear(false)
    .Columns([
        new TaskDescriptionColumn(),
        new ProgressBarColumn(),
        new PercentageColumn(),
        new RemainingTimeColumn(),
        new ElapsedTimeColumn(),
        new SpinnerColumn()

    ])
    .StartAsync(async ctx =>
{

    var cpuInfo = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString();
    var osInfo = System.Runtime.InteropServices.RuntimeInformation.OSDescription.ToString();
    var hasVectorSupport = Vector.IsHardwareAccelerated;
    AnsiConsole.WriteLine($"CPU Architecture        : {cpuInfo}");
    AnsiConsole.WriteLine($"OS Info                 : {osInfo}");
    AnsiConsole.WriteLine($"Vector Hardware Support : {hasVectorSupport}");

    var mainTask = ctx.AddTask("Total Progress", new ProgressTaskSettings { MaxValue = 4 });
    mainTask.Increment(1);

    List<TestCase?> data = new();
    
        /*
         Loads the MNIST Dataset ref https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
         */
        data = FileReaderMNIST.LoadImagesAndLables(
         @"Assets/train-labels-idx1-ubyte.gz"
        , @"Assets/train-images-idx3-ubyte.gz"
    ).ToList();
    // var labels = new byte[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }; // Uncomment to train full. Warning, 30 epochs, on a Ryzen7700 takes +/- 15m

    
    data = data.Where(z => labels.Any(l => l == z.Label)).ToList();

    var dataAsSpan = data.ToArray().AsSpan();

    // Randomize set
    new Random().Shuffle(dataAsSpan);

    data  = dataAsSpan.ToArray().ToList();

    var loadingSet = ctx.AddTask("Converting Images into Training Set:", new ProgressTaskSettings { MaxValue = data.Count() });
    foreach (var node in data) {
        trainingDataSet.Add(new TrainingData(node.Image.Flatten2DMatrix() , node.Label.ByteToFlatOutput(labels.Length)));
        loadingSet.Increment(1);
    }

    mainTask.Increment(1);

    var trainingParameters = new TrainingParameters(
            TrainingDataSet: trainingDataSet.ToArray(),
            Epochs: 3,
            TrainingRate: 0.01
        );

    

    var trainingTask = ctx.AddTask("Training Network", new ProgressTaskSettings { MaxValue = trainingDataSet.Count * trainingParameters.Epochs });

        mnistNetwork = new NeuralNetwork(new NetworkDefinition(
           InputNodes: 784,
           HiddenLayerNodeDescription: [128],
           OutputNodes: labels.Length, // Size of the label set will dictate the length
           //ActivationStrategy: new ReLuActivationStrategy(   // ReLu still WIP
           //    seed: 1337,
           //    randomWeightExpression: (r) => 0,
           //    biasInitializationExpression: (r) => 0.01),
           ActivationStrategy: new SigmoidActivationStrategy(seed: 1337),
           NotificationCallback: (current, total, description) => { trainingTask.Increment(1); }
        ));

    await mnistNetwork.Train(trainingParameters);
    
    mainTask.Increment(1);

    AnsiConsole.WriteLine("Writing Weights.");

    File.WriteAllText("weights.json", mnistNetwork.ExportWith<JSONExporter, string>());
    mainTask.Increment(1);

    
});

AnsiConsole.MarkupLineInterpolated($"[bold green]Training Completed. Testing Inference in 10 random runs[/]");
var random = new Random();
foreach (var index in Enumerable.Range(0, 10))
{
    var table = new Table();
    table.Title = new TableTitle($"Test Run #{index}");

    table.AddColumn("Positions:", c => {
        c.Alignment = Justify.Left;
    });

    foreach (var item in Enumerable.Range(0, labels.Length))
    {
        table.AddColumn($"{item}", c => {
            c.Alignment = Justify.Left;
        });
    }

    table.AddColumn("Confidence");

    set = trainingDataSet[new Random().Next(trainingDataSet.Count)];
    var result = mnistNetwork!.Predict(set.input);

    CreateRows(table, "Predicting", set.expectedOutput, "-");
    CreateRows(table, "Output", result, Evaluate(set.expectedOutput, result));


    AnsiConsole.Write(table);
    AnsiConsole.WriteLine();
}

AnsiConsole.WriteLine("Done!. Press Any Key to Continue...");
Console.ReadLine();


static void CreateRows(Table table, string label, double[] values, string eval) {

    var row = new List<string>() { label };
    row.AddRange(values.Select(v => v.ToString("n")));
    row.Add(eval);
    table.AddRow(row.Select(v => new Markup(v))) ;
}


static string Evaluate(double[] expected, double[] predicted) {

    var expectedIndex = Array.FindIndex(expected, match => match == 1);

    var max = predicted.Max();
    var indexOfMax = Array.FindIndex(predicted, match => match == max);

    if (indexOfMax != expectedIndex) return $"[red]Incorrect[/]";

    return predicted[expectedIndex]  switch
    {   
        >= 0d and < 0.3d     => $"[red]Low[/]",
        > 0.3d and < 0.7d      => $"[yellow]Weak[/]",
        >=0.7d                   => $"[green]High[/]",
        _ => throw new NotImplementedException(),
    };
}