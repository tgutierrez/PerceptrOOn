using MNIST.IO;
using PerceptrOOn;
using PerceptrOOn.Exporters;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using Spectre.Console;
using Spectre.Console.Rendering;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.Drawing.Imaging;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Channels;
using SixLabors.ImageSharp.Processing;

internal class Program
{
    private static Grid? SystemInfo;
    private static CircularBuffer.CircularBuffer<string> LogBuffer = new CircularBuffer.CircularBuffer<string>(20);
    private static string LastMessage = String.Empty;
    private static async Task Main(string[] args)
    {


        NeuralNetwork? mnistNetwork = null;
        var trainingDataSet = new List<TrainingData>();


        //var labels = new byte[] { 0, 1, 2, 3 }; // Demo training with a subset of 4 labels
        var labels = new byte[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }; // Uncomment to train full. Warning, 30 epochs, on a Ryzen7700 takes +/- 1-2m

        var cpuInfo = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString();
        var osInfo = System.Runtime.InteropServices.RuntimeInformation.OSDescription.ToString();
        var hasVectorSupport = Vector.IsHardwareAccelerated;

        SystemInfo = new Grid();
        SystemInfo.AddColumn()
            .AddColumn();

        SystemInfo.AddRow("CPU Architecture", cpuInfo);
        SystemInfo.AddRow("OS Info", osInfo);
        SystemInfo.AddRow("Vector Hardware Support", hasVectorSupport.ToString());

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
            .UseRenderHook((renderable, tasks) => RenderHook(tasks, renderable))
            .StartAsync(async ctx =>
        {
            var mainTask = ctx.AddTask("Total Progress", new ProgressTaskSettings { MaxValue = 4 });
            mainTask.Increment(1);

            List<TestCase?> data = new();
            LogBuffer.PushFront("Started");
            /*
             Loads the MNIST Dataset ref https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
             */
            data = FileReaderMNIST.LoadImagesAndLables(
                 @"Assets/train-labels-idx1-ubyte.gz"
                , @"Assets/train-images-idx3-ubyte.gz"
            ).ToList();

            data = data.Where(z => labels.Any(l => l == z!.Label)).ToList();

            var dataAsSpan = data.ToArray().AsSpan();

            // Randomize set
            new Random().Shuffle(dataAsSpan);

            data = dataAsSpan.ToArray().ToList();

            var loadingSet = ctx.AddTask("Converting Images into Training Set:", new ProgressTaskSettings { MaxValue = data.Count() });
            foreach (var node in data)
            {
                trainingDataSet.Add(new TrainingDataPreservingOriginal<TestCase>(node!,node!.Image.Flatten2DMatrix(), node.Label.ByteToFlatOutput(labels.Length)));
                loadingSet.Increment(1);
            }

            var random = new Random(42); // Fixed seed for reproducibility
            var validationSplit = 0.2; // 20% for validation
            var dataCount = trainingDataSet.Count;
            var validationCount = (int)(dataCount * validationSplit);

            // Shuffle and split data
            var shuffledData = trainingDataSet.OrderBy(x => random.Next()).ToArray();
            var trainingData = shuffledData.Take(dataCount - validationCount).ToArray();
            var validationData = shuffledData.Skip(dataCount - validationCount).ToArray();
            //var trainingData = trainingDataSet.ToArray();
            //var validationData = trainingDataSet.ToArray();
            LogBuffer.PushFront("Training Data Ready");
            mainTask.Increment(1);

            var trainingParameters = new TrainingParameters
            {
                TrainingDataSet = trainingData,
                ValidationSet = validationData,
                Epochs = 50,                    // More epochs since we have early stopping
                InitialLearningRate = 0.01,     // Higher initial learning rate
                BatchSize = 32,                // Add mini-batch training
                LossFunction = new CategoricalCrossEntropy(),  // Appropriate for digit classification
                LearningRateScheduler = new LearningRateScheduler(
                    initialRate: 0.01,
                    decayRate: 0.95,           // Slower decay
                    decaySteps: 5            // Decay every 5 epochs
                ),
                EarlyStoppingPatience = 5     // Stop if no improvement for 5 epochs
            };



            var trainingTask = ctx.AddTask("Training Network", new ProgressTaskSettings { MaxValue = trainingDataSet.Count * trainingParameters.Epochs });

           mnistNetwork = new NeuralNetwork(new NetworkDefinition(
           InputNodes: 784,
           HiddenLayerNodeDescription: [32, 16],
           OutputNodes: labels.Length, // Size of the label set will dictate the length
           Strategies: new Strategies(new ReLuActivationStrategy(   // ReLu
               seed: 1337,
               biasInitializationExpression: (r) => 0),
                new DefaultComputeStrategy()
           ),
               //Strategies: new Strategies(new SigmoidActivationStrategy(seed: 1337), new DefaultComputeStrategy()),
               NotificationCallback: (current, total, description) => { 
                   trainingTask.Increment(1);
                   LogBuffer.PushBack(description);
               }
            ));
            LogBuffer.PushFront("Starting Training");
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

            table.AddColumn("Positions:", c =>
            {
                c.Alignment = Justify.Left;
            });

            foreach (var item in Enumerable.Range(0, labels.Length))
            {
                table.AddColumn($"{item}", c =>
                {
                    c.Alignment = Justify.Left;
                });
            }

            table.AddColumn("Confidence");

            TrainingDataPreservingOriginal<TestCase> set = (trainingDataSet[new Random().Next(trainingDataSet.Count)] as TrainingDataPreservingOriginal<TestCase>)!;
            var result = await mnistNetwork!.Predict(set.input);
            using var memstream = new MemoryStream();
            ByteArrayToImage(set.Original.Image)
                .SaveAsPng(memstream, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
            memstream.Position = 0;
            var image = new CanvasImage(memstream);


            CreateRows(table, "Predicting", set.expectedOutput, image);
            CreateRows(table, "Output", result, new Markup(Evaluate(set.expectedOutput, result)));


            AnsiConsole.Write(table);
            AnsiConsole.WriteLine();
        }

        AnsiConsole.WriteLine("Done!. Press Any Key to Continue...");
        Console.ReadLine();


        static void CreateRows(Table table, string label, double[] values, IRenderable rend)
        {

            var row = new List<IRenderable>() { new Markup(label) };
            row.AddRange(values.Select(v => new Markup(v.ToString("n"))));
            row.Add(rend);
            table.AddRow(row);
        }


        static string Evaluate(double[] expected, double[] predicted)
        {

            var expectedIndex = Array.FindIndex(expected, match => match == 1);

            var max = predicted.Max();
            var indexOfMax = Array.FindIndex(predicted, match => match == max);

            if (indexOfMax != expectedIndex) return $"[red]Incorrect[/]";

            return predicted[expectedIndex] switch
            {
                >= 0d and < 0.3d => $"[red]Low[/]",
                > 0.3d and < 0.7d => $"[yellow]Weak[/]",
                >= 0.7d => $"[green]High[/]",
                _ => throw new NotImplementedException(),
            };
        }

        

        static IRenderable RenderHook(IReadOnlyList<ProgressTask> tasks, IRenderable renderable)
        {
            var log = new Rows(LogBuffer.Select(x => new Text(x.ToString(), new Style(Spectre.Console.Color.Green, Spectre.Console.Color.Black, Decoration.Dim))));

            var header = new Panel(SystemInfo!) { Expand = true };
            var content = new Panel(renderable) { Expand = true};
            var footer = new Panel(log) { Expand = true };
            return new Rows(header, content, footer);
        }
    }

    public static Image<L8> ByteArrayToImage(byte[,] data)
    {
        int width = data.GetLength(0);
        int height = data.GetLength(1);

        // Create an image with the grayscale pixel format (L8)
        var image = new Image<L8>(width, height);

        // Set each pixel
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                byte value = data[x, y];
                image[x, y] = new L8(value);
            }
        }
        image.Mutate(ctx => ctx.RotateFlip(RotateMode.Rotate90, FlipMode.Horizontal));
        return image;
    }


}