// See https://aka.ms/new-console-template for more information

using MNIST.IO;
using PerceptrOOn;
using PerceptrOOn.Exporters;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp;
using Spectre.Console;
using Spectre.Console.Rendering;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Numerics;
using System.Reflection.Emit;
using System.Text.Json;

var cpuInfo = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString();
var osInfo = System.Runtime.InteropServices.RuntimeInformation.OSDescription.ToString();
var hasVectorSupport = Vector.IsHardwareAccelerated;

var table = new Table() { Expand = true };
table.AddColumn("[bold]Testing Inference with Saved weights and MNIST dataset[/]");



var systemInfo = new Grid();
systemInfo.AddColumn()
    .AddColumn();

systemInfo.AddRow("CPU Architecture", cpuInfo);
systemInfo.AddRow("OS Info", osInfo);
systemInfo.AddRow("Vector Hardware Support", hasVectorSupport.ToString());

table.AddRow(new Panel(systemInfo){ Expand = true });

List<TestCase> data = null;


AnsiConsole.Status()
    .Start("Loading Training Images", ctx =>
    {
        data = FileReaderMNIST.LoadImagesAndLables(
         @"Assets/train-labels-idx1-ubyte.gz"
        , @"Assets/train-images-idx3-ubyte.gz"
    ).ToList();

    });

NeuralNetwork mnistNetwork = null;

AnsiConsole.Status()
    .Start("Loading Checkpoint and Creating Network", ctx =>
    {
        var mnistJSON = File.ReadAllText(@"Assets/mnist.json");
        var importer = new JSONImporter();
        var layers = importer.Import(mnistJSON);
        mnistNetwork = new NeuralNetwork(layers, new Strategies(new SigmoidActivationStrategy(), new DefaultComputeStrategy()));
    });

await AnsiConsole.Live(table)
    .StartAsync(async ctx =>
    {
        
        var random = new Random();

        foreach (var index in Enumerable.Range(1, 10))
        {
            var testCase = data![random.Next(data.Count)];

            var testImage = testCase.Image.Flatten2DMatrix();
            var testLabel = testCase.Label.ByteToFlatOutput(10);
            var result = await mnistNetwork!.Predict(testImage);


            var innerTable = new Table();
            innerTable.Title = new TableTitle($"[bold]Test Run #{index}[/]");


            innerTable.AddColumn("Positions:", c =>
            {
                c.Alignment = Justify.Left;
            });

            foreach (var item in Enumerable.Range(0, 10))
            {
                innerTable.AddColumn($"{item}", c =>
                {
                    c.Alignment = Justify.Left;
                });
            }

            innerTable.AddColumn("Confidence");

            using var memstream = new MemoryStream();
            ByteArrayToImage(testCase.Image)
                .SaveAsPng(memstream, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
            memstream.Position = 0;
            var image = new CanvasImage(memstream);

            CreateRows(innerTable, "Predicting", testLabel, image);
            CreateRows(innerTable, "Output", result, new Markup(Evaluate(testLabel, result)));

            table.AddRow(innerTable);
            ctx.Refresh();
        }
    });
Console.WriteLine("Press any key to exit...");
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

static Image<L8> ByteArrayToImage(byte[,] data)
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