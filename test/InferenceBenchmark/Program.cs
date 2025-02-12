// See https://aka.ms/new-console-template for more information

using PerceptrOOn.Exporters;
using PerceptrOOn;
using System.Text.Json;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

BenchmarkRunner.Run<InferenceBenchmark>();

public class InferenceBenchmark {

    private readonly string mnistJSON;
    private readonly double[][] testcases;
    private readonly ILayer[] layers;
    public InferenceBenchmark()
    {
        mnistJSON = File.ReadAllText(@"weights_sigmoid.json");
        testcases = JsonSerializer.Deserialize<double[][]>(File.ReadAllText("sample.json")) ?? throw new InvalidOperationException("cannot deserialize test values");
        var importer = new JSONImporter();
        layers = new JSONImporter().Import(mnistJSON);
    }

    [Benchmark]
    public async Task<double[]> DefaultComputeStrategy() {
        
        var xorNetwork = new NeuralNetwork(layers, new Strategies(new SigmoidActivationStrategy(), new DefaultComputeStrategy()));
        double[]? lastresult = null;
        foreach (var testcase in testcases) {
            lastresult = await xorNetwork.Predict(testcase);
        }

        return lastresult!; // Returns last result as recommended by BanchmarkDotNet
    }

    [Benchmark]
    public async Task<double[]> DefaultComputeStrategyStandardSum()
    {

        var xorNetwork = new NeuralNetwork(layers, new Strategies(new SigmoidActivationStrategy(), new DefaultComputeStrategy(false)));
        double[]? lastresult = null;
        foreach (var testcase in testcases)
        {
            lastresult = await xorNetwork.Predict(testcase);
        }

        return lastresult!; // Returns last result as recommended by BanchmarkDotNet
    }


}