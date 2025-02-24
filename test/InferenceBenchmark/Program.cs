// See https://aka.ms/new-console-template for more information

using PerceptrOOn.Exporters;
using PerceptrOOn;
using System.Text.Json;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using System.ComponentModel;

BenchmarkRunner.Run<InferenceBenchmark>();

public class InferenceBenchmark {

    private readonly string mnistJSON;
    private readonly double[][] testcases;
    private readonly ILayer[] layers;
    public InferenceBenchmark()
    {
        mnistJSON = File.ReadAllText(@"weights_relu.json");
        testcases = JsonSerializer.Deserialize<double[][]>(File.ReadAllText("sample.json")) ?? throw new InvalidOperationException("cannot deserialize test values");
        var importer = new JSONImporter();
        layers = new JSONImporter().Import(mnistJSON);
    }

    public async Task<double[]> DefaultComputeStrategy() {
        
        var xorNetwork = new NeuralNetwork(layers, new Strategies(new ReLuActivationStrategy(), new DefaultComputeStrategy(true)));
        double[]? lastresult = null;
        foreach (var testcase in testcases) {
            lastresult = await xorNetwork.Predict(testcase);
        }

        return lastresult!; // Returns last result as recommended by BanchmarkDotNet
    }

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

    [Benchmark]
    public async Task<double[]> BenchamarkSimpleTraining() {

        var xorNetwork = new NeuralNetwork(new NetworkDefinition(
       InputNodes: 3,
       HiddenLayerNodeDescription: [4],
       OutputNodes: 2,
       Strategies: new Strategies(new ReLuActivationStrategy(0, x => 0.5, x => 0.5), new DefaultComputeStrategy()),
       UseSoftMaxOutput: true
    ));

        var trainingParameters = new TrainingParameters(
                TrainingDataSet: [
                    new TrainingData([0d, 0d, 1d], [1d, 0d]),
                        new TrainingData([1d, 1d, 1d], [0d, 1d]),
                        new TrainingData([1d, 0d, 1d], [0d, 1d]),
                        new TrainingData([0d, 1d, 0d], [1d, 0d]),
                ],
                Epochs: 10000,
                TrainingRate: 0.01
            );

        await xorNetwork.Train(trainingParameters);

        return [0d, 0d, 0d];
    }


}