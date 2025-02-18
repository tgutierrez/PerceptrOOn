#region Base Definitions

using PerceptrOOn;
using System.Collections.Concurrent;

public interface ILayer {
    int Id { get; }
    int Size { get; }
    INode[] Content {  get; }

    Task BackPropagate(IBackPropagationInput[] backPropagationInput, double rate);
}

public interface INode
{
    int Id { get; }
    public double Value { get; }

    public Task ComputeValue();

    public MutableArray<Weight> OutputWeights { get; }

    public MutableArray<Weight> InputWeights { get; }

}

public class Weight
{
    private readonly INode linkedFrom;
    private readonly INode linksTo;

    public Weight(INode linkedFrom, INode linksTo, double initialValue)
    {
        this.linkedFrom = linkedFrom;
        this.linksTo = linksTo;
        Value = initialValue;
        // add to previous node
        linkedFrom.OutputWeights.Add(this);
    }

    public INode LinkedFrom  => linkedFrom;
    public INode LinksTo => linksTo;

    public double Value { get; private set; }

    public double Compute()
    {
        return Value * LinkedFrom.Value;
    }

    public void SetWeightTo(double value)
    {
        Value = value;
    }
};

public interface IActivationStrategy
{
    string Name { get; }
    double ComputeActivation(double input);
    double ComputeActivationDerivative (double input);
    double GetRandomWeight(int inputs);
    double GetInitialBias();
}

public interface IComputeStrategies 
{
    string Name { get; }
    Task ComputeLayer<T>(IActivationStrategy strategy, T layer) where T : ILayer;
    Task<double> ComputeNode<T>(IActivationStrategy strategy, T node) where T : INode;
}

public interface INetworkExporter<T>
{
    public T Export(ILayer[] layer, Strategies strategies);
}

public interface INetworkImporter<T> 
{
    public ILayer[] Import(T networkData, Func<Strategies>? strategyProvider);

}

public delegate void Notify(int current, int total, string description);

public class Strategies(IActivationStrategy activationStrategy, IComputeStrategies computeStrategies) { 
    public IActivationStrategy ActivationStrategy => activationStrategy;
    public IComputeStrategies ComputeStrategies => computeStrategies;
};

public record NetworkDefinition(int InputNodes, int[] HiddenLayerNodeDescription, int OutputNodes, Strategies Strategies, Notify? NotificationCallback = null, bool UseSoftMaxOutput = false);
public record TrainingData(double[] input, double[] expectedOutput);
public record TrainingDataPreservingOriginal<T>(T Original, double[] input, double[] expectedOutput) : TrainingData(input, expectedOutput);
public record TrainingParameters(TrainingData[] TrainingDataSet, int Epochs, double TrainingRate);

#endregion

public interface IBackPropagationInput {

    double Value { get; }

    INode Node { get; }

    void AdjustWeights();

}

#region Activation strategies

public class SigmoidActivationStrategy : IActivationStrategy
{
    private Random rand; 
    private Func<Random, double> BiasInit;
    public SigmoidActivationStrategy(int? seed = default, Func<Random, double>? biasInitialization = null) {
        rand = seed.HasValue ? new Random(seed.Value) : new Random();
        BiasInit = biasInitialization ?? new Func<Random, double>((r) => rand.NextDouble() * 2 - 1);
    }
    public string Name => "Sigmoid";

    public double ComputeActivation(double x) => 1.0 / (1.0 + Math.Exp(-x));

    public double ComputeActivationDerivative (double x) => x * (1 - x);

    public double GetRandomWeight(int inputs) => rand.NextDouble() *2 -1;

    public double GetInitialBias() => BiasInit(rand);
}

public class ReLuActivationStrategy : IActivationStrategy
{
    public ReLuActivationStrategy(int? seed = default, Func<Random, double>? biasInitializationExpression = null) {
        rand = seed.HasValue? new Random(seed.Value) : new Random();
        this.getBiasInitExpression  = biasInitializationExpression ?? new Func<Random, double>((r) => 0);
    }

    private readonly Func<Random, double> getBiasInitExpression;

    private readonly Random rand;

    public string Name => "ReLu";

    public double ComputeActivation(double input) => Math.Max(0, input);

    public double ComputeActivationDerivative(double input) => input switch {
        > 0 => 1,
        _ => 0,
    };

    public double GetRandomWeight(int inputs) {

        // Generate two uniform random values in (0, 1)
        double u1 = 1.0 - rand.NextDouble();
        double u2 = 1.0 - rand.NextDouble();
        // Box-Muller transform to get a standard normal value (mean = 0, std = 1)
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        // Scale by sqrt(2 / fanIn) for He initialization
        double stdDev = Math.Sqrt(2.0 / inputs);
        return randStdNormal * stdDev;

    }
    public double GetInitialBias() => getBiasInitExpression(rand);
}

public static class ActivationStrategyFactory {

    public static IActivationStrategy Create(string name, int? seed = default)
        => name switch
        {
            "Sigmoid" => new SigmoidActivationStrategy(seed),
            "ReLu" => new ReLuActivationStrategy(), // ReLu with random weights -1 ... 1
            _ => throw new NotImplementedException()
        };
}



public static class ComputeStrategyFactory {
    public static IComputeStrategies Create(string name, int? seed = default)
    => name switch
    {
        _ => new DefaultComputeStrategy(),
    };
}

#endregion

/// <summary>
/// A neural network consisting of an input layer, N number of hidden layers and an output.
/// It encapsulates the logic that will train / infer / initialize the network
/// </summary>
/// <param name="input">Input data for the netwrokk</param>
/// <param name="numberOfLayers"></param>
/// <param name="actionFunctions"></param>
public class NeuralNetwork
{
    private readonly ILayer[] layers;
    private readonly Strategies strategies;
    
    private InputLayer InputLayer => (layers[0] as InputLayer)!;
    private Layer OutputLayer => (layers[^1] as Layer)!;
    private Layer[] HiddenLayer => layers[1..^1].Cast<Layer>().ToArray()!;

    public NetworkDefinition Definition { get; internal set; }

    /// <summary>
    /// Builds the network using definitions
    /// </summary>
    /// <param name="definition"></param>
    public NeuralNetwork(NetworkDefinition definition) { 
        this.strategies = definition.Strategies;
        layers = BuildLayers(definition);
        this.Definition = definition;
    }

    /// <summary>
    /// Builds the network using imported layers
    /// </summary>
    /// <param name="layers"></param>
    /// <param name="activationStrategy"></param>
    /// <exception cref="NotImplementedException"></exception>
    public NeuralNetwork(ILayer[] layers, Strategies strategies) { 
        this.layers = layers;
        this.strategies = strategies;
        var hiddenDefinitions = HiddenLayer.ToList().Select(h => h.Neurons.Length).ToArray();
        bool hasSoftMaxOutput = (layers[^1] is SoftMaxOutputLayer);
        this.Definition = new NetworkDefinition(InputLayer.Size, hiddenDefinitions, this.OutputLayer.Size, strategies,null, hasSoftMaxOutput); // infer definition based on the layer structure
        // TODO: Validate input
    }

    private ILayer[] BuildLayers(NetworkDefinition definition)
    {
        var layerList = new List<ILayer>();

        ILayer previousLayer = new InputLayer(definition.InputNodes);
        layerList.Add(previousLayer);

        int layerIndex = 1;
        foreach (var description in definition.HiddenLayerNodeDescription) {
            var newHiddenLayer = new HiddenLayer(previousLayer, description, this.strategies);
            layerList.Add(newHiddenLayer);
            previousLayer = newHiddenLayer;
            layerIndex++;
        }

        var outputLayer = new OutputLayer(previousLayer, definition.OutputNodes, strategies);
        layerList.Add(outputLayer);

        if (definition.UseSoftMaxOutput)
        {
            layerList.Add(new SoftMaxOutputLayer(outputLayer, definition.OutputNodes, strategies));
        }

        return layerList.ToArray();
    }

    public async Task<double[]> Predict(double[] input)
    {
        // Collect inputs
        InputLayer.CollectInput(input);

        // Compute Hidden Layers
        foreach (var hiddenLayer in this.HiddenLayer)
        {
            await hiddenLayer.Compute();
        }

        // Compute Output
        await OutputLayer.Compute();

        // Collect Result and  return
        return OutputLayer.CollectOutput();
    }

    /// <summary>
    /// Train the network using backpropagation
    /// </summary>
    /// <param name="trainingParameters">Parameters</param>
    public async Task Train(TrainingParameters trainingParameters)
    {
        foreach (var epoch in Enumerable.Range(0, trainingParameters.Epochs))
        {
            Definition.NotificationCallback?.Invoke(epoch, trainingParameters.TrainingDataSet.Length, $"Training Epoch {epoch + 1}/{trainingParameters.Epochs}");
            await TrainingEpoch(trainingParameters, epoch, trainingParameters.TrainingRate);
        }      
    }

    private async Task TrainingEpoch(TrainingParameters trainingParameters, int epoch, double rate) {
        foreach (var trainingSet in trainingParameters.TrainingDataSet)
        {
            await TrainingCycle(trainingSet, epoch, rate);
        }
    }

    private async Task TrainingCycle(TrainingData trainingSet, int epoch, double rate)
    {
        // 1 - Compute
        await this.Predict(trainingSet.input);
        // 2 - Collect errors (predicted vs actual)
        var outputErrors = this.OutputLayer
                        .Neurons
                        .Index()
                        .Select((e) => new OutputErrorAdjustment(e.Item, trainingSet.expectedOutput[e.Index] - e.Item.Value)).ToArray();

        // 3 - Feed the errors to the output layer and start the process
        await this.OutputLayer.BackPropagate(outputErrors, rate);
    }

    public T ExportWith<E, T>() where E : INetworkExporter<T>, new()
        => new E().Export(layers, strategies);


    public T Export<T>(INetworkExporter<T> exporter) => exporter.Export(layers, strategies);
}

/// <summary>
/// an input layer that holds constant values used as parameters for a network
/// </summary>
public class InputLayer : ILayer
{
    private readonly InputNode[] inputNodes;
    private readonly int numberOfParameters;
    public INode[] Content => inputNodes;
    public int Size => numberOfParameters;

    public int Id => 0; // Input Layer Id will always be 0.

    /// <param name="input"></param>
    public InputLayer(int numberOfParameters)
    {
        this.numberOfParameters = numberOfParameters;
        inputNodes = new InputNode[numberOfParameters];
        foreach (var index in Enumerable.Range(0, numberOfParameters))
        {
            inputNodes[index] = new InputNode(index);
        }
    }

    public InputLayer(InputNode[] inputNodes) 
    { 
        this.numberOfParameters += inputNodes.Length;
        this.inputNodes = inputNodes;
    }

    public InputNode[] InputNodes { 
        get => inputNodes;
    }

    public void CollectInput(double[] input) {

        if (input.Length != this.Size) throw new ArgumentException("Invalid number of inputs");
        // fill input layer
        foreach (var index in Enumerable.Range(0, input.Length))
        {
            this.InputNodes[index].SetValue(input[index]);
        }
    }

    public Task BackPropagate(IBackPropagationInput[] backPropagationInput, double gradient) => Task.CompletedTask;
}

/// <summary>
/// A neuron layer, could be hidden or output 
/// </summary>
/// <param name="inputLayer">previous layer</param>
/// <param name="size">layer size</param>
/// <param name="activationStrategy">set of action definitions</param>
public abstract class Layer : ILayer 
{
    private readonly Neuron[] neurons;
    internal Strategies strategies;
    public Layer(ILayer inputLayer, int size, Strategies strategies) {
        this.Id = inputLayer.Id+1;
        this.InputLayer = inputLayer;
        this.Size = size;
        this.strategies = strategies;
        neurons = new Neuron[size];
        foreach (int index in Enumerable.Range(0, size))
        {
            neurons[index] = new Neuron(strategies, this.InputLayer, index);
        }
    }

    public Layer(Neuron[] neurons, Strategies strategies, ILayer inputLayer, int id)
    {
        this.neurons = neurons;
        this.strategies = strategies;
        InputLayer = inputLayer;
        Id = id;
        Size = this.neurons.Length;
    }

    public ILayer InputLayer { get; init; }

    public Neuron[] Neurons 
    { get => neurons.ToArray(); 
    }

    public int Id { get; init; }
    public int Size { get; init; }

    public INode[] Content => Neurons;

    public virtual async Task Compute() {
        await strategies.ComputeStrategies.ComputeLayer(strategies.ActivationStrategy, this);
    }

    public virtual double[] CollectOutput() =>
        neurons.Select(w => w.Value).ToArray();

    public abstract Task BackPropagate(IBackPropagationInput[] backPropagationInput, double gradient);
    
}

public class HiddenLayer : Layer
{
    public HiddenLayer(ILayer inputLayer, int size, Strategies strategies) : base(inputLayer, size, strategies) { }
    public HiddenLayer(Neuron[] neurons, Strategies strategies, ILayer inputLayer, int id)  : base(neurons, strategies, inputLayer, id) { }
    public override async Task BackPropagate(IBackPropagationInput[] gradients, double rate) 
    {
        var hiddenGradients = new ConcurrentBag<Tuple<int, GradientAdjustment>>();
        var neurons = this.Neurons;

        Parallel.ForEach(Enumerable.Range(0, neurons.Length), (i) =>    // Use the input length to create parallel tasks 
        {
            var neuron = neurons[i];

            double error = neuron.OutputWeights.Select(k => gradients[k.LinksTo.Id].Value * k.Value).ToArray().Fast_Sum();

            var hiddenGradient = new GradientAdjustment(neuron, error * strategies.ActivationStrategy.ComputeActivationDerivative(neuron.Value) * rate);
            hiddenGradient.AdjustWeights();
            hiddenGradients.Add(new Tuple<int, GradientAdjustment>(i, hiddenGradient));
        });

        await this.InputLayer.BackPropagate(hiddenGradients.OrderBy(t => t.Item1).Select(v => v.Item2).ToArray(), rate); // Ensure results are ordered, since gradients were calculated in parallel.
    }
}


public class OutputLayer : Layer 
{
    public OutputLayer(ILayer inputLayer, int size, Strategies strategies) : base(inputLayer, size, strategies) { }
    public OutputLayer(Neuron[] neurons, Strategies strategies, ILayer inputLayer, int id) : base(neurons, strategies, inputLayer, id) { }
    public override async Task BackPropagate(IBackPropagationInput[] backPropagationInput, double rate) 
    {
        var gradients = new List<IBackPropagationInput>();
        foreach (OutputErrorAdjustment input in backPropagationInput)
        {
            var outputGradient = input.CreateOuput(this.strategies, rate);
            outputGradient.AdjustWeights();
            gradients.Add(outputGradient);
        }

        // Backpropagate
        await this.InputLayer.BackPropagate(gradients.ToArray(), rate);
    }
}

public class SoftMaxOutputLayer : Layer
{
    public SoftMaxOutputLayer(ILayer inputLayer, int size, Strategies strategies) : base(inputLayer, size, strategies) { }

    public override Task BackPropagate(IBackPropagationInput[] backPropagationInput, double gradient)
    {
        throw new NotImplementedException();
    }

    public override Task Compute()
    {
        var maxValueNode = this.InputLayer.Content.Max(p => p.Value);
        var exps = this.InputLayer.Content.Select(n => new { Node = n, Exp = Math.Exp(n.Value - maxValueNode) });
        double sumExps = exps.Sum(s => s.Exp);

        foreach (var item in exps)
        {
            (this.Content[item.Node.Id] as Neuron)!.SetValue(item.Exp / sumExps);
        }

        return Task.CompletedTask;
    }
}

/// <summary>
/// Defines a constant value node that will receive inputs
/// </summary>
/// <param name="value"></param>
public class InputNode(int id) : INode
{
    public int Id => id;
    public double Value { get; private set; }

    public MutableArray<Weight> OutputWeights { get; private set; } = new MutableArray<Weight>();

    public MutableArray<Weight> InputWeights { get; private set; } = new MutableArray<Weight>();

    public Task ComputeValue() => Task.CompletedTask;

    public void SetValue(double value) { 
        this.Value = value;
    }
}

/// <summary>
/// Defines a neuron node with connected weights
/// </summary>
public class Neuron: INode
{

    private readonly Strategies strategies;
    
    public Neuron(Strategies strategies, ILayer inputLayer, int id)
    {
        this.strategies = strategies;
        this.InputLayer = inputLayer;
        this.Id = id;

        InitializeWeights();

        Bias = strategies.ActivationStrategy.GetInitialBias();
    }

    public Neuron(Strategies strategies, MutableArray<Weight> inputWeights, MutableArray<Weight> outputWeights, double bias, ILayer inputLayer, int id)
    {
        this.strategies = strategies;
        InputWeights = inputWeights;
        OutputWeights = outputWeights;
        Bias = bias;
        InputLayer = inputLayer;
        Id = id;
    }

    private void InitializeWeights()
    {
        foreach (var input in InputLayer.Content)
        {
            this.InputWeights.Add(new Weight(input, this, strategies.ActivationStrategy.GetRandomWeight(InputLayer.Content.Length)));
        }
    }

    public async Task ComputeValue() => this.Value = await strategies.ComputeStrategies.ComputeNode(strategies.ActivationStrategy, this); // strategies.ActivationStrategy.ComputeActivation(InputWeights.Select(x => x.Compute()).Fast_Sum() + Bias);

    public MutableArray<Weight> OutputWeights { get; private set; } = new MutableArray<Weight>();

    public MutableArray<Weight> InputWeights { get; private set; } = new MutableArray<Weight>();

    public double Value { get; private set;}

    public void SetValue(double value) => this.Value = value;

    public double Bias { get; set; }    

    public ILayer InputLayer { get; init; }

    public int Id { get; init; }

}

#region 

#endregion

#region BackPropagation Training

public class OutputErrorAdjustment(Neuron Neuron, double Error) : IBackPropagationInput
{
    public INode Node => Neuron;

    public double Value => Error;

    public IBackPropagationInput CreateOuput(Strategies strategies, double rate)
    {
        return new GradientAdjustment(Neuron, Value* strategies.ActivationStrategy.ComputeActivationDerivative(Neuron.Value)*rate);
    }

    public void AdjustWeights()
    {
        throw new NotImplementedException();
    }

}

public class GradientAdjustment(Neuron Neuron, double Gradient) : IBackPropagationInput
{
    public INode Node => Neuron;
    public double Value => Gradient;

    public void AdjustWeights()
    {
        // Adjust Weights
        Neuron.InputWeights.Apply(SetWeight);

        // Adjust Bias
        Neuron.Bias += Gradient;
    }

    private void SetWeight(Weight weight)
    {
        double delta = Gradient * weight.LinkedFrom.Value;
        weight.SetWeightTo(weight.Value + delta);
    }
}

#endregion


