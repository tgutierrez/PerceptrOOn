#region Base Definitions

public interface ILayer {
    int Id { get; }
    int Size { get; }
    INode[] Content {  get; }

    void BackPropagate(IBackPropagationInput[] backPropagationInput, double rate);
}

public interface INode
{
    int Id { get; }
    public double Value { get; }

    public void ComputeValue();

    public List<Weight> OutputWeights { get; }

    public List<Weight> InputWeights { get; }

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
    double ComputeActivation(double input);
    double ComputeActivationDerivative (double input);
    double GetRandomWeight();
}


public delegate void Notify(int current, int total, string description);


public record NetworkDefinition(int InputNodes, int[] HiddenLayerNodeDescription, int OutputNodes, IActivationStrategy ActivationStrategy, Notify? NotificationCallback = null);
public record TrainingData(double[] input, double[] expectedOutput);
public record TrainingParameters(TrainingData[] TrainingDataSet, int Epochs, double TrainingRate);

#endregion

public interface IBackPropagationInput {

    double Value { get; }

    INode Node { get; }

    void AdjustWeights();
}



#region Activation strategies

public class SigmoidActivationStrategy(int seed) : IActivationStrategy
{
    private Random rand = new(seed);

    public double ComputeActivation(double x) => 1.0 / (1.0 + Math.Exp(-x));

    public double ComputeActivationDerivative (double x) => x * (1 - x);

    public double GetRandomWeight() => rand.NextDouble() *2 -1;
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
    private readonly IActivationStrategy activationStrategy;
    
    private InputLayer InputLayer => (layers[0] as InputLayer)!;
    private Layer OutputLayer => (layers[^1] as Layer)!;
    private Layer[] HiddenLayer => layers[1..^1].Cast<Layer>().ToArray()!;

    public NetworkDefinition Definition { get; internal set; }

    public NeuralNetwork(NetworkDefinition definition) { 
        this.activationStrategy = definition.ActivationStrategy;
        layers = BuildLayers(definition);
        this.Definition = definition;
    }

    private ILayer[] BuildLayers(NetworkDefinition definition)
    {
        var layerList = new List<ILayer>();

        ILayer previousLayer = new InputLayer(definition.InputNodes);
        layerList.Add(previousLayer);

        int layerIndex = 1;
        foreach (var description in definition.HiddenLayerNodeDescription) {
            var newHiddenLayer = new HiddenLayer(previousLayer, description, this.activationStrategy);
            layerList.Add(newHiddenLayer);
            previousLayer = newHiddenLayer;
            layerIndex++;
        }

        layerList.Add(new OutputLayer(previousLayer, definition.OutputNodes, activationStrategy));

        return layerList.ToArray();
    }

    public double[] Predict(double[] input)
    {
        // Collect inputs
        InputLayer.CollectInput(input);

        // Compute Hidden Layers
        foreach (var hiddenLayer in this.HiddenLayer)
        {
            hiddenLayer.Compute();
        }

        // Compute Output
        OutputLayer.Compute();

        // Collect Result and  return
        return OutputLayer.CollectOutput();
    }

    /// <summary>
    /// Train the network using backpropagation
    /// </summary>
    /// <param name="trainingParameters">Parameters</param>
    public void Train(TrainingParameters trainingParameters)
    {
        foreach (var epoch in Enumerable.Range(0, trainingParameters.Epochs))
        {
            TrainingEpoch(trainingParameters, epoch, trainingParameters.TrainingRate);
        }      
    }

    private void TrainingEpoch(TrainingParameters trainingParameters, int epoch, double rate) {
        int cnt = 0;
        foreach (var trainingSet in trainingParameters.TrainingDataSet)
        {
            TrainingCycle(trainingSet, epoch, rate);
            Definition.NotificationCallback?.Invoke(cnt, trainingParameters.TrainingDataSet.Length, "Performing Cycle");
        }
    }

    private void TrainingCycle(TrainingData trainingSet, int epoch, double rate)
    {
        // 1 - Compute
        this.Predict(trainingSet.input);
        // 2 - Collect errors (predicted vs actual)
        var outputErrors = this.OutputLayer
                        .Neurons
                        .Index()
                        .Select((e) => new OutputErrorAdjustment(e.Item, trainingSet.expectedOutput[e.Index] - e.Item.Value)).ToArray();

        // 3 - Feed the errors to the output layer and start the process
        this.OutputLayer.BackPropagate(outputErrors, rate  );

    }
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

    public void BackPropagate(IBackPropagationInput[] backPropagationInput, double gradient) { /*No-Op, nothing to backpropagate*/ }
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
    internal IActivationStrategy activationStrategy;
    public Layer(ILayer inputLayer, int size, IActivationStrategy activationStrategy) {
        this.Id = inputLayer.Id+1;
        this.InputLayer = inputLayer;
        this.Size = size;
        this.activationStrategy = activationStrategy;
        neurons = new Neuron[size];
        foreach (int index in Enumerable.Range(0, size))
        {
            neurons[index] = new Neuron(activationStrategy, this.InputLayer, index);
        }
    }

    public ILayer InputLayer { get; init; }

    public Neuron[] Neurons 
    { get => neurons.ToArray(); 
    }

    public int Id { get; init; }
    public int Size { get; init; }

    public INode[] Content => Neurons;

    public void Compute() {
        foreach (var neuron in neurons) {
            neuron.ComputeValue();
        }
    }

    public double[] CollectOutput() =>
        neurons.Select(w => w.Value).ToArray();

    public abstract void BackPropagate(IBackPropagationInput[] backPropagationInput, double gradient);
    
}

public class HiddenLayer : Layer
{
    public HiddenLayer(ILayer inputLayer, int size, IActivationStrategy activationStrategy) : base(inputLayer, size, activationStrategy) { }

    public override void BackPropagate(IBackPropagationInput[] gradients, double rate) 
    {
        var hiddenGradients = new List<IBackPropagationInput>();

        // Calculate errors from Previous layer
        foreach (var neuron in this.Neurons)
        {
            double error = 0;
            foreach (var outputWeight in neuron.OutputWeights)
            {
                error += gradients[outputWeight.LinksTo.Id].Value * outputWeight.Value;
            }
            
            var hiddenGradient = new GradientAdjustment(neuron, error*activationStrategy.ComputeActivationDerivative(neuron.Value)*rate);
            hiddenGradient.AdjustWeights();
            hiddenGradients.Add(hiddenGradient);
        }

        this.InputLayer.BackPropagate(hiddenGradients.ToArray(), rate);
    }
}


public class OutputLayer : Layer 
{
    public OutputLayer(ILayer inputLayer, int size, IActivationStrategy activationStrategy) : base(inputLayer, size, activationStrategy) { }

    public override void BackPropagate(IBackPropagationInput[] backPropagationInput, double rate) 
    {
        var gradients = new List<IBackPropagationInput>();
        foreach (OutputErrorAdjustment input in backPropagationInput)
        {
            var outputGradient = input.CreateOuput(this.activationStrategy, rate);
            outputGradient.AdjustWeights();
            gradients.Add(outputGradient);
        }

        // Backpropagate
        this.InputLayer.BackPropagate(gradients.ToArray(), rate);
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

    public List<Weight> OutputWeights { get; private set; } = new List<Weight>();

    public List<Weight> InputWeights { get; private set; } = new List<Weight>();

    public void ComputeValue() { }

    public void SetValue(double value) { 
        this.Value = value;
    }
}

/// <summary>
/// Defines a neuron node with connected weights
/// </summary>
public class Neuron: INode
{

    private readonly IActivationStrategy actions;
    
    public Neuron(IActivationStrategy activationStrategy, ILayer inputLayer, int id)
    {
        this.actions = activationStrategy;
        this.InputLayer = inputLayer;
        this.InputWeights = new List<Weight>();
        this.OutputWeights = new List<Weight>();
        this.Id = id;

        InitializeWeights();

        Bias = activationStrategy.GetRandomWeight();
    }

    private void InitializeWeights()
    {
        foreach (var input in InputLayer.Content)
        {
            this.InputWeights.Add(new Weight(input, this, actions.GetRandomWeight()));
        }
    }

    public void ComputeValue() => this.Value = actions.ComputeActivation(InputWeights.Sum(x => x.Compute()) + Bias);

    public List<Weight> InputWeights { get; private set; }

    public List<Weight> OutputWeights { get; private set; }

    public double Value { get; private set;}

    public double Bias { get; set; }    

    public ILayer InputLayer { get; init; }

    public int Id { get; init; }

}

#region BackPropagation Training

public class OutputErrorAdjustment(Neuron Neuron, double Error) : IBackPropagationInput
{
    public INode Node => Neuron;

    public double Value => Error;

    public IBackPropagationInput CreateOuput(IActivationStrategy activationStrategy, double rate)
    {
        return new GradientAdjustment(Neuron, Value*activationStrategy.ComputeActivationDerivative(Neuron.Value)*rate);
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
        foreach (var weight in Neuron.InputWeights)
        {
            double delta = Gradient * weight.LinkedFrom.Value;
            weight.SetWeightTo(weight.Value + delta);
        }
        // Adjust Bias
        Neuron.Bias += Gradient;
    }

}

#endregion