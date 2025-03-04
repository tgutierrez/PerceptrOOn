using PerceptrOOn.Exporters;
using PerceptrOOn;
using Perceptr00n.WebUI.App;
using System.Text.Json;
using System.Text;

namespace WebUI.App
{
    public class InferenceHandler
    {
        private readonly Guid _id;
        private readonly NeuralNetwork _currentNetwork;
        public InferenceHandler(Guid id)
        {
            _id = id;

            var mnistJSON = File.ReadAllText(@"Checkpoints/mnist.json");
            var importer = new JSONImporter();
            var layers = importer.Import(mnistJSON);
            _currentNetwork = new NeuralNetwork(layers, new Strategies(new ReLuActivationStrategy(), new DefaultComputeStrategy()));
        }


        public async Task<double[]> Infer(double[] input)
        {
            return await _currentNetwork.Predict(input.Normalize());
        }

    }

    public static class InferenceExtensions
    {
        public static IEndpointRouteBuilder AddInferenceAPI(this IEndpointRouteBuilder builder) {

            builder.MapPost("/api/infer", InferService);

            return builder;
        }

        private static async Task InferService(double[] values, HttpContext context, InferenceSessionHandler inferenceSessionHandler)
        {
            var results = await context.GetInferenceSession().Infer(values);

            context.Response.ContentType = "application/json";
            await context.Response.Body.WriteAsync(Encoding.UTF8.GetBytes(JsonSerializer.Serialize(results)));

            return;
        }
    }

}
