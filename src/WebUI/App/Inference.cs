using PerceptrOOn.Exporters;
using PerceptrOOn;
using Perceptr00n.WebUI.App;
using System.Text.Json;
using System.Text;
using System.Drawing;
using System.Collections;
using System.Reflection.Emit;

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
            return await _currentNetwork.Predict(input);
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
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = values[i] switch
                {
                    > 0.46 => 1,
                    _ => values[i]
                };
            }


            var results = await context.GetInferenceSession().Infer(values);
#if DEBUG
            LogImage(values);
#endif
            Directory.CreateDirectory("ImageLog");



            context.Response.ContentType = "application/json";
            await context.Response.Body.WriteAsync(Encoding.UTF8.GetBytes(JsonSerializer.Serialize(results)));

            return;
        }

        private static void LogImage(double[] values)
        {
            var bytePixels = values.Select(v => (byte)(v * 255)).ToArray();
            Directory.CreateDirectory("Images");
            var fileName = $"Images/{Guid.NewGuid()}.bmp";
            unsafe
            {
                fixed (byte* ptr = bytePixels)
                {
                    var bmp = new Bitmap(28, 28, 28, System.Drawing.Imaging.PixelFormat.Format8bppIndexed, (IntPtr)ptr);
                    bmp.Save(fileName);
                }
            }
        }
    }

}
