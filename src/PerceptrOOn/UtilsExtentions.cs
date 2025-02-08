using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptrOOn
{
    public static class UtilsExtentions
    {
        public static double[] ByteToFlatOutput(this byte label, int outputSize)
        {
            double[] output = new double[outputSize];
            output[label] = 1;
            return output;
        }

        public static double[] Flatten2DMatrix(this byte[,] image)
        {
            var flat = new List<double>();
            foreach (var item in image)
            {
                flat.Add(item.Normalize());
            }

            return flat.ToArray();
        }

        public static double Normalize(this byte input) => input / 255d;
    }
}
