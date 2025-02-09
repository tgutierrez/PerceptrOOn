using HPCsharp.ParallelAlgorithms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;
using System.Transactions;

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


        /// <summary>
        /// If hardware support is available, it will route the sum method to the fast version
        /// </summary>
        /// <remarks>
        /// only x86 is supported at this moment.
        /// </remarks>
        /// <param name="array"></param>
        /// <returns></returns>
        public static double Fast_Sum(this double[] array) =>
            Sse.IsSupported ? array.SumSse() :              
            array.Sum();
    }
}
