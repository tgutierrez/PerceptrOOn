using HPCsharp.ParallelAlgorithms;
using Microsoft.VisualBasic;
using System;
using System.Collections;
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


    /// <summary>
    /// encapsulates an array so it can be mutated and accessed. 
    /// </summary>
    /// <remarks>
    /// Explicitly restricts enumerations to purge away foreach but keeping the linq-esque semantics
    /// Add is expensive!
    /// </remarks>
    /// <typeparam name="T"></typeparam>
    public class MutableArray<T>
    {
        public MutableArray() { }

        private MutableArray(T[] values) {
            this.values = values;
        }

        private T[] values = Array.Empty<T>();

        private readonly Lock _lock = new();
        public T this[int index] { 
            get => values[index];
            set => values[index] = value;
        }

        public int Count => values.Length;

        public void Add(T element) {
            lock (_lock)
            {
                var list = values.ToList();
                list.Add(element);
                values = list.ToArray();
            }
        }

        public Proj[] Select<Proj>(Func<T, Proj> selector) {

            if (values.Length == 0) return Array.Empty<Proj>();
            var span = values.AsSpan();
            Proj[] projection = new Proj[values.Length];    
            int i = 0;


            do
            {
                projection[i] = selector(span[i]);
                i++;
            } while (i < projection.Length);

            return projection;
        }

        public void Apply(Action<T> action) 
        {
            if (values.Length == 0) return;
            var span = values.AsSpan();
            int i = 0;
            do
            {
                action(span[i]);
                i++;
            } while (i < span.Length);
        }

        public T[] Values { get => values; }

        public static implicit operator T[](MutableArray<T> array)  => array.Values;

        public static implicit operator MutableArray<T>(T[] array) => new MutableArray<T>(array);

    }
}
