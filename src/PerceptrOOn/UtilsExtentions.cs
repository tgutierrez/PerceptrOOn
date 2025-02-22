﻿using System.Numerics;

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

        #region Fast Sum Methods. Lifted from https://github.com/DragonSpit/HPCsharp/blob/81837935698a8b1d412ee6de8a5f04b04a721838/HPCsharp/SumParallel.cs

        // All Credits to https://github.com/DragonSpit/HPCsharp (C) Victor J. Duvanenko.

        /// <summary>
        /// If hardware support is available, it will route the sum method to the fast version
        /// </summary>
        /// <remarks>
        /// only x86 is supported at this moment.
        /// </remarks>
        /// <param name="array"></param>
        /// <returns></returns>
        public static double Fast_Sum(this double[] array) =>
            Vector.IsHardwareAccelerated ? array.SumSse() :              
            array.Sum();

        /// <summary>
        /// Summation of double[] array, using data parallel SIMD/SSE instructions for higher performance on a single core.
        /// </summary>
        /// <param name="arrayToSum">An array to sum up</param>
        /// <returns>double sum</returns>
        public static double SumSse(this double[] arrayToSum)
        {
            if (arrayToSum == null)
                throw new ArgumentNullException(nameof(arrayToSum));
            return arrayToSum.SumSseInner(0, arrayToSum.Length - 1);
        }

        /// <summary>
        /// Summation of double[] array, using data parallel SIMD/SSE instructions for higher performance on a single core.
        /// </summary>
        /// <param name="arrayToSum">An array to sum up</param>
        /// <param name="startIndex">index of the starting element for the summation</param>
        /// <param name="length">number of array elements to sum up</param>
        /// <returns>double sum</returns>

        public static double SumSse(this double[] arrayToSum, int startIndex, int length)
        {
            if (arrayToSum == null)
                throw new ArgumentNullException(nameof(arrayToSum));
            return arrayToSum.SumSseInner(startIndex, startIndex + length - 1);
        }

        private static double SumSseInner(this double[] arrayToSum, int l, int r)
        {
            var sumVector = new Vector<double>();
            int sseIndexEnd = l + ((r - l + 1) / Vector<double>.Count) * Vector<double>.Count;
            int i;
            for (i = l; i < sseIndexEnd; i += Vector<double>.Count)
            {
                var inVector = new Vector<double>(arrayToSum, i);
                sumVector += inVector;
            }
            double overallSum = 0;
            for (; i <= r; i++)
                overallSum += arrayToSum[i];
            for (i = 0; i < Vector<double>.Count; i++)
                overallSum += sumVector[i];
            return overallSum;
        }

        #endregion
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
