using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace PerceptrOOn.Exporters
{
    public class JSONImporter : INetworkImporter<string>
    {
        public ILayer[] Import(string networkData)
        {
            var layers = new List<Layer>(); 
            var exportableNetwork = JsonSerializer.Deserialize<ExportableNetwork>(networkData, SourceGenerationContext.Default.ExportableNetwork);



            return layers.ToArray();    
        }
    }
}
