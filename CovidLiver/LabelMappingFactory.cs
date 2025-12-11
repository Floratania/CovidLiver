using Microsoft.ML.Transforms;
using Microsoft.ML.Data;

namespace CovidLiver;

[CustomMappingFactoryAttribute("LabelMapping")]
public class LabelMappingFactory : CustomMappingFactory<ModelInput, LabelOutput>
{
    public override Action<ModelInput, LabelOutput> GetMapping()
        => (input, output) =>
        {
            output.Label = input.Alive_Dead?.Trim().ToLower() == "alive";
        };
}

public class LabelOutput
{
    public bool Label { get; set; }
}
