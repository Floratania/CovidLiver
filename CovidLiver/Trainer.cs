using Microsoft.ML;
using Microsoft.ML.Data;

namespace CovidLiver;

public class Trainer
{
    private readonly MLContext ml = new();
    public string ModelPath = "liverModel.zip";

    public void Train(string path)
    {
        Console.WriteLine("Loading data...");
        var data = ml.Data.LoadFromTextFile<ModelInput>(
            path,
            hasHeader: true,
            separatorChar: ',',
            allowQuoting: true,
            trimWhitespace: true
        );


        var rows = ml.Data.CreateEnumerable<ModelInput>(data, false)
            .Select(r =>
            {
                r.HCC_TNM_Stage = Fix(r.HCC_TNM_Stage);
                r.HCC_BCLC_Stage = Fix(r.HCC_BCLC_Stage);
                r.ICC_TNM_Stage = Fix(r.ICC_TNM_Stage);
                r.Type_of_incidental_finding = Fix(r.Type_of_incidental_finding);
                r.Surveillance_programme = Fix(r.Surveillance_programme);
                r.Surveillance_effectiveness = Fix(r.Surveillance_effectiveness);
                r.Mode_of_surveillance_detection = Fix(r.Mode_of_surveillance_detection);
                return r;
            })
            .ToList();

        var split = StratifiedSplit(rows, 0.2f);

   
        var pipeline =
           
            ml.Transforms.CustomMapping(new LabelMappingFactory().GetMapping(), contractName: "LabelMapping")

          
            .Append(ml.Transforms.ReplaceMissingValues("Month"))
            .Append(ml.Transforms.ReplaceMissingValues("Age"))
            .Append(ml.Transforms.ReplaceMissingValues("Size"))
            .Append(ml.Transforms.ReplaceMissingValues("Survival_fromMDM"))
            .Append(ml.Transforms.ReplaceMissingValues("Time_diagnosis_1st_Tx"))
            .Append(ml.Transforms.ReplaceMissingValues("PS"))
            .Append(ml.Transforms.ReplaceMissingValues("Time_MDM_1st_treatment"))
            .Append(ml.Transforms.ReplaceMissingValues("Time_decisiontotreat_1st_treatment"))
            .Append(ml.Transforms.ReplaceMissingValues("Months_from_last_surveillance"))

    
            .Append(ml.Transforms.Text.FeaturizeText("CancerF", "Cancer"))
            .Append(ml.Transforms.Text.FeaturizeText("YearF", "Year"))
            .Append(ml.Transforms.Text.FeaturizeText("BleedF", "Bleed"))
            .Append(ml.Transforms.Text.FeaturizeText("ModeF", "Mode_Presentation"))
            .Append(ml.Transforms.Text.FeaturizeText("GenderF", "Gender"))
            .Append(ml.Transforms.Text.FeaturizeText("EtiologyF", "Etiology"))
            .Append(ml.Transforms.Text.FeaturizeText("CirrhosisF", "Cirrhosis"))

   
            .Append(ml.Transforms.Concatenate("Features",
                "Month", "Age", "Size", "Survival_fromMDM",
                "CancerF", "YearF", "BleedF", "ModeF",
                "GenderF", "EtiologyF", "CirrhosisF"))

  
            .Append(ml.BinaryClassification.Trainers.LbfgsLogisticRegression(
                labelColumnName: "Label",
                featureColumnName: "Features"
            ));

        Console.WriteLine("\n Training model...");
        var model = pipeline.Fit(split.Train);

        Console.WriteLine("\n Evaluating...");
        var predictions = model.Transform(split.Test);
        var metrics = ml.BinaryClassification.Evaluate(predictions, "Label");


        Console.ForegroundColor = ConsoleColor.Cyan;
        
        Console.WriteLine($"Accuracy:    {metrics.Accuracy:P2}");
        Console.WriteLine($"Precision:   {metrics.PositivePrecision:P2}");
        Console.WriteLine($"Recall:      {metrics.PositiveRecall:P2}");
        Console.WriteLine($"F1 Score:    {metrics.F1Score:P2}");

        if (double.IsNaN(metrics.AreaUnderRocCurve))
            Console.WriteLine("AUC:          Not defined (both classes required)");
        else
            Console.WriteLine($"AUC:         {metrics.AreaUnderRocCurve:P2}");

 
        Console.ResetColor();

        Console.WriteLine("\n Saving model...");
        ml.Model.Save(model, split.Train.Schema, ModelPath);

        Console.WriteLine(" Model trained & saved!");
    }

    private string Fix(string? v)
    {
        if (string.IsNullOrWhiteSpace(v) || v.Trim() == "NA")
            return "Unknown";
        return v.Trim();
    }

    private (IDataView Train, IDataView Test) StratifiedSplit(List<ModelInput> rows, float testFraction)
    {
        var alive = rows.Where(r => r.Alive_Dead == "Alive").ToList();
        var dead = rows.Where(r => r.Alive_Dead == "Dead").ToList();

        int aliveTestCount = Math.Max(1, (int)(alive.Count * testFraction));
        int deadTestCount = Math.Max(1, (int)(dead.Count * testFraction));

        var rnd = new Random();
        alive = alive.OrderBy(_ => rnd.Next()).ToList();
        dead = dead.OrderBy(_ => rnd.Next()).ToList();

        var test = new List<ModelInput>();
        test.AddRange(alive.Take(aliveTestCount));
        test.AddRange(dead.Take(deadTestCount));

        var train = new List<ModelInput>();
        train.AddRange(alive.Skip(aliveTestCount));
        train.AddRange(dead.Skip(deadTestCount));

        // Shuffle
        train = train.OrderBy(_ => rnd.Next()).ToList();
        test = test.OrderBy(_ => rnd.Next()).ToList();

        Console.WriteLine("\n SPLIT");
        Console.WriteLine($"Train Alive: {train.Count(x => x.Alive_Dead == "Alive")}");
        Console.WriteLine($"Train Dead : {train.Count(x => x.Alive_Dead == "Dead")}");
        Console.WriteLine($"Test Alive : {test.Count(x => x.Alive_Dead == "Alive")}");
        Console.WriteLine($"Test Dead  : {test.Count(x => x.Alive_Dead == "Dead")}");
  
        return (
            ml.Data.LoadFromEnumerable(train),
            ml.Data.LoadFromEnumerable(test)
        );
    }
}
