using Microsoft.ML;

namespace CovidLiver;

public class Predictor
{
    private readonly MLContext ml = new();
    private readonly string modelPath = "liverModel.zip";

    public void PredictSample()
    {
        if (!File.Exists(modelPath))
        {
            Console.WriteLine(" No model found!");
            return;
        }

        Console.WriteLine(" Loading model...");
        var model = ml.Model.Load(modelPath, out _);

        var engine = ml.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

        var sample = new ModelInput
        {
            Cancer = "Y",
            Year = "Prepandemic",
            Month = 1,
            Bleed = "N",
            Mode_Presentation = "Surveillance",
            Age = 60,
            Gender = "M",
            Etiology = "NAFLD",
            Cirrhosis = "Y",
            Size = 30,
            HCC_TNM_Stage = "II",
            HCC_BCLC_Stage = "A",
            ICC_TNM_Stage = "NA",
            Treatment_grps = "Ablation",
            Survival_fromMDM = 20,

          
            Alive_Dead = "Dead"
        };

        //var sample = new ModelInput
        //{
        //    Cancer = "Y",
        //    Year = "Prepandemic",
        //    Month = 3,
        //    Bleed = "N",
        //    Mode_Presentation = "Incidental",
        //    Age = 78,
        //    Gender = "F",
        //    Etiology = "NAFLD",
        //    Cirrhosis = "Y",
        //    Size = 45,
        //    HCC_TNM_Stage = "IIIA+IIIB",
        //    HCC_BCLC_Stage = "C",
        //    ICC_TNM_Stage = "Unknown",
        //    Treatment_grps = "Supportive care",
        //    Survival_fromMDM = 4
        //};


        var result = engine.Predict(sample);

        Console.WriteLine($"\nPrediction: {(result.Prediction ? "Alive" : "Dead")}");
        Console.WriteLine($"Probability: {result.Probability:P2}");
    }
}
