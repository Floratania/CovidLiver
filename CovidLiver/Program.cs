using CovidLiver;

class Program
{
    static void Main()
    {
        Console.WriteLine("==== COVID LIVER ML APP ====");
        Console.WriteLine("1 - Train model");
        Console.WriteLine("2 - Predict sample");
        Console.Write("Choose: ");

        var choice = Console.ReadLine();
        var trainer = new Trainer();
        var predictor = new Predictor();

        switch (choice)
        {
            case "1":
                trainer.Train("Data/covid-liver.csv");
                break;

            case "2":
                predictor.PredictSample();
                break;
        }
    }
}
