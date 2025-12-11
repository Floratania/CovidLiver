using Microsoft.ML.Data;

namespace CovidLiver;

public class ModelInput
{
    [LoadColumn(0)] public string Cancer { get; set; }
    [LoadColumn(1)] public string Year { get; set; }
    [LoadColumn(2)] public float Month { get; set; }
    [LoadColumn(3)] public string Bleed { get; set; }
    [LoadColumn(4)] public string Mode_Presentation { get; set; }
    [LoadColumn(5)] public float Age { get; set; }
    [LoadColumn(6)] public string Gender { get; set; }
    [LoadColumn(7)] public string Etiology { get; set; }
    [LoadColumn(8)] public string Cirrhosis { get; set; }
    [LoadColumn(9)] public float Size { get; set; }
    [LoadColumn(10)] public string HCC_TNM_Stage { get; set; }
    [LoadColumn(11)] public string HCC_BCLC_Stage { get; set; }
    [LoadColumn(12)] public string ICC_TNM_Stage { get; set; }
    [LoadColumn(13)] public string Treatment_grps { get; set; }
    [LoadColumn(14)] public float Survival_fromMDM { get; set; }
    [LoadColumn(15)] public string Alive_Dead { get; set; }
    [LoadColumn(16)] public string Type_of_incidental_finding { get; set; }
    [LoadColumn(17)] public string Surveillance_programme { get; set; }
    [LoadColumn(18)] public string Surveillance_effectiveness { get; set; }
    [LoadColumn(19)] public string Mode_of_surveillance_detection { get; set; }
    [LoadColumn(20)] public float Time_diagnosis_1st_Tx { get; set; }
    [LoadColumn(21)] public string Date_incident_surveillance_scan { get; set; }
    [LoadColumn(22)] public float PS { get; set; }
    [LoadColumn(23)] public float Time_MDM_1st_treatment { get; set; }
    [LoadColumn(24)] public float Time_decisiontotreat_1st_treatment { get; set; }
    [LoadColumn(25)] public string Prev_known_cirrhosis { get; set; }
    [LoadColumn(26)] public float Months_from_last_surveillance { get; set; }
}
