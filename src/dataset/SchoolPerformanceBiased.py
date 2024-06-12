from .base import CSVDataSource


class SchoolPerformanceBiased(CSVDataSource):
    name = "SchoolPerformanceBiased"
    delimiter = ","

    # Some values are nans, which will be deleted later. If we would take Gender as categorical, then these nans are
    # very tricky to avoid. This is why we don't make Gender a categorical column, i.e. it is just column [0].
    binary_sens_cols = [0, 1]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self):
        # Walc: Weekly alcohol consumption
        # goout: amount of time the student goes out per week, ranges from 1 (never) to 4 (thrice or more per week)
        return super().read('CompleteDataAndBiases.csv',
                            sens_columns=["sex", "Parents_edu"],
                            label_column=["Predicted_Pass_PassFailStrategy"],
                            drop_columns=["index", "ParticipantID", "name", "G3", "PredictedGrade", "PredictedRank",
                                          "StereotypeActivation"],
                            categorical_values=["sex", "studytime", "freetime", "Walc", "Parents_edu", "absences",
                                                "reason"],
                            mapping={"romantic": {"no": False, "yes": True}},
                            mapping_label=None,
                            normalise_columns=None,
                            drop_rows=None)
