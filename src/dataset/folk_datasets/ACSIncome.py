from folktables import ACSIncome as FolkACSIncome

from .Folktables_base import ACSDataSource


class ACSIncome(ACSDataSource):
    name = "ACSIncome"
    binary_sens_cols = [0, 1]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self):
        data = super().get_data()
        features, labels, _ = FolkACSIncome.df_to_pandas(data)

        # From Retiring Adult: New Datasets for Fair Machine Learning
        # AGEP: Age of person   COW: Class of worker    SCHL: Educational attainment
        # MAR: Marital status   OCCP: Occupation        POBP: Place of birth
        # RELP: Relationship    WKHP: Usual hours worked per week past 12 months
        # SEX: 1 - male, 2 - female                     RA1CP: Recorded detailed race code
        # PINCP: Total person's income: > 50 000
        df = features.join(labels)

        return super()._preprocess_df(df,
                                      sens_columns=["SEX", "AGEP", "MAR", "RAC1P"],
                                      label_column=["PINCP"],
                                      drop_columns=["POBP", "RELP"],
                                      categorical_values=["COW", "MAR", "SEX", "RAC1P"],
                                      mapping={"RAC1P": {3: 8, 4: 8, 5: 8, 7: 8}},
                                      mapping_label={},
                                      normalise_columns=["AGEP", "SCHL", "OCCP", "WKHP"],
                                      drop_rows={})
