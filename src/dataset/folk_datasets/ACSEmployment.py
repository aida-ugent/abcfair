from folktables import ACSEmployment as FolkACSEmployment

from .Folktables_base import ACSDataSource


class ACSEmployment(ACSDataSource):
    name = "ACSEmployment"
    binary_sens_cols = [0, 1]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self):
        data = super().get_data()
        features, labels, _ = FolkACSEmployment.df_to_pandas(data)

        # From Retiring Adult: New Datasets for Fair Machine Learning
        # AGEP: Age of person   SCHL: Educational attainment
        # MAR: Marital status   
        # RELP: Relationship    WKHP: Usual hours worked per week past 12 months
        # DIS: Disability (binary)                      ESP: Employment state of the parent
        # CIT: Citizenship status                       MIG: Mobility status
        # MIL: Military service ANC: Ancestry recode    NATIVITY: natitivity
        # DEAR: Hearing diff    DEYE: Vision diff       DREM: Cognitive disability
        # SEX: 1 - male, 2 - female                     RA1CP: Recorded detailed race code
        # ESR: Employment status
        df = features.join(labels)

        return super()._preprocess_df(df,
                                      sens_columns=["SEX", "AGEP", "MAR", "RAC1P", "DIS"],
                                      label_column=["ESR"],
                                      drop_columns=["RELP"],
                                      categorical_values=["MAR", "SEX", "RAC1P", "DIS", "ESP", "CIT", "MIG", "MIL",
                                                          "NATIVITY", "DEAR", "DEYE", "DREM"],
                                      mapping={"RAC1P": {3: 8, 4: 8, 5: 8, 7: 8}},
                                      mapping_label={},
                                      normalise_columns=["AGEP"],
                                      drop_rows={})
