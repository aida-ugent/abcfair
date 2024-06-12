from folktables import ACSTravelTime as FolkACSTravelTime

from .Folktables_base import ACSDataSource


class ACSTravelTime(ACSDataSource):
    name = "ACSTravelTime"
    binary_sens_cols = [0, 1]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self):
        data = super().get_data()
        features, labels, _ = FolkACSTravelTime.df_to_pandas(data)

        # AGEP: Age of person       SCHL: Educational attainment
        # MAR: Marital status       SEX: 1 - male, 2 - female
        # DIS: Disability (binary)  ESP: Employment state of the parent
        # CIT: Citizenship status   MIL: Military service
        # ANC: Ancestry             NATIVITY: natitivity
        # RELP: Relationship        DEAR: Hearing diff
        # DEYE: Vision diff         DREM: Cognitive disability
        # RAC1P: Recorded detailed race code
        # GCL: Grandparents living with grandchildren
        # COW: Class of worker      ESR: Employment status
        # WKHP: Usual hours worked per week past 12 months
        # JWMNP: Travel time to work
        # PINCP: Person's income    ST: State
        # OCCP: Occupation code     JWTR: Mode of transportation
        # POVPIP: Income-to-poverty ratio
        df = features.join(labels)

        # 'POVPIP',

        return super()._preprocess_df(df,
                                      sens_columns=["SEX", "AGEP", "RAC1P", "DIS"],
                                      label_column=["JWMNP"],
                                      drop_columns=["RELP", "ESP"],
                                      categorical_values=["SCHL", "MAR", "SEX", "RAC1P", "DIS", "ST", "CIT", "JWTR",
                                                          "PUMA", "POWPUMA", "OCCP"],
                                      mapping={"RAC1P": {3: 8, 4: 8, 5: 8, 7: 8}},
                                      mapping_label={},
                                      normalise_columns=["AGEP", "POVPIP"],
                                      drop_rows={})