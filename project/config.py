# config.py

# Note: Update these relative paths to match your local or repository data structure.
DATASETS_CONFIG = [
    {
        'name': 'eSpCas9',
        'feature_folder': "data/eSpCas9/eSpCas9-fm",
        'excel_file': "data/eSpCas9/eSpCas9_lable.xlsx",
        'sequence_file': "data/eSpCas9/eSpCas9(1).csv",
        'id_column': 'ID', 'label_column': 'label', 'sequence_column': '43mer'
    },

    {
        'name': 'Cas9',
        'feature_folder': "data/Cas9/Cas9-fm",
        'excel_file': "data/Cas9/Cas9_lable.xlsx",
        'sequence_file': "data/Cas9/Cas9.csv",
        'id_column': 'ID', 'label_column': 'label', 'sequence_column': '43mer'
    },

    {
        'name': 'knoRecA_Cas9',
        'feature_folder': "data/knoRecA_Cas9/knoRecA_Cas9-fm",
        'excel_file': "data/knoRecA_Cas9/knoRecA_Cas9_lable.xlsx",
        'sequence_file': "data/knoRecA_Cas9/knoRecA_Cas9.csv",
        'id_column': 'ID', 'label_column': 'label', 'sequence_column': '43mer'
    }
]