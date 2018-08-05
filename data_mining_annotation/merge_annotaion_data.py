import pandas as pd


data_path_anno = r"C:\Users\lei_jia.NUANCE\Desktop\emp\merged_transandgen.goldset.wfas"
data_path_utter = r"C:\Users\lei_jia.NUANCE\Desktop\emp\merged_transandgen.goldset.4tool"
data_path_final = r"C:\Users\lei_jia.NUANCE\Desktop\emp\merged_transandgen.goldset.review"

utters = pd.read_table(data_path_utter)
annos = pd.read_table(data_path_anno)

annos_need = annos.iloc[:, 2:4]
utters_need = utters.text

final_list = pd.concat([utters_need, annos_need], axis=1)
final_list.to_csv(data_path_final)
