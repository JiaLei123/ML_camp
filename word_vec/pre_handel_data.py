import gzip
import re
import zipfile
import pandas as pd

data_path = "E:\\ML_learning\\NLP_data\\news_sohusite_xml.full.tar.gz"
# zip_file = zipfile.ZipFile(data_path, 'r')

output_list = list()
line_regex = re.compile(r"<content>.*?</content>")
with gzip.open(data_path, 'r') as input:
    for line in input:
       try:
           line = line.decode("GBK").strip()
           match = re.search(line_regex, line)
           if match:
               output_list.append(match.group())
           # if line.startswith("<content>") and len(line) > 20:
           #     output_list.append(line)
       except Exception as e:
            print(e)

output_list = pd.Series(output_list)
output_list.to_csv("E:\\ML_learning\\NLP_data\\corpus_pc.txt", encoding='utf-8')

# with open("E:\\ML_learning\\NLP_data\\corpus.txt", 'w', encoding='utf-8') as write_file:
#     for file_name in zip_file.namelist():
#         with zip_file.open(file_name, 'r') as zf:
#             for line in zf:
#                 try:
#                     line = line.decode("GBK")
#                     if line.startswith("<content>") and len(line) > 20:
#                         write_file.write(line)
#                 except Exception as e:
#                     print(e)