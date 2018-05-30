import zipfile

zip_file = zipfile.ZipFile("news_sohusite_xml.full.zip", 'r')
with open("corpus.txt", 'w', encoding='utf-8') as write_file:
    for file_name in zip_file.namelist():
        with zip_file.open(file_name, 'r') as zf:
            for line in zf:
                try:
                    line = line.decode("GB2312")
                    if line.startswith("<content>") and len(line) > 20:
                        write_file.write(line)
                except Exception as e:
                    print(e)