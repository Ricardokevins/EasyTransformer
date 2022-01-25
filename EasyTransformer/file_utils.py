# record some file operation
def read_txt_lines(path):
    f = open(path,'r',encoding = 'utf-8')
    lines = f.readlines()
    lines = [i.strip() for i in lines]
    return lines

def read_json_lines(path):
    import json
    lines = read_txt_lines(path)
    data = []
    for i in lines:
        data_dict = json.loads(i)
        data.append(data_dict)
    return data

def read_xlsx_file(path):
    pass

def write_txt_file(path,lines):
    f = open(path,'w',encoding = 'utf-8')
    for i in lines:
        f.write(i + '\n')

def write_json_file(path,data):
    import json
    lines = []
    for i in data:
        temp = json.dumps(i)
        lines.append(temp)
    write_txt_file(path, lines)

def write_xlsx_file(path,data):
    pass

def read_fairseq_file(path):
    pass

def write_fairseq_file(path,data):
    pass
