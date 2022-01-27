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

def read_xlsx_file(path,sheet_name = 'Sheet1'):
    import xlrd
    # required xlrd == 1.2.0 else not support excel
    path = "D:\\Python_plus\\data_set\\400.xlsx"
    wb = xlrd.open_workbook(path)
    #按工作簿定位工作表
    sh = wb.sheet_by_name(sheet_name)
    # print(sh.nrows)#有效数据行数
    # print(sh.ncols)#有效数据列数
    # print(sh.cell(0,0).value)#输出第一行第一列的值
    # print(sh.row_values(0))#输出第一行的所有值
    # #将数据和标题组合成字典
    # print(dict(zip(sh.row_values(0),sh.row_values(1))))
    data_dict = []
    for i in range(1,sh.nrows):
        data_dict.append(dict(zip(sh.row_values(0),sh.row_values(i))))
    return data_dict



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

def read_fairseq_file(source_path,target_path):
    source = read_txt_lines(source_path)
    target = read_txt_lines(target_path)
    return source,target

def write_fairseq_file(source_path,target_path,source_data,target_data):
    write_txt_file(source_path, source_data)
    write_txt_file(target_path, target_data)
