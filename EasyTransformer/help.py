def file_API():
    f = open("file_utils.py",'r',encoding = 'utf-8')
    lines = f.readlines()
    for i in lines:
        if i[:3] == "def":
            print(i[4:])

def return_file_API():
    print('''read_txt_lines(path):

    read_json_lines(path):

    read_xlsx_file(path,sheet_name = 'Sheet1'):

    write_txt_file(path,lines):

    write_json_file(path,data):

    write_xlsx_file(path,data):

    read_fairseq_file(source_path,target_path):

    write_fairseq_file(source_path,target_path,source_data,target_data):
    ''')


if __name__ == "__main__":
    # execute only if run as a script
    file_API()
#file_API()