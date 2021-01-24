# use this module to draw the pic I want
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
from matplotlib.font_manager import FontProperties
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

sns.set(context='notebook', style='whitegrid', palette='colorblind', font='sans-serif', font_scale=1, color_codes=False, rc=None)

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


def DrawLine(Lines, Names, index, Title,  xlabel_name="X",y_label_name='Y',Colors=None,):
    if Colors==None:
        Colors=['lightseagreen','orange','red','darkblue','green' ]
    plt.rcParams['figure.figsize'] = (12.0, 7.0)
    for i in range(len(Lines)):
        plt.plot(index,Lines[i],Colors[i],label=Names[i],linewidth=0.8)

    English=0
    if English:
        Title_font = {'family': 'fantasy', 'size': 20}
    else:
        Title_font = {'family': "STXinwei", 'size': 20}

    plt.title(Title, Title_font)

    if English:
        font1 = {'family': 'Tahoma','size': 14}
    else:
        font1 = {'family': "FangSong", 'size': 14}
    plt.xlabel('决策概率/‰', font1)
    plt.ylabel('期望收益/元', font1,rotation=90)
    plt.legend()

    show_num=20
    x_len=len(index)/show_num
    x_major_locator = MultipleLocator(x_len)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()

def DrawLineWithSeaBorn(Lines,Names,index,Title):
    #index = pd.date_range("1 1 2000", periods=100,freq="m", name="date")
    Lines=np.array(Lines)
    Lines=np.transpose(Lines)
    wide_df = pd.DataFrame(Lines, index, Names)
    f, ax = plt.subplots(figsize=(12.0, 7.0))
    ax.set_title(Title, fontsize=22, position=(0.5, 1.05))
    #ax.invert_yaxis()
    ax.set_xlabel('X Label', fontsize=18)
    ax.set_ylabel('Y Label', fontsize=18)
    sns.lineplot(data=wide_df)
    plt.show()

def DrawBar(lines,labels,title,x_label='X',y_label='Y'):
    index = [i+1 for i in range(len(lines[0]))]
    bar_width = 0.3  # 条形宽度
    cur_width = 0
    colors = ['tomato','lightseagreen','orange','red','darkblue','green' ]
    for i in range(len(lines)):
        base_index = np.arange(len(index))
        cur_index = base_index + cur_width
        cur_width += bar_width
        plt.bar(cur_index , height=lines[i], width=bar_width, color=colors[i], label=labels[i])
    
    plt.legend()  # 显示图例
    plt.xticks(np.arange(len(index)) + bar_width/2, index)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    plt.ylabel(y_label)  # 纵坐标轴标题
    plt.xlabel(x_label)  # 纵坐标轴标题
    plt.title(title)  # 图形标题
    plt.show()

def DrawScatter(xpos, ypos, names,dot_size=5,xlabel='x', ylabel='y', title='distribute'):
    plt.figure(figsize=(12, 8))
    for i in range(len(xpos)):
        plt.scatter(xpos[i], ypos[i], s=dot_size, label=names[i])
    plt.xlabel(xlabel, fontdict={'size': 16})
    plt.ylabel(ylabel, fontdict={'size': 16})
    plt.title(title, fontdict={'size': 20})
    plt.legend(loc='best')
    plt.show()

def test():
    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    C,S = np.cos(X), np.sin(X)
    DrawLine([C,S],['Cos','Sin'],X,"Test")
    DrawLineWithSeaBorn([C,S],['Cos','Sin'],X,"Sin And Cos")

def test2():
    X = np.linspace(-np.pi, np.pi, 8, endpoint=True)
    C, S = np.cos(X), np.sin(X)
    DrawBar([C,S],['Cos','Sin'],"test")

def test3():
    X = np.linspace(-np.pi, np.pi, 64, endpoint=True)
    C, S = np.cos(X), np.sin(X)
    DrawScatter([X,X],[C,S],["Cos","Sin"])

