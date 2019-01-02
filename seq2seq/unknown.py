1.pickle.dump以二进制保存文件
2.pickle.load以二进制打开文件
3.sorted(self.items(),key=_itemgetter(1),reverse=True) 对self.item()的元素的第1项降序排列
4.<< (位运算符号) x=4 <<1 x=8 是将x的二进制表示左移一位，相当于原数x乘2
5.list1 = ['Google', 'Runoob', 'Taobao']
  list_pop=list1.pop(1)
  print("删除的项为 :", list_pop) ----'Runoob',
  list.pop(obj)
  参数
  obj -- 可选参数，要移除列表元素的索引值，不能超过列表总长度，默认为 index=-1，删除最后一个列表值，返回值为删除项。
6.random.seed( )
    用于指定随机数生成时所用算法开始的整数值。
    1.如果使用相同的seed( )值，则每次生成的随即数都相同；
    2.如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
    3.设置的seed()值仅一次有效
7. assert:
    如果你断言的 语句正确 则什么反应都没有
    但是如果你出错之后 就会报出    AssertionError 并且错误可以自己填写
8. words_count.items() 加载word_count（）中的元素，并以列表的形式展现出来，列表中的元素为元组类型
9. numpy.random.rand(d0, d1, ..., dn)：生成一个[0,1)之间的随机浮点数或N维浮点数组。
   numpy.random.randn(d0, d1, ..., dn)：生成一个浮点数或N维浮点数组，取数范围：正态分布的随机样本数
   numpy.random.standard_normal(size=None)：生产一个浮点数或N维浮点数组，取数范围：标准正态分布随机样本
   numpy.random.randint(low, high=None, size=None, dtype='l')：生成一个整数或N维整数数组，取数范围：若high不为None时，取[low,high)之间随机整数，否则取值[0,low)之间随机整数。
   numpy.random.random_integers(low, high=None, size=None)：生成一个整数或一个N维整数数组，取值范围：若high不为None，则取[low,high]之间随机整数，否则取[1,low]之间随机整数。
   numpy.random.random_sample(size=None)：生成一个[0,1)之间随机浮点数或N维浮点数组。

10. numpy.apply_along_axis()
    函数返回的是一个根据func()函数以及维度axis运算后得到的的数组.
11. numpy.nditer
    迭代对象nditer提供了一种灵活访问一个或者多个数组的方式.
12.#random.getstate() #返回对象捕获发生器的当前内部状态。这个对象可以传递给setstate（）来恢复状态
   #random.setstate(state) #状态应该已经从以前的调用中获得getstate（），
   #以及setstate（）恢复发生器的getstate（）被调用的时候的内部状态。
13.[:,None]

    None表示该维不进行切片，而是将该维整体作为数组元素处理。

    所以，[:,None]的效果就是将二维数组按每行分割，最后形成一个三维数组
14.array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
    arr[:,0] # 取第0列的数据，以行的形式返回的
    out:
    array([0, 4, 8])

    arr[:,:1] # 取第0列的数据，以列的形式返回的
    out:
    array([[0],
           [4],
           [8]])
15.numpy提供了numpy.concatenate((a1,a2,...),axis=0)函数。能够一次完成多个数组的拼接。其中a1,a2,...是数组类型的参数,concatenate()效率更高，适合大规模的数据拼接

16.tf.squeeze(input, squeeze_dims=None, name=None)
给定张量输入，此操作返回相同类型的张量，并删除所有尺寸为1的尺寸。 如果不想删除所有尺寸1尺寸，可以通过指定squeeze_dims来删除特定尺寸1尺寸。
如果不想删除所有大小是1的维度，可以通过squeeze_dims指定。

17.logspac用于创建等比数列(起始点，结束点，数量)
>>> a = np.logspace(0,9,10)
>>> a
array([  1.00000000e+00,   1.00000000e+01,   1.00000000e+02,
         1.00000000e+03,   1.00000000e+04,   1.00000000e+05,
         1.00000000e+06,   1.00000000e+07,   1.00000000e+08,
         1.00000000e+09])


