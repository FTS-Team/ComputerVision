# Intelligent Scissors MFC
- Open Image 打开图片
- Show 选择显示图片样式
    - Original image：原图
    - Image with gaussian 5x5 blur：高斯5x5模糊
    - Image with gaussian 9x9 blur：高斯9x9模糊
    - Laplacian zero-crossing image：拉普拉斯交零点特征值图
    - Gradient magnitude image：梯度值图
    - Gradient direction(minimum) image：梯度方向特征值（选最小的）图
    - Cost(minimum) image：选cost值最小的图
- Process 进行处理
    - 鼠标左键：选取seed point
    - 鼠标右键：撤销上一个seed point
    - 鼠标移动：实时显示路径
    - 控制台：显示信息