# 智能环保项目案例：基于神经网络的垃圾分类
![](https://github.com/KXCY-AI/AI-Case-Studies/blob/main/%E5%9F%BA%E4%BA%8E%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E5%9E%83%E5%9C%BE%E5%88%86%E7%B1%BB/img/1.png)

垃圾分类，一般是指按一定规定或标准将垃圾分类储存、分类投放和分类搬运，从而转变成公共资源的一系列活动的总称。分类的目的是提高垃圾的资源价值和经济价值，力争物尽其用。垃圾分为四类，分别是：有害垃圾、厨余垃圾、可回收物和其他垃圾，其对应的垃圾桶颜色分别是红、绿、蓝、黑。

本次项目使用的数据集包括硬纸板、玻璃制品、金属、纸制品、塑料和废物垃圾等 6 种常见的生活垃圾共计 2,247 张。项目分为三部分。数据处理、模型建立和模型测试。经过迭代模型在训练集上准确率达到 89.89% 左右，在验证集上准确率达到 77.68% 左右。

本案例适合作为深度学习实践课程配套教学案例，能够达到以下教学效果：

- **提升学生数据预处理能力：** 通过对样本图片进行缩放、翻转、选择等操作对样本进行扩充，使得样本数据更加丰富，有利于模型性能的提高。
- **提升学生深度学习建模的能力：** 提升学生通过 TensorFlow 框架建立深度神经网络模型的能力，通过相对直观的 tf.keras 序贯模型建立垃圾分类的卷积神经网络模型，并进行编译和训练。。
- **帮助学生掌握模型评估常用手段：** 通过数据绘图，实现模型评估常用指标：准确率和损失值的可视化评估。
- **帮助学生掌握模型测试与推理方法：** 通过对完成训练后的模型进行部署与加载，使用训练/测试集以外的第三方图片进行推理验证。

## 案例依赖库
本案例主要采用 tf.keras 进行 TensorFlow 深度神经网络模型构建。Keras 最初是由 Google AI 开发人员/研究人员 Francois Chollet 创建并开发的。Francois 于 2015 年 3 月 27 日将 Keras 的第一个版本 commit 并 release 到他的 [GitHub](https://github.com/fchollet)。一开始，Francois 开发 Keras 是为了方便他自己的研究和实验。但是，随着深度学习的普及，许多开发人员、程序员和机器学习从业人员都因其易于使用的 API 而涌向 Keras。

同时，为了训练自己自定义的神经网络，Keras 需要一个后端。后端是一个计算引擎——它可以构建网络的图和拓扑结构，运行优化器，并执行具体的数字运算。要理解后端的概念，可以试想你需要从头开始构建一个网站。你可以使用 PHP 编程语言和 SQL 数据库。这个 SQL 数据库就是是后端。你可以使用 MySQL，PostgreSQL 或者 SQL Server 作为你的数据库；但是，用于与数据库交互的 PHP 代码是不会变的。PHP 并不关心正在使用哪个数据库，只要它符合 PHP 的规则即可。Keras 也是如此。你可以把后台看作是你的数据库，Keras 是你用来访问数据库的编程语言。你可以把后端替换成任何你喜欢的后端，只要它遵守某些规则，你的代码就不需要更改。因此，你可以把 Keras 看作是一组用来简化深度学习操作的封装（Abstraction）。在 v1.1.0 之前，Keras 的默认后端都是 Theano。与此同时，Google 发布了 TensorFlow，这是一个用于机器学习和神经网络训练的符号数学库。Keras 开始支持 TensorFlow 作为后端。渐渐地，TensorFlow 成为最受欢迎的后端，这也就使得 TensorFlow 从 Keras v1.1.0 发行版开始成为 Keras 的默认后端。

![](https://github.com/KXCY-AI/AI-Case-Studies/blob/main/%E5%9F%BA%E4%BA%8E%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E5%9E%83%E5%9C%BE%E5%88%86%E7%B1%BB/img/tf.keras.png)

当谷歌在 2019 年 6 月发布 TensorFlow 2.0 时，他们宣布 Keras 现在是 TensorFlow 的官方高阶 API，用于快速简单的模型设计和训练。另一方面，随着 Keras 2.3.0 的发布，[Francois 声明](https://github.com/keras-team/keras/releases/tag/2.3.0)：

- 这是 Keras 首个与 tf.keras 同步的版本
- 这也是 Keras 支持多个后端（即 Theano，CNTK 等）的最终版本
- 最重要的是，所有深度学习从业人员都应将其代码转换成 TensorFlow 2.0 和 tf.keras 软件包
- 原始的 keras 软件包仍会接收 bug 并修复，但请向前看，你应该开始使用 tf.keras 了

因此，在本案例中，我们直接使用 tf.keras 进行 TensorFlow 神经网络的构造：

```tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import SGD
```


原创制作：广州跨象乘云软件技术有限公司（版权所有，不得转载）

公司网站：https://www.080910t.com/
