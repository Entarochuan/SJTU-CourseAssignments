## OS Lab3 

#### 520030910393 马逸川



### 练习题1: 

内核从完成必要的初始化到用户态程序的过程是怎么样的？尝试描述一下调用关系。

参考`main.c`代码，在这一过程中，首先调用`create_root_thread();`创建线程，`eret_to_thread`切换到选中的线程。



### 练习题2:
