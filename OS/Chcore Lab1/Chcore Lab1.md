# Chcore Lab1

#### 520030910393 马逸川



首先感谢助教学长的指导。



### 思考题1:

Refrence:[https://developer.arm.com/](https://developer.arm.com/documentation/ddi0595/2021-12/AArch64-Registers/MPIDR-EL1--Multiprocessor-Affinity-Register?lang=en#fieldset_0-63_40)

_start函数开头部分如下:

![](D:\SJTU-Course-Assignments\OS\Chcore Lab1\pictures\Q1_1.png)

其中， `mpidr_el1`寄存器对每个cpu核不同，在多处理器的系统中为调度提供标识。

在`mpidr_el1`的各位中，Aff0(bits[7:0])用于区分各个核心，thread1.1的标识为0，其余均不为0。基于此，在`0x80008`判断最后一位是否为0，若是则进入初始化，否则陷入死循环产生异常中断，暂停执行。



### 练习题2:

添加代码如下:

![image-20221023202717436](C:\Users\YichuanMa\AppData\Roaming\Typora\typora-user-images\image-20221023202717436.png)

由下方几行的代码得知取得的CurrentEL放在x9寄存器中。

![image-20221023203543991](C:\Users\YichuanMa\AppData\Roaming\Typora\typora-user-images\image-20221023203543991.png)



### 练习题3:

添加代码如下:

![image-20221023204702625](C:\Users\YichuanMa\AppData\Roaming\Typora\typora-user-images\image-20221023204702625.png)

这段代码的实现参考了课件和`.Lin_el2`代码段下的内容。

具体完成的思路是：

前两行:将`.Ltarget`位置地址取到x9寄存器中，作为el3的异常状态返回地址，存储于`elr_el3`中。

后两行:将el1状态的程序状态存储到x9中，再存储于`spsr_el3`，从而修改返回后的程序状态。



### 思考题4:

在进入`init_c.c`函数之前，通过以下的代码段设置栈。

![image-20221023164133289](C:\Users\YichuanMa\AppData\Roaming\Typora\typora-user-images\image-20221023164133289.png)

参考`kernel.img`的反汇编。

![image-20221023165744787](C:\Users\YichuanMa\AppData\Roaming\Typora\typora-user-images\image-20221023165744787.png)

在进入`_init.c`之前，为`init_c.c`中的栈boot_cpu_stack提供足够大小的栈空间。

如果不设置，则随程序执行，sp会和存储在这段栈空间的数据发生冲突。



### 思考题5:

参考附录，查看.bss段信息:

![image-20221023171922047](C:\Users\YichuanMa\AppData\Roaming\Typora\typora-user-images\image-20221023171922047.png)

如果未清零.bss段，内核在使用全局变量时可能调用随机的初始值，可能会为变量分配错误的值和大小。



### 练习题6:

阅读`uart.c`中实现的函数，在TODO处填写如下代码:

![image-20221029171822844](C:\Users\YichuanMa\AppData\Roaming\Typora\typora-user-images\image-20221029171822844.png)

初始化UART，取每个字符并输出，结果如下:

![image-20221029171758578](C:\Users\YichuanMa\AppData\Roaming\Typora\typora-user-images\image-20221029171758578.png)

### 练习题7:

![image-20221023211247680](C:\Users\YichuanMa\AppData\Roaming\Typora\typora-user-images\image-20221023211247680.png)

这一段参考下方的几行实现的。可以看到调用`SCTLR_EL1`寄存器各个字段的方式，参考得出调用M字段的代码。

![image-20221023211906834](C:\Users\YichuanMa\AppData\Roaming\Typora\typora-user-images\image-20221023211906834.png)

