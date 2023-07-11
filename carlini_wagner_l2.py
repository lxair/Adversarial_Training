"""The CarliniWagnerL2 attack."""
import torch


INF = float("inf")


def carlini_wagner_l2(
    model_fn,
    x,
    n_classes,
    y=None,
    targeted=False,
    lr=0.005,
    confidence=0,
    clip_min=0,
    clip_max=1,
    initial_const=1e-2, #用于初始化二分搜索的常数项。
    binary_search_steps=5, #二分搜索的步数，用于搜索最小的常数项
    max_iterations=50, #最大迭代次数，表示生成对抗样本的最大迭代次数。
):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644

    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.

    :param model_fn: a callable that takes an input tensor and returns
              the model logits. The logits should be a tensor of shape
              (n_examples, n_classes).
    :param x: input tensor of shape (n_examples, ...), where ... can
              be any arbitrary dimension that is compatible with
              model_fn.
    :param n_classes: the number of classes.
    :param y: (optional) Tensor with true labels. If targeted is true,
              then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when
              crafting adversarial samples. Otherwise, model predictions
              are used as labels to avoid the "label leaking" effect
              (explained in this paper:
              https://arxiv.org/abs/1611.01236). If provide y, it
              should be a 1D tensor of shape (n_examples, ).
              Default is None.
    :param targeted: (optional) bool. Is the attack targeted or
              untargeted? Untargeted, the default, will try to make the
              label incorrect. Targeted will instead try to move in the
              direction of being more like y.
    :param lr: (optional) float. The learning rate for the attack
              algorithm. Default is 5e-3.
    :param confidence: (optional) float. Confidence of adversarial
              examples: higher produces examples with larger l2
              distortion, but more strongly classified as adversarial.
              Default is 0.
    :param clip_min: (optional) float. Minimum float value for
              adversarial example components. Default is 0.
    :param clip_max: (optional) float. Maximum float value for
              adversarial example components. Default is 1.
    :param initial_const: The initial tradeoff-constant to use to tune the
              relative importance of size of the perturbation and
              confidence of classification. If binary_search_steps is
              large, the initial constant is not important. A smaller
              value of this constant gives lower distortion results.
              Default is 1e-2.
    :param binary_search_steps: (optional) int. The number of times we
              perform binary search to find the optimal tradeoff-constant
              between norm of the perturbation and confidence of the
              classification. Default is 5.
    :param max_iterations: (optional) int. The maximum number of
              iterations. Setting this to a larger value will produce
              lower distortion results. Using only a few iterations
              requires a larger learning rate, and will produce larger
              distortion results. Default is 1000.
    """

    def compare(pred, label, is_logits=False):
        """
        A helper function to compare prediction against a label.
        Returns true if the attack is considered successful.

        :param pred: can be either a 1D tensor of logits or a predicted
                class (int).
        :param label: int. A label to compare against.
        :param is_logits: (optional) bool. If True, treat pred as an
                array of logits. Default is False.
        """

        # Convert logits to predicted class if necessary
        if is_logits:
            pred_copy = pred.clone().detach()
            pred_copy[label] += -confidence if targeted else confidence
            pred = torch.argmax(pred_copy)

        return pred == label if targeted else pred != label

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        pred = model_fn(x)
        y = torch.argmax(pred, 1)

    # Initialize some values needed for binary search on const
    lower_bound = [0.0] * len(x)
    upper_bound = [1e10] * len(x)  # 限制对抗干扰的上限和下限范围
    const = x.new_ones(len(x), 1) * initial_const #const是一个与输入x形状相同的张量，每个元素都初始化为initial_const的值。这个张量用于初始化二分搜索的常数项。

    o_bestl2 = [INF] * len(x) #是一个列表，长度为x的数量，初始时每个元素被设置为正无穷大（INF）。它用于记录生成的最佳对抗样本的L2距离。
    o_bestscore = [-1.0] * len(x) #也是一个列表，长度为x的数量，初始时每个元素被设置为-1.0。它用于记录生成的最佳对抗样本的得分（分数）。
    x = torch.clamp(x, clip_min, clip_max)
    ox = x.clone().detach()  # 它用于保存原始的输入图像，以便在需要时进行比较和恢复。
    o_bestattack = x.clone().detach() #它用于保存生成的最佳对抗样本，以便在更新时进行替换。

    # Map images into the tanh-space
    x = (x - clip_min) / (clip_max - clip_min) #归一化，放到0-1之间。
    x = torch.clamp(x, 0, 1) # 裁剪，确保指在0-1之间。
    x = x * 2 - 1 #将值缩放到 [-1, 1] 的范围内，即将值映射到 [-1, 1] 的范围。
    x = torch.arctanh(x * 0.999999) #进行反双曲正切变换，将其转换为位于 (-inf, inf) 范围内的值。该变换有助于保持输入数据的均匀分布，以便在对抗攻击算法中进行优化。

    # Prepare some variables
    modifier = torch.zeros_like(x, requires_grad=True) #义了一个和输入 x 相同大小的张量 modifier，并将其设置为需要梯度计算。这个 modifier 将用于存储对输入 x 进行扰动的值。
    
    # 离散化处理，将收益率映射到整数标签
    returns_min = y.min().item()
    returns_max = y.max().item()
    returns_range = returns_max - returns_min
    y = ((y - returns_min) / returns_range * (n_classes - 1)).to(torch.int)

    y_onehot = torch.nn.functional.one_hot(y.to(torch.int64), n_classes).to(torch.float)#这样的转换将目标标签表示为一个向量，其中只有目标类别对应的位置为1，其他位置为0。

    # Define loss functions and optimizer
    f_fn = lambda real, other, targeted: torch.max(
        ((other - real) if targeted else (real - other)) + confidence,
        torch.tensor(0.0).to(real.device), # confidence是一个常数项。然后将计算结果和零取最大值，使用 torch.max 函数进行比较。
    )
    l2dist_fn = lambda x, y: torch.pow(x - y, 2).sum(list(range(len(x.size())))[1:])#list(range(len(x.size())))[1:] 来忽略批次维度，只计算特征维度上的差的平方和。
    optimizer = torch.optim.Adam([modifier], lr=lr)#l2dist_fn用于计算两个张量之间的 L2 距离。计算它们之间的差的平方和

    # Outer loop performing binary search on const
    for outer_step in range(binary_search_steps):
        # Initialize some values needed for the inner loop
        bestl2 = [INF] * len(x) #bestl2 列表中的每个元素被设置为一个较大的值 INF，以确保在内部循环中找到的最小L2距离可以被更新。
        bestscore = [-1.0] * len(x)  # bestscore 列表中的每个元素被设置为 -1.0，以确保在内部循环中找到的最佳分数可以被更新。

        # Inner loop performing attack iterations
        for i in range(max_iterations):
            # One attack step
            new_x = (torch.tanh(modifier + x) + 1) / 2 #根据 modifier 对输入 x 进行扰动，
            new_x = new_x * (clip_max - clip_min) + clip_min  # 逆归一化的操作，使得数值返回到原始的数值。
            logits = model_fn(new_x) 
            #y_onehot 与模型的输出 logits 进行相乘。这样得到的结果是一个张量，其中每个元素表示对应样本的目标类别得分。沿着第一维度求和，得到每个样本的目标类别得分 real。
            real = torch.sum(y_onehot * logits, 1) 
            other, _ = torch.max((1 - y_onehot) * logits - y_onehot * 1e4, 1) #计算其他类别的最大得分 other。
            #计算对抗攻击的损失函数
            optimizer.zero_grad()
            f = f_fn(real, other, targeted)
            l2 = l2dist_fn(new_x, ox) #计算了对抗样本 new_x 与原始样本 ox 之间的 L2 距离 l2。
            loss = (const * f + l2).sum() #将损失函数定义为 const * f + l2，其中 const 是用于平衡两个项的常数。
            loss.backward()
            optimizer.step()

            # Update best results
            for n, (l2_n, logits_n, new_x_n) in enumerate(zip(l2, logits, new_x)):
                y_n = y[n] #获取样本的真实标签 y_n。
                succeeded = compare(logits_n, y_n, is_logits=True)#检查预测结果是否与真实标签一致，且返回的 succeeded 表示攻击是否成功。
                if l2_n < o_bestl2[n] and succeeded:
                    pred_n = torch.argmax(logits_n)
                    o_bestl2[n] = l2_n
                    o_bestscore[n] = pred_n
                    o_bestattack[n] = new_x_n
                    # l2_n < o_bestl2[n] implies l2_n < bestl2[n] so we modify inner loop variables too
                    bestl2[n] = l2_n
                    bestscore[n] = pred_n
                elif l2_n < bestl2[n] and succeeded:
                    bestl2[n] = l2_n
                    bestscore[n] = torch.argmax(logits_n)

        # Binary search step
        for n in range(len(x)):
            y_n = y[n]
        #如果最佳预测结果与真实标签一致且不为 -1，表示攻击成功，此时将 const[n] 的上界 upper_bound[n] 更新为当前 const[n] 与 upper_bound[n] 的较小值，即缩小 const[n] 的取值范围。
            if compare(bestscore[n], y_n) and bestscore[n] != -1:#并确保最佳预测结果不为 -1（表示攻击成功）。
                # Success, divide const by two
                upper_bound[n] = min(upper_bound[n], const[n])
                if upper_bound[n] < 1e9:
                    const[n] = (lower_bound[n] + upper_bound[n]) / 2
            else:
                # Failure, either multiply by 10 if no solution found yet
                # or do binary search with the known upper bound
                lower_bound[n] = max(lower_bound[n], const[n])
                if upper_bound[n] < 1e9:
                    const[n] = (lower_bound[n] + upper_bound[n]) / 2
                else:
                    const[n] *= 10
        # 上面的代码就是这段代码用于控制对抗攻击中的上界和下界，这里的目标是找到一个合适的 const 值，以控制对抗攻击的强度。const 是一个用于控制损失函数中两个部分（即对抗项和 L2 距离项）权重的常数项。
    return o_bestattack.detach()


if __name__ == "__main__":
    x = torch.clamp(torch.randn(5, 10), 0, 1)
    y = torch.randint(0, 9, (5,))
    model_fn = lambda x: x

    # targeted
    new_x = carlini_wagner_l2(model_fn, x, 10, targeted=True, y=y)
    new_pred = model_fn(new_x)
    new_pred = torch.argmax(new_pred, 1)

    # untargeted
    new_x_untargeted = carlini_wagner_l2(model_fn, x, 10, targeted=False, y=y)
    new_pred_untargeted = model_fn(new_x_untargeted)
    new_pred_untargeted = torch.argmax(new_pred_untargeted, 1)
