# 大涨大跌最终冻结运行包

这是冻结后的纯现货运行包。V156 负责大跌，V189 负责大涨；运行时只应用已经冻结的 base index、coverage 和 Development 阈值，不重新遍历候选池或根据 Test 重新选参。

在本目录执行：

    python src/run_extreme_side.py down <现货文件> <输出目录>
    python src/run_extreme_side.py up <现货文件> <输出目录>
