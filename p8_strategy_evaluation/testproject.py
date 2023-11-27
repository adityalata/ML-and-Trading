from experiment1 import exp1
from experiment2 import exp2
from ManualStrategy import ManualStrategy

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "alata6"  # replace tb34 with your Georgia Tech username


if __name__ == "__main__":
    manualStrategy = ManualStrategy()
    manualStrategy.compare_manual_strategy_with_benchmark()
    exp1()
    exp2()
