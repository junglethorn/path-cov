import calculator


class CalculatorTest:
    def __init__(self):
        self.myCal = calculator.Calculator()

    def testPlus(self):
        result = self.myCal.plus(9, 12)
        assert result == 21, "result is 21"
        result = self.myCal.plus(12, 9)
        assert result == 21, "result is 21"
        result = self.myCal.plus(21, 12)
        assert result == 33, "result is 33"
        result = self.myCal.plus(21, -1)
        assert result == 20, "result is 20"
        result = self.myCal.plus(-1, -1)
        assert result == -2, "result is -2"
        result = self.myCal.plus(9, 12)
        assert result == 21, "result is 21"
        result = self.myCal.plus(12, 9)
        assert result == 21, "result is 21"
        result = self.myCal.plus(21, 12)
        assert result == 33, "result is 33"
        result = self.myCal.plus(21, -1)
        assert result == 20, "result is 20"
        result = self.myCal.plus(-1, -1)
        assert result == -2, "result is -2"

    def testMax(self):
        x = 9
        y = 12
        result = self.myCal.max(x, y)
        assert result == 12, "result is 12"

    def testMin(self):
        x = 9
        y = 12
        result = self.myCal.min(x, y)
        assert result == 9, "result is 9"


def main():
    test = CalculatorTest()
    test.testPlus()
    test.testMax()
    return


if __name__ == '__main__':
    main()
