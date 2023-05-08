class Calculator:
    def plus(self, x, y):
        return x + y

    def minus(self, x, y):
        return x - y

    def divide(self, x, y):
        if y == 0:
            raise ZeroDivisionError
        return x / y

    def max(self, x, y):
        if x < y:
            return y
        if x > y:
            return x
        if x == y:
            return x

    def min(self, x, y):
        return x if x < y else y
